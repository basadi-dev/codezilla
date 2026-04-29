mod discovery;
mod turn;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock as AsyncRwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::llm::client::UnifiedClient;
use crate::llm::LlmClient;

// Agent subsystem — types used only internally in this file
use super::agent::model_gateway::build_compaction_messages;
use super::agent::supervisor::{AgentSupervisor, SpawnAgentToolProviderReal};
use super::agent::{
    ApprovalManager, BashToolProvider, EventBus, ExtensionManager, FileToolProvider,
    ImageToolProvider, ListDirToolProvider, ModelGateway, PermissionManager,
    RequestUserInputToolProvider, SandboxManager, SearchToolProvider, ShellToolProvider,
    ToolOrchestrator, WebToolProvider,
};
use super::intel::RepoMap;
// Agent types re-exported for callers outside runtime.rs
#[allow(unused_imports)]
pub use super::agent::{AutoReviewer, EventFilter, EventSubscription, ModelDescription};

use super::config::EffectiveConfig;
use super::domain::{
    now_millis, now_seconds, AccountSession, ApprovalPolicy, CompactionStrategy, ConversationItem,
    ItemId, ItemKind, MemoryMode, ModelSettings, PathString, PermissionProfile, PersistedThread,
    RuntimeEvent, RuntimeEventKind, SessionId, ThreadFilter, ThreadId, ThreadMetadata,
    ThreadStatus, ToolCall, ToolCallId, TurnId, TurnMetadata, TurnStatus, UserInput,
};
use super::persistence::PersistenceManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadStartParams {
    pub cwd: Option<PathString>,
    pub model_settings: Option<ModelSettings>,
    pub approval_policy: Option<ApprovalPolicy>,
    pub permission_profile: Option<PermissionProfile>,
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadStartResult {
    pub metadata: ThreadMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadResumeParams {
    pub thread_id: ThreadId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadResumeResult {
    pub metadata: ThreadMetadata,
    #[serde(default)]
    pub turns: Vec<TurnMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadForkParams {
    pub thread_id: ThreadId,
    pub ephemeral: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadForkResult {
    pub metadata: ThreadMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReadParams {
    pub thread_id: ThreadId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadReadResult {
    pub thread: PersistedThread,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListParams {
    pub cwd: Option<PathString>,
    pub archived: Option<bool>,
    pub search_term: Option<String>,
    pub limit: Option<i32>,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListResult {
    #[serde(default)]
    pub threads: Vec<ThreadMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadCompactParams {
    pub thread_id: ThreadId,
    pub strategy: CompactionStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadCompactResult {
    pub thread_id: ThreadId,
    pub summary_item_id: Option<ItemId>,
    pub items_removed: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadRollbackParams {
    pub thread_id: ThreadId,
    pub to_turn_id: TurnId,
    pub discard: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadRollbackResult {
    pub thread_id: ThreadId,
    pub turns_removed: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMemoryModeParams {
    pub thread_id: ThreadId,
    pub memory_mode: MemoryMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnStartParams {
    pub thread_id: ThreadId,
    #[serde(default)]
    pub input: Vec<UserInput>,
    pub cwd: Option<PathString>,
    pub model_settings: Option<ModelSettings>,
    pub approval_policy: Option<ApprovalPolicy>,
    pub permission_profile: Option<PermissionProfile>,
    pub output_schema: Option<Value>,
    /// Agent nesting depth — 0 for top-level, incremented by spawn_agent.
    #[serde(default)]
    pub agent_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnStartResult {
    pub turn: TurnMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnInterruptParams {
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnInterruptResult {
    pub turn_id: TurnId,
    pub interrupted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnSteerParams {
    pub thread_id: ThreadId,
    pub expected_turn_id: TurnId,
    #[serde(default)]
    pub input: Vec<UserInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnSteerResult {
    pub turn_id: TurnId,
    pub queued_items: usize,
}

// ModelDescription, ModelRequest, ModelStreamEvent, ModelGateway and
// ExtensionManager live in agent/model_gateway.rs and agent/extensions.rs —
// imported at the top of this file via `super::agent::{...}`.
// TurnExecutor lives in agent/executor.rs.

// ─── Session state (runtime-internal) ────────────────────────────────────────

pub(crate) struct LoadedTurn {
    pub(crate) metadata: TurnMetadata,
    pub(crate) items: Vec<ConversationItem>,
    pub(crate) status: TurnStatus,
    #[allow(dead_code)]
    pub(crate) pending_tool_calls: HashMap<ToolCallId, ToolCall>,
    #[allow(dead_code)]
    pub(crate) stream_buffer: Vec<RuntimeEvent>,
    pub(crate) cancel_token: CancellationToken,
}

pub(crate) struct ThreadSession {
    pub(crate) metadata: ThreadMetadata,
    pub(crate) turns: Vec<LoadedTurn>,
    pub(crate) active_turn_id: Option<TurnId>,
    #[allow(dead_code)]
    pub(crate) subscribed_clients: HashSet<String>,
    #[allow(dead_code)]
    pub(crate) owner_connection_id: Option<String>,
    pub(crate) pending_steering: Vec<UserInput>,
    /// Live approval-policy override; updated by the TUI without restarting the turn.
    pub(crate) approval_policy_override: Option<ApprovalPolicy>,
    /// Items that don't belong to any real turn (e.g. ReasoningSummary produced by
    /// compact_thread with a synthetic turn_id like "compaction"). These are prepended
    /// to the context on every turn so the model always sees the compaction summary.
    pub(crate) prefix_items: Vec<ConversationItem>,
}

// ─── RuntimeInner ─────────────────────────────────────────────────────────────

pub(crate) struct RuntimeInner {
    #[allow(dead_code)]
    pub(crate) runtime_id: SessionId,
    pub(crate) effective_config: EffectiveConfig,
    #[allow(dead_code)]
    pub(crate) account_session: AccountSession,
    pub(crate) loaded_threads: AsyncRwLock<HashMap<ThreadId, Arc<Mutex<ThreadSession>>>>,
    pub(crate) event_bus: EventBus,
    pub(crate) tool_orchestrator: ToolOrchestrator,
    pub(crate) permission_manager: Arc<PermissionManager>,
    pub(crate) approval_manager: Arc<ApprovalManager>,
    pub(crate) persistence_manager: Arc<PersistenceManager>,
    pub(crate) model_gateway: Arc<ModelGateway>,
    pub(crate) extension_manager: Arc<ExtensionManager>,
    /// Codebase intelligence: repo map builder + SHA2-keyed symbol cache.
    pub(crate) repo_map: Arc<RepoMap>,
}

// ─── ConversationRuntime (thin coordinator) ───────────────────────────────────

#[derive(Clone)]
pub struct ConversationRuntime {
    pub(crate) inner: Arc<RuntimeInner>,
}

impl ConversationRuntime {
    pub async fn new(
        effective_config: EffectiveConfig,
        account_session: AccountSession,
    ) -> Result<Self> {
        let llm_client: Arc<dyn LlmClient> = Arc::new(
            UnifiedClient::new(effective_config.llm.clone())
                .map_err(|e| anyhow!("llm_client_init_failed: {e}"))?,
        );
        Self::new_with_llm_client(effective_config, account_session, llm_client).await
    }

    /// Construct a runtime with a caller-supplied `LlmClient`.
    ///
    /// Used by the fake-model test harness to drive deterministic agent loops
    /// without contacting a real LLM provider.
    pub async fn new_with_llm_client(
        effective_config: EffectiveConfig,
        account_session: AccountSession,
        llm_client: Arc<dyn LlmClient>,
    ) -> Result<Self> {
        let persistence = Arc::new(PersistenceManager::new(
            std::path::Path::new(&effective_config.app_home).join("state"),
            std::path::Path::new(&effective_config.app_home).join("memories"),
            std::path::Path::new(&effective_config.app_home).join("logs"),
        )?);

        let sandbox = Arc::new(SandboxManager::new());
        let permissions = Arc::new(PermissionManager::new(&effective_config.trusted_projects));
        let tool_orchestrator = ToolOrchestrator::new();
        tool_orchestrator.register_provider(Arc::new(ShellToolProvider::new(
            sandbox.clone(),
            permissions.clone(),
        )));
        tool_orchestrator.register_provider(Arc::new(BashToolProvider::new(
            sandbox.clone(),
            permissions.clone(),
        )));
        tool_orchestrator.register_provider(Arc::new(ListDirToolProvider));

        // Build the intel cache early so FileToolProvider can invalidate entries
        // after write operations.
        let repo_map = Arc::new(RepoMap::new(200));
        let intel_cache = repo_map.cache();

        tool_orchestrator.register_provider(Arc::new(
            FileToolProvider::new(sandbox.clone(), permissions.clone())
                .with_intel_cache(intel_cache),
        ));
        tool_orchestrator.register_provider(Arc::new(SearchToolProvider));
        tool_orchestrator.register_provider(Arc::new(ImageToolProvider));
        // NOTE: SpawnAgentToolProvider is registered *after* Self is constructed (late registration)
        // so it can hold a ConversationRuntime clone. See below.
        tool_orchestrator.register_provider(Arc::new(RequestUserInputToolProvider));
        tool_orchestrator.register_provider(Arc::new(WebToolProvider::new()));

        let mcp_registry = Arc::new(super::mcp::McpRegistry::new());
        for srv_cfg in &effective_config.mcp_servers {
            if let Err(e) = mcp_registry.start_server(srv_cfg.clone()).await {
                // Log and continue
                eprintln!("Failed to start MCP server {}: {}", srv_cfg.name, e);
            }
        }
        tool_orchestrator.register_provider(mcp_registry);

        let extensions = Arc::new(ExtensionManager::new());
        extensions
            .reload_all(&effective_config.working_directory)
            .await?;

        // ── Build Self first so the runtime handle is available ────────────
        // SpawnAgentToolProvider needs a ConversationRuntime clone, so we
        // construct Self before registering it, then swap the stub out.
        let inner = RuntimeInner {
            runtime_id: format!("runtime_{}", Uuid::new_v4().simple()),
            effective_config,
            account_session,
            loaded_threads: AsyncRwLock::new(HashMap::new()),
            event_bus: EventBus::new(),
            tool_orchestrator,
            permission_manager: permissions,
            approval_manager: Arc::new(ApprovalManager::new(
                super::domain::ApprovalsReviewerKind::User,
            )),
            persistence_manager: persistence,
            model_gateway: Arc::new(ModelGateway::new(llm_client)),
            extension_manager: extensions,
            // Re-use the repo_map that shares its cache with FileToolProvider
            // so write invalidations are visible to the map builder.
            repo_map,
        };
        let me = Self {
            inner: Arc::new(inner),
        };

        let agent_cfg = &me.inner.effective_config.agent;
        tracing::debug!(
            max_iterations = agent_cfg.max_iterations,
            max_consecutive_failures = agent_cfg.max_consecutive_failures,
            max_no_tool_nudges = agent_cfg.max_no_tool_nudges,
            max_consecutive_read_only_rounds = agent_cfg.max_consecutive_read_only_rounds,
            max_empty_responses = agent_cfg.max_empty_responses,
            max_total_nudges = agent_cfg.max_total_nudges,
            max_response_chars = agent_cfg.max_response_chars,
            max_child_agents = agent_cfg.max_child_agents,
            max_spawn_depth = agent_cfg.max_spawn_depth,
            child_timeout_secs = agent_cfg.child_timeout_secs,
            max_child_timeout_secs = agent_cfg.max_child_timeout_secs,
            "runtime: agent config loaded"
        );

        // Late-register the real SpawnAgentToolProvider with a runtime clone.
        me.inner
            .tool_orchestrator
            .register_provider(Arc::new(SpawnAgentToolProviderReal::new(
                AgentSupervisor::new(me.clone(), me.inner.effective_config.agent.max_child_agents),
            )));

        Ok(me)
    }

    #[allow(dead_code)]
    pub fn runtime_id(&self) -> &str {
        &self.inner.runtime_id
    }

    pub fn effective_config(&self) -> &EffectiveConfig {
        &self.inner.effective_config
    }

    pub fn event_bus(&self) -> &EventBus {
        &self.inner.event_bus
    }

    pub async fn start_thread(&self, params: ThreadStartParams) -> Result<ThreadStartResult> {
        let model = params
            .model_settings
            .unwrap_or_else(|| self.inner.effective_config.model_settings.clone());
        let cwd = params
            .cwd
            .unwrap_or_else(|| self.inner.effective_config.working_directory.clone());
        let metadata = ThreadMetadata {
            thread_id: format!("thread_{}", Uuid::new_v4().simple()),
            title: None,
            created_at: now_seconds(),
            updated_at: now_seconds(),
            cwd: Some(cwd.clone()),
            model_id: model.model_id.clone(),
            provider_id: model.provider_id.clone(),
            status: ThreadStatus::Idle,
            forked_from_id: None,
            archived: false,
            ephemeral: params.ephemeral,
            memory_mode: MemoryMode::Enabled,
        };
        self.inner.persistence_manager.create_thread(&metadata)?;

        let thread = Arc::new(Mutex::new(ThreadSession {
            metadata: metadata.clone(),
            turns: Vec::new(),
            active_turn_id: None,
            subscribed_clients: HashSet::new(),
            owner_connection_id: None,
            pending_steering: Vec::new(),
            approval_policy_override: params.approval_policy.clone(),
            prefix_items: Vec::new(),
        }));
        self.inner
            .loaded_threads
            .write()
            .await
            .insert(metadata.thread_id.clone(), thread);

        self.publish_event(
            RuntimeEventKind::ThreadStarted,
            Some(metadata.thread_id.clone()),
            None,
            serde_json::to_value(&metadata)?,
        )
        .await?;

        Ok(ThreadStartResult { metadata })
    }

    pub async fn resume_thread(&self, params: ThreadResumeParams) -> Result<ThreadResumeResult> {
        let persisted = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;
        self.load_persisted_thread(&persisted).await?;
        Ok(ThreadResumeResult {
            metadata: persisted.metadata,
            turns: persisted.turns,
        })
    }

    pub async fn fork_thread(&self, params: ThreadForkParams) -> Result<ThreadForkResult> {
        let source = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;
        let new_id = format!("thread_{}", Uuid::new_v4().simple());
        let metadata = ThreadMetadata {
            thread_id: new_id.clone(),
            title: source.metadata.title.clone(),
            created_at: now_seconds(),
            updated_at: now_seconds(),
            cwd: source.metadata.cwd.clone(),
            model_id: source.metadata.model_id.clone(),
            provider_id: source.metadata.provider_id.clone(),
            status: ThreadStatus::Idle,
            forked_from_id: Some(source.metadata.thread_id.clone()),
            archived: false,
            ephemeral: params.ephemeral,
            memory_mode: source.metadata.memory_mode,
        };
        self.inner.persistence_manager.create_thread(&metadata)?;

        let mut turn_map = HashMap::new();
        for turn in source.turns {
            let new_turn = TurnMetadata {
                turn_id: format!("turn_{}", Uuid::new_v4().simple()),
                thread_id: new_id.clone(),
                created_at: turn.created_at,
                updated_at: turn.updated_at,
                status: turn.status,
                started_by_surface: turn.started_by_surface,
                token_usage: turn.token_usage,
            };
            turn_map.insert(turn.turn_id, new_turn.clone());
            self.inner.persistence_manager.create_turn(&new_turn)?;
            self.inner.persistence_manager.update_turn(&new_turn)?;
        }
        for item in source.items {
            let new_item = ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: new_id.clone(),
                turn_id: turn_map
                    .get(&item.turn_id)
                    .map(|t| t.turn_id.clone())
                    .unwrap_or_else(|| item.turn_id.clone()),
                created_at: item.created_at,
                kind: item.kind,
                payload: item.payload,
            };
            self.inner.persistence_manager.append_item(&new_item)?;
        }

        self.resume_thread(ThreadResumeParams {
            thread_id: new_id.clone(),
        })
        .await?;
        Ok(ThreadForkResult { metadata })
    }

    pub async fn read_thread(&self, params: ThreadReadParams) -> Result<ThreadReadResult> {
        Ok(ThreadReadResult {
            thread: self
                .inner
                .persistence_manager
                .read_thread(&params.thread_id)?,
        })
    }

    pub async fn list_threads(&self, params: ThreadListParams) -> Result<ThreadListResult> {
        Ok(ThreadListResult {
            threads: self.inner.persistence_manager.list_threads(ThreadFilter {
                cwd: params.cwd,
                archived: params.archived,
                search_term: params.search_term,
                limit: params.limit.unwrap_or(20),
                cursor: params.cursor,
            })?,
        })
    }

    pub async fn compact_thread(&self, params: ThreadCompactParams) -> Result<ThreadCompactResult> {
        // ── 1. Read the full thread from persistence ──────────────────────────
        let persisted = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;

        let item_count_before = persisted.items.len();
        if item_count_before == 0 {
            return Ok(ThreadCompactResult {
                thread_id: params.thread_id,
                summary_item_id: None,
                items_removed: 0,
            });
        }

        // ── 2. Build system instructions (same as a normal turn) ───────────────
        let cwd = persisted
            .metadata
            .cwd
            .clone()
            .unwrap_or_else(|| self.inner.effective_config.working_directory.clone());
        let skills = self.inner.extension_manager.list_skills(&cwd).await;
        let mut system_instructions = vec![self.inner.effective_config.system_prompt.clone()];
        for skill in skills {
            if skill.enabled {
                system_instructions.push(format!("Skill {}: {}", skill.name, skill.description));
            }
        }

        // ── 3. Ask the LLM to summarise ────────────────────────────────────────
        // Use the thread's configured model for compaction.
        let model_settings = ModelSettings {
            model_id: persisted.metadata.model_id.clone(),
            provider_id: persisted.metadata.provider_id.clone(),
            reasoning_effort: None, // no extended thinking needed for summaries
            summary_mode: None,
            service_tier: None,
            web_search_enabled: false,
            context_window: self.inner.effective_config.model_settings.context_window,
        };

        let compaction_messages = build_compaction_messages(&system_instructions, &persisted.items);

        let llm_response = self
            .inner
            .model_gateway
            .inner_client()
            .complete(
                &model_settings.provider_id,
                &compaction_messages,
                &[], // no tools needed for summarisation
                &model_settings.model_id,
                0.2,
                None,
                2048,
            )
            .await
            .map_err(|e| anyhow!("compact_thread_llm_failed: {e}"))?;

        let summary_text = if llm_response.content.trim().is_empty() {
            "[Compaction summary unavailable]".to_string()
        } else {
            llm_response.content.trim().to_string()
        };

        // ── 4. Tombstone all existing items ────────────────────────────────────
        let items_removed = self
            .inner
            .persistence_manager
            .tombstone_all_items(&params.thread_id)? as i32;

        // ── 5. Insert the summary as a single ReasoningSummary item ────────────
        let summary_item_id = format!("item_{}", Uuid::new_v4().simple());
        let summary_item = ConversationItem {
            item_id: summary_item_id.clone(),
            thread_id: params.thread_id.clone(),
            turn_id: "compaction".into(),
            created_at: now_seconds(),
            kind: ItemKind::ReasoningSummary,
            payload: json!({
                "text": summary_text,
                "strategy": params.strategy,
                "items_compacted": items_removed,
            }),
        };
        self.inner.persistence_manager.append_item(&summary_item)?;

        // ── 6. Reload the in-memory session so the live thread sees the summary ─
        {
            let mut loaded = self.inner.loaded_threads.write().await;
            loaded.remove(&params.thread_id);
        }
        // Warm the cache again so the next turn picks up the new state.
        let _ = self.load_thread(&params.thread_id).await;

        self.publish_event(
            RuntimeEventKind::ItemCompleted,
            Some(params.thread_id.clone()),
            Some(summary_item.turn_id.clone()),
            serde_json::to_value(&summary_item)?,
        )
        .await?;

        Ok(ThreadCompactResult {
            thread_id: params.thread_id,
            summary_item_id: Some(summary_item_id),
            items_removed,
        })
    }

    pub async fn rollback_thread(
        &self,
        params: ThreadRollbackParams,
    ) -> Result<ThreadRollbackResult> {
        let persisted = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;
        let turns_removed = persisted
            .turns
            .iter()
            .filter(|turn| {
                turn.created_at
                    > persisted
                        .turns
                        .iter()
                        .find(|target| target.turn_id == params.to_turn_id)
                        .map(|t| t.created_at)
                        .unwrap_or(i64::MAX)
            })
            .count() as i32;
        Ok(ThreadRollbackResult {
            thread_id: params.thread_id,
            turns_removed,
        })
    }

    pub async fn set_thread_memory_mode(&self, params: ThreadMemoryModeParams) -> Result<()> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        {
            let mut thread = thread.lock().await;
            thread.metadata.memory_mode = params.memory_mode;
            thread.metadata.updated_at = now_seconds();
            self.inner
                .persistence_manager
                .update_thread(&thread.metadata)?;
        }
        Ok(())
    }

    /// Update the live approval-policy override for a running thread so that
    /// mid-turn policy changes (e.g. toggling auto-approve) take effect on the
    /// next tool-call evaluation without restarting the turn.
    pub async fn set_thread_approval_policy(
        &self,
        thread_id: &str,
        policy: Option<ApprovalPolicy>,
    ) -> Result<()> {
        if let Some(thread) = self.load_thread(thread_id).await? {
            let mut guard = thread.lock().await;
            guard.approval_policy_override = policy;
        }
        Ok(())
    }

    pub async fn archive_thread(&self, thread_id: &str) -> Result<()> {
        self.inner.persistence_manager.archive_thread(thread_id)?;
        if let Some(thread) = self.load_thread(thread_id).await? {
            let mut thread = thread.lock().await;
            thread.metadata.archived = true;
            thread.metadata.status = ThreadStatus::Archived;
            thread.metadata.updated_at = now_seconds();
        }
        Ok(())
    }

    pub(crate) async fn load_thread(
        &self,
        thread_id: &str,
    ) -> Result<Option<Arc<Mutex<ThreadSession>>>> {
        if let Some(thread) = self
            .inner
            .loaded_threads
            .read()
            .await
            .get(thread_id)
            .cloned()
        {
            return Ok(Some(thread));
        }
        if let Ok(persisted) = self.inner.persistence_manager.read_thread(thread_id) {
            let thread = self.load_persisted_thread(&persisted).await?;
            return Ok(Some(thread));
        }
        Ok(None)
    }

    async fn load_persisted_thread(
        &self,
        persisted: &PersistedThread,
    ) -> Result<Arc<Mutex<ThreadSession>>> {
        if let Some(thread) = self
            .inner
            .loaded_threads
            .read()
            .await
            .get(&persisted.metadata.thread_id)
            .cloned()
        {
            return Ok(thread);
        }

        let mut turns_map: HashMap<String, LoadedTurn> = persisted
            .turns
            .iter()
            .map(|turn| {
                (
                    turn.turn_id.clone(),
                    LoadedTurn {
                        metadata: turn.clone(),
                        items: Vec::new(),
                        status: turn.status,
                        pending_tool_calls: HashMap::new(),
                        stream_buffer: Vec::new(),
                        cancel_token: CancellationToken::new(),
                    },
                )
            })
            .collect();

        // Items whose turn_id does not match any real turn (e.g. the ReasoningSummary
        // written by compact_thread with turn_id "compaction") are collected into
        // prefix_items so the executor can prepend them to the model context.
        let mut prefix_items: Vec<ConversationItem> = Vec::new();
        for item in &persisted.items {
            if let Some(turn) = turns_map.get_mut(&item.turn_id) {
                turn.items.push(item.clone());
            } else {
                prefix_items.push(item.clone());
            }
        }

        let thread = Arc::new(Mutex::new(ThreadSession {
            metadata: persisted.metadata.clone(),
            turns: turns_map.into_values().collect(),
            active_turn_id: None,
            subscribed_clients: HashSet::new(),
            owner_connection_id: None,
            pending_steering: Vec::new(),
            approval_policy_override: None,
            prefix_items,
        }));
        self.inner
            .loaded_threads
            .write()
            .await
            .insert(persisted.metadata.thread_id.clone(), thread.clone());
        Ok(thread)
    }

    pub(crate) async fn publish_event(
        &self,
        kind: RuntimeEventKind,
        thread_id: Option<ThreadId>,
        turn_id: Option<TurnId>,
        payload: Value,
    ) -> Result<()> {
        let sequence = if let Some(thread_id) = &thread_id {
            self.inner
                .persistence_manager
                .bump_thread_sequence(thread_id)?
        } else {
            0
        };
        self.inner.event_bus.publish(RuntimeEvent {
            event_id: format!("evt_{}", Uuid::new_v4().simple()),
            kind,
            thread_id,
            turn_id,
            sequence,
            payload,
            emitted_at: now_millis(),
        });
        Ok(())
    }

    pub async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.inner.loaded_threads.write().await.remove(thread_id);
        self.inner.persistence_manager.delete_thread(thread_id)?;
        Ok(())
    }
}

// ─── Fake-model end-to-end harness tests ──────────────────────────────────────

#[cfg(test)]
mod fake_model_tests {
    use super::*;
    use crate::system::agent::fake_model::{FakeLlmClient, ScriptedResponse};
    use crate::system::config::{AgentConfig, AutoCompactionConfig, EffectiveConfig, LlmConfig};
    use crate::system::domain::{
        ApprovalDecision, ApprovalPolicy, ApprovalsReviewerKind, ItemKind, ModelSettings,
        PermissionProfile, SandboxMode, SurfaceKind, UserInput,
    };
    use crate::system::intel::CodebaseIntelConfig;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    /// Per-test scratch space under the OS temp directory.
    fn unique_app_home() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "codezilla-test-{}-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
            n
        ));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        dir
    }

    fn test_config(app_home: &std::path::Path) -> EffectiveConfig {
        let app_home_str = app_home.to_string_lossy().to_string();
        EffectiveConfig {
            app_home: app_home_str.clone(),
            sqlite_home: app_home.join("state").to_string_lossy().to_string(),
            model_settings: ModelSettings {
                model_id: "fake-model".into(),
                provider_id: "fake".into(),
                reasoning_effort: None,
                summary_mode: None,
                service_tier: None,
                web_search_enabled: false,
                context_window: Some(32_000),
            },
            approval_policy: ApprovalPolicy::default(),
            approvals_reviewer: ApprovalsReviewerKind::User,
            permission_profile: PermissionProfile {
                sandbox_mode: SandboxMode::WorkspaceWrite,
                writable_roots: Vec::new(),
                network_enabled: false,
                allowed_domains: Vec::new(),
                allowed_unix_sockets: Vec::new(),
            },
            add_dirs: Vec::new(),
            notifications_enabled: false,
            mcp_servers: Vec::new(),
            plugins_enabled: false,
            apps_enabled: false,
            features: Default::default(),
            trusted_projects: vec![app_home_str.clone()],
            working_directory: app_home_str,
            system_prompt: String::new(),
            llm: LlmConfig::default(),
            log_level: "off".into(),
            log_file: app_home.join("test.log").to_string_lossy().to_string(),
            models: Vec::new(),
            auto_compaction: AutoCompactionConfig {
                enabled: false,
                ..Default::default()
            },
            codebase_intel: CodebaseIntelConfig {
                enabled: false,
                ..Default::default()
            },
            agent: AgentConfig {
                // Tighten limits so loop tests fail fast.
                max_iterations: 8,
                max_empty_responses: 2,
                ..Default::default()
            },
        }
    }

    async fn build_runtime(script: Vec<ScriptedResponse>) -> (ConversationRuntime, PathBuf) {
        build_runtime_with(script, |_| {}).await
    }

    /// Build a runtime, applying `tweak` to the test config before construction.
    async fn build_runtime_with(
        script: Vec<ScriptedResponse>,
        tweak: impl FnOnce(&mut EffectiveConfig),
    ) -> (ConversationRuntime, PathBuf) {
        let app_home = unique_app_home();
        let mut cfg = test_config(&app_home);
        tweak(&mut cfg);
        let client = Arc::new(FakeLlmClient::new(script));
        let runtime =
            ConversationRuntime::new_with_llm_client(cfg, AccountSession::default(), client)
                .await
                .expect("runtime build");
        (runtime, app_home)
    }

    /// Drive a single user message through the runtime and wait for the turn
    /// to terminate. Returns the final event so callers can assert on its kind
    /// and payload.
    async fn run_turn_and_wait(
        runtime: &ConversationRuntime,
        user_text: &str,
    ) -> (ThreadId, RuntimeEvent) {
        run_turn_and_wait_at_depth(runtime, user_text, 0).await
    }

    async fn run_turn_and_wait_at_depth(
        runtime: &ConversationRuntime,
        user_text: &str,
        agent_depth: u32,
    ) -> (ThreadId, RuntimeEvent) {
        let mut sub = runtime.event_bus().subscribe(
            "test".into(),
            crate::system::agent::EventFilter { thread_id: None },
        );

        let started = runtime
            .start_thread(ThreadStartParams {
                cwd: Some(runtime.inner.effective_config.working_directory.clone()),
                model_settings: None,
                approval_policy: None,
                permission_profile: None,
                ephemeral: true,
            })
            .await
            .expect("start_thread");
        let thread_id = started.metadata.thread_id.clone();

        runtime
            .start_turn(
                TurnStartParams {
                    thread_id: thread_id.clone(),
                    input: vec![UserInput::from_text(user_text)],
                    cwd: None,
                    model_settings: None,
                    approval_policy: None,
                    permission_profile: None,
                    output_schema: None,
                    agent_depth,
                },
                SurfaceKind::Exec,
            )
            .await
            .expect("start_turn");

        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            assert!(!remaining.is_zero(), "timed out waiting for turn end");
            match tokio::time::timeout(remaining, sub.receiver.recv()).await {
                Ok(Some(evt)) => match evt.kind {
                    RuntimeEventKind::TurnCompleted | RuntimeEventKind::TurnFailed => {
                        return (thread_id, evt);
                    }
                    _ => continue,
                },
                Ok(None) => panic!("event_bus closed before turn ended"),
                Err(_) => panic!("timed out waiting for turn end"),
            }
        }
    }

    #[tokio::test]
    async fn fake_model_completes_text_only_turn() {
        let (runtime, _home) =
            build_runtime(vec![ScriptedResponse::Text("hello there".into())]).await;
        let (thread_id, evt) = run_turn_and_wait(&runtime, "hello").await;

        assert_eq!(
            evt.kind,
            RuntimeEventKind::TurnCompleted,
            "expected TurnCompleted, got {:?}",
            evt
        );

        let read = runtime
            .read_thread(ThreadReadParams { thread_id })
            .await
            .unwrap();
        let agent_texts: Vec<String> = read
            .thread
            .items
            .iter()
            .filter(|i| i.kind == ItemKind::AgentMessage)
            .filter_map(|i| {
                i.payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
        assert!(
            agent_texts.iter().any(|t| t.contains("hello there")),
            "expected assistant text in items, got {agent_texts:?}",
        );
    }

    #[tokio::test]
    async fn fake_model_executes_tool_call_then_finishes() {
        let (runtime, home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![(
                "list_dir".into(),
                serde_json::json!({ "path": "." }),
            )]),
            ScriptedResponse::Text("done listing".into()),
        ])
        .await;
        // list_dir reads the cwd. Make sure there's at least one entry so the
        // tool succeeds deterministically.
        std::fs::write(home.join("marker.txt"), "x").unwrap();

        let (thread_id, evt) = run_turn_and_wait(&runtime, "list this directory").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);

        let read = runtime
            .read_thread(ThreadReadParams { thread_id })
            .await
            .unwrap();
        let mut tool_calls = 0;
        let mut tool_results = 0;
        let mut final_text = String::new();
        for item in &read.thread.items {
            match item.kind {
                ItemKind::ToolCall => {
                    tool_calls += 1;
                    assert_eq!(
                        item.payload.get("toolName").and_then(|v| v.as_str()),
                        Some("list_dir")
                    );
                }
                ItemKind::ToolResult => tool_results += 1,
                ItemKind::AgentMessage => {
                    if let Some(t) = item.payload.get("text").and_then(|v| v.as_str()) {
                        final_text = t.to_string();
                    }
                }
                _ => {}
            }
        }
        assert_eq!(tool_calls, 1, "expected exactly one tool call");
        assert_eq!(tool_results, 1, "expected exactly one tool result");
        assert!(
            final_text.contains("done listing"),
            "expected final assistant text, got {final_text:?}"
        );
    }

    #[tokio::test]
    async fn fake_model_empty_responses_fail_turn() {
        // max_empty_responses = 2 → after the 2nd empty response the turn fails.
        let (runtime, _home) = build_runtime(vec![
            ScriptedResponse::Empty,
            ScriptedResponse::Empty,
            ScriptedResponse::Empty,
        ])
        .await;

        let (_thread_id, evt) = run_turn_and_wait(&runtime, "hello").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnFailed, "got {:?}", evt);

        let kind = evt
            .payload
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let reason = evt
            .payload
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert_eq!(
            kind, "empty_response",
            "expected empty_response failure, got kind={kind:?} reason={reason:?}",
        );
    }

    #[tokio::test]
    async fn spawn_agent_at_max_depth_returns_capped_error() {
        // max_spawn_depth = 1; start the turn at agent_depth = 1 so the
        // spawn_agent guard fires immediately on the first tool call.
        let (runtime, _home) = build_runtime_with(
            vec![
                ScriptedResponse::ToolCalls(vec![(
                    "spawn_agent".into(),
                    serde_json::json!({ "prompt": "do a thing" }),
                )]),
                ScriptedResponse::Text("can't go deeper, finishing up".into()),
            ],
            |cfg| {
                cfg.agent.max_spawn_depth = 1;
            },
        )
        .await;

        let (thread_id, evt) = run_turn_and_wait_at_depth(&runtime, "kick off", 1).await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);

        let read = runtime
            .read_thread(ThreadReadParams { thread_id })
            .await
            .unwrap();

        // The spawn_agent tool call must produce a ToolResult with ok=false
        // and an error message that mentions the depth cap.
        let tool_result = read
            .thread
            .items
            .iter()
            .find(|i| i.kind == ItemKind::ToolResult)
            .expect("tool result item missing");
        assert_eq!(
            tool_result.payload.get("ok").and_then(|v| v.as_bool()),
            Some(false),
            "spawn_agent should fail at the depth cap, payload={}",
            tool_result.payload
        );
        let err_msg = tool_result
            .payload
            .get("errorMessage")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            err_msg.contains("nested") && err_msg.contains("1"),
            "expected depth-cap error, got {err_msg:?}"
        );
    }

    #[tokio::test]
    async fn bash_exec_blocks_on_approval_and_denied_short_circuits() {
        // OnRequest is the default; bash_exec maps to SandboxEscalation,
        // so the executor must wait for a resolution before running the tool.
        let (runtime, _home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![(
                "bash_exec".into(),
                serde_json::json!({ "command": "echo unreached" }),
            )]),
            ScriptedResponse::Text("acknowledged the deny".into()),
        ])
        .await;

        // Handler task: deny every ApprovalRequested event the runtime emits.
        // This MUST be subscribed before start_turn so we don't miss the event.
        let mut sub = runtime.event_bus().subscribe(
            "test-approver".into(),
            crate::system::agent::EventFilter { thread_id: None },
        );
        let runtime_for_handler = runtime.clone();
        let approval_seen = Arc::new(tokio::sync::Notify::new());
        let approval_seen_clone = approval_seen.clone();
        let handler = tokio::spawn(async move {
            while let Some(evt) = sub.receiver.recv().await {
                if evt.kind == RuntimeEventKind::ApprovalRequested {
                    let approval_id = evt
                        .payload
                        .get("request")
                        .and_then(|r| r.get("approvalId"))
                        .and_then(|v| v.as_str())
                        .expect("ApprovalRequested payload missing approvalId")
                        .to_string();
                    runtime_for_handler
                        .resolve_approval(&approval_id, ApprovalDecision::Denied, None)
                        .await
                        .expect("resolve_approval");
                    approval_seen_clone.notify_one();
                }
                if matches!(
                    evt.kind,
                    RuntimeEventKind::TurnCompleted | RuntimeEventKind::TurnFailed
                ) {
                    break;
                }
            }
        });

        let (thread_id, evt) = run_turn_and_wait(&runtime, "run a command").await;
        // Make sure the handler actually saw and resolved the approval before
        // we tear down — otherwise a regression that bypasses approval would
        // race with the wait helper and look like a pass.
        tokio::time::timeout(Duration::from_secs(2), approval_seen.notified())
            .await
            .expect("ApprovalRequested event was never emitted");
        let _ = handler.await;

        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);

        let read = runtime
            .read_thread(ThreadReadParams { thread_id })
            .await
            .unwrap();
        let tool_result = read
            .thread
            .items
            .iter()
            .find(|i| i.kind == ItemKind::ToolResult)
            .expect("tool result item missing");
        assert_eq!(
            tool_result.payload.get("ok").and_then(|v| v.as_bool()),
            Some(false),
            "denied bash_exec should report ok=false, payload={}",
            tool_result.payload
        );
        let err_msg = tool_result
            .payload
            .get("errorMessage")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            err_msg.contains("Denied"),
            "expected Denied error, got {err_msg:?}"
        );

        // Final assistant text must come from the *second* scripted response,
        // confirming the model saw the denied tool result and continued.
        let final_text = read
            .thread
            .items
            .iter()
            .filter(|i| i.kind == ItemKind::AgentMessage)
            .filter_map(|i| i.payload.get("text").and_then(|v| v.as_str()))
            .last()
            .unwrap_or_default()
            .to_string();
        assert!(
            final_text.contains("acknowledged the deny"),
            "expected post-deny assistant text, got {final_text:?}"
        );
    }

    #[tokio::test]
    async fn parallel_safe_tool_calls_in_one_round_all_dispatch() {
        // Both list_dir calls are parallel-safe; the executor groups them into
        // one batch and runs them via join_all. End-to-end we just verify both
        // calls and both results made it into the transcript.
        let (runtime, home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![
                ("list_dir".into(), serde_json::json!({ "path": "." })),
                ("list_dir".into(), serde_json::json!({ "path": "." })),
            ]),
            ScriptedResponse::Text("listed twice".into()),
        ])
        .await;
        std::fs::write(home.join("a.txt"), "x").unwrap();

        let (thread_id, evt) = run_turn_and_wait(&runtime, "list it twice").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);

        let read = runtime
            .read_thread(ThreadReadParams { thread_id })
            .await
            .unwrap();
        let tool_call_count = read
            .thread
            .items
            .iter()
            .filter(|i| i.kind == ItemKind::ToolCall)
            .count();
        let tool_result_count = read
            .thread
            .items
            .iter()
            .filter(|i| i.kind == ItemKind::ToolResult)
            .count();
        assert_eq!(tool_call_count, 2, "expected 2 ToolCall items");
        assert_eq!(tool_result_count, 2, "expected 2 ToolResult items");

        // ToolCall items must precede their ToolResult items in the transcript
        // — the executor persists all calls before any results regardless of
        // batching.
        let positions: Vec<(usize, ItemKind)> = read
            .thread
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, it)| match it.kind {
                ItemKind::ToolCall | ItemKind::ToolResult => Some((i, it.kind)),
                _ => None,
            })
            .collect();
        let last_call_pos = positions
            .iter()
            .rev()
            .find(|(_, k)| *k == ItemKind::ToolCall)
            .map(|(p, _)| *p)
            .unwrap();
        let first_result_pos = positions
            .iter()
            .find(|(_, k)| *k == ItemKind::ToolResult)
            .map(|(p, _)| *p)
            .unwrap();
        assert!(
            last_call_pos < first_result_pos,
            "all ToolCall items must precede all ToolResult items; positions={positions:?}"
        );
    }

    #[tokio::test]
    async fn event_ordering_for_tool_round_is_stable() {
        // Verifies the runtime emits events in a predictable sequence that
        // consumers (TUI, server, benchmarks) rely on:
        //   TurnStarted → ItemStarted(ToolCall) → ItemCompleted(ToolCall)
        //              → ItemStarted(ToolResult) → ItemCompleted(ToolResult)
        //              → ItemStarted(AgentMessage) → ItemUpdated* → ItemCompleted
        //              → TurnCompleted
        // Also exercises the typed payload API end-to-end.
        use crate::system::domain::ItemKind as IK;
        use crate::system::event_payload::RuntimeEventPayload;

        let (runtime, home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![(
                "list_dir".into(),
                serde_json::json!({ "path": "." }),
            )]),
            ScriptedResponse::Text("done".into()),
        ])
        .await;
        std::fs::write(home.join("file.txt"), "x").unwrap();

        let mut sub = runtime.event_bus().subscribe(
            "ordering-test".into(),
            crate::system::agent::EventFilter { thread_id: None },
        );

        let started = runtime
            .start_thread(ThreadStartParams {
                cwd: Some(runtime.inner.effective_config.working_directory.clone()),
                model_settings: None,
                approval_policy: None,
                permission_profile: None,
                ephemeral: true,
            })
            .await
            .unwrap();
        runtime
            .start_turn(
                TurnStartParams {
                    thread_id: started.metadata.thread_id.clone(),
                    input: vec![UserInput::from_text("go")],
                    cwd: None,
                    model_settings: None,
                    approval_policy: None,
                    permission_profile: None,
                    output_schema: None,
                    agent_depth: 0,
                },
                SurfaceKind::Exec,
            )
            .await
            .unwrap();

        // Collect events filtered to this turn (drops ThreadStarted from
        // the earlier start_thread call so we assert only the turn's stream).
        let mut kinds: Vec<RuntimeEventKind> = Vec::new();
        // For each item-lifecycle event, record (kind_of_event, item_kind).
        let mut item_events: Vec<(RuntimeEventKind, IK)> = Vec::new();
        let turn_completed_payload: crate::system::event_payload::TurnCompletedPayload;
        let mut got_turn_started = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            assert!(!remaining.is_zero(), "timed out collecting events");
            let event = match tokio::time::timeout(remaining, sub.receiver.recv()).await {
                Ok(Some(e)) => e,
                Ok(None) => panic!("event_bus closed"),
                Err(_) => panic!("timeout"),
            };
            // Skip ThreadStarted and anything else not tied to this thread.
            if event.thread_id.as_deref() != Some(started.metadata.thread_id.as_str()) {
                continue;
            }
            if !got_turn_started {
                if event.kind == RuntimeEventKind::TurnStarted {
                    got_turn_started = true;
                } else {
                    continue;
                }
            }
            kinds.push(event.kind);
            let parsed = event
                .parsed_payload()
                .unwrap_or_else(|e| panic!("typed decode failed for {:?}: {e}", event.kind));
            match parsed {
                RuntimeEventPayload::ItemStarted(env) | RuntimeEventPayload::ItemCompleted(env) => {
                    item_events.push((event.kind, env.kind));
                }
                RuntimeEventPayload::TurnCompleted(p) => {
                    turn_completed_payload = p;
                    break;
                }
                RuntimeEventPayload::TurnFailed(p) => {
                    panic!("unexpected TurnFailed: {p:?}");
                }
                _ => {}
            }
        }
        // The turn-level frame: starts with TurnStarted, ends with TurnCompleted.
        assert_eq!(
            kinds.first(),
            Some(&RuntimeEventKind::TurnStarted),
            "first turn event should be TurnStarted, got {kinds:?}"
        );
        assert_eq!(
            kinds.last(),
            Some(&RuntimeEventKind::TurnCompleted),
            "last turn event should be TurnCompleted, got {kinds:?}"
        );

        // Item-completion ordering: ToolCall completes before ToolResult,
        // and both complete before the final AgentMessage. (Tool items are
        // persisted in one shot — single ItemCompleted, no Started event.)
        let completed_in_order: Vec<IK> = item_events
            .iter()
            .filter(|(k, _)| *k == RuntimeEventKind::ItemCompleted)
            .map(|(_, ik)| *ik)
            .collect();
        let first_pos = |target: IK| completed_in_order.iter().position(|k| *k == target);
        let tool_call_pos = first_pos(IK::ToolCall).expect("ToolCall ItemCompleted missing");
        let tool_result_pos = first_pos(IK::ToolResult).expect("ToolResult ItemCompleted missing");
        let agent_msg_pos =
            first_pos(IK::AgentMessage).expect("AgentMessage ItemCompleted missing");
        assert!(
            tool_call_pos < tool_result_pos && tool_result_pos < agent_msg_pos,
            "expected ToolCall < ToolResult < AgentMessage in completion order; got {completed_in_order:?}"
        );

        // The streaming AgentMessage should also have an ItemStarted *before*
        // its ItemCompleted (proves the streaming lifecycle is intact).
        let agent_started_idx = item_events
            .iter()
            .position(|(k, ik)| *k == RuntimeEventKind::ItemStarted && *ik == IK::AgentMessage)
            .expect("AgentMessage ItemStarted missing");
        let agent_completed_idx = item_events
            .iter()
            .position(|(k, ik)| *k == RuntimeEventKind::ItemCompleted && *ik == IK::AgentMessage)
            .expect("AgentMessage ItemCompleted missing");
        assert!(
            agent_started_idx < agent_completed_idx,
            "AgentMessage ItemStarted must precede ItemCompleted; events={item_events:?}"
        );

        // Typed TurnCompleted payload should round-trip the stable fields.
        let payload = turn_completed_payload;
        assert_eq!(payload.status, TurnStatus::Completed);
        assert_eq!(payload.thread_id, started.metadata.thread_id);
    }

    #[tokio::test]
    async fn late_subscriber_replays_completed_turn_from_ring() {
        // No subscription before/during the turn. After it completes, a
        // late subscriber asks for replay from sequence 0 and should see
        // the entire event stream of the turn reconstructed from the bus's
        // in-memory ring.
        let (runtime, home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![(
                "list_dir".into(),
                serde_json::json!({ "path": "." }),
            )]),
            ScriptedResponse::Text("done".into()),
        ])
        .await;
        std::fs::write(home.join("present.txt"), "x").unwrap();

        // Drive the turn synchronously (the helper subscribes briefly and
        // tears down; the bus's replay ring keeps a copy of every event).
        let (thread_id, evt) = run_turn_and_wait(&runtime, "list it").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted);

        // Now subscribe with replay from before-the-beginning. Should
        // immediately receive every retained event for this thread.
        let mut late = runtime.event_bus().subscribe_with_replay(
            "late".into(),
            crate::system::agent::EventFilter {
                thread_id: Some(thread_id.clone()),
            },
            Some(0),
        );

        let mut kinds = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, late.receiver.recv()).await {
                Ok(Some(e)) => {
                    kinds.push(e.kind);
                    if e.kind == RuntimeEventKind::TurnCompleted {
                        break;
                    }
                }
                _ => break,
            }
        }

        // The replay must contain (at minimum) TurnStarted at the front and
        // TurnCompleted at the back. Item lifecycle in between proves we
        // got the full transcript.
        assert!(
            kinds.contains(&RuntimeEventKind::TurnStarted),
            "replay missing TurnStarted: {kinds:?}"
        );
        assert!(
            kinds.contains(&RuntimeEventKind::TurnCompleted),
            "replay missing TurnCompleted: {kinds:?}"
        );
        let item_completed = kinds
            .iter()
            .filter(|k| **k == RuntimeEventKind::ItemCompleted)
            .count();
        assert!(
            item_completed >= 3,
            "expected ≥3 ItemCompleted events (user msg, tool call, tool result, agent msg); got {kinds:?}"
        );

        // The cursor helper should report the same final sequence the
        // subscription saw, so a follow-up subscriber can resume cleanly.
        let last_seq = runtime.event_bus().last_sequence_for_thread(&thread_id);
        assert!(last_seq.is_some(), "expected a recorded last sequence");
    }
}
