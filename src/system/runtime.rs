use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock as AsyncRwLock, Semaphore};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::llm::client::UnifiedClient;
use crate::llm::LlmClient;

// Agent subsystem — types used only internally in this file
use super::agent::model_gateway::build_compaction_messages;
use super::agent::{
    ApprovalManager, BashToolProvider, EventBus, ExtensionManager, FileToolProvider,
    ImageToolProvider, ListDirToolProvider, ModelGateway, PermissionManager,
    RequestUserInputToolProvider, SandboxManager, SearchToolProvider, ShellToolProvider,
    ToolOrchestrator, TurnExecutor, WebToolProvider,
};
use super::intel::RepoMap;
// Agent types re-exported for callers outside runtime.rs
#[allow(unused_imports)]
pub use super::agent::{AutoReviewer, EventFilter, EventSubscription, ModelDescription};

use super::config::EffectiveConfig;
use super::domain::{
    now_millis, now_seconds, AccountSession, ApprovalDecision, ApprovalPolicy, ApprovalResolution,
    CompactionStrategy, ConnectorDefinition, ConversationItem, ItemId, ItemKind,
    McpServerDefinition, MemoryMode, ModelSettings, PathString, PermissionProfile, PersistedThread,
    PluginDefinition, PrefixRule, RuntimeEvent, RuntimeEventKind, SessionId, SkillDefinition,
    SurfaceKind, ThreadFilter, ThreadId, ThreadMetadata, ThreadStatus, TokenUsage, ToolCall,
    ToolCallId, TurnId, TurnMetadata, TurnStatus, UserInput,
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
        let persistence = Arc::new(PersistenceManager::new(
            std::path::Path::new(&effective_config.app_home).join("state"),
            std::path::Path::new(&effective_config.app_home).join("memories"),
            std::path::Path::new(&effective_config.app_home).join("logs"),
        )?);

        let llm_client: Arc<dyn LlmClient> = Arc::new(
            UnifiedClient::new(effective_config.llm.clone())
                .map_err(|e| anyhow!("llm_client_init_failed: {e}"))?,
        );

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

        // Late-register the real SpawnAgentToolProvider with a runtime clone.
        me.inner
            .tool_orchestrator
            .register_provider(Arc::new(SpawnAgentToolProviderReal::new(
                AgentSupervisor::new(me.clone(), 4),
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

    pub async fn start_turn(
        &self,
        params: TurnStartParams,
        surface: SurfaceKind,
    ) -> Result<TurnStartResult> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;

        let turn = {
            let mut thread_guard = thread.lock().await;
            if thread_guard.metadata.archived {
                bail!("thread_archived: {}", thread_guard.metadata.thread_id);
            }
            if thread_guard.active_turn_id.is_some() {
                bail!("turn_already_active: {}", thread_guard.metadata.thread_id);
            }
            let turn = TurnMetadata {
                turn_id: format!("turn_{}", Uuid::new_v4().simple()),
                thread_id: thread_guard.metadata.thread_id.clone(),
                created_at: now_seconds(),
                updated_at: now_seconds(),
                status: TurnStatus::Created,
                started_by_surface: surface,
                token_usage: TokenUsage::default(),
            };
            self.inner.persistence_manager.create_turn(&turn)?;
            thread_guard.active_turn_id = Some(turn.turn_id.clone());
            thread_guard.metadata.status = ThreadStatus::Running;
            thread_guard.approval_policy_override = params.approval_policy.clone();
            thread_guard.turns.push(LoadedTurn {
                metadata: turn.clone(),
                items: Vec::new(),
                status: TurnStatus::Created,
                pending_tool_calls: HashMap::new(),
                stream_buffer: Vec::new(),
                cancel_token: CancellationToken::new(),
            });
            turn
        };

        self.publish_event(
            RuntimeEventKind::TurnStarted,
            Some(params.thread_id.clone()),
            Some(turn.turn_id.clone()),
            serde_json::to_value(&turn)?,
        )
        .await?;

        let runtime = self.clone();
        let params_clone = params.clone();
        let thread_id_for_task = params.thread_id.clone();
        let turn_id = turn.turn_id.clone();
        let turn_for_result = turn.clone();
        tokio::spawn(async move {
            let executor = TurnExecutor { runtime };
            if let Err(error) = executor.run_turn(params_clone, turn_id.clone()).await {
                let error_text = error.to_string();
                let err = crate::system::error::from_raw(&error_text);
                let _ = executor
                    .fail_turn(
                        &thread_id_for_task,
                        &turn_id,
                        err.kind.label(),
                        &err.message,
                    )
                    .await;
            }
        });

        Ok(TurnStartResult {
            turn: turn_for_result,
        })
    }

    pub async fn interrupt_turn(&self, params: TurnInterruptParams) -> Result<TurnInterruptResult> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        let cancelled = {
            let mut thread_guard = thread.lock().await;
            let Some(active_id) = &thread_guard.active_turn_id else {
                return Ok(TurnInterruptResult {
                    turn_id: params.turn_id,
                    interrupted: false,
                });
            };
            if active_id != &params.turn_id {
                bail!("turn_not_found: {}", params.turn_id);
            }
            let turn = thread_guard
                .turns
                .iter_mut()
                .find(|t| t.metadata.turn_id == params.turn_id)
                .ok_or_else(|| anyhow!("turn_not_found: {}", params.turn_id))?;
            turn.cancel_token.cancel();
            true
        };

        let approvals = self
            .inner
            .approval_manager
            .cancel_for_turn(&params.thread_id, &params.turn_id)
            .await;
        for resolution in approvals {
            self.publish_event(
                RuntimeEventKind::ApprovalResolved,
                Some(params.thread_id.clone()),
                Some(params.turn_id.clone()),
                serde_json::to_value(&resolution)?,
            )
            .await?;
        }

        Ok(TurnInterruptResult {
            turn_id: params.turn_id,
            interrupted: cancelled,
        })
    }

    pub async fn steer_turn(&self, params: TurnSteerParams) -> Result<TurnSteerResult> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        let queued = {
            let mut guard = thread.lock().await;
            if guard.active_turn_id.as_deref() != Some(params.expected_turn_id.as_str()) {
                bail!("turn_mismatch: {}", params.expected_turn_id);
            }
            let count = params.input.len();
            guard.pending_steering.extend(params.input);
            count
        };
        Ok(TurnSteerResult {
            turn_id: params.expected_turn_id,
            queued_items: queued,
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

    pub async fn resolve_approval(
        &self,
        approval_id: &str,
        decision: ApprovalDecision,
        rule: Option<PrefixRule>,
    ) -> Result<Option<ApprovalResolution>> {
        Ok(self
            .inner
            .approval_manager
            .resolve_approval(approval_id, decision, rule)
            .await)
    }

    pub fn list_models(&self) -> Vec<ModelDescription> {
        self.inner
            .model_gateway
            .list_models(&self.inner.effective_config.model_settings)
    }

    pub async fn list_skills(&self) -> Vec<SkillDefinition> {
        self.inner
            .extension_manager
            .list_skills(&self.inner.effective_config.working_directory)
            .await
    }

    pub async fn list_plugins(&self) -> Vec<PluginDefinition> {
        self.inner.extension_manager.list_plugins().await
    }

    pub async fn list_connectors(&self) -> Vec<ConnectorDefinition> {
        self.inner.extension_manager.list_connectors().await
    }

    pub async fn list_mcp_servers(&self) -> Vec<McpServerDefinition> {
        self.inner.extension_manager.list_mcp_servers().await
    }

    pub fn reset_memories(&self) -> Result<()> {
        self.inner.persistence_manager.reset_memories()
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

    // ── Sub-agent support ──────────────────────────────────────────────────────

    /// Subscribe to the event bus and block until a specific child turn reaches
    /// a terminal state. The persisted turn status is authoritative; events only
    /// wake this waiter so fast terminal transitions cannot be missed.
    pub async fn await_child_turn_completion(
        &self,
        thread_id: &str,
        turn_id: &str,
        timeout_secs: u64,
    ) -> Result<TurnCompletionOutcome> {
        if let Some(outcome) = self.child_turn_terminal_outcome(thread_id, turn_id)? {
            return Ok(outcome);
        }

        let subscriber_id = format!("spawn_agent_{}", Uuid::new_v4().simple());
        let mut sub = self.inner.event_bus.subscribe(
            subscriber_id.clone(),
            super::agent::EventFilter {
                thread_id: Some(thread_id.to_string()),
            },
        );
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);

        let outcome = loop {
            if let Some(outcome) = self.child_turn_terminal_outcome(thread_id, turn_id)? {
                break outcome;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break TurnCompletionOutcome::TimedOut;
            }
            match tokio::time::timeout(remaining, sub.receiver.recv()).await {
                Ok(Ok(event)) => {
                    if event.thread_id.as_deref() != Some(thread_id) {
                        continue;
                    }
                    match event.kind {
                        RuntimeEventKind::TurnCompleted => {
                            if event.turn_id.as_deref() == Some(turn_id) {
                                continue;
                            }
                        }
                        RuntimeEventKind::TurnFailed => {
                            if event.turn_id.as_deref() == Some(turn_id) {
                                continue;
                            }
                        }
                        _ => continue,
                    }
                }
                Ok(Err(_)) => break TurnCompletionOutcome::Failed("event_bus_closed".into()),
                Err(_) => break TurnCompletionOutcome::TimedOut,
            }
        };

        self.inner.event_bus.unsubscribe(&subscriber_id);
        Ok(outcome)
    }

    fn child_turn_terminal_outcome(
        &self,
        thread_id: &str,
        turn_id: &str,
    ) -> Result<Option<TurnCompletionOutcome>> {
        let persisted = self.inner.persistence_manager.read_thread(thread_id)?;
        let Some(turn) = persisted.turns.iter().find(|turn| turn.turn_id == turn_id) else {
            return Ok(Some(TurnCompletionOutcome::Failed(format!(
                "child turn not found: {turn_id}"
            ))));
        };

        let outcome = match turn.status {
            TurnStatus::Completed => Some(TurnCompletionOutcome::Completed),
            TurnStatus::Failed => Some(TurnCompletionOutcome::Failed(
                last_error_message(&persisted.items, turn_id).unwrap_or_else(|| "failed".into()),
            )),
            TurnStatus::Interrupted => Some(TurnCompletionOutcome::Interrupted),
            TurnStatus::Created | TurnStatus::Running | TurnStatus::WaitingForApproval => None,
        };
        Ok(outcome)
    }

    async fn cancel_child_turn(
        &self,
        thread_id: &str,
        turn_id: &str,
        grace_secs: u64,
    ) -> Result<TurnCompletionOutcome> {
        let _ = self
            .interrupt_turn(TurnInterruptParams {
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
            })
            .await?;

        self.await_child_turn_completion(thread_id, turn_id, grace_secs)
            .await
            .or_else(|_| Ok(TurnCompletionOutcome::TimedOut))
    }

    /// Read the last `AgentMessage` item from a thread's persisted history.
    pub fn read_last_agent_message(&self, thread_id: &str) -> Result<Option<String>> {
        let persisted = self.inner.persistence_manager.read_thread(thread_id)?;
        let text = persisted
            .items
            .iter()
            .rev()
            .find(|i| i.kind == ItemKind::AgentMessage)
            .and_then(|i| i.payload.get("text").and_then(Value::as_str))
            .map(|s| s.to_string());
        Ok(text)
    }

    pub async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.inner.loaded_threads.write().await.remove(thread_id);
        self.inner.persistence_manager.delete_thread(thread_id)?;
        Ok(())
    }
}

/// Outcome of awaiting a sub-agent turn.
pub enum TurnCompletionOutcome {
    Completed,
    Failed(String),
    Interrupted,
    TimedOut,
}

fn last_error_message(items: &[ConversationItem], turn_id: &str) -> Option<String> {
    items
        .iter()
        .rev()
        .find(|item| item.turn_id == turn_id && item.kind == ItemKind::Error)
        .and_then(|item| {
            item.payload
                .get("message")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
}

#[derive(Clone)]
pub(crate) struct AgentSupervisor {
    runtime: ConversationRuntime,
    child_slots: Arc<Semaphore>,
}

pub(crate) struct ChildAgentRequest {
    prompt: String,
    cwd: String,
    approval_policy: ApprovalPolicy,
    permission_profile: PermissionProfile,
    timeout_secs: u64,
    agent_depth: u32,
}

pub(crate) struct ChildAgentRun {
    child_thread_id: ThreadId,
    child_turn_id: TurnId,
    result_text: String,
    outcome: TurnCompletionOutcome,
}

impl AgentSupervisor {
    pub fn new(runtime: ConversationRuntime, max_concurrent_children: usize) -> Self {
        Self {
            runtime,
            child_slots: Arc::new(Semaphore::new(max_concurrent_children.max(1))),
        }
    }

    pub async fn run_child(&self, request: ChildAgentRequest) -> Result<ChildAgentRun> {
        let _slot = self
            .child_slots
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| anyhow!("child_agent_slots_closed: {e}"))?;

        let child = self
            .runtime
            .start_thread(ThreadStartParams {
                cwd: Some(request.cwd),
                model_settings: None,
                approval_policy: Some(request.approval_policy.clone()),
                permission_profile: Some(request.permission_profile.clone()),
                ephemeral: true,
            })
            .await?;
        let child_thread_id = child.metadata.thread_id;

        let child_turn = self
            .runtime
            .start_turn(
                TurnStartParams {
                    thread_id: child_thread_id.clone(),
                    input: vec![UserInput::from_text(&request.prompt)],
                    cwd: None,
                    model_settings: None,
                    approval_policy: Some(request.approval_policy),
                    permission_profile: Some(request.permission_profile),
                    output_schema: None,
                    agent_depth: request.agent_depth + 1,
                },
                SurfaceKind::Exec,
            )
            .await?;
        let child_turn_id = child_turn.turn.turn_id;

        let mut outcome = self
            .runtime
            .await_child_turn_completion(&child_thread_id, &child_turn_id, request.timeout_secs)
            .await?;

        if matches!(outcome, TurnCompletionOutcome::TimedOut) {
            let _ = self
                .runtime
                .cancel_child_turn(&child_thread_id, &child_turn_id, 5)
                .await;
            outcome = TurnCompletionOutcome::TimedOut;
        }

        let result_text = self
            .runtime
            .read_last_agent_message(&child_thread_id)?
            .unwrap_or_else(|| "[sub-agent produced no output]".into());

        if matches!(outcome, TurnCompletionOutcome::Completed) {
            if let Err(e) = self.runtime.delete_thread(&child_thread_id).await {
                tracing::warn!(thread_id = %child_thread_id, "failed to delete successful ephemeral sub-agent thread: {e}");
            }
        }

        Ok(ChildAgentRun {
            child_thread_id,
            child_turn_id,
            result_text,
            outcome,
        })
    }
}

// ─── SpawnAgentToolProviderReal ───────────────────────────────────────────────
//
// Registered *after* ConversationRuntime is constructed so it can hold a
// runtime clone without creating a circular dependency.

use crate::system::agent::tools::ToolProvider;
use crate::system::domain::{
    ToolDefinition, ToolExecutionContext, ToolListingContext, ToolProviderKind, ToolResult,
};
use async_trait::async_trait;

pub(crate) struct SpawnAgentToolProviderReal {
    supervisor: AgentSupervisor,
}

impl SpawnAgentToolProviderReal {
    pub fn new(supervisor: AgentSupervisor) -> Self {
        Self { supervisor }
    }
}

#[async_trait]
impl ToolProvider for SpawnAgentToolProviderReal {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "spawn_agent".into(),
            description: "Spawn an independent sub-agent to work on a task in parallel. \
                The sub-agent runs with full tool access and returns its final answer as text. \
                Use this to parallelise independent sub-tasks — for example, analysing multiple \
                files simultaneously. Multiple spawn_agent calls in the same response run \
                concurrently."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description for the sub-agent. Be specific and self-contained."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Maximum seconds to wait for the sub-agent (default 120, max 600)."
                    }
                },
                "required": ["prompt"]
            }),
            requires_approval: false,
            supports_parallel_calls: true,
            provider_kind: ToolProviderKind::Builtin,
        }]
    }

    async fn execute(&self, call: &ToolCall, ctx: &ToolExecutionContext) -> Result<ToolResult> {
        use anyhow::anyhow;

        // Depth guard: prevent unbounded recursive agent spawning.
        if ctx.agent_depth >= 3 {
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({ "error": "max sub-agent depth reached" }),
                error_message: Some("spawn_agent cannot be nested more than 3 levels deep".into()),
            });
        }

        let prompt = call
            .arguments
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("spawn_agent: prompt is required"))?
            .to_string();

        let timeout_secs = call
            .arguments
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(120)
            .min(600);

        let run = self
            .supervisor
            .run_child(ChildAgentRequest {
                prompt,
                cwd: ctx.cwd.clone(),
                approval_policy: ctx.approval_policy.clone(),
                permission_profile: ctx.permission_profile.clone(),
                timeout_secs,
                agent_depth: ctx.agent_depth,
            })
            .await?;

        match run.outcome {
            TurnCompletionOutcome::Completed => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: true,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                }),
                error_message: None,
            }),
            TurnCompletionOutcome::Failed(reason) => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                    "error": reason,
                }),
                error_message: Some(format!("sub-agent failed: {reason}")),
            }),
            TurnCompletionOutcome::Interrupted => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                    "error": "interrupted",
                }),
                error_message: Some("sub-agent interrupted".into()),
            }),
            TurnCompletionOutcome::TimedOut => Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: json!({
                    "thread_id": run.child_thread_id,
                    "turn_id": run.child_turn_id,
                    "result": run.result_text,
                    "error": "timeout",
                }),
                error_message: Some(format!("sub-agent timed out after {timeout_secs}s")),
            }),
        }
    }
}
