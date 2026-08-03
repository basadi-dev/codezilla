mod builder;
mod discovery;
mod thread;
mod turn;

pub use builder::RuntimeBuilder;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock as AsyncRwLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::llm::LlmClient;

// Agent subsystem — types used only internally in this file
use super::agent::supervisor::{AgentOrchestrationToolProvider, AgentSupervisor};
use super::agent::{
    ApprovalManager, BashToolProvider, EventBus, ExtensionManager, FileToolProvider,
    GraphToolProvider, ImageToolProvider, ListDirToolProvider, MemoryToolProvider, ModelGateway,
    PatternMiner, PermissionManager, RequestUserInputToolProvider, SandboxManager,
    SearchToolProvider, ShellToolProvider, ToolOrchestrator, WebToolProvider,
};
use super::intel::RepoMap;
// Agent types re-exported for callers outside runtime.rs
#[allow(unused_imports)]
pub use super::agent::{AutoReviewer, EventFilter, EventSubscription, ModelDescription};

use super::config::EffectiveConfig;
use super::domain::{
    now_millis, AccountSession, ApprovalPolicy, CompactionStrategy, ConversationItem, ItemId,
    MemoryMode, ModelSettings, PathString, PermissionProfile, PersistedThread, RuntimeEvent,
    RuntimeEventKind, SessionId, ThreadId, ThreadMetadata, ToolCall, ToolCallId, TurnId,
    TurnMetadata, TurnStatus, UserInput,
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
pub struct ThreadModelSettingsParams {
    pub thread_id: ThreadId,
    pub model_id: String,
    pub provider_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RepoMapVerbosity {
    Lean,
    Verbose,
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
    #[serde(default)]
    pub repo_map_verbosity: Option<RepoMapVerbosity>,
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
    /// Phase 8: pre-state snapshots of file-changing tool calls. Lets the
    /// caller `undo_tool_call(id)` after a bad edit.
    #[allow(dead_code)]
    pub(crate) checkpoint_store: Arc<super::agent::checkpoint::CheckpointStore>,
    /// Behavioural pattern miner — learns recurring habits from transcripts
    /// and injects them as system-prompt hints on future turns.
    pub(crate) pattern_miner: Arc<PatternMiner>,
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
        // Default constructor goes through the builder so the same path is
        // exercised in production and tests; embedders can use `RuntimeBuilder`
        // directly when they need to inject a custom client or extra tools.
        RuntimeBuilder::new(effective_config, account_session)
            .build()
            .await
    }

    /// Construct a runtime with a caller-supplied `LlmClient`.
    ///
    /// Used by the fake-model test harness to drive deterministic agent loops
    /// without contacting a real LLM provider. New code should prefer
    /// [`RuntimeBuilder::new(...).with_llm_client(...).build()`]; this
    /// signature is kept for the existing call sites.
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
        let checkpoint_store = Arc::new(super::agent::checkpoint::CheckpointStore::new());

        tool_orchestrator.register_provider(Arc::new(
            FileToolProvider::new(sandbox.clone(), permissions.clone())
                .with_intel_cache(intel_cache)
                .with_checkpoint_store(checkpoint_store.clone()),
        ));
        tool_orchestrator.register_provider(Arc::new(SearchToolProvider));
        tool_orchestrator.register_provider(Arc::new(MemoryToolProvider::new(persistence.clone())));
        tool_orchestrator.register_provider(Arc::new(ImageToolProvider));
        // NOTE: SpawnAgentToolProvider is registered *after* Self is constructed (late registration)
        // so it can hold a ConversationRuntime clone. See below.
        tool_orchestrator.register_provider(Arc::new(RequestUserInputToolProvider));
        tool_orchestrator.register_provider(Arc::new(WebToolProvider::new()));
        tool_orchestrator.register_provider(Arc::new(GraphToolProvider::new(
            std::path::PathBuf::from(&effective_config.app_home),
        )));

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
        let pattern_miner = {
            let db_path = std::path::Path::new(&effective_config.app_home)
                .join("state")
                .join("patterns.sqlite3");
            Arc::new(
                PatternMiner::open(&db_path).unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "pattern_miner: failed to open db, using in-memory fallback");
                    // Safety: open_in_memory only fails on OOM — acceptable panic.
                    PatternMiner::open_in_memory().expect("in-memory pattern miner")
                }),
            )
        };
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
            checkpoint_store,
            pattern_miner,
        };
        let me = Self {
            inner: Arc::new(inner),
        };

        let agent_cfg = &me.inner.effective_config.agent;
        tracing::debug!(
            turn_policy_mode = ?agent_cfg.turn_policy_mode,
            max_iterations = agent_cfg.max_iterations,
            max_consecutive_failures = agent_cfg.max_consecutive_failures,
            max_no_tool_nudges = agent_cfg.max_no_tool_nudges,
            max_consecutive_read_only_rounds = agent_cfg.max_consecutive_read_only_rounds,
            max_empty_responses = agent_cfg.max_empty_responses,
            max_total_nudges = agent_cfg.max_total_nudges,
            max_response_chars = agent_cfg.max_response_chars,
            max_concurrent_agents = agent_cfg.max_concurrent_agents,
            max_child_agents = agent_cfg.max_child_agents,
            max_concurrent_child_agents = agent_cfg.max_concurrent_child_agents(),
            max_spawn_depth = agent_cfg.max_spawn_depth,
            child_timeout_secs = agent_cfg.child_timeout_secs,
            max_child_timeout_secs = agent_cfg.max_child_timeout_secs,
            teams_enabled = agent_cfg.teams_enabled,
            team_max_members = agent_cfg.team_max_members,
            "runtime: agent config loaded"
        );

        // Late-register agent orchestration tools with a runtime clone.
        me.inner.tool_orchestrator.register_provider(Arc::new(
            AgentOrchestrationToolProvider::new(AgentSupervisor::new(
                me.clone(),
                me.inner
                    .effective_config
                    .agent
                    .max_concurrent_child_agents(),
            )),
        ));

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

    /// List all stored behavioural patterns.
    pub fn list_patterns(&self) -> Result<Vec<super::agent::pattern_miner::BehaviourPattern>> {
        self.inner.pattern_miner.list_all_patterns()
    }

    /// Delete a single pattern by ID. Returns true if removed.
    pub fn delete_pattern(&self, pattern_id: &str) -> Result<bool> {
        self.inner.pattern_miner.delete_pattern(pattern_id)
    }

    /// Delete all stored patterns. Returns count removed.
    pub fn delete_all_patterns(&self) -> Result<usize> {
        self.inner.pattern_miner.delete_all_patterns()
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
}

// ─── Fake-model end-to-end harness tests ──────────────────────────────────────
