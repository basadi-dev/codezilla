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
    /// Phase 8: pre-state snapshots of file-changing tool calls. Lets the
    /// caller `undo_tool_call(id)` after a bad edit.
    #[allow(dead_code)]
    pub(crate) checkpoint_store: Arc<super::agent::checkpoint::CheckpointStore>,
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
            checkpoint_store,
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

    /// Per-test scratch space under the OS temp directory. Canonicalised so
    /// macOS's `/private/` symlink prefix doesn't mismatch the sandbox's
    /// writable-root check (the sandbox canonicalises target paths before
    /// validating, so writable_roots must use the same form).
    pub(super) fn unique_app_home() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "codezilla-test-{}-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
            n
        ));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        std::fs::canonicalize(&dir).unwrap_or(dir)
    }

    pub(super) fn test_config(app_home: &std::path::Path) -> EffectiveConfig {
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
    async fn spawn_agent_redirects_directory_inventory_to_direct_tools() {
        let (runtime, _home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![(
                "spawn_agent".into(),
                serde_json::json!({
                    "prompt": "List the contents of the `bin` directory recursively (depth 3). For each file, note its path and type."
                }),
            )]),
            ScriptedResponse::Text("Done.".into()),
        ])
        .await;

        let (thread_id, evt) = run_turn_and_wait(&runtime, "inventory bin").await;
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
            Some(false)
        );
        let output = tool_result.payload.get("output").unwrap();
        assert_eq!(
            output.get("status").and_then(|v| v.as_str()),
            Some("not_spawned")
        );
        assert_eq!(
            output.get("suggested_tool").and_then(|v| v.as_str()),
            Some("list_dir")
        );
        assert_eq!(
            output
                .get("suggested_arguments")
                .and_then(|v| v.get("path"))
                .and_then(|v| v.as_str()),
            Some("bin")
        );
        assert!(
            output.get("thread_id").is_none(),
            "redirect should not spawn a child thread: {output}"
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

    // ── Phase 8: undo / rollback tests ────────────────────────────────────────

    #[tokio::test]
    async fn undo_tool_call_restores_modified_file() {
        let app_home = unique_app_home();
        let path = app_home.join("modified.txt");
        std::fs::write(&path, "ORIGINAL").unwrap();
        let path_str = path.to_string_lossy().into_owned();

        // write_file requires approval under OnRequest; flip to Never so the
        // turn runs unattended and we can exercise the undo path.
        let mut cfg = test_config(&app_home);
        cfg.approval_policy = ApprovalPolicy {
            kind: crate::system::domain::ApprovalPolicyKind::Never,
            granular: None,
        };
        let client = Arc::new(FakeLlmClient::new(vec![
            ScriptedResponse::ToolCalls(vec![(
                "write_file".into(),
                serde_json::json!({
                    "path": path_str,
                    "content": "new bytes",
                }),
            )]),
            ScriptedResponse::Text("wrote it".into()),
        ]));
        let runtime =
            ConversationRuntime::new_with_llm_client(cfg, AccountSession::default(), client)
                .await
                .expect("runtime build");

        let (thread_id, evt) = run_turn_and_wait(&runtime, "modify the file").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);

        // Diagnostic: look at the ToolResult to see if the write actually
        // succeeded — failures here surface as a clearer error message than
        // "left ≠ right" on the file-content compare.
        let read = runtime
            .read_thread(ThreadReadParams { thread_id })
            .await
            .unwrap();
        let result_item = read
            .thread
            .items
            .iter()
            .find(|i| i.kind == ItemKind::ToolResult)
            .expect("expected a ToolResult");
        let ok = result_item.payload.get("ok").and_then(|v| v.as_bool());
        assert_eq!(
            ok,
            Some(true),
            "write_file should have succeeded; payload={}",
            result_item.payload
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new bytes");

        // The fake client assigns call_1 for the first tool call.
        let summary = runtime.undo_tool_call("call_1").unwrap();
        assert_eq!(summary.restored.len(), 1, "expected one restore");
        assert!(summary.deleted.is_empty());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "ORIGINAL",
            "undo should restore prior bytes"
        );

        // Idempotent: undoing again is a no-op.
        let again = runtime.undo_tool_call("call_1").unwrap();
        assert!(again.is_empty());
    }

    #[tokio::test]
    async fn undo_tool_call_deletes_files_that_were_created() {
        let app_home = unique_app_home();
        let path = app_home.join("fresh.txt");
        // Make sure it doesn't exist before the turn runs.
        let _ = std::fs::remove_file(&path);
        let path_str = path.to_string_lossy().into_owned();

        let mut cfg = test_config(&app_home);
        cfg.approval_policy = ApprovalPolicy {
            kind: crate::system::domain::ApprovalPolicyKind::Never,
            granular: None,
        };
        let client = Arc::new(FakeLlmClient::new(vec![
            ScriptedResponse::ToolCalls(vec![(
                "write_file".into(),
                serde_json::json!({
                    "path": path_str,
                    "content": "new file",
                }),
            )]),
            ScriptedResponse::Text("created".into()),
        ]));
        let runtime =
            ConversationRuntime::new_with_llm_client(cfg, AccountSession::default(), client)
                .await
                .expect("runtime build");

        let (_thread_id, evt) = run_turn_and_wait(&runtime, "create the file").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);
        assert!(path.exists(), "fake model should have written the file");

        let summary = runtime.undo_tool_call("call_1").unwrap();
        assert!(summary.restored.is_empty());
        assert_eq!(summary.deleted.len(), 1);
        assert!(
            !path.exists(),
            "undo should delete files the tool call created"
        );
    }

    #[tokio::test]
    async fn rollback_turn_undoes_every_tool_call_in_order() {
        // Two write_file calls in one response → both snapshotted; rollback
        // should restore both files.
        let app_home = unique_app_home();
        let path_a = app_home.join("a.txt");
        let path_b = app_home.join("b.txt");
        std::fs::write(&path_a, "ORIG-A").unwrap();
        std::fs::write(&path_b, "ORIG-B").unwrap();
        let a_str = path_a.to_string_lossy().into_owned();
        let b_str = path_b.to_string_lossy().into_owned();

        let mut cfg = test_config(&app_home);
        cfg.approval_policy = ApprovalPolicy {
            kind: crate::system::domain::ApprovalPolicyKind::Never,
            granular: None,
        };
        let client = Arc::new(FakeLlmClient::new(vec![
            ScriptedResponse::ToolCalls(vec![
                (
                    "write_file".into(),
                    serde_json::json!({
                        "path": a_str,
                        "content": "new-a",
                    }),
                ),
                (
                    "write_file".into(),
                    serde_json::json!({
                        "path": b_str,
                        "content": "new-b",
                    }),
                ),
            ]),
            ScriptedResponse::Text("done".into()),
        ]));
        let runtime =
            ConversationRuntime::new_with_llm_client(cfg, AccountSession::default(), client)
                .await
                .expect("runtime build");

        let (thread_id, evt) = run_turn_and_wait(&runtime, "modify both").await;
        assert_eq!(evt.kind, RuntimeEventKind::TurnCompleted, "got {:?}", evt);
        let turn_id = evt.turn_id.clone().expect("TurnCompleted carries turn_id");

        assert_eq!(std::fs::read_to_string(&path_a).unwrap(), "new-a");
        assert_eq!(std::fs::read_to_string(&path_b).unwrap(), "new-b");

        let summary = runtime.rollback_turn(&thread_id, &turn_id).unwrap();
        assert_eq!(
            summary.restored.len(),
            2,
            "expected both files restored, got {summary:?}"
        );
        assert_eq!(std::fs::read_to_string(&path_a).unwrap(), "ORIG-A");
        assert_eq!(std::fs::read_to_string(&path_b).unwrap(), "ORIG-B");
    }

    #[tokio::test]
    async fn spawn_agent_publishes_child_agent_spawned_event() {
        // Drive a parent turn that calls spawn_agent. The supervisor should
        // emit a `ChildAgentSpawned` event tying the child thread/turn back
        // to the parent's tool_call_id, so the TUI's activity tree can pick
        // it up.
        let (runtime, _home) = build_runtime(vec![
            ScriptedResponse::ToolCalls(vec![(
                "spawn_agent".into(),
                serde_json::json!({ "prompt": "summarise main.rs" }),
            )]),
            ScriptedResponse::Text("done".into()),
            // The child agent's fake-model script: it shares the same
            // FakeLlmClient queue as the parent, so the child consumes the
            // next entry. One Empty response → child fails fast on
            // empty_response, which is fine for this test (we only care
            // about the spawn event firing).
            ScriptedResponse::Empty,
            ScriptedResponse::Empty,
        ])
        .await;

        let mut sub = runtime.event_bus().subscribe(
            "spawn-test".into(),
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
                    input: vec![UserInput::from_text("kick off")],
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

        // Wait for the spawn event to fire.
        let mut spawn_payload = None;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            assert!(
                !remaining.is_zero(),
                "timed out waiting for ChildAgentSpawned"
            );
            match tokio::time::timeout(remaining, sub.receiver.recv()).await {
                Ok(Some(evt)) => {
                    if evt.kind == RuntimeEventKind::ChildAgentSpawned {
                        spawn_payload = Some(evt.payload.clone());
                        break;
                    }
                    if evt.kind == RuntimeEventKind::TurnCompleted
                        && evt.thread_id.as_deref() == Some(started.metadata.thread_id.as_str())
                    {
                        break;
                    }
                }
                _ => break,
            }
        }

        let payload = spawn_payload.expect("ChildAgentSpawned event was not emitted");
        // The parent thread/turn should match what we started.
        assert_eq!(
            payload.get("parentThreadId").and_then(|v| v.as_str()),
            Some(started.metadata.thread_id.as_str())
        );
        assert!(payload.get("parentToolCallId").is_some());
        assert!(payload.get("childThreadId").is_some());
        assert!(payload.get("childTurnId").is_some());
        let label = payload
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(label.contains("summarise"), "label was {label:?}");

        // Typed payload also decodes.
        use crate::system::event_payload::RuntimeEventPayload;
        let evt = RuntimeEvent {
            event_id: "x".into(),
            kind: RuntimeEventKind::ChildAgentSpawned,
            thread_id: Some(started.metadata.thread_id.clone()),
            turn_id: None,
            sequence: 0,
            payload: payload.clone(),
            emitted_at: 0,
        };
        match evt.parsed_payload().unwrap() {
            RuntimeEventPayload::ChildAgentSpawned(p) => {
                assert_eq!(p.parent_thread_id, started.metadata.thread_id);
            }
            _ => panic!("wrong variant"),
        }
    }
}
