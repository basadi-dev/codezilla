use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub type JsonValue = Value;
pub type TimestampSeconds = i64;
pub type TimestampMillis = i64;
pub type PathString = String;
pub type UrlString = String;
pub type ThreadId = String;
pub type TurnId = String;
pub type ItemId = String;
pub type ApprovalId = String;
pub type ToolCallId = String;
pub type ProcessId = String;
pub type SessionId = String;
pub type ModelId = String;
pub type ProviderId = String;
pub type FeatureKey = String;
pub type PluginId = String;
pub type SkillId = String;
pub type McpServerName = String;
pub type ConnectorId = String;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SurfaceKind {
    Interactive,
    Exec,
    AppServer,
    Review,
    McpServer,
    ExecServer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ThreadStatus {
    NotLoaded,
    Idle,
    Running,
    WaitingForApproval,
    Interrupted,
    Archived,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TurnStatus {
    Created,
    Running,
    WaitingForApproval,
    Completed,
    Failed,
    Interrupted,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ItemKind {
    UserMessage,
    UserAttachment,
    AgentMessage,
    SystemMessage,
    ReasoningText,
    ReasoningSummary,
    ToolCall,
    ToolResult,
    CommandExecution,
    CommandOutput,
    FileChange,
    Error,
    ReviewMarker,
    Status,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
    External,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApprovalPolicyKind {
    Never,
    UnlessTrusted,
    OnFailure,
    OnRequest,
    Granular,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApprovalsReviewerKind {
    User,
    AutoReviewer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApprovalCategory {
    SandboxEscalation,
    FileChange,
    RulesChange,
    SkillApproval,
    RequestPermissions,
    McpTool,
    ConnectorAction,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ApprovalDecision {
    Approved,
    Denied,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AuthMode {
    None,
    ApiToken,
    BrowserSession,
    DeviceCode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ToolProviderKind {
    Builtin,
    Mcp,
    Connector,
    Plugin,
    Dynamic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OutputMode {
    Human,
    Jsonl,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[allow(dead_code)]
pub enum SessionStartMode {
    New,
    Resume,
    Fork,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MemoryMode {
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelSettings {
    pub model_id: ModelId,
    pub provider_id: ProviderId,
    pub reasoning_effort: Option<String>,
    pub summary_mode: Option<String>,
    pub service_tier: Option<String>,
    #[serde(default)]
    pub web_search_enabled: bool,
    #[serde(default)]
    pub context_window: Option<usize>,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            model_id: "glm-5.1:cloud".into(),
            provider_id: "ollama".into(),
            reasoning_effort: None,
            summary_mode: None,
            service_tier: None,
            web_search_enabled: false,
            context_window: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WritableRoot {
    pub path: PathString,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionProfile {
    pub sandbox_mode: SandboxMode,
    #[serde(default)]
    pub writable_roots: Vec<WritableRoot>,
    #[serde(default)]
    pub network_enabled: bool,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub allowed_unix_sockets: Vec<PathString>,
}

impl Default for PermissionProfile {
    fn default() -> Self {
        Self {
            sandbox_mode: SandboxMode::WorkspaceWrite,
            writable_roots: Vec::new(),
            network_enabled: false,
            allowed_domains: Vec::new(),
            allowed_unix_sockets: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GranularApprovalPolicy {
    pub sandbox_approval: bool,
    pub rules_approval: bool,
    pub skill_approval: bool,
    pub request_permissions: bool,
    pub mcp_tool_approval: bool,
    pub connector_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalPolicy {
    pub kind: ApprovalPolicyKind,
    pub granular: Option<GranularApprovalPolicy>,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        Self {
            kind: ApprovalPolicyKind::OnRequest,
            granular: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ConversationPathSet {
    pub app_home: PathString,
    pub sessions_root: PathString,
    pub archived_sessions_root: PathString,
    pub memories_root: PathString,
    pub config_path: PathString,
    pub state_root: PathString,
    pub control_socket_path: PathString,
    pub logs_root: PathString,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountSession {
    pub auth_mode: AuthMode,
    pub account_id: Option<String>,
    pub email: Option<String>,
    pub access_token: Option<String>,
    pub api_token: Option<String>,
    pub expires_at: Option<TimestampSeconds>,
}

impl Default for AccountSession {
    fn default() -> Self {
        Self {
            auth_mode: AuthMode::None,
            account_id: None,
            email: None,
            access_token: None,
            api_token: None,
            expires_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMetadata {
    pub thread_id: ThreadId,
    pub title: Option<String>,
    pub created_at: TimestampSeconds,
    pub updated_at: TimestampSeconds,
    pub cwd: Option<PathString>,
    pub model_id: ModelId,
    pub provider_id: ProviderId,
    pub status: ThreadStatus,
    pub forked_from_id: Option<ThreadId>,
    pub archived: bool,
    pub ephemeral: bool,
    pub memory_mode: MemoryMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnMetadata {
    pub turn_id: TurnId,
    pub thread_id: ThreadId,
    pub created_at: TimestampSeconds,
    pub updated_at: TimestampSeconds,
    pub status: TurnStatus,
    pub started_by_surface: SurfaceKind,
    pub token_usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserInputText {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserInputImage {
    pub path: PathString,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UserInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<UserInputText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<UserInputImage>,
}

impl UserInput {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(UserInputText { text: text.into() }),
            image: None,
        }
    }

    #[allow(dead_code)]
    pub fn from_image(path: impl Into<String>) -> Self {
        Self {
            text: None,
            image: Some(UserInputImage { path: path.into() }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub tool_call_id: ToolCallId,
    pub provider_kind: ToolProviderKind,
    pub tool_name: String,
    pub arguments: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResult {
    pub tool_call_id: ToolCallId,
    pub ok: bool,
    pub output: JsonValue,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CommandExecutionRecord {
    pub process_id: ProcessId,
    pub command: Vec<String>,
    pub cwd: PathString,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct FileChangeRecord {
    pub path: PathString,
    pub change_kind: String,
    pub old_path: Option<PathString>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationItem {
    pub item_id: ItemId,
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub created_at: TimestampSeconds,
    pub kind: ItemKind,
    pub payload: JsonValue,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RuntimeEventKind {
    ThreadStarted,
    TurnStarted,
    ItemStarted,
    ItemUpdated,
    ItemCompleted,
    TurnCompleted,
    TurnFailed,
    ApprovalRequested,
    ApprovalResolved,
    Warning,
    Disconnected,
    CompactionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeEvent {
    pub event_id: String,
    pub kind: RuntimeEventKind,
    pub thread_id: Option<ThreadId>,
    pub turn_id: Option<TurnId>,
    pub sequence: i64,
    pub payload: JsonValue,
    pub emitted_at: TimestampMillis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub provider_kind: ToolProviderKind,
    pub description: String,
    pub input_schema: JsonValue,
    pub requires_approval: bool,
    pub supports_parallel_calls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolListingContext {
    pub thread_id: ThreadId,
    pub cwd: PathString,
    #[serde(default)]
    pub features: HashMap<FeatureKey, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionContext {
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub cwd: PathString,
    pub permission_profile: PermissionProfile,
    pub approval_policy: ApprovalPolicy,
    /// Nesting depth: 0 = top-level agent, 1 = sub-agent, etc.
    /// Used to prevent unbounded recursive agent spawning.
    #[serde(default)]
    pub agent_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionDescriptor {
    pub action_type: String,
    pub command: Option<Vec<String>>,
    #[serde(default)]
    pub paths: Vec<PathString>,
    #[serde(default)]
    pub domains: Vec<String>,
    pub category: ApprovalCategory,
}

impl Default for ActionDescriptor {
    fn default() -> Self {
        Self {
            action_type: String::new(),
            command: None,
            paths: Vec::new(),
            domains: Vec::new(),
            category: ApprovalCategory::Other,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SandboxRequest {
    pub sandbox_mode: Option<SandboxMode>,
    #[serde(default)]
    pub writable_roots: Vec<PathString>,
    #[serde(default)]
    pub network_enabled: bool,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub allowed_unix_sockets: Vec<PathString>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalRequest {
    pub approval_id: ApprovalId,
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub category: ApprovalCategory,
    pub title: String,
    pub justification: String,
    pub action: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingApproval {
    pub request: ApprovalRequest,
    pub created_at: TimestampSeconds,
    pub reviewer_kind: ApprovalsReviewerKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrefixRule {
    pub pattern: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalResolution {
    pub approval_id: ApprovalId,
    pub decision: ApprovalDecision,
    pub persisted_rule: Option<PrefixRule>,
    pub reviewer_kind: ApprovalsReviewerKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerConfig {
    pub name: McpServerName,
    pub command: Option<Vec<String>>,
    pub url: Option<UrlString>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub supports_parallel_tool_calls: bool,
    #[serde(default)]
    pub default_requires_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillDefinition {
    pub skill_id: SkillId,
    pub name: String,
    pub description: String,
    pub root_path: PathString,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorDefinition {
    pub connector_id: ConnectorId,
    pub name: String,
    pub description: String,
    pub installed: bool,
    pub authenticated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginDefinition {
    pub plugin_id: PluginId,
    pub name: String,
    pub version: String,
    pub description: String,
    pub enabled: bool,
    #[serde(default)]
    pub skills: Vec<SkillDefinition>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub connectors: Vec<ConnectorDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerDefinition {
    pub config: McpServerConfig,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompactionStrategy {
    SummarizePrefix,
    SummarizeAll,
    TruncatePrefix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistedThread {
    pub metadata: ThreadMetadata,
    pub turns: Vec<TurnMetadata>,
    pub items: Vec<ConversationItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ThreadFilter {
    pub cwd: Option<PathString>,
    pub archived: Option<bool>,
    pub search_term: Option<String>,
    pub limit: i32,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessStartRequest {
    pub process_id: ProcessId,
    pub argv: Vec<String>,
    pub cwd: PathString,
    #[serde(default)]
    pub env: HashMap<String, String>,
    pub tty: bool,
    pub pipe_stdin: bool,
    pub arg0_override: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[allow(dead_code)]
pub enum ReviewTargetKind {
    GitRef,
    GitRange,
    PatchFile,
    WorkingTree,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct ReviewTarget {
    pub kind: ReviewTargetKind,
    pub r#ref: String,
    pub base_path: PathString,
}

pub fn now_seconds() -> TimestampSeconds {
    chrono::Utc::now().timestamp()
}

pub fn now_millis() -> TimestampMillis {
    chrono::Utc::now().timestamp_millis()
}
