use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::intel::CodebaseIntelConfig;

// ── Agent orchestration config ────────────────────────────────────────────────

/// Controls the agent loop, child-agent fan-out, and model-output guards.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AgentConfig {
    /// Absolute per-turn loop backstop.
    #[serde(default = "default_agent_max_iterations")]
    pub max_iterations: usize,
    /// Stop after this many consecutive all-failed tool rounds.
    #[serde(default = "default_agent_max_consecutive_failures")]
    pub max_consecutive_failures: usize,
    /// Number of intent-without-tool retries before accepting the response.
    #[serde(default = "default_agent_max_no_tool_nudges")]
    pub max_no_tool_nudges: usize,
    /// Read-only rounds before nudging the model to act.
    #[serde(default = "default_agent_max_consecutive_read_only_rounds")]
    pub max_consecutive_read_only_rounds: usize,
    /// Empty model responses before failing the turn.
    #[serde(default = "default_agent_max_empty_responses")]
    pub max_empty_responses: usize,
    /// Total nudges before failing the turn.
    #[serde(default = "default_agent_max_total_nudges")]
    pub max_total_nudges: usize,
    /// Streaming text guard for a single model response.
    #[serde(default = "default_agent_max_response_chars")]
    pub max_response_chars: usize,
    /// Maximum total concurrent agents (top-level + child agents).
    #[serde(default = "default_agent_max_concurrent_agents")]
    pub max_concurrent_agents: usize,
    /// Maximum concurrent child agents spawned by one runtime.
    #[serde(default = "default_agent_max_child_agents")]
    pub max_child_agents: usize,
    /// Maximum nesting depth for `spawn_agent`.
    #[serde(default = "default_agent_max_spawn_depth")]
    pub max_spawn_depth: u32,
    /// Default child-agent timeout.
    #[serde(default = "default_agent_child_timeout_secs")]
    pub child_timeout_secs: u64,
    /// Upper bound accepted from a `spawn_agent` tool call.
    #[serde(default = "default_agent_max_child_timeout_secs")]
    pub max_child_timeout_secs: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_agent_max_iterations(),
            max_consecutive_failures: default_agent_max_consecutive_failures(),
            max_no_tool_nudges: default_agent_max_no_tool_nudges(),
            max_consecutive_read_only_rounds: default_agent_max_consecutive_read_only_rounds(),
            max_empty_responses: default_agent_max_empty_responses(),
            max_total_nudges: default_agent_max_total_nudges(),
            max_response_chars: default_agent_max_response_chars(),
            max_concurrent_agents: default_agent_max_concurrent_agents(),
            max_child_agents: default_agent_max_child_agents(),
            max_spawn_depth: default_agent_max_spawn_depth(),
            child_timeout_secs: default_agent_child_timeout_secs(),
            max_child_timeout_secs: default_agent_max_child_timeout_secs(),
        }
    }
}

impl AgentConfig {
    fn normalized(mut self) -> Self {
        self.max_iterations = self.max_iterations.max(1);
        self.max_consecutive_failures = self.max_consecutive_failures.max(1);
        self.max_consecutive_read_only_rounds = self.max_consecutive_read_only_rounds.max(1);
        self.max_empty_responses = self.max_empty_responses.max(1);
        self.max_total_nudges = self.max_total_nudges.max(1);
        self.max_response_chars = self.max_response_chars.max(1);
        self.max_concurrent_agents = self.max_concurrent_agents.max(1);
        self.max_child_agents = self.max_child_agents.max(1);
        self.child_timeout_secs = self.child_timeout_secs.max(1);
        self.max_child_timeout_secs = self.max_child_timeout_secs.max(self.child_timeout_secs);
        self
    }

    /// Child-agent concurrency derived from the total-agent budget.
    /// Reserves one slot for the top-level agent.
    pub fn max_concurrent_child_agents(&self) -> usize {
        self.max_child_agents
            .min(self.max_concurrent_agents.saturating_sub(1))
    }
}

fn default_agent_max_iterations() -> usize {
    150
}

fn default_agent_max_consecutive_failures() -> usize {
    5
}

fn default_agent_max_no_tool_nudges() -> usize {
    2
}

fn default_agent_max_consecutive_read_only_rounds() -> usize {
    4
}

fn default_agent_max_empty_responses() -> usize {
    2
}

fn default_agent_max_total_nudges() -> usize {
    4
}

fn default_agent_max_response_chars() -> usize {
    256_000
}

fn default_agent_max_concurrent_agents() -> usize {
    4
}

fn default_agent_max_child_agents() -> usize {
    4
}

fn default_agent_max_spawn_depth() -> u32 {
    3
}

fn default_agent_child_timeout_secs() -> u64 {
    120
}

fn default_agent_max_child_timeout_secs() -> u64 {
    600
}

// ── Auto-compaction config ────────────────────────────────────────────────────

/// Controls automatic context compaction triggered after each completed turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AutoCompactionConfig {
    /// Whether auto-compaction is active. Default: true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Compact when estimated context usage exceeds this % of the prompt budget.
    /// Expressed as an integer 1–100. Default: 70.
    #[serde(default = "default_compact_threshold")]
    pub threshold_pct: u8,
    /// Per-model overrides: model_id → threshold_pct.
    /// Larger-context models can wait longer; smaller ones should compact earlier.
    #[serde(default)]
    pub model_thresholds: HashMap<String, u8>,
}

impl Default for AutoCompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_pct: default_compact_threshold(),
            model_thresholds: HashMap::new(),
        }
    }
}

fn default_compact_threshold() -> u8 {
    70
}

// ── LLM connection config (API keys, provider URLs) ───────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmConfig {
    #[serde(default)]
    pub api_keys: LlmApiKeysConfig,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub openai: OpenAiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmApiKeysConfig {
    #[serde(default)]
    pub openai: String,
    #[serde(default)]
    pub gemini: String,
    #[serde(default)]
    pub anthropic: String,
    #[serde(default)]
    pub ollama: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
    #[serde(default)]
    pub auth_type: Option<String>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiConfig {
    #[serde(default)]
    pub base_url: Option<String>,
}

fn default_ollama_url() -> String {
    "https://ollama.com".into()
}

// ── Domain imports ────────────────────────────────────────────────────────────

use super::domain::{
    now_seconds, AccountSession, ApprovalPolicy, ApprovalsReviewerKind, AuthMode,
    ConversationPathSet, FeatureKey, McpServerConfig, ModelPreset, ModelSettings, PathString,
    PermissionProfile, ReasoningEffort, SandboxMode,
};

// ── Process context ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ProcessContext {
    pub argv: Vec<String>,
    pub env: HashMap<String, String>,
    pub stdin_is_tty: bool,
    pub stdout_is_tty: bool,
    pub stderr_is_tty: bool,
    pub cwd: PathString,
    pub paths: ConversationPathSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ConfigResolutionInput {
    pub process_context: SerializableProcessContext,
    pub profile_name: Option<String>,
    #[serde(default)]
    pub cli_overrides: HashMap<String, Value>,
    #[serde(default)]
    pub command_defaults: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SerializableProcessContext {
    pub argv: Vec<String>,
    pub env: HashMap<String, String>,
    pub stdin_is_tty: bool,
    pub stdout_is_tty: bool,
    pub stderr_is_tty: bool,
    pub cwd: PathString,
    pub paths: ConversationPathSet,
}

impl From<&ProcessContext> for SerializableProcessContext {
    fn from(value: &ProcessContext) -> Self {
        Self {
            argv: value.argv.clone(),
            env: value.env.clone(),
            stdin_is_tty: value.stdin_is_tty,
            stdout_is_tty: value.stdout_is_tty,
            stderr_is_tty: value.stderr_is_tty,
            cwd: value.cwd.clone(),
            paths: value.paths.clone(),
        }
    }
}

// ── Effective config (resolved, ready to use) ─────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EffectiveConfig {
    pub app_home: PathString,
    pub sqlite_home: PathString,
    pub model_settings: ModelSettings,
    pub approval_policy: ApprovalPolicy,
    pub approvals_reviewer: ApprovalsReviewerKind,
    pub permission_profile: PermissionProfile,
    #[serde(default)]
    pub add_dirs: Vec<PathString>,
    #[serde(default = "default_true")]
    pub notifications_enabled: bool,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    #[serde(default = "default_true")]
    pub plugins_enabled: bool,
    #[serde(default = "default_true")]
    pub apps_enabled: bool,
    #[serde(default)]
    pub features: HashMap<FeatureKey, bool>,
    #[serde(default)]
    pub trusted_projects: Vec<PathString>,
    pub working_directory: PathString,
    pub system_prompt: String,
    pub llm: LlmConfig,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_log_file")]
    pub log_file: String,
    /// User-defined model presets shown in the /model autocomplete list.
    #[serde(default)]
    pub models: Vec<ModelPreset>,
    #[serde(default)]
    pub auto_compaction: AutoCompactionConfig,
    #[serde(default)]
    pub codebase_intel: CodebaseIntelConfig,
    #[serde(default)]
    pub agent: AgentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub struct ConfigEdit {
    pub path: String,
    pub value: Value,
}

// ── Raw spec config (parsed from config file) ─────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
struct RawSpecConfig {
    pub app_home: Option<String>,
    pub model_settings: Option<ModelSettings>,
    pub approval_policy: Option<ApprovalPolicy>,
    pub approvals_reviewer: Option<ApprovalsReviewerKind>,
    pub permission_profile: Option<PermissionProfile>,
    #[serde(default)]
    pub add_dirs: Vec<PathString>,
    pub notifications_enabled: Option<bool>,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    pub plugins_enabled: Option<bool>,
    pub apps_enabled: Option<bool>,
    #[serde(default)]
    pub features: HashMap<FeatureKey, bool>,
    #[serde(default)]
    pub trusted_projects: Vec<PathString>,
    #[serde(default)]
    pub profiles: HashMap<String, RawSpecProfile>,
    pub managed: Option<ManagedRequirements>,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_log_file")]
    pub log_file: String,
    #[serde(default)]
    pub models: Vec<ModelPreset>,
    #[serde(default)]
    pub auto_compaction: AutoCompactionConfig,
    #[serde(default)]
    pub codebase_intel: CodebaseIntelConfig,
    #[serde(default)]
    pub agent: Option<AgentConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
struct RawSpecProfile {
    pub model_settings: Option<ModelSettings>,
    pub approval_policy: Option<ApprovalPolicy>,
    pub approvals_reviewer: Option<ApprovalsReviewerKind>,
    pub permission_profile: Option<PermissionProfile>,
    #[serde(default)]
    pub add_dirs: Vec<PathString>,
    pub notifications_enabled: Option<bool>,
    #[serde(default)]
    pub features: HashMap<FeatureKey, bool>,
    #[serde(default)]
    pub trusted_projects: Vec<PathString>,
    #[serde(default)]
    pub agent: Option<AgentConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
struct ManagedRequirements {
    #[serde(default)]
    pub immutable_paths: Vec<String>,
}

// ── Config manager ────────────────────────────────────────────────────────────

pub struct ConfigManager {
    config_path: PathBuf,
    default_config: Value,
    cached_effective_config: RwLock<Option<EffectiveConfig>>,
}

impl ConfigManager {
    pub fn new(config_path: PathBuf) -> Self {
        Self {
            config_path,
            default_config: default_spec_config_json(),
            cached_effective_config: RwLock::new(None),
        }
    }

    pub fn load_effective_config(&self, input: ConfigResolutionInput) -> Result<EffectiveConfig> {
        let file_json = self.read_config_file()?;
        let mut merged = self.default_config.clone();
        deep_merge(&mut merged, &file_json);

        if let Some(profile_name) = &input.profile_name {
            let profile_value = merged
                .get("profiles")
                .and_then(|v| v.get(profile_name))
                .cloned()
                .ok_or_else(|| anyhow!("unknown profile: {profile_name}"))?;
            deep_merge(&mut merged, &profile_value);
        }

        for (path, value) in input.command_defaults {
            apply_path_override(&mut merged, &path, value)?;
        }

        let immutable = immutable_paths(&merged);
        for (path, value) in input.cli_overrides {
            if immutable.contains(&path) {
                bail!("config_invalid: immutable field override attempted for {path}");
            }
            apply_path_override(&mut merged, &path, value)?;
        }

        let raw: RawSpecConfig = serde_json::from_value(merged.clone())?;
        let app_home = raw
            .app_home
            .clone()
            .unwrap_or_else(|| default_app_home().to_string_lossy().to_string());
        let sqlite_home = Path::new(&app_home)
            .join("state")
            .to_string_lossy()
            .to_string();

        let mut llm = raw.llm;
        apply_env_overrides(&mut llm);

        let mut model_settings = raw.model_settings.unwrap_or_default();

        // If the models preset list has an entry matching the active model_id,
        // backfill any fields that weren't explicitly set in model_settings.
        // This means context_window (and other per-model settings) only need
        // to be defined once in the models list — not duplicated in model_settings.
        if let Some(preset) = raw
            .models
            .iter()
            .find(|m| m.model_id == model_settings.model_id)
        {
            if model_settings.context_window.is_none() {
                model_settings.context_window = preset.context_window;
            }
            if model_settings.reasoning_effort == ReasoningEffort::Auto {
                if let Some(effort) = preset.reasoning_effort {
                    model_settings.reasoning_effort = effort;
                }
            }
        }

        let effective = EffectiveConfig {
            app_home,
            sqlite_home,
            model_settings,
            approval_policy: raw.approval_policy.unwrap_or_default(),
            approvals_reviewer: raw
                .approvals_reviewer
                .unwrap_or(ApprovalsReviewerKind::User),
            permission_profile: raw
                .permission_profile
                .unwrap_or_else(|| default_permission_profile(&input.process_context.cwd)),
            add_dirs: raw.add_dirs,
            notifications_enabled: raw.notifications_enabled.unwrap_or(true),
            mcp_servers: raw.mcp_servers,
            plugins_enabled: raw.plugins_enabled.unwrap_or(true),
            apps_enabled: raw.apps_enabled.unwrap_or(true),
            features: raw.features,
            trusted_projects: raw.trusted_projects,
            working_directory: input.process_context.cwd.clone(),
            system_prompt: raw.system_prompt,
            llm,
            log_level: raw.log_level,
            log_file: raw.log_file,
            models: raw.models,
            auto_compaction: raw.auto_compaction,
            codebase_intel: raw.codebase_intel,
            agent: raw.agent.unwrap_or_default().normalized(),
        };

        *self.cached_effective_config.write().unwrap() = Some(effective.clone());
        Ok(effective)
    }

    #[allow(dead_code)]
    pub fn reload(&self) -> Result<EffectiveConfig> {
        let cached = self.cached_effective_config.read().unwrap().clone();
        let Some(cached) = cached else {
            bail!("no cached effective config");
        };

        self.load_effective_config(ConfigResolutionInput {
            process_context: SerializableProcessContext {
                argv: Vec::new(),
                env: HashMap::new(),
                stdin_is_tty: true,
                stdout_is_tty: true,
                stderr_is_tty: true,
                cwd: cached.working_directory.clone(),
                paths: resolve_paths(
                    Some(PathBuf::from(&cached.app_home)),
                    self.config_path.clone(),
                ),
            },
            profile_name: None,
            cli_overrides: HashMap::new(),
            command_defaults: HashMap::new(),
        })
    }

    pub fn read_config_file(&self) -> Result<Value> {
        if !self.config_path.exists() {
            return Ok(json!({}));
        }
        let content = fs::read_to_string(&self.config_path)
            .with_context(|| format!("reading config file {}", self.config_path.display()))?;
        let yaml: serde_yaml::Value = serde_yaml::from_str(&content)
            .with_context(|| format!("parsing config file {}", self.config_path.display()))?;
        Ok(serde_json::to_value(yaml)?)
    }

    #[allow(dead_code)]
    pub fn write_config_edits(&self, edits: Vec<ConfigEdit>) -> Result<EffectiveConfig> {
        let mut value = self.read_config_file()?;
        for edit in edits {
            apply_path_override(&mut value, &edit.path, edit.value)?;
        }

        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let yaml = serde_yaml::to_string(&value)?;
        fs::write(&self.config_path, yaml)?;

        let paths = resolve_paths(None, self.config_path.clone());
        let ctx = ProcessContext {
            argv: Vec::new(),
            env: HashMap::new(),
            stdin_is_tty: true,
            stdout_is_tty: true,
            stderr_is_tty: true,
            cwd: std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .to_string_lossy()
                .to_string(),
            paths,
        };

        self.load_effective_config(ConfigResolutionInput {
            process_context: SerializableProcessContext::from(&ctx),
            profile_name: None,
            cli_overrides: HashMap::new(),
            command_defaults: HashMap::new(),
        })
    }
}

// ── Auth manager ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AuthRequirement {
    None,
    RemoteAccountRequired,
    ApiTokenAllowed,
    BrowserSessionAllowed,
}

pub struct AuthManager {
    session_path: PathBuf,
    session: Arc<RwLock<AccountSession>>,
}

impl AuthManager {
    pub fn new(paths: &ConversationPathSet) -> Result<Self> {
        let session_path = Path::new(&paths.app_home).join("auth").join("session.json");
        if let Some(parent) = session_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let session = if session_path.exists() {
            serde_json::from_str::<AccountSession>(&fs::read_to_string(&session_path)?)?
        } else {
            AccountSession::default()
        };
        Ok(Self {
            session_path,
            session: Arc::new(RwLock::new(session)),
        })
    }

    pub fn login_with_api_token(&self, api_token: String) -> Result<AccountSession> {
        let session = AccountSession {
            auth_mode: AuthMode::ApiToken,
            account_id: Some(format!("local-{}", now_seconds())),
            email: None,
            access_token: None,
            api_token: Some(api_token),
            expires_at: None,
        };
        self.persist(session)
    }

    pub fn login_with_browser_session(&self) -> Result<AccountSession> {
        let session = AccountSession {
            auth_mode: AuthMode::BrowserSession,
            account_id: Some(format!("browser-{}", now_seconds())),
            email: None,
            access_token: Some("browser-session".into()),
            api_token: None,
            expires_at: None,
        };
        self.persist(session)
    }

    pub fn login_with_device_code(&self) -> Result<AccountSession> {
        let session = AccountSession {
            auth_mode: AuthMode::DeviceCode,
            account_id: Some(format!("device-{}", now_seconds())),
            email: None,
            access_token: Some("device-session".into()),
            api_token: None,
            expires_at: None,
        };
        self.persist(session)
    }

    pub fn logout(&self) -> Result<()> {
        self.persist(AccountSession::default())?;
        Ok(())
    }

    pub fn get_session(&self) -> AccountSession {
        self.session.read().unwrap().clone()
    }

    #[allow(dead_code)]
    pub fn ensure_authenticated(&self, requirement: AuthRequirement) -> Result<AccountSession> {
        let session = self.get_session();
        match requirement {
            AuthRequirement::None => Ok(session),
            AuthRequirement::RemoteAccountRequired => {
                if session.auth_mode == AuthMode::None {
                    bail!("auth_required: remote account required");
                }
                Ok(session)
            }
            AuthRequirement::ApiTokenAllowed => {
                if session.auth_mode == AuthMode::None {
                    bail!("auth_required: api token or session required");
                }
                Ok(session)
            }
            AuthRequirement::BrowserSessionAllowed => {
                if session.auth_mode == AuthMode::None {
                    bail!("auth_required: browser session required");
                }
                Ok(session)
            }
        }
    }

    fn persist(&self, session: AccountSession) -> Result<AccountSession> {
        *self.session.write().unwrap() = session.clone();
        fs::write(&self.session_path, serde_json::to_string_pretty(&session)?)?;
        Ok(session)
    }
}

// ── Path resolution ───────────────────────────────────────────────────────────

pub fn resolve_paths(
    app_home_override: Option<PathBuf>,
    config_path: PathBuf,
) -> ConversationPathSet {
    let app_home = app_home_override.unwrap_or_else(default_app_home);
    let state_root = app_home.join("state");
    ConversationPathSet {
        app_home: app_home.to_string_lossy().to_string(),
        sessions_root: app_home.join("sessions").to_string_lossy().to_string(),
        archived_sessions_root: app_home.join("archived").to_string_lossy().to_string(),
        memories_root: app_home.join("memories").to_string_lossy().to_string(),
        config_path: config_path.to_string_lossy().to_string(),
        state_root: state_root.to_string_lossy().to_string(),
        control_socket_path: app_home.join("control.sock").to_string_lossy().to_string(),
        logs_root: app_home.join("logs").to_string_lossy().to_string(),
    }
}

fn default_app_home() -> PathBuf {
    if let Ok(path) = std::env::var("CODEZILLA_HOME") {
        return PathBuf::from(path);
    }
    if let Ok(cwd) = std::env::current_dir() {
        return cwd.join(".codezilla");
    }
    if let Some(path) = dirs::data_local_dir() {
        path.join("codezilla")
    } else {
        PathBuf::from(".codezilla")
    }
}

// ── Defaults ──────────────────────────────────────────────────────────────────

fn default_spec_config_json() -> Value {
    json!({
        "model_settings": {
            "model_id": "glm-5.1:cloud",
            "provider_id": "ollama",
            "web_search_enabled": false
        },
        "llm": {
            "ollama": {
                "base_url": "https://ollama.com",
                "auth_type": "bearer"
            }
        },
        "approval_policy": { "kind": "ON_REQUEST" },
        "approvals_reviewer": "USER",
        "permission_profile": {
            "sandboxMode": "workspace-write",
            "writableRoots": [],
            "networkEnabled": false,
            "allowedDomains": [],
            "allowedUnixSockets": []
        },
        "notifications_enabled": true,
        "plugins_enabled": true,
        "apps_enabled": true,
        "agent": {
            "max_iterations": 150,
            "max_consecutive_failures": 5,
            "max_no_tool_nudges": 2,
            "max_consecutive_read_only_rounds": 4,
            "max_empty_responses": 2,
            "max_total_nudges": 4,
            "max_response_chars": 256000,
            "max_concurrent_agents": 4,
            "max_child_agents": 4,
            "max_spawn_depth": 3,
            "child_timeout_secs": 120,
            "max_child_timeout_secs": 600
        },
        "auto_compaction": {
            "enabled": true,
            "threshold_pct": 70,
            "model_thresholds": {
                "claude-opus-4-5": 80,
                "claude-sonnet-4-5": 80,
                "claude-opus-4-7": 80,
                "claude-sonnet-4-6": 80,
                "gpt-4o": 75,
                "gpt-4o-mini": 70,
                "qwen3-coder:480b": 72,
                "kimi-k2.6:cloud": 65,
                "glm-5.1:cloud": 65,
                "deepseek-v4-flash:cloud": 68
            }
        },
        "models": [
            { "model_id": "glm-5.1:cloud", "provider_id": "ollama", "context_window": 198000, "modalities": ["text"] },
            { "model_id": "qwen3-coder-next:cloud", "provider_id": "ollama", "context_window": 256000, "modalities": ["text"] },
            { "model_id": "kimi-k2.6:cloud", "provider_id": "ollama", "context_window": 256000, "modalities": ["text", "vision"] },
            { "model_id": "deepseek-v4-flash:cloud", "provider_id": "ollama", "context_window": 1000000, "modalities": ["text"] },
            { "model_id": "deepseek-v4-pro:cloud", "provider_id": "ollama", "context_window": 1000000, "modalities": ["text"] },
            { "model_id": "claude-opus-4-5", "provider_id": "anthropic", "reasoning_effort": "medium", "modalities": ["text", "vision"] },
            { "model_id": "gemma4:cloud", "provider_id": "ollama", "context_window": 256000, "modalities": ["text", "vision"] }
        ],
        "features": {},
        "trusted_projects": []
    })
}

fn default_true() -> bool {
    true
}

fn default_permission_profile(cwd: &str) -> PermissionProfile {
    PermissionProfile {
        sandbox_mode: SandboxMode::WorkspaceWrite,
        writable_roots: vec![super::domain::WritableRoot { path: cwd.into() }],
        network_enabled: false,
        allowed_domains: Vec::new(),
        allowed_unix_sockets: Vec::new(),
    }
}

fn default_log_level() -> String {
    "info".into()
}

fn default_log_file() -> String {
    "logs/codezilla.log".into()
}

// ── Environment variable overrides ────────────────────────────────��───────────

fn apply_env_overrides(llm: &mut LlmConfig) {
    let _ = dotenvy::dotenv();

    if let Ok(v) = std::env::var("OLLAMA_API_KEY") {
        if !v.is_empty() {
            llm.api_keys.ollama = v;
            if llm.ollama.auth_type.is_none() {
                llm.ollama.auth_type = Some("bearer".into());
            }
        }
    }
    if let Ok(v) = std::env::var("OLLAMA_USERNAME") {
        if !v.is_empty() {
            llm.ollama.username = Some(v);
            llm.ollama.auth_type = Some("basic".into());
        }
    }
    if let Ok(v) = std::env::var("OLLAMA_PASSWORD") {
        if !v.is_empty() {
            llm.ollama.password = Some(v);
        }
    }
    if let Ok(v) = std::env::var("OLLAMA_BASE_URL") {
        if !v.is_empty() {
            llm.ollama.base_url = v;
        }
    }
    if let Ok(v) = std::env::var("OPENAI_BASE_URL") {
        if !v.is_empty() {
            llm.openai.base_url = Some(v);
        }
    }
    if let Ok(v) = std::env::var("OPENAI_API_KEY") {
        if !v.is_empty() {
            llm.api_keys.openai = v;
        }
    }
    if let Ok(v) = std::env::var("ANTHROPIC_API_KEY") {
        if !v.is_empty() {
            llm.api_keys.anthropic = v;
        }
    }
    if let Ok(v) = std::env::var("GEMINI_API_KEY") {
        if !v.is_empty() {
            llm.api_keys.gemini = v;
        }
    }
}

// ── Config helpers ────────────────────────────────────────────────────────────

fn immutable_paths(value: &Value) -> Vec<String> {
    value
        .get("managed")
        .and_then(|v| v.get("immutablePaths"))
        .and_then(Value::as_array)
        .map(|paths| {
            paths
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn deep_merge(target: &mut Value, overlay: &Value) {
    match (target, overlay) {
        (Value::Object(target), Value::Object(overlay)) => {
            for (key, value) in overlay {
                match target.get_mut(key) {
                    Some(existing) => deep_merge(existing, value),
                    None => {
                        target.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (target, overlay) => *target = overlay.clone(),
    }
}

fn apply_path_override(root: &mut Value, path: &str, value: Value) -> Result<()> {
    let mut parts = path.split('.').peekable();
    let mut cursor = root;
    while let Some(part) = parts.next() {
        let is_last = parts.peek().is_none();
        if is_last {
            ensure_object(cursor)?;
            cursor
                .as_object_mut()
                .unwrap()
                .insert(part.to_string(), value);
            return Ok(());
        }

        ensure_object(cursor)?;
        let object = cursor.as_object_mut().unwrap();
        cursor = object
            .entry(part.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
    }
    Ok(())
}

fn ensure_object(value: &mut Value) -> Result<()> {
    if value.is_null() {
        *value = Value::Object(Map::new());
    }
    if !value.is_object() {
        bail!("config_invalid: expected object while applying config override");
    }
    Ok(())
}
