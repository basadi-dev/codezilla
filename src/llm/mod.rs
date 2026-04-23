#![allow(async_fn_in_trait)]

pub mod client;
pub mod providers;

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ── Core message types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub result: Option<String>,
    pub error: Option<String>,
}

/// A single message in the conversation. Models the Go `agent.Message` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolResult>,
    /// Internal reasoning block — stored for logging but never sent to LLM.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub think_content: String,
    #[serde(default)]
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: vec![],
            tool_result: None,
            think_content: String::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}

// ── Tool schema types (for sending tool definitions to the LLM) ───────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub typ: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ── LLM response types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct LlmResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// ── Stream chunk ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    Usage(TokenUsage),
    Done,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Non-streaming completion.
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
        reasoning_effort: Option<&str>,
        max_tokens: usize,
    ) -> Result<LlmResponse>;

    /// Streaming completion — returns a channel of `StreamChunk`s.
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
        reasoning_effort: Option<&str>,
        max_tokens: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>>;
}
