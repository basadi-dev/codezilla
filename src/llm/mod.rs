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

/// A single content part within a message — supports multi-modal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { mime_type: String, data: String }, // base64-encoded
}
impl ContentPart {
    #[allow(dead_code)]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
    pub fn image(mime_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image {
            mime_type: mime_type.into(),
            data: data.into(),
        }
    }
}
/// A single message in the conversation. Models the Go `agent.Message` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    /// Primary text content. For backward compatibility this is always populated;
    /// multi-modal content (images) is carried in `content_parts`.
    pub content: String,
    /// Multi-modal content parts. When empty, providers serialize `content` as a
    /// plain string. When non-empty, providers serialize as a content array and
    /// `content` is treated as the first text part (if present).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_parts: Vec<ContentPart>,
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
    /// Create a user message with both text and image attachments.
    pub fn user_with_images(text: impl Into<String>, images: Vec<ContentPart>) -> Self {
        let text_str = text.into();
        Self {
            role: Role::User,
            content: text_str,
            content_parts: images, // text lives in `content`; providers add it separately
            tool_calls: vec![],
            tool_result: None,
            think_content: String::new(),
            timestamp: chrono::Utc::now(),
        }
    }
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            content_parts: vec![],
            tool_calls: vec![],
            tool_result: None,
            think_content: String::new(),
            timestamp: chrono::Utc::now(),
        }
    }
    /// Returns true when this message carries image content parts.
    pub fn has_images(&self) -> bool {
        self.content_parts
            .iter()
            .any(|p| matches!(p, ContentPart::Image { .. }))
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

// ── LLM response types ───���────────────────────────────────────────────────────

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
    #[serde(default)]
    pub cached_tokens: u64,
}

// ── Stream chunk ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    Thinking(String),
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
    #[allow(clippy::too_many_arguments)]
    async fn complete(
        &self,
        provider_id: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
        reasoning_effort: Option<&str>,
        max_tokens: usize,
    ) -> Result<LlmResponse>;

    /// Streaming completion — returns a channel of `StreamChunk`s.
    #[allow(clippy::too_many_arguments)]
    async fn stream(
        &self,
        provider_id: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
        reasoning_effort: Option<&str>,
        max_tokens: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>>;
}

// ── Provider trait abstraction ────────────────────────────────────────────────

/// Flattened request struct that replaces the 8-argument provider signatures.
/// Every provider takes the same parameters — this makes the contract explicit.
#[derive(Debug, Clone)]
pub struct CompletionRequest<'a> {
    pub messages: &'a [Message],
    pub tools: &'a [ToolDefinition],
    pub model: &'a str,
    pub temperature: f32,
    pub reasoning_effort: Option<&'a str>,
    pub max_tokens: usize,
}

/// Capability flags for a provider. Used by the executor to avoid sending
/// unsupported parameters (e.g. `reasoning_effort` to a provider that ignores it).
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ProviderCaps {
    /// Whether the provider supports streaming responses.
    pub streaming: bool,
    /// Whether the provider supports reasoning effort / thinking configuration.
    pub reasoning_effort: bool,
    /// Whether the provider supports vision (image inputs).
    pub vision: bool,
}

/// Trait implemented by each LLM provider. New providers implement this trait
/// and register with the `UnifiedClient` — no need to edit the dispatch match.
#[async_trait::async_trait]
#[allow(dead_code)]
pub trait LlmProvider: Send + Sync {
    /// Unique identifier for this provider (e.g. "ollama", "openai").
    fn id(&self) -> &str;

    /// Declare what this provider supports.
    fn capabilities(&self) -> ProviderCaps;

    /// Non-streaming completion.
    async fn complete(&self, req: CompletionRequest<'_>) -> Result<LlmResponse>;

    /// Streaming completion — returns a channel of `StreamChunk`s.
    async fn stream(
        &self,
        req: CompletionRequest<'_>,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>>;
}
