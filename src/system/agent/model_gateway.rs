use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::llm::{self, LlmClient, StreamChunk};
use crate::system::domain::{
    ConversationItem, ItemKind, ModelId, ModelSettings, TokenUsage, ToolCall, ToolDefinition,
    ToolProviderKind,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescription {
    pub model_id: ModelId,
    pub provider_id: String,
    pub display_name: String,
    pub hidden: bool,
    pub is_default: bool,
    #[serde(default)]
    pub reasoning_efforts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRequest {
    pub model_settings: ModelSettings,
    #[serde(default)]
    pub system_instructions: Vec<String>,
    #[serde(default)]
    pub conversation_items: Vec<ConversationItem>,
    #[serde(default)]
    pub tool_definitions: Vec<ToolDefinition>,
    pub output_schema: Option<Value>,
}

#[derive(Debug, Clone)]
pub enum ModelStreamEvent {
    AssistantDelta(String),
    #[allow(dead_code)]
    ReasoningDelta(String),
    ToolCalls(Vec<ToolCall>),
    Completed(TokenUsage),
    Failed(String),
}

pub struct ModelGateway {
    client: Arc<dyn LlmClient>,
}

impl ModelGateway {
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self { client }
    }

    /// Direct access to the underlying LLM client (used by compact_thread).
    pub fn inner_client(&self) -> &Arc<dyn LlmClient> {
        &self.client
    }

    pub fn list_models(&self, settings: &ModelSettings) -> Vec<ModelDescription> {
        vec![ModelDescription {
            model_id: settings.model_id.clone(),
            provider_id: settings.provider_id.clone(),
            display_name: settings.model_id.clone(),
            hidden: false,
            is_default: true,
            reasoning_efforts: vec!["low".into(), "medium".into(), "high".into()],
        }]
    }

    pub async fn start_response(
        &self,
        request: ModelRequest,
    ) -> Result<mpsc::Receiver<ModelStreamEvent>> {
        let messages =
            build_llm_messages(&request.system_instructions, &request.conversation_items);
        let tools = request
            .tool_definitions
            .iter()
            .map(|t| llm::ToolDefinition {
                typ: "function".into(),
                function: llm::FunctionDefinition {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect::<Vec<_>>();

        let (tx, rx) = mpsc::channel(256);
        let client = self.client.clone();
        let model_settings = request.model_settings.clone();

        tokio::spawn(async move {
            match client
                .stream(
                    &model_settings.provider_id,
                    &messages,
                    &tools,
                    &model_settings.model_id,
                    0.2,
                    model_settings.reasoning_effort.as_deref(),
                    8192,
                )
                .await
            {
                Ok(mut stream) => {
                    let mut usage = TokenUsage::default();
                    let mut partial_tools: HashMap<
                        usize,
                        (Option<String>, Option<String>, String),
                    > = HashMap::new();
                    while let Some(chunk) = stream.recv().await {
                        match chunk {
                            StreamChunk::Text(text) => {
                                let _ = tx.send(ModelStreamEvent::AssistantDelta(text)).await;
                            }
                            StreamChunk::Thinking(thought) => {
                                let _ = tx.send(ModelStreamEvent::ReasoningDelta(thought)).await;
                            }
                            StreamChunk::ToolCallDelta {
                                index,
                                id,
                                name,
                                arguments_delta,
                            } => {
                                let entry = partial_tools.entry(index).or_default();
                                if let Some(id) = id {
                                    entry.0 = Some(id);
                                }
                                if let Some(name) = name {
                                    entry.1 = Some(name);
                                }
                                if let Some(args) = arguments_delta {
                                    entry.2.push_str(&args);
                                }
                            }
                            StreamChunk::Usage(tokens) => {
                                usage = TokenUsage {
                                    input_tokens: tokens.prompt_tokens as i64,
                                    output_tokens: tokens.completion_tokens as i64,
                                    cached_tokens: 0,
                                };
                            }
                            StreamChunk::Done => break,
                        }
                    }
                    if !partial_tools.is_empty() {
                        let calls = partial_tools
                            .into_iter()
                            .map(|(_, (id, name, args))| ToolCall {
                                tool_call_id: id.unwrap_or_else(|| {
                                    format!("call_{}", uuid::Uuid::new_v4().simple())
                                }),
                                provider_kind: ToolProviderKind::Builtin,
                                tool_name: name.unwrap_or_default(),
                                arguments: serde_json::from_str(&args)
                                    .unwrap_or_else(|_| serde_json::json!({})),
                            })
                            .collect::<Vec<_>>();
                        let _ = tx.send(ModelStreamEvent::ToolCalls(calls)).await;
                    }
                    let _ = tx.send(ModelStreamEvent::Completed(usage)).await;
                }
                Err(stream_err) => {
                    let stream_err_text = stream_err.to_string();
                    if is_non_retryable_stream_error(&stream_err_text) {
                        let _ = tx
                            .send(ModelStreamEvent::Failed(humanize_stream_error(
                                &stream_err_text,
                            )))
                            .await;
                        return;
                    }

                    match client
                        .complete(
                            &model_settings.provider_id,
                            &messages,
                            &tools,
                            &model_settings.model_id,
                            0.2,
                            model_settings.reasoning_effort.as_deref(),
                            8192,
                        )
                        .await
                    {
                        Ok(response) => {
                            if !response.content.is_empty() {
                                let _ = tx
                                    .send(ModelStreamEvent::AssistantDelta(response.content))
                                    .await;
                            }
                            if !response.tool_calls.is_empty() {
                                let calls = response
                                    .tool_calls
                                    .into_iter()
                                    .map(|c| ToolCall {
                                        tool_call_id: c.id,
                                        provider_kind: ToolProviderKind::Builtin,
                                        tool_name: c.function.name,
                                        arguments: serde_json::from_str(&c.function.arguments)
                                            .unwrap_or_else(|_| serde_json::json!({})),
                                    })
                                    .collect::<Vec<_>>();
                                let _ = tx.send(ModelStreamEvent::ToolCalls(calls)).await;
                            }
                            let usage = response.usage.unwrap_or_default();
                            let _ = tx
                                .send(ModelStreamEvent::Completed(TokenUsage {
                                    input_tokens: usage.prompt_tokens as i64,
                                    output_tokens: usage.completion_tokens as i64,
                                    cached_tokens: 0,
                                }))
                                .await;
                        }
                        Err(complete_err) => {
                            let _ = tx
                                .send(ModelStreamEvent::Failed(format!(
                                    "{}; fallback complete failed: {}",
                                    humanize_stream_error(&stream_err_text),
                                    humanize_stream_error(&complete_err.to_string())
                                )))
                                .await;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

pub fn is_context_overflow_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("context exceeded")
        || lower.contains("maximum context length")
        || lower.contains("context window")
        || lower.contains("prompt is too long")
        || lower.contains("prompt too long")
        || (lower.contains("api error 400") && lower.contains("context"))
}

fn is_non_retryable_stream_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("api error 400")
        || lower.contains("api error 401")
        || lower.contains("api error 403")
        || lower.contains("api error 404")
        || lower.contains("api error 413")
        || lower.contains("api error 422")
        || is_context_overflow_error(err)
}

fn humanize_stream_error(err: &str) -> String {
    if is_context_overflow_error(err) {
        format!(
            "{err}. The conversation exceeded the model context window. \
             Start a new thread, compact this one with /compact, or use a model with a larger context."
        )
    } else {
        err.to_string()
    }
}

// ─── Token budget constants ───────────────────────────────────────────────────

/// Conservative estimate: 4 characters ≈ 1 token (GPT/Claude average).
const CHARS_PER_TOKEN: usize = 4;

/// Maximum context tokens we ever attempt to fill.
/// 100 k is a safe ceiling that fits within every model we support;
/// the actual model limit may be larger but we stay well clear of it.
const MAX_CONTEXT_TOKENS: usize = 100_000;

/// Tokens we always reserve for the model's output.
const RESERVED_OUTPUT_TOKENS: usize = 8_192;

/// Token budget available for the full prompt (system + history).
const PROMPT_TOKEN_BUDGET: usize = MAX_CONTEXT_TOKENS - RESERVED_OUTPUT_TOKENS;

/// Rough token count for a single `ConversationItem`.
fn item_token_estimate(item: &ConversationItem) -> usize {
    item.payload.to_string().len().saturating_add(32) / CHARS_PER_TOKEN
}

/// Apply the token-budget guard and convert items to LLM messages.
///
/// Algorithm:
/// 1. Always include system instructions.
/// 2. Group items into "turns" (sequences that belong together — user message,
///    optional tool-call/result pairs, and the assistant reply).
/// 3. Walk turns newest → oldest, adding them while budget permits.
///    The most recent turn is *always* kept regardless of size.
/// 4. If any turns were dropped, prepend a synthetic system notice.
pub fn build_llm_messages(
    system_instructions: &[String],
    items: &[ConversationItem],
) -> Vec<llm::Message> {
    // ── 1. Measure system tokens ──────────────────────────────────────────────
    let system_tokens: usize = system_instructions
        .iter()
        .map(|s| s.len().saturating_add(16) / CHARS_PER_TOKEN)
        .sum();
    let mut remaining_budget = PROMPT_TOKEN_BUDGET.saturating_sub(system_tokens);

    // ── 2. Group consecutive items into logical turns ─────────────────────────
    // A "turn boundary" is any UserMessage or a ReasoningSummary (compaction
    // marker). Everything between boundaries belongs to the same turn.
    let mut turns: Vec<Vec<&ConversationItem>> = Vec::new();
    let mut current: Vec<&ConversationItem> = Vec::new();

    for item in items {
        let is_boundary = matches!(
            item.kind,
            ItemKind::UserMessage | ItemKind::ReasoningSummary
        );
        if is_boundary && !current.is_empty() {
            turns.push(std::mem::take(&mut current));
        }
        current.push(item);
    }
    if !current.is_empty() {
        turns.push(current);
    }

    // ── 3. Walk newest → oldest, keep turns that fit in budget ────────────────
    let mut kept_indices: Vec<usize> = Vec::new();
    for (idx, turn) in turns.iter().enumerate().rev() {
        let cost: usize = turn.iter().map(|i| item_token_estimate(i)).sum();
        if kept_indices.is_empty() {
            // Always keep the most recent turn.
            kept_indices.push(idx);
            remaining_budget = remaining_budget.saturating_sub(cost);
        } else if cost <= remaining_budget {
            kept_indices.push(idx);
            remaining_budget = remaining_budget.saturating_sub(cost);
        } else {
            break; // Older turns won't fit either.
        }
    }
    kept_indices.sort_unstable();
    let truncated = kept_indices.len() < turns.len();
    let dropped_turns = turns.len().saturating_sub(kept_indices.len());

    // ── 4. Assemble messages ──────────────────────────────────────────────────
    let mut messages = Vec::new();

    for instruction in system_instructions {
        messages.push(llm::Message::system(instruction.clone()));
    }

    if truncated {
        messages.push(llm::Message::system(format!(
            "[Note: {dropped_turns} earlier conversation turn(s) were omitted to fit within \
             the context window. The conversation continues from a later point.]"
        )));
    }

    for &idx in &kept_indices {
        for item in &turns[idx] {
            if let Some(msg) = item_to_llm_message(item) {
                messages.push(msg);
            }
        }
    }

    messages
}

/// Convert a single `ConversationItem` to an `llm::Message`, returning `None`
/// for item kinds that have no LLM representation (Status, Error, etc.).
fn item_to_llm_message(item: &ConversationItem) -> Option<llm::Message> {
    match item.kind {
        ItemKind::UserMessage => {
            let text = item.payload.get("text").and_then(Value::as_str)?;
            Some(llm::Message::user(text.to_string()))
        }
        ItemKind::UserAttachment => {
            let path = item.payload.get("path").and_then(Value::as_str)?;
            Some(llm::Message::user(format!("[image:{}]", path)))
        }
        ItemKind::AgentMessage | ItemKind::ReasoningSummary => {
            let text = item.payload.get("text").and_then(Value::as_str)?;
            Some(llm::Message::assistant(text.to_string()))
        }
        ItemKind::ToolCall => {
            let tc = serde_json::from_value::<ToolCall>(item.payload.clone()).ok()?;
            Some(llm::Message {
                role: llm::Role::Assistant,
                content: String::new(),
                tool_calls: vec![llm::ToolCall {
                    id: tc.tool_call_id,
                    function: llm::FunctionCall {
                        name: tc.tool_name,
                        arguments: tc.arguments.to_string(),
                    },
                }],
                tool_result: None,
                think_content: String::new(),
                timestamp: chrono::Utc::now(),
            })
        }
        ItemKind::ToolResult => {
            let tr =
                serde_json::from_value::<crate::system::domain::ToolResult>(item.payload.clone())
                    .ok()?;
            Some(llm::Message {
                role: llm::Role::Tool,
                content: String::new(),
                tool_calls: Vec::new(),
                tool_result: Some(llm::ToolResult {
                    tool_call_id: tr.tool_call_id,
                    name: String::new(),
                    result: Some(tr.output.to_string()),
                    error: tr.error_message,
                }),
                think_content: String::new(),
                timestamp: chrono::Utc::now(),
            })
        }
        _ => None,
    }
}

/// Build a summarization prompt for `compact_thread`. Returns the messages that
/// should be sent to a fast model to produce a conversation summary.
pub fn build_compaction_messages(
    system_instructions: &[String],
    items: &[ConversationItem],
) -> Vec<llm::Message> {
    // Render the raw history as a simple text transcript.
    let mut transcript = String::new();
    for item in items {
        match item.kind {
            ItemKind::UserMessage => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    transcript.push_str(&format!("USER: {text}\n\n"));
                }
            }
            ItemKind::AgentMessage => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    transcript.push_str(&format!("ASSISTANT: {text}\n\n"));
                }
            }
            ItemKind::ToolCall => {
                if let Some(name) = item.payload.get("toolName").and_then(Value::as_str) {
                    transcript.push_str(&format!("TOOL CALL: {name}\n\n"));
                }
            }
            ItemKind::ToolResult => {
                let ok = item.payload.get("ok").and_then(Value::as_bool).unwrap_or(true);
                let out = item.payload.get("output").map(|v| v.to_string()).unwrap_or_default();
                let status = if ok { "OK" } else { "ERROR" };
                // Keep tool outputs brief in the compaction prompt.
                let snippet: String = out.chars().take(512).collect();
                transcript.push_str(&format!("TOOL RESULT ({status}): {snippet}\n\n"));
            }
            _ => {}
        }
    }

    let system_ctx = system_instructions.first().cloned().unwrap_or_default();
    let prompt = format!(
        "You are summarizing a conversation for context compaction. \
         Produce a concise but complete summary that preserves: \
         (1) key decisions made, \
         (2) files created or modified and their purpose, \
         (3) open tasks or next steps, \
         (4) any important context the assistant needs to continue effectively. \
         Write in third person past tense. \
         Do NOT include pleasantries or meta-commentary about the summary itself.\n\n\
         SYSTEM CONTEXT:\n{system_ctx}\n\n\
         CONVERSATION:\n{transcript}"
    );

    vec![llm::Message::user(prompt)]
}
