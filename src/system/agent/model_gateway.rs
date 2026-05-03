use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::llm::{self, LlmClient, StreamChunk};
use crate::system::domain::{
    ConversationItem, ItemKind, ModelId, ModelSettings, TokenUsage, ToolCall, ToolDefinition,
    ToolProviderKind,
};
use crate::system::error as cod_error;

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
    /// Incremental token usage update during streaming (live, may be partial).
    StreamingUsage(TokenUsage),
    Completed(TokenUsage),
    Failed {
        display: String,
        raw: String,
    },
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
        cancel_token: CancellationToken,
    ) -> Result<mpsc::Receiver<ModelStreamEvent>> {
        let prompt_budget = calculate_prompt_budget(request.model_settings.context_window);
        let messages = build_llm_messages(
            &request.system_instructions,
            &request.conversation_items,
            prompt_budget,
        );
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
            let stream_result = tokio::select! {
                _ = cancel_token.cancelled() => return,
                result = client.stream(
                    &model_settings.provider_id,
                    &messages,
                    &tools,
                    &model_settings.model_id,
                    0.2,
                    model_settings.reasoning_effort.as_deref(),
                    8192,
                ) => result,
            };

            match stream_result {
                Ok(mut stream) => {
                    let mut usage = TokenUsage::default();
                    // Estimated usage that is updated in real time from streaming text.
                    let mut estimated_usage = TokenUsage {
                        input_tokens: estimate_messages_tokens(&messages) as i64,
                        output_tokens: 0,
                        cached_tokens: 0,
                    };
                    // Publish initial prompt estimate immediately (before first token arrives).
                    let _ = send_model_event(
                        &tx,
                        &cancel_token,
                        ModelStreamEvent::StreamingUsage(estimated_usage.clone()),
                    )
                    .await;
                    let mut partial_tools: HashMap<
                        usize,
                        (Option<String>, Option<String>, String),
                    > = HashMap::new();
                    loop {
                        let chunk = tokio::select! {
                            _ = cancel_token.cancelled() => return,
                            chunk = stream.recv() => chunk,
                        };
                        let Some(chunk) = chunk else {
                            break;
                        };
                        match chunk {
                            StreamChunk::Text(text) => {
                                // Live output estimate: update on every streamed text delta.
                                estimated_usage.output_tokens += estimate_text_tokens(&text) as i64;
                                if !send_model_event(
                                    &tx,
                                    &cancel_token,
                                    ModelStreamEvent::AssistantDelta(text),
                                )
                                .await
                                {
                                    return;
                                }
                                let _ = send_model_event(
                                    &tx,
                                    &cancel_token,
                                    ModelStreamEvent::StreamingUsage(estimated_usage.clone()),
                                )
                                .await;
                            }
                            StreamChunk::Thinking(thought) => {
                                if !send_model_event(
                                    &tx,
                                    &cancel_token,
                                    ModelStreamEvent::ReasoningDelta(thought),
                                )
                                .await
                                {
                                    return;
                                }
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
                                // Only accept provider usage if it carries real (non-zero)
                                // completion tokens — some providers send usage chunks
                                // with completion_tokens = 0 mid-stream, which would
                                // overwrite our live estimate with zeros and cause the
                                // token display to flicker/disappear.
                                if tokens.completion_tokens > 0 {
                                    usage = TokenUsage {
                                        input_tokens: tokens.prompt_tokens as i64,
                                        output_tokens: tokens.completion_tokens as i64,
                                        cached_tokens: tokens.cached_tokens as i64,
                                    };
                                    // If provider usage arrives, treat it as authoritative while streaming.
                                    estimated_usage = usage.clone();
                                    // Forward the real usage to the TUI immediately so
                                    // ctx % updates even if no more text chunks arrive.
                                    let _ = send_model_event(
                                        &tx,
                                        &cancel_token,
                                        ModelStreamEvent::StreamingUsage(estimated_usage.clone()),
                                    )
                                    .await;
                                } else if tokens.prompt_tokens > 0 {
                                    // Provider gave us input tokens but no output yet —
                                    // preserve the input count but keep our output estimate.
                                    estimated_usage.input_tokens = tokens.prompt_tokens as i64;
                                    estimated_usage.cached_tokens = tokens.cached_tokens as i64;
                                    // Forward input token count to TUI so ctx % updates.
                                    let _ = send_model_event(
                                        &tx,
                                        &cancel_token,
                                        ModelStreamEvent::StreamingUsage(estimated_usage.clone()),
                                    )
                                    .await;
                                }
                            }
                            StreamChunk::Done => break,
                        }
                    }
                    if !partial_tools.is_empty() {
                        if cancel_token.is_cancelled() {
                            return;
                        }
                        let mut calls = Vec::new();
                        for (_, (id, name, args)) in partial_tools {
                            let tool_name = name.unwrap_or_default();
                            if tool_name.is_empty() {
                                tracing::warn!("model_gateway: discarding tool call with empty name (args: {args:?})");
                                continue;
                            }
                            let base_id = id.unwrap_or_else(|| {
                                format!("call_{}", uuid::Uuid::new_v4().simple())
                            });
                            let argument_list = parse_tool_arguments_lenient(&tool_name, &args);
                            for (i, arguments) in argument_list.into_iter().enumerate() {
                                let tool_call_id = if i == 0 {
                                    base_id.clone()
                                } else {
                                    format!("call_{}", uuid::Uuid::new_v4().simple())
                                };
                                calls.push(ToolCall {
                                    tool_call_id,
                                    provider_kind: ToolProviderKind::Builtin,
                                    tool_name: tool_name.clone(),
                                    arguments,
                                });
                            }
                        }
                        if !calls.is_empty()
                            && !send_model_event(
                                &tx,
                                &cancel_token,
                                ModelStreamEvent::ToolCalls(calls),
                            )
                            .await
                        {
                            return;
                        }
                    }
                    // If provider did not return final usage, keep `usage` at zero.
                    // The executor/TUI can then continue to use estimated usage.
                    let _ =
                        send_model_event(&tx, &cancel_token, ModelStreamEvent::Completed(usage))
                            .await;
                }
                Err(stream_err) => {
                    let stream_err_text = stream_err.to_string();
                    let stream_kind = cod_error::classify(&stream_err_text);

                    // Non-retryable errors (4xx, context overflow, auth) skip the
                    // non-streaming fallback and surface directly to the executor.
                    if stream_kind.is_fatal()
                        || matches!(
                            stream_kind,
                            cod_error::ErrorKind::ContextOverflow
                                | cod_error::ErrorKind::AuthError
                                | cod_error::ErrorKind::ApiError
                        )
                    {
                        let msg = cod_error::humanize(&stream_err_text, stream_kind);
                        let _ = send_model_event(
                            &tx,
                            &cancel_token,
                            ModelStreamEvent::Failed {
                                display: msg,
                                raw: stream_err_text.clone(),
                            },
                        )
                        .await;
                        return;
                    }

                    // Retryable stream errors: fall back to non-streaming complete().
                    tracing::warn!(
                        error = %stream_err_text,
                        "model_gateway: stream failed — attempting non-streaming fallback"
                    );
                    let complete_result = tokio::select! {
                        _ = cancel_token.cancelled() => return,
                        result = client.complete(
                            &model_settings.provider_id,
                            &messages,
                            &tools,
                            &model_settings.model_id,
                            0.2,
                            model_settings.reasoning_effort.as_deref(),
                            8192,
                        ) => result,
                    };
                    match complete_result {
                        Ok(response) => {
                            if !response.content.is_empty()
                                && !send_model_event(
                                    &tx,
                                    &cancel_token,
                                    ModelStreamEvent::AssistantDelta(response.content),
                                )
                                .await
                            {
                                return;
                            }
                            if !response.tool_calls.is_empty() {
                                let mut calls = Vec::new();
                                for c in response.tool_calls {
                                    if c.function.name.is_empty() {
                                        tracing::warn!("model_gateway: discarding tool call with empty name in complete() fallback");
                                        continue;
                                    }
                                    let argument_list = parse_tool_arguments_lenient(
                                        &c.function.name,
                                        &c.function.arguments,
                                    );
                                    for (i, arguments) in argument_list.into_iter().enumerate() {
                                        let tool_call_id = if i == 0 {
                                            c.id.clone()
                                        } else {
                                            format!("call_{}", uuid::Uuid::new_v4().simple())
                                        };
                                        calls.push(ToolCall {
                                            tool_call_id,
                                            provider_kind: ToolProviderKind::Builtin,
                                            tool_name: c.function.name.clone(),
                                            arguments,
                                        });
                                    }
                                }
                                if !calls.is_empty()
                                    && !send_model_event(
                                        &tx,
                                        &cancel_token,
                                        ModelStreamEvent::ToolCalls(calls),
                                    )
                                    .await
                                {
                                    return;
                                }
                            }
                            let usage = response.usage.unwrap_or_default();
                            let _ = send_model_event(
                                &tx,
                                &cancel_token,
                                ModelStreamEvent::Completed(TokenUsage {
                                    input_tokens: usage.prompt_tokens as i64,
                                    output_tokens: usage.completion_tokens as i64,
                                    cached_tokens: 0,
                                }),
                            )
                            .await;
                        }
                        Err(complete_err) => {
                            let complete_err_text = complete_err.to_string();
                            let complete_kind = cod_error::classify(&complete_err_text);
                            let stream_msg = cod_error::humanize(&stream_err_text, stream_kind);
                            let fallback_msg =
                                cod_error::humanize(&complete_err_text, complete_kind);
                            let _ = send_model_event(
                                &tx,
                                &cancel_token,
                                ModelStreamEvent::Failed {
                                    display: format!(
                                        "{stream_msg} (fallback also failed: {fallback_msg})"
                                    ),
                                    raw: format!(
                                        "stream_error: {stream_err_text}; fallback_error: {complete_err_text}"
                                    ),
                                },
                            )
                            .await;
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

async fn send_model_event(
    tx: &mpsc::Sender<ModelStreamEvent>,
    cancel_token: &CancellationToken,
    event: ModelStreamEvent,
) -> bool {
    tokio::select! {
        _ = cancel_token.cancelled() => false,
        result = tx.send(event) => result.is_ok(),
    }
}

/// Re-export for callers (e.g. executor.rs) that check context overflow.
pub fn is_context_overflow_error(err: &str) -> bool {
    cod_error::is_context_overflow(err)
}

// ─── Token budget constants ───────────────────────────────────────────────────

/// Conservative estimate: 4 characters ≈ 1 token (GPT/Claude average).
const CHARS_PER_TOKEN: usize = 4;

/// Tokens we always reserve for the model's output.
const RESERVED_OUTPUT_TOKENS: usize = 8_192;

/// Default context window if not specified by the model settings.
pub const DEFAULT_CONTEXT_WINDOW: usize = 100_000;

fn estimate_text_tokens(text: &str) -> usize {
    text.len().saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

fn estimate_len_tokens(len: usize) -> usize {
    len.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

fn estimate_messages_tokens(messages: &[llm::Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            let tool_calls_len: usize = m
                .tool_calls
                .iter()
                .map(|c| c.id.len() + c.function.name.len() + c.function.arguments.len())
                .sum();
            let tool_result_len = m
                .tool_result
                .as_ref()
                .map(|r| {
                    r.tool_call_id.len()
                        + r.name.len()
                        + r.result.as_deref().unwrap_or("").len()
                        + r.error.as_deref().unwrap_or("").len()
                })
                .unwrap_or(0);
            estimate_text_tokens(&m.content)
                + estimate_text_tokens(&m.think_content)
                + estimate_text_tokens(&m.role.to_string())
                + estimate_len_tokens(tool_calls_len + tool_result_len)
                + 8
        })
        .sum()
}

pub fn calculate_prompt_budget(context_window: Option<usize>) -> usize {
    let window = context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    window.saturating_sub(RESERVED_OUTPUT_TOKENS)
}

/// Rough token count for a single `ConversationItem`.
fn item_token_estimate(item: &ConversationItem) -> usize {
    item.payload.to_string().len().saturating_add(32) / CHARS_PER_TOKEN
}

/// Returns estimated context usage as a percentage of the given `prompt_budget`.
/// Used by auto-compaction to decide when to compact.
pub fn estimate_items_token_pct(items: &[ConversationItem], prompt_budget: usize) -> f64 {
    let used: usize = items.iter().map(item_token_estimate).sum();
    (used as f64 / prompt_budget as f64) * 100.0
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
    prompt_budget: usize,
) -> Vec<llm::Message> {
    // ── 1. Measure system tokens ──────────────────────────────────────────────
    let system_tokens: usize = system_instructions
        .iter()
        .map(|s| s.len().saturating_add(16) / CHARS_PER_TOKEN)
        .sum();
    let mut remaining_budget = prompt_budget.saturating_sub(system_tokens);

    // ── 2. Group consecutive items into logical turns ─────────────────────────
    // A "turn boundary" is any UserMessage or a ReasoningSummary (compaction
    // marker). Everything between boundaries belongs to the same turn.
    let mut turns: Vec<Vec<&ConversationItem>> = Vec::new();
    let mut current: Vec<&ConversationItem> = Vec::new();

    for item in items {
        let is_boundary = matches!(
            item.kind,
            ItemKind::UserMessage | ItemKind::SystemMessage | ItemKind::ReasoningSummary
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
        ItemKind::SystemMessage => {
            let text = item.payload.get("text").and_then(Value::as_str)?;
            Some(llm::Message::user(format!("[SYSTEM] {text}")))
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

/// Parse raw tool-call argument strings into one or more `Value` objects.
///
/// Returns a `Vec` because the model sometimes concatenates multiple JSON
/// objects into a single argument field (e.g. `{"path":"a"}{"path":"b"}`).
/// In that case every object is treated as a separate invocation of the same
/// tool, which is almost always what was intended.
fn parse_tool_arguments_lenient(tool_name: &str, raw_args: &str) -> Vec<Value> {
    let trimmed = raw_args.trim();
    if trimmed.is_empty() {
        tracing::warn!(
            tool_name = %tool_name,
            "model_gateway: tool call had empty arguments; using empty object"
        );
        return vec![serde_json::json!({})];
    }

    match serde_json::from_str::<Value>(trimmed) {
        Ok(v) if v.is_object() => vec![v],
        Ok(v) => {
            tracing::warn!(
                tool_name = %tool_name,
                parsed_args = %v,
                "model_gateway: tool arguments were valid JSON but not an object; coercing"
            );
            vec![coerce_non_object_tool_arguments(tool_name, v)]
        }
        Err(_) => {
            // Attempt to split concatenated JSON objects produced when the
            // model packs multiple parallel calls into one argument string.
            // serde_json's streaming deserializer handles adjacent objects.
            let objects: Vec<Value> = serde_json::Deserializer::from_str(trimmed)
                .into_iter::<Value>()
                .filter_map(|r| r.ok())
                .filter(|v| v.is_object())
                .collect();

            if !objects.is_empty() {
                tracing::warn!(
                    tool_name = %tool_name,
                    raw_args = %trimmed,
                    count = objects.len(),
                    "model_gateway: split {} concatenated JSON objects into separate tool calls",
                    objects.len()
                );
                return objects;
            }

            // Truly unparseable — fall back to best-effort coercion.
            tracing::warn!(
                tool_name = %tool_name,
                raw_args = %trimmed,
                "model_gateway: tool arguments were not valid JSON; preserving as best-effort arguments"
            );
            vec![coerce_raw_tool_arguments(tool_name, trimmed)]
        }
    }
}

fn coerce_non_object_tool_arguments(tool_name: &str, value: Value) -> Value {
    match (tool_name, value) {
        ("bash_exec", Value::String(command)) => serde_json::json!({ "command": command }),
        ("shell_exec", Value::String(argv)) => serde_json::json!({ "argv": argv }),
        (_, value) => serde_json::json!({ "_raw_arguments": value }),
    }
}

fn coerce_raw_tool_arguments(tool_name: &str, raw_args: &str) -> Value {
    match tool_name {
        "bash_exec" => serde_json::json!({ "command": raw_args }),
        "shell_exec" => serde_json::json!({ "argv": raw_args }),
        _ => serde_json::json!({ "_raw_arguments": raw_args }),
    }
}

/// Build a summarization prompt for `compact_thread`. Returns the messages that
/// should be sent to the model to produce a structured coding-session summary.
///
/// The prompt is optimised for coding LLMs: it captures the concrete state
/// needed to resume work — exact file paths, function/type names, error messages,
/// task progress, and architectural decisions — rather than a prose retelling.
pub fn build_compaction_messages(
    system_instructions: &[String],
    items: &[ConversationItem],
) -> Vec<llm::Message> {
    // ── Separate prior summary (baseline) from new items ──────────────────────
    // Find the last ReasoningSummary; everything after it is "new" activity.
    let prior_summary_idx = items
        .iter()
        .rposition(|i| i.kind == ItemKind::ReasoningSummary);

    let (baseline_text, new_items) = match prior_summary_idx {
        Some(idx) => {
            let text = items[idx]
                .payload
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            (Some(text), &items[idx + 1..])
        }
        None => (None, items),
    };

    // ── Build a focused transcript of new activity only ───────────────────────
    let mut transcript = String::new();
    let mut last_tool_name = "";

    for item in new_items {
        match item.kind {
            ItemKind::UserMessage => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    let snippet: String = text.chars().take(1200).collect();
                    transcript.push_str(&format!("[USER]\n{snippet}\n\n"));
                }
            }
            ItemKind::SystemMessage => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    let snippet: String = text.chars().take(1200).collect();
                    transcript.push_str(&format!("[SYSTEM]\n{snippet}\n\n"));
                }
            }
            ItemKind::AgentMessage => {
                if let Some(text) = item.payload.get("text").and_then(Value::as_str) {
                    // Skip pure acknowledgement messages to save space.
                    let trimmed = text.trim();
                    if trimmed.len() > 20 {
                        let snippet: String = trimmed.chars().take(600).collect();
                        transcript.push_str(&format!("[ASSISTANT]\n{snippet}\n\n"));
                    }
                }
            }
            ItemKind::ToolCall => {
                let name = item
                    .payload
                    .get("toolName")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                last_tool_name = name;
                let args = item
                    .payload
                    .get("arguments")
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                // Extract just the key identifying info from arguments (file_path, command, etc.)
                let args_snippet = extract_key_tool_args(name, &args);
                transcript.push_str(&format!("[TOOL: {name}] {args_snippet}\n"));
            }
            ItemKind::ToolResult => {
                let ok = item
                    .payload
                    .get("ok")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let out = item
                    .payload
                    .get("output")
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                let err = item
                    .payload
                    .get("errorMessage")
                    .and_then(Value::as_str)
                    .unwrap_or("");

                if ok {
                    // For successful reads/edits/writes, record just a one-liner; skip verbose output
                    // unless it looks like test/build output worth preserving.
                    let is_diagnostic = looks_like_diagnostic(&out, last_tool_name);
                    if is_diagnostic {
                        let snippet: String = out.chars().take(800).collect();
                        transcript.push_str(&format!("[OK] {snippet}\n\n"));
                    } else {
                        transcript.push_str("[OK]\n\n");
                    }
                } else {
                    let combined = if err.is_empty() {
                        out.clone()
                    } else {
                        format!("{err}\n{out}")
                    };
                    let snippet: String = combined.chars().take(1600).collect();
                    transcript.push_str(&format!("[ERROR]\n{snippet}\n\n"));
                }
            }
            _ => {}
        }
    }

    let system_ctx = system_instructions.first().cloned().unwrap_or_default();

    let baseline_section = match &baseline_text {
        Some(text) => format!(
            "PRIOR SUMMARY (already compacted — treat as your starting baseline, extend it with new activity below):\n{text}\n\n"
        ),
        None => String::new(),
    };

    let incremental_note = if baseline_text.is_some() {
        "You have a PRIOR SUMMARY above. Merge new activity into it — update changed sections in place rather than appending. Do not repeat information already captured unless it changed."
    } else {
        "Summarise the full conversation below."
    };

    let prompt = format!(
        r#"You are compacting a coding-agent conversation into a structured context summary.
Your output REPLACES the full conversation history for the next agent turn — it must be self-contained.
The agent is a coding assistant that reads, writes, and runs code.

LENGTH BUDGET: Keep your entire response under 1400 tokens (~5600 chars). Dense bullet points, no prose padding.

CRITICAL REQUIREMENTS:
- Be SPECIFIC: exact file paths, function/type names, error messages (quote errors verbatim).
- NEVER omit unresolved errors or incomplete tasks — those are the most important things to preserve.
- NO meta-commentary, caveats, or pleasantries.
- Present tense for ongoing state; past tense for completed actions.

{incremental_note}

OUTPUT FORMAT (use exactly these section headers, omit a section only if truly empty):

## Working Context
1–2 sentences: what is being built/fixed, language/framework, cwd.

## Environment
- Language/toolchain version if known (e.g. `rustc 1.78`, `node 20`, `python 3.12`)
- Key dependencies or workspace layout details that affect the work
Omit if unknown.

## Files Modified
- `path/to/file` — what changed — status: complete | in-progress | broken
If none: "None."

## Current Errors & Failures
- Exact error text (first relevant line + location)
If clean: "None."

## Completed Steps
- Short bullet per finished item (file path where relevant)

## In Progress / Next Steps
- What was being done when compaction triggered
- Immediate next action
Use [BLOCKED: reason] for blocked items.

## Key Decisions & Constraints
- Architectural choices, API contracts, naming rules that must be respected
If none: "None."

---

SYSTEM CONTEXT:
{system_ctx}

{baseline_section}NEW ACTIVITY TO INCORPORATE:
{transcript}
---
Write the structured summary now."#
    );

    vec![llm::Message::user(prompt)]
}

/// Extract a short identifying string from tool call arguments for the compaction transcript.
fn extract_key_tool_args(tool_name: &str, args_json: &str) -> String {
    // Try to parse as JSON and pull the most informative field.
    if let Ok(v) = serde_json::from_str::<Value>(args_json) {
        // File-targeting tools: show the path.
        for key in &["file_path", "path", "filePath"] {
            if let Some(p) = v.get(key).and_then(Value::as_str) {
                return format!("← {p}");
            }
        }
        // Shell/bash: show the command.
        if tool_name.to_lowercase().contains("bash") || tool_name.to_lowercase().contains("shell") {
            if let Some(cmd) = v.get("command").and_then(Value::as_str) {
                let short: String = cmd.chars().take(120).collect();
                return format!("$ {short}");
            }
        }
        // Search tools: show the query/pattern.
        for key in &["query", "pattern", "glob", "search"] {
            if let Some(q) = v.get(key).and_then(Value::as_str) {
                let short: String = q.chars().take(80).collect();
                return format!("search: {short}");
            }
        }
    }
    // Fallback: first 100 chars of raw JSON.
    args_json.chars().take(100).collect()
}

/// Returns true if the tool output looks like compiler/test/lint output worth preserving.
fn looks_like_diagnostic(output: &str, tool_name: &str) -> bool {
    if tool_name.to_lowercase().contains("bash") || tool_name.to_lowercase().contains("shell") {
        // Heuristic: contains keywords typical in compiler/test output.
        let lower = output.to_lowercase();
        return lower.contains("error")
            || lower.contains("warning")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("test result")
            || lower.contains("✓")
            || lower.contains("✗")
            || lower.contains("passed")
            || lower.contains("FAILED");
    }
    false
}
