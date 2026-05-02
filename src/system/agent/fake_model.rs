//! Test-only fake `LlmClient` for driving deterministic agent loops.
//!
//! Each call to [`LlmClient::stream`] pops the next [`ScriptedResponse`] from
//! the script and replays it as `StreamChunk`s. Tests can therefore exercise
//! the full executor → model_gateway → tool_dispatch pipeline without
//! contacting a real provider.

use std::sync::Mutex;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::llm::{
    LlmClient, LlmResponse, Message, StreamChunk, TokenUsage as LlmTokenUsage, ToolDefinition,
};

/// One scripted assistant turn the fake model will return on the next call.
#[derive(Debug, Clone)]
pub enum ScriptedResponse {
    /// Plain assistant text. Emitted as a single chunk followed by `Done`.
    Text(String),
    /// One or more tool calls. Each entry is `(tool_name, arguments_json)`.
    /// `tool_call_id` is auto-assigned (`call_1`, `call_2`, …).
    ToolCalls(Vec<(String, Value)>),
    /// Text followed by tool calls in the same response.
    #[allow(dead_code)]
    TextThenTools(String, Vec<(String, Value)>),
    /// Empty assistant response (no text, no tool calls). Used to exercise the
    /// `max_empty_responses` guard.
    Empty,
}

/// A `LlmClient` implementation that replays a predetermined script.
///
/// Construct with [`FakeLlmClient::new`] and a `Vec<ScriptedResponse>`. Each
/// call to `stream` consumes the next entry. If the script is exhausted, the
/// next call returns an error so tests fail loudly instead of hanging.
pub struct FakeLlmClient {
    script: Mutex<std::collections::VecDeque<ScriptedResponse>>,
    /// Captures every `(provider_id, model_id)` we were called with — useful
    /// for asserting the executor wires settings through correctly.
    pub calls: Mutex<Vec<(String, String)>>,
}

impl FakeLlmClient {
    pub fn new(script: Vec<ScriptedResponse>) -> Self {
        Self {
            script: Mutex::new(script.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn next_response(&self) -> Result<ScriptedResponse> {
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow!("fake_model: script exhausted"))
    }

    fn record_call(&self, provider_id: &str, model: &str) {
        self.calls
            .lock()
            .unwrap()
            .push((provider_id.to_string(), model.to_string()));
    }
}

#[async_trait]
impl LlmClient for FakeLlmClient {
    async fn complete(
        &self,
        provider_id: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        model: &str,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _max_tokens: usize,
    ) -> Result<LlmResponse> {
        // The agent path goes through `stream`; `complete` is only used as a
        // non-streaming fallback. Tests that need it can override.
        self.record_call(provider_id, model);
        Ok(LlmResponse::default())
    }

    async fn stream(
        &self,
        provider_id: &str,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        model: &str,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _max_tokens: usize,
    ) -> Result<mpsc::Receiver<StreamChunk>> {
        self.record_call(provider_id, model);
        let response = self.next_response()?;
        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            emit_response(&tx, response).await;
            // Always send a usage chunk so callers can observe accounting.
            let _ = tx
                .send(StreamChunk::Usage(LlmTokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    cached_tokens: 0,
                }))
                .await;
            let _ = tx.send(StreamChunk::Done).await;
        });

        Ok(rx)
    }
}

async fn emit_response(tx: &mpsc::Sender<StreamChunk>, response: ScriptedResponse) {
    match response {
        ScriptedResponse::Text(text) => {
            let _ = tx.send(StreamChunk::Text(text)).await;
        }
        ScriptedResponse::ToolCalls(calls) => {
            for (idx, (name, args)) in calls.into_iter().enumerate() {
                let _ = tx
                    .send(StreamChunk::ToolCallDelta {
                        index: idx,
                        id: Some(format!("call_{}", idx + 1)),
                        name: Some(name),
                        arguments_delta: Some(args.to_string()),
                    })
                    .await;
            }
        }
        ScriptedResponse::TextThenTools(text, calls) => {
            let _ = tx.send(StreamChunk::Text(text)).await;
            for (idx, (name, args)) in calls.into_iter().enumerate() {
                let _ = tx
                    .send(StreamChunk::ToolCallDelta {
                        index: idx,
                        id: Some(format!("call_{}", idx + 1)),
                        name: Some(name),
                        arguments_delta: Some(args.to_string()),
                    })
                    .await;
            }
        }
        ScriptedResponse::Empty => {
            // Send no content. The executor should treat this as an empty
            // response and (after `max_empty_responses`) terminate the turn.
        }
    }
}
