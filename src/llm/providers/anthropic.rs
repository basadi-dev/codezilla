/// Anthropic provider — Claude API with streaming support.
use anyhow::{Context as AnyhowContext, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::llm::{
    FunctionCall, LlmResponse, Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition,
};
use crate::system::config::LlmConfig as Config;

const ANTHROPIC_API: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";

fn build_anthropic_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
    let mut system_prompt = None;
    let mut msgs = vec![];

    for msg in messages {
        match msg.role {
            Role::System => {
                system_prompt = Some(msg.content.clone());
            }
            Role::User => {
                // Anthropic vision: when images are present, serialize as content array.
                if msg.has_images() {
                    let mut parts: Vec<Value> = vec![];
                    if !msg.content.is_empty() {
                        parts.push(json!({ "type": "text", "text": msg.content }));
                    }
                    for part in &msg.content_parts {
                        match part {
                            crate::llm::ContentPart::Text { text } => {
                                if !text.is_empty() {
                                    parts.push(json!({ "type": "text", "text": text }));
                                }
                            }
                            crate::llm::ContentPart::Image { mime_type, data } => {
                                parts.push(json!({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": data,
                                    }
                                }));
                            }
                        }
                    }
                    msgs.push(json!({ "role": "user", "content": parts }));
                } else {
                    msgs.push(json!({ "role": "user", "content": msg.content }));
                }
            }
            Role::Assistant => {
                let mut entry = json!({ "role": "assistant", "content": msg.content });
                if !msg.tool_calls.is_empty() {
                    let tool_use: Vec<Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            let input: Value =
                                serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                            json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": input,
                            })
                        })
                        .collect();
                    entry["content"] = json!(tool_use);
                }
                msgs.push(entry);
            }
            Role::Tool => {
                if let Some(tr) = &msg.tool_result {
                    let content = tr
                        .result
                        .as_deref()
                        .or(tr.error.as_deref())
                        .unwrap_or("null");
                    msgs.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": content,
                        }]
                    }));
                }
            }
        }
    }
    (system_prompt, msgs)
}

fn build_anthropic_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "name": t.function.name,
                "description": t.function.description,
                "input_schema": t.function.parameters,
            })
        })
        .collect()
}

fn parse_anthropic_response(resp: &Value) -> Result<LlmResponse> {
    let mut content = String::new();
    let mut tool_calls = vec![];

    if let Some(blocks) = resp["content"].as_array() {
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(text) = block["text"].as_str() {
                        content.push_str(text);
                    }
                }
                Some("tool_use") => {
                    let id = block["id"].as_str().unwrap_or("").to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    let arguments = serde_json::to_string(&block["input"]).unwrap_or_default();
                    tool_calls.push(ToolCall {
                        id,
                        function: FunctionCall { name, arguments },
                    });
                }
                _ => {}
            }
        }
    }

    let usage = resp.get("usage").map(|u| {
        let input = u["input_tokens"].as_u64().unwrap_or(0);
        let output = u["output_tokens"].as_u64().unwrap_or(0);
        let cache_creation = u
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read = u
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        TokenUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
            cached_tokens: cache_creation + cache_read,
        }
    });

    Ok(LlmResponse {
        content,
        tool_calls,
        usage,
    })
}

#[allow(clippy::too_many_arguments)]
pub async fn complete(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    reasoning_effort: Option<&str>,
    max_tokens: usize,
) -> Result<LlmResponse> {
    let (system, msgs) = build_anthropic_messages(messages);
    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": msgs,
    });
    if let Some(sys) = system {
        body["system"] = json!(sys);
    }
    if !tools.is_empty() {
        body["tools"] = json!(build_anthropic_tools(tools));
    }
    if let Some(effort) = reasoning_effort {
        if effort == "low" || effort == "medium" || effort == "high" {
            body["thinking"] = json!({ "type": "enabled", "budget_tokens": 5000 });
        }
    }

    let resp: Value = http
        .post(format!("{ANTHROPIC_API}/messages"))
        .header("x-api-key", &cfg.api_keys.anthropic)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("sending Anthropic request")?
        .error_for_status()
        .context("Anthropic API error")?
        .json()
        .await?;

    parse_anthropic_response(&resp)
}

#[allow(clippy::too_many_arguments)]
pub async fn stream(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    _reasoning_effort: Option<&str>,
    max_tokens: usize,
) -> Result<mpsc::Receiver<StreamChunk>> {
    let (system, msgs) = build_anthropic_messages(messages);
    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": msgs,
        "stream": true,
    });
    if let Some(sys) = system {
        body["system"] = json!(sys);
    }
    if !tools.is_empty() {
        body["tools"] = json!(build_anthropic_tools(tools));
    }

    let response = http
        .post(format!("{ANTHROPIC_API}/messages"))
        .header("x-api-key", &cfg.api_keys.anthropic)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .context("sending Anthropic stream request")?
        .error_for_status()
        .context("Anthropic API error")?;

    let (tx, rx) = mpsc::channel::<StreamChunk>(256);
    let mut byte_stream = response.bytes_stream();

    tokio::spawn(async move {
        // State for accumulating tool_use blocks
        let mut tool_name: Option<String> = None;
        let mut tool_id: Option<String> = None;
        let mut tool_idx: usize = 0;
        let mut buf = String::new();
        // Accumulate input/cache tokens from message_start; merge with output_tokens from message_delta.
        let mut input_tokens: u64 = 0;
        let mut cache_creation_tokens: u64 = 0;
        let mut cache_read_tokens: u64 = 0;

        while let Some(chunk) = byte_stream.next().await {
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            let mut start = 0;
            while let Some(nl) = buf[start..].find('\n') {
                let line = buf[start..start + nl].trim().to_string();
                start += nl + 1;
                if !line.starts_with("data: ") {
                    continue;
                }
                let data = line.trim_start_matches("data: ").trim();
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };

                match v["type"].as_str() {
                    Some("message_start") => {
                        // Capture input_tokens and cache tokens from the initial message usage.
                        if let Some(usage) = v.get("message").and_then(|m| m.get("usage")) {
                            input_tokens = usage
                                .get("input_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            cache_creation_tokens = usage
                                .get("cache_creation_input_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            cache_read_tokens = usage
                                .get("cache_read_input_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                        }
                    }
                    Some("content_block_start") => {
                        let block = &v["content_block"];
                        if block["type"] == "tool_use" {
                            tool_id = block["id"].as_str().map(String::from);
                            tool_name = block["name"].as_str().map(String::from);
                            tool_idx = v["index"].as_u64().unwrap_or(0) as usize;
                        }
                    }
                    Some("content_block_delta") => {
                        let delta = &v["delta"];
                        match delta["type"].as_str() {
                            Some("text_delta") => {
                                if let Some(text) = delta["text"].as_str() {
                                    let _ = tx.send(StreamChunk::Text(text.to_string())).await;
                                }
                            }
                            Some("thinking_delta") => {
                                if let Some(t) = delta["thinking"].as_str() {
                                    let _ = tx
                                        .send(StreamChunk::Text(format!("<think>{t}</think>")))
                                        .await;
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(j) = delta["partial_json"].as_str() {
                                    let _ = tx
                                        .send(StreamChunk::ToolCallDelta {
                                            index: tool_idx,
                                            id: tool_id.clone(),
                                            name: tool_name.clone(),
                                            arguments_delta: Some(j.to_string()),
                                        })
                                        .await;
                                }
                            }
                            _ => {}
                        }
                    }
                    Some("message_delta") => {
                        if let Some(usage) = v["usage"].as_object() {
                            let output_tokens = usage
                                .get("output_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let total = input_tokens + output_tokens;
                            let cached = cache_creation_tokens + cache_read_tokens;
                            let _ = tx
                                .send(StreamChunk::Usage(TokenUsage {
                                    prompt_tokens: input_tokens,
                                    completion_tokens: output_tokens,
                                    total_tokens: total,
                                    cached_tokens: cached,
                                }))
                                .await;
                        }
                    }
                    Some("message_stop") => {
                        let _ = tx.send(StreamChunk::Done).await;
                        return;
                    }
                    _ => {}
                }
            }
            buf = buf[start..].to_string();
        }
        let _ = tx.send(StreamChunk::Done).await;
    });

    Ok(rx)
}
