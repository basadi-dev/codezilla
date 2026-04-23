/// Anthropic provider — Claude API with streaming support.
use anyhow::{Context as AnyhowContext, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::system::config::LlmConfig as Config;
use crate::llm::{
    FunctionCall, LlmResponse, Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition,
};

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
                msgs.push(json!({ "role": "user", "content": msg.content }));
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

    let usage = resp.get("usage").map(|u| TokenUsage {
        prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0),
        completion_tokens: u["output_tokens"].as_u64().unwrap_or(0),
        total_tokens: u["input_tokens"].as_u64().unwrap_or(0)
            + u["output_tokens"].as_u64().unwrap_or(0),
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
                            let _ = tx
                                .send(StreamChunk::Usage(TokenUsage {
                                    prompt_tokens: 0,
                                    completion_tokens: usage
                                        .get("output_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    total_tokens: 0,
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
