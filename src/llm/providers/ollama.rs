/// Ollama provider — uses the native `/api/chat` endpoint.
use anyhow::{anyhow, Context as AnyhowContext, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::llm::{
    FunctionCall, LlmResponse, Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition,
};
use crate::system::config::LlmConfig as Config;

fn chat_url(cfg: &Config) -> String {
    let base = cfg.ollama.base_url.trim_end_matches('/');
    if base.ends_with("/api/chat") {
        base.to_string()
    } else if base.ends_with("/api") {
        format!("{base}/chat")
    } else {
        format!("{base}/api/chat")
    }
}

fn auth_headers(cfg: &Config) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = vec![];
    match cfg.ollama.auth_type.as_deref() {
        Some("bearer") if !cfg.api_keys.ollama.is_empty() => {
            headers.push((
                "Authorization".into(),
                format!("Bearer {}", cfg.api_keys.ollama),
            ));
        }
        Some("basic") => {
            let user = cfg.ollama.username.as_deref().unwrap_or("");
            let pass = cfg.ollama.password.as_deref().unwrap_or("");
            let encoded = base64::encode(format!("{user}:{pass}"));
            headers.push(("Authorization".into(), format!("Basic {encoded}")));
        }
        _ => {}
    }
    for (k, v) in &cfg.ollama.headers {
        headers.push((k.clone(), v.clone()));
    }
    headers
}

fn build_chat_body(
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
    reasoning_effort: Option<&str>,
) -> Value {
    let msgs = build_chat_messages(messages);
    let think = matches!(reasoning_effort, Some(e) if e != "off");
    let mut body = json!({
        "model": model,
        "messages": msgs,
        "stream": stream,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    body
}

fn build_chat_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System | Role::User | Role::Assistant => {
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => unreachable!(),
                };
                let mut entry = json!({
                    "role": role,
                    "content": msg.content,
                });
                if !msg.tool_calls.is_empty() {
                    entry["tool_calls"] = json!(msg
                        .tool_calls
                        .iter()
                        .map(|tc| json!({
                            "function": {
                                "name": tc.function.name,
                                "arguments": parse_arguments_object(&tc.function.arguments),
                            }
                        }))
                        .collect::<Vec<_>>());
                }
                out.push(entry);
            }
            Role::Tool => {
                if let Some(tr) = &msg.tool_result {
                    let content = tr
                        .result
                        .as_deref()
                        .or(tr.error.as_deref())
                        .unwrap_or("null");
                    out.push(json!({
                        "role": "tool",
                        "content": content,
                    }));
                }
            }
        }
    }
    out
}

pub fn build_openai_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = vec![];
    for msg in messages {
        let role_str = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };

        if msg.role == Role::Tool {
            if let Some(tr) = &msg.tool_result {
                let content = tr
                    .result
                    .as_deref()
                    .or(tr.error.as_deref())
                    .unwrap_or("null");
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": content,
                }));
                continue;
            }
        }

        let mut entry = json!({
            "role": role_str,
            "content": msg.content,
        });
        if !msg.tool_calls.is_empty() {
            entry["tool_calls"] = json!(msg
                .tool_calls
                .iter()
                .map(|tc| json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }))
                .collect::<Vec<_>>());
        }
        out.push(entry);
    }
    out
}

pub async fn complete(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    max_tokens: usize,
    reasoning_effort: Option<&str>,
) -> Result<LlmResponse> {
    let url = chat_url(cfg);
    let body = build_chat_body(messages, tools, model, temperature, max_tokens, false, reasoning_effort);

    let mut req = http.post(&url).json(&body);
    for (k, v) in auth_headers(cfg) {
        req = req.header(&k, &v);
    }

    let resp = req.send().await.context("sending Ollama request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "could not read error body".to_string());
        anyhow::bail!("Ollama API error {}: {}", status, body);
    }

    let resp_val: Value = resp.json().await.context("parsing Ollama response")?;

    parse_chat_response(&resp_val)
}

pub async fn stream(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    max_tokens: usize,
    reasoning_effort: Option<&str>,
) -> Result<mpsc::Receiver<StreamChunk>> {
    let url = chat_url(cfg);
    let body = build_chat_body(messages, tools, model, temperature, max_tokens, true, reasoning_effort);

    let mut req = http.post(&url).json(&body);
    for (k, v) in auth_headers(cfg) {
        req = req.header(&k, &v);
    }

    let response = req.send().await.context("sending Ollama stream request")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "could not read error body".to_string());
        anyhow::bail!("Ollama API error {}: {}", status, body);
    }

    let (tx, rx) = mpsc::channel::<StreamChunk>(256);
    let mut byte_stream = response.bytes_stream();

    tokio::spawn(async move {
        let mut buf = String::new();
        while let Some(chunk) = byte_stream.next().await {
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));

            let mut start = 0;
            while let Some(nl) = buf[start..].find('\n') {
                let line = buf[start..start + nl].trim().to_string();
                start += nl + 1;

                if line.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    for chunk in parse_chat_chunk(&v) {
                        let _ = tx.send(chunk).await;
                    }
                    if v.get("done").and_then(Value::as_bool) == Some(true) {
                        let _ = tx.send(StreamChunk::Done).await;
                        return;
                    }
                }
            }
            buf = buf[start..].to_string();
        }
        if !buf.trim().is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(buf.trim()) {
                for chunk in parse_chat_chunk(&v) {
                    let _ = tx.send(chunk).await;
                }
            }
        }
        let _ = tx.send(StreamChunk::Done).await;
    });

    Ok(rx)
}

fn parse_chat_response(resp: &Value) -> Result<LlmResponse> {
    let message = resp
        .get("message")
        .ok_or_else(|| anyhow!("invalid Ollama response: missing message"))?;

    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let tool_calls = parse_ollama_tool_calls(message.get("tool_calls"));

    let usage = usage_from_ollama(resp);

    Ok(LlmResponse {
        content,
        tool_calls,
        usage,
    })
}

fn parse_chat_chunk(v: &Value) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    if let Some(thinking) = v
        .get("message")
        .and_then(|m| m.get("thinking"))
        .and_then(Value::as_str)
    {
        if !thinking.is_empty() {
            chunks.push(StreamChunk::Thinking(thinking.to_string()));
        }
    }
    if let Some(text) = v
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
    {
        if !text.is_empty() {
            chunks.push(StreamChunk::Text(text.to_string()));
        }
    }

    if let Some(tool_calls) = v
        .get("message")
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for (index, tc) in tool_calls.iter().enumerate() {
            let function = tc.get("function").unwrap_or(tc);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let arguments = function
                .get("arguments")
                .map(arguments_json)
                .unwrap_or_else(|| "{}".into());
            chunks.push(StreamChunk::ToolCallDelta {
                index,
                id: tc.get("id").and_then(Value::as_str).map(ToOwned::to_owned),
                name,
                arguments_delta: Some(arguments),
            });
        }
    }

    if let Some(usage) = usage_from_ollama(v) {
        chunks.push(StreamChunk::Usage(usage));
    }

    chunks
}

/// Parse an OpenAI-format SSE delta object into StreamChunks.
pub fn parse_sse_delta(v: &Value) -> Vec<StreamChunk> {
    let mut chunks = vec![];
    if let Some(choices) = v["choices"].as_array() {
        for choice in choices {
            let delta = &choice["delta"];
            if let Some(text) = delta["content"].as_str() {
                if !text.is_empty() {
                    chunks.push(StreamChunk::Text(text.to_string()));
                }
            }
            // Also handle reasoning content (for models that expose it)
            if let Some(reasoning) = delta["reasoning_content"].as_str() {
                if !reasoning.is_empty() {
                    chunks.push(StreamChunk::Text(format!("<think>{reasoning}</think>")));
                }
            }
            if let Some(tool_calls) = delta["tool_calls"].as_array() {
                for tc in tool_calls {
                    let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                    let id = tc["id"].as_str().map(|s| s.to_string());
                    let name = tc["function"]["name"].as_str().map(|s| s.to_string());
                    let args = tc["function"]["arguments"].as_str().map(|s| s.to_string());
                    chunks.push(StreamChunk::ToolCallDelta {
                        index: idx,
                        id,
                        name,
                        arguments_delta: args,
                    });
                }
            }
        }
    }
    // Usage in last chunk
    if let Some(usage) = v.get("usage").filter(|u| !u.is_null()) {
        chunks.push(StreamChunk::Usage(TokenUsage {
            prompt_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0),
            completion_tokens: usage["completion_tokens"].as_u64().unwrap_or(0),
            total_tokens: usage["total_tokens"].as_u64().unwrap_or(0),
        }));
    }
    chunks
}

/// Parse a complete OpenAI-format JSON response into LlmResponse.
pub fn parse_openai_response(resp: &Value) -> Result<LlmResponse> {
    let choice = &resp["choices"][0];
    let msg = &choice["message"];

    let content = msg["content"].as_str().unwrap_or("").to_string();

    let mut tool_calls = vec![];
    if let Some(tcs) = msg["tool_calls"].as_array() {
        for (i, tc) in tcs.iter().enumerate() {
            let id = tc["id"]
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("call_{i}_{}", uuid::Uuid::new_v4().simple()));
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let arguments = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("{}")
                .to_string();
            tool_calls.push(ToolCall {
                id,
                function: FunctionCall { name, arguments },
            });
        }
    }

    let usage = resp
        .get("usage")
        .filter(|u| !u.is_null())
        .map(|u| TokenUsage {
            prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
            completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
            total_tokens: u["total_tokens"].as_u64().unwrap_or(0),
        });

    Ok(LlmResponse {
        content,
        tool_calls,
        usage,
    })
}

fn parse_ollama_tool_calls(value: Option<&Value>) -> Vec<ToolCall> {
    let Some(tool_calls) = value.and_then(Value::as_array) else {
        return Vec::new();
    };

    tool_calls
        .iter()
        .enumerate()
        .map(|(index, tc)| {
            let function = tc.get("function").unwrap_or(tc);
            let id = tc
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("call_{index}_{}", uuid::Uuid::new_v4().simple()));
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let arguments = function
                .get("arguments")
                .map(arguments_json)
                .unwrap_or_else(|| "{}".into());

            ToolCall {
                id,
                function: FunctionCall { name, arguments },
            }
        })
        .collect()
}

fn usage_from_ollama(value: &Value) -> Option<TokenUsage> {
    let prompt_tokens = value
        .get("prompt_eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = value.get("eval_count").and_then(Value::as_u64).unwrap_or(0);
    if prompt_tokens == 0 && completion_tokens == 0 {
        None
    } else {
        Some(TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        })
    }
}

fn parse_arguments_object(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

fn arguments_json(arguments: &Value) -> String {
    if let Some(text) = arguments.as_str() {
        text.to_string()
    } else {
        arguments.to_string()
    }
}

// base64 helper
mod base64 {
    use ::base64::engine::Engine;
    pub fn encode(s: impl AsRef<[u8]>) -> String {
        ::base64::engine::general_purpose::STANDARD.encode(s.as_ref())
    }
}
