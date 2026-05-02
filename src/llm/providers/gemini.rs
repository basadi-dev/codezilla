/// Gemini provider — Google Generative AI REST API.
use anyhow::{Context as AnyhowContext, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::llm::{
    FunctionCall, LlmResponse, Message, Role, StreamChunk, TokenUsage, ToolCall, ToolDefinition,
};
use crate::system::config::LlmConfig as Config;

fn api_url(cfg: &Config, model: &str, stream: bool) -> String {
    let key = &cfg.api_keys.gemini;
    let method = if stream {
        "streamGenerateContent"
    } else {
        "generateContent"
    };
    let alt = if stream { "?alt=sse&key=" } else { "?key=" };
    format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}{alt}{key}")
}

fn build_gemini_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
    let mut system = None;
    let mut contents = vec![];

    for msg in messages {
        match msg.role {
            Role::System => {
                system = Some(msg.content.clone());
            }
            Role::User => {
                contents.push(json!({
                    "role": "user",
                    "parts": [{ "text": msg.content }]
                }));
            }
            Role::Assistant => {
                if !msg.tool_calls.is_empty() {
                    let parts: Vec<Value> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            let args: Value =
                                serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                            json!({ "functionCall": { "name": tc.function.name, "args": args } })
                        })
                        .collect();
                    contents.push(json!({ "role": "model", "parts": parts }));
                } else {
                    contents.push(json!({
                        "role": "model",
                        "parts": [{ "text": msg.content }]
                    }));
                }
            }
            Role::Tool => {
                if let Some(tr) = &msg.tool_result {
                    let result: Value = tr
                        .result
                        .as_ref()
                        .and_then(|r| serde_json::from_str(r).ok())
                        .unwrap_or_else(|| json!(tr.result));
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "functionResponse": { "name": tr.name, "response": result } }]
                    }));
                }
            }
        }
    }
    (system, contents)
}

fn build_gemini_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    if tools.is_empty() {
        return vec![];
    }
    let declarations: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            })
        })
        .collect();
    vec![json!({ "functionDeclarations": declarations })]
}

fn parse_gemini_response(resp: &Value) -> Result<LlmResponse> {
    let candidate = &resp["candidates"][0];
    let parts = candidate["content"]["parts"].as_array();

    let mut content = String::new();
    let mut tool_calls = vec![];

    if let Some(parts) = parts {
        for part in parts {
            if let Some(text) = part["text"].as_str() {
                content.push_str(text);
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc["name"].as_str().unwrap_or("").to_string();
                let arguments = serde_json::to_string(&fc["args"]).unwrap_or_default();
                tool_calls.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                    function: FunctionCall { name, arguments },
                });
            }
        }
    }

    let usage = resp.get("usageMetadata").map(|u| TokenUsage {
        prompt_tokens: u["promptTokenCount"].as_u64().unwrap_or(0),
        completion_tokens: u["candidatesTokenCount"].as_u64().unwrap_or(0),
        total_tokens: u["totalTokenCount"].as_u64().unwrap_or(0),
        cached_tokens: u
            .get("cachedContentTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
    });

    Ok(LlmResponse {
        content,
        tool_calls,
        usage,
    })
}

pub async fn complete(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    max_tokens: usize,
) -> Result<LlmResponse> {
    let (system, contents) = build_gemini_messages(messages);
    let mut body = json!({
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    });
    if let Some(sys) = system {
        body["systemInstruction"] = json!({ "parts": [{ "text": sys }] });
    }
    if !tools.is_empty() {
        body["tools"] = json!(build_gemini_tools(tools));
    }

    let resp: Value = http
        .post(api_url(cfg, model, false))
        .json(&body)
        .send()
        .await
        .context("sending Gemini request")?
        .error_for_status()
        .context("Gemini API error")?
        .json()
        .await?;

    parse_gemini_response(&resp)
}

pub async fn stream(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    max_tokens: usize,
) -> Result<mpsc::Receiver<StreamChunk>> {
    let (system, contents) = build_gemini_messages(messages);
    let mut body = json!({
        "contents": contents,
        "generationConfig": { "temperature": temperature, "maxOutputTokens": max_tokens },
    });
    if let Some(sys) = system {
        body["systemInstruction"] = json!({ "parts": [{ "text": sys }] });
    }
    if !tools.is_empty() {
        body["tools"] = json!(build_gemini_tools(tools));
    }

    let response = http
        .post(api_url(cfg, model, true))
        .json(&body)
        .send()
        .await
        .context("sending Gemini stream request")?
        .error_for_status()
        .context("Gemini API error")?;

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
                if line.starts_with("data: ") {
                    let data = line.trim_start_matches("data: ").trim();
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        // Gemini SSE sends full response objects each time
                        if let Ok(resp) = parse_gemini_response(&v) {
                            if !resp.content.is_empty() {
                                let _ = tx.send(StreamChunk::Text(resp.content)).await;
                            }
                            for tc in resp.tool_calls {
                                let _ = tx
                                    .send(StreamChunk::ToolCallDelta {
                                        index: 0,
                                        id: Some(tc.id),
                                        name: Some(tc.function.name),
                                        arguments_delta: Some(tc.function.arguments),
                                    })
                                    .await;
                            }
                            if let Some(u) = resp.usage {
                                let _ = tx.send(StreamChunk::Usage(u)).await;
                            }
                        }
                    }
                }
            }
            buf = buf[start..].to_string();
        }
        let _ = tx.send(StreamChunk::Done).await;
    });

    Ok(rx)
}
