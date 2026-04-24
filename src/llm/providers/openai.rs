/// OpenAI provider — also used for any OpenAI-compatible endpoint.
use anyhow::{Context as AnyhowContext, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::ollama::{build_openai_messages, parse_openai_response, parse_sse_delta};
use crate::llm::{LlmResponse, Message, StreamChunk, ToolDefinition};
use crate::system::config::LlmConfig as Config;

fn api_base(cfg: &Config) -> String {
    cfg.openai
        .base_url
        .clone()
        .unwrap_or_else(|| "https://api.openai.com/v1".into())
}

fn auth_header(cfg: &Config) -> Option<String> {
    if !cfg.api_keys.openai.is_empty() {
        Some(format!("Bearer {}", cfg.api_keys.openai))
    } else {
        None
    }
}

fn build_body(
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    reasoning_effort: Option<&str>,
    max_tokens: usize,
    stream: bool,
) -> Value {
    let msgs = build_openai_messages(messages);
    let mut body = json!({
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    if let Some(effort) = reasoning_effort {
        if !effort.is_empty() && effort != "none" {
            body["reasoning_effort"] = json!(effort);
        }
    }
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
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
    let url = format!("{}/chat/completions", api_base(cfg));
    let body = build_body(
        messages,
        tools,
        model,
        temperature,
        reasoning_effort,
        max_tokens,
        false,
    );

    let mut req = http.post(&url).json(&body);
    if let Some(auth) = auth_header(cfg) {
        req = req.header("Authorization", auth);
    }

    let resp: Value = req
        .send()
        .await
        .context("sending OpenAI request")?
        .error_for_status()
        .context("OpenAI API error")?
        .json()
        .await
        .context("parsing OpenAI response")?;

    parse_openai_response(&resp)
}

#[allow(clippy::too_many_arguments)]
pub async fn stream(
    http: &Client,
    cfg: &Config,
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    temperature: f32,
    reasoning_effort: Option<&str>,
    max_tokens: usize,
) -> Result<mpsc::Receiver<StreamChunk>> {
    let url = format!("{}/chat/completions", api_base(cfg));
    let body = build_body(
        messages,
        tools,
        model,
        temperature,
        reasoning_effort,
        max_tokens,
        true,
    );

    let mut req = http.post(&url).json(&body);
    if let Some(auth) = auth_header(cfg) {
        req = req.header("Authorization", auth);
    }

    let response = req
        .send()
        .await
        .context("sending OpenAI stream request")?
        .error_for_status()
        .context("OpenAI API error")?;

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
                    if data == "[DONE]" {
                        let _ = tx.send(StreamChunk::Done).await;
                        return;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        for chunk in parse_sse_delta(&v) {
                            let _ = tx.send(chunk).await;
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
