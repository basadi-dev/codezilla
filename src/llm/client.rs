use anyhow::{bail, Context as AnyhowContext, Result};
use reqwest::Client;
use std::time::Duration;

use crate::llm::providers::{anthropic, gemini, ollama, openai};
use crate::llm::{LlmClient, LlmResponse, Message, StreamChunk, ToolDefinition};
use crate::system::config::LlmConfig;

/// Unified LLM client that dispatches to provider implementations.
pub struct UnifiedClient {
    pub http: Client,
    pub cfg: LlmConfig,
}

impl UnifiedClient {
    pub fn new(cfg: LlmConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .context("building HTTP client")?;

        Ok(Self { http, cfg })
    }
}

#[async_trait::async_trait]
impl LlmClient for UnifiedClient {
    async fn complete(
        &self,
        provider_id: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
        reasoning_effort: Option<&str>,
        max_tokens: usize,
    ) -> Result<LlmResponse> {
        match provider_id {
            "ollama" => {
                ollama::complete(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    max_tokens,
                    reasoning_effort,
                )
                .await
            }
            "openai" | "openai-compat" => {
                openai::complete(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    reasoning_effort,
                    max_tokens,
                )
                .await
            }
            "anthropic" => {
                anthropic::complete(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    reasoning_effort,
                    max_tokens,
                )
                .await
            }
            "gemini" => {
                gemini::complete(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    max_tokens,
                )
                .await
            }
            p => bail!("unknown LLM provider: {p}"),
        }
    }

    async fn stream(
        &self,
        provider_id: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        temperature: f32,
        reasoning_effort: Option<&str>,
        max_tokens: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>> {
        match provider_id {
            "ollama" => {
                ollama::stream(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    max_tokens,
                    reasoning_effort,
                )
                .await
            }
            "openai" | "openai-compat" => {
                openai::stream(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    reasoning_effort,
                    max_tokens,
                )
                .await
            }
            "anthropic" => {
                anthropic::stream(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    reasoning_effort,
                    max_tokens,
                )
                .await
            }
            "gemini" => {
                gemini::stream(
                    &self.http,
                    &self.cfg,
                    messages,
                    tools,
                    model,
                    temperature,
                    max_tokens,
                )
                .await
            }
            p => bail!("unknown LLM provider: {p}"),
        }
    }
}
