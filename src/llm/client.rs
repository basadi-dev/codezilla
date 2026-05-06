use anyhow::{bail, Context as AnyhowContext, Result};
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::llm::providers::{anthropic, gemini, ollama, openai};
use crate::llm::{
    CompletionRequest, LlmClient, LlmProvider, LlmResponse, Message, StreamChunk, ToolDefinition,
};
use crate::system::config::LlmConfig;

/// Unified LLM client that dispatches to provider implementations.
///
/// Supports two dispatch modes:
/// - **Registry-based** (preferred): Providers implement [`LlmProvider`] and are
///   registered via `register()`. Lookup is O(1) by provider ID.
/// - **Fallback match**: For providers not yet migrated to the trait, the legacy
///   `match provider_id` dispatch is used.
///
/// This hybrid approach allows incremental migration — new providers only need
/// the trait, and existing ones continue to work without changes.
pub struct UnifiedClient {
    pub http: Client,
    pub cfg: LlmConfig,
    /// Registry of trait-based providers, keyed by provider ID.
    providers: HashMap<String, Arc<dyn LlmProvider>>,
}

impl UnifiedClient {
    pub fn new(cfg: LlmConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .context("building HTTP client")?;

        Ok(Self {
            http,
            cfg,
            providers: HashMap::new(),
        })
    }

    /// Register a provider for trait-based dispatch.
    /// If a provider with the same ID already exists, it is replaced.
    #[allow(dead_code)]
    pub fn register(&mut self, provider: Arc<dyn LlmProvider>) {
        self.providers.insert(provider.id().to_string(), provider);
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
        // Try registry-based dispatch first.
        if let Some(provider) = self.providers.get(provider_id) {
            return provider
                .complete(CompletionRequest {
                    messages,
                    tools,
                    model,
                    temperature,
                    reasoning_effort,
                    max_tokens,
                })
                .await;
        }

        // Fallback: legacy match dispatch for providers not yet migrated.
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
        // Try registry-based dispatch first.
        if let Some(provider) = self.providers.get(provider_id) {
            return provider
                .stream(CompletionRequest {
                    messages,
                    tools,
                    model,
                    temperature,
                    reasoning_effort,
                    max_tokens,
                })
                .await;
        }

        // Fallback: legacy match dispatch.
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
