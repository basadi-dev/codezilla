use anyhow::{Context as AnyhowContext, Result};
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::llm::providers::{anthropic, gemini, ollama, openai};
use crate::llm::{
    CompletionRequest, LlmClient, LlmProvider, LlmResponse, Message, StreamChunk, ToolDefinition,
};
use crate::system::config::LlmConfig;
use crate::system::domain::{ModelPreset, ReasoningCapability};

/// Unified LLM client that dispatches to provider implementations.
///
/// Providers are registered by ID and dispatched through a single registry.
pub struct UnifiedClient {
    /// Registry of trait-based providers, keyed by provider ID.
    providers: HashMap<String, Arc<dyn LlmProvider>>,
}

impl UnifiedClient {
    pub fn new(cfg: LlmConfig, model_presets: &[ModelPreset]) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .context("building HTTP client")?;

        let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        providers.insert(
            "ollama".into(),
            Arc::new(ollama::OllamaProvider {
                http: http.clone(),
                cfg: cfg.clone(),
                configured_capabilities: model_presets
                    .iter()
                    .filter(|preset| preset.provider_id == "ollama")
                    .filter_map(|preset| {
                        preset
                            .reasoning_capability
                            .clone()
                            .map(|capability| (preset.model_id.clone(), capability))
                    })
                    .collect::<HashMap<String, ReasoningCapability>>(),
                discovered_capabilities: tokio::sync::RwLock::new(HashMap::new()),
            }),
        );
        let openai = Arc::new(openai::OpenAiProvider {
            http: http.clone(),
            cfg: cfg.clone(),
        });
        providers.insert("openai".into(), openai.clone());
        providers.insert("openai-compat".into(), openai);
        providers.insert(
            "anthropic".into(),
            Arc::new(anthropic::AnthropicProvider {
                http: http.clone(),
                cfg: cfg.clone(),
            }),
        );
        providers.insert(
            "gemini".into(),
            Arc::new(gemini::GeminiProvider {
                http: http.clone(),
                cfg: cfg.clone(),
            }),
        );

        Ok(Self { providers })
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
        let provider = self
            .providers
            .get(provider_id)
            .ok_or_else(|| anyhow::anyhow!("unknown LLM provider: {provider_id}"))?;
        provider
            .complete(CompletionRequest {
                messages,
                tools,
                model,
                temperature,
                reasoning_effort,
                max_tokens,
            })
            .await
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
        let provider = self
            .providers
            .get(provider_id)
            .ok_or_else(|| anyhow::anyhow!("unknown LLM provider: {provider_id}"))?;
        provider
            .stream(CompletionRequest {
                messages,
                tools,
                model,
                temperature,
                reasoning_effort,
                max_tokens,
            })
            .await
    }
}
