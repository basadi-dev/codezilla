//! Fluent builder for [`ConversationRuntime`].
//!
//! The phase-4 deferred work: replace the long-form `new` /
//! `new_with_llm_client` calls with a fluent builder that lets callers
//! customise construction without forking the function. The two existing
//! constructors stay as thin wrappers so call sites don't change.
//!
//! What the builder lets you customise:
//!   - `with_llm_client(...)` — inject a fake or alternate `LlmClient`
//!     (used by the test harness; previously required a separate
//!     constructor).
//!   - `with_extra_provider(...)` — append a custom `ToolProvider` to the
//!     orchestrator after the builtin set is registered. Useful for
//!     embedding the runtime in a host app that ships its own tools.
//!
//! Order of registration is unchanged — extras are appended at the very
//! end, after `MCP` and after the late-bound `SpawnAgentToolProviderReal`,
//! so extras can override behaviour but won't accidentally shadow the
//! builtin parallel/safety contracts the executor relies on.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::ConversationRuntime;
use crate::llm::client::UnifiedClient;
use crate::llm::LlmClient;
use crate::system::agent::tools::ToolProvider;
use crate::system::config::EffectiveConfig;
use crate::system::domain::AccountSession;

pub struct RuntimeBuilder {
    effective_config: EffectiveConfig,
    account_session: AccountSession,
    llm_client: Option<Arc<dyn LlmClient>>,
    extra_providers: Vec<Arc<dyn ToolProvider>>,
}

impl RuntimeBuilder {
    pub fn new(effective_config: EffectiveConfig, account_session: AccountSession) -> Self {
        Self {
            effective_config,
            account_session,
            llm_client: None,
            extra_providers: Vec::new(),
        }
    }

    /// Inject an `LlmClient`. If unset, the builder constructs the default
    /// `UnifiedClient` from the effective config's `llm` block.
    #[allow(dead_code)] // covered by tests; production paths still use the legacy constructor signature
    pub fn with_llm_client(mut self, client: Arc<dyn LlmClient>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Append a custom tool provider. Multiple calls accumulate; the order
    /// they're registered in is preserved.
    #[allow(dead_code)] // public ergonomic API for embedding the runtime
    pub fn with_extra_provider(mut self, provider: Arc<dyn ToolProvider>) -> Self {
        self.extra_providers.push(provider);
        self
    }

    pub async fn build(self) -> Result<ConversationRuntime> {
        let RuntimeBuilder {
            effective_config,
            account_session,
            llm_client,
            extra_providers,
        } = self;

        let llm_client = match llm_client {
            Some(c) => c,
            None => Arc::new(
                UnifiedClient::new(effective_config.llm.clone())
                    .map_err(|e| anyhow!("llm_client_init_failed: {e}"))?,
            ),
        };

        let runtime =
            ConversationRuntime::new_with_llm_client(effective_config, account_session, llm_client)
                .await?;

        for provider in extra_providers {
            runtime.inner.tool_orchestrator.register_provider(provider);
        }

        Ok(runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::agent::fake_model::{FakeLlmClient, ScriptedResponse};
    use crate::system::agent::tools::ToolProvider;
    use crate::system::domain::{
        ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext, ToolProviderKind,
        ToolResult,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Minimal custom provider that records execute calls — proves the
    /// extras are reachable through the orchestrator after `build()`.
    struct EchoProvider {
        executed: AtomicUsize,
    }

    #[async_trait]
    impl ToolProvider for EchoProvider {
        fn get_kind(&self) -> ToolProviderKind {
            ToolProviderKind::Builtin
        }

        fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "echo_test".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "test-only echo tool".into(),
                input_schema: json!({"type":"object","properties":{}}),
                requires_approval: false,
                supports_parallel_calls: true,
            }]
        }

        async fn execute(
            &self,
            call: &ToolCall,
            _ctx: &ToolExecutionContext,
        ) -> Result<ToolResult> {
            self.executed.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: true,
                output: json!({"echoed": true}),
                error_message: None,
            })
        }
    }

    fn ctx_thread() -> ToolListingContext {
        ToolListingContext {
            thread_id: "t".into(),
            cwd: ".".into(),
            features: Default::default(),
        }
    }

    #[tokio::test]
    async fn builder_with_llm_client_round_trips() {
        // Reuse the test_config helper from runtime/mod.rs by going through
        // the harness — the cleanest way to get an EffectiveConfig in tests.
        let app_home = super::super::fake_model_tests::unique_app_home();
        let cfg = super::super::fake_model_tests::test_config(&app_home);
        let client = Arc::new(FakeLlmClient::new(vec![ScriptedResponse::Text(
            "hello".into(),
        )]));

        let runtime = RuntimeBuilder::new(cfg, AccountSession::default())
            .with_llm_client(client)
            .build()
            .await
            .expect("builder should succeed");

        // Builtin tools must still be registered — sanity-check one.
        let listing = runtime
            .inner
            .tool_orchestrator
            .list_available_tools(&ctx_thread());
        assert!(
            listing.iter().any(|t| t.name == "list_dir"),
            "builtin tools missing after builder.build(): {:?}",
            listing.iter().map(|t| &t.name).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn builder_extra_provider_is_registered_after_builtins() {
        let app_home = super::super::fake_model_tests::unique_app_home();
        let cfg = super::super::fake_model_tests::test_config(&app_home);
        let client = Arc::new(FakeLlmClient::new(vec![ScriptedResponse::Text(
            "hello".into(),
        )]));
        let echo = Arc::new(EchoProvider {
            executed: AtomicUsize::new(0),
        });

        let runtime = RuntimeBuilder::new(cfg, AccountSession::default())
            .with_llm_client(client)
            .with_extra_provider(echo.clone())
            .build()
            .await
            .expect("builder build");

        let listing = runtime
            .inner
            .tool_orchestrator
            .list_available_tools(&ctx_thread());
        let names: Vec<&str> = listing.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"echo_test"),
            "extra provider missing: {names:?}"
        );
        assert!(
            names.contains(&"list_dir"),
            "builtins still present: {names:?}"
        );
    }
}
