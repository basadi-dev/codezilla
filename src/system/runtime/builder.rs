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
//! end, after `MCP` and after the late-bound `AgentOrchestrationToolProvider`,
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
                UnifiedClient::new(effective_config.llm.clone(), &effective_config.models)
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
