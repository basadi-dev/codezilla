use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::system::agent::model_gateway::ModelRequest;
use crate::system::domain::{
    ConversationItem, ModelSettings, PermissionProfile, ToolDefinition, ToolListingContext, TurnId,
};
use crate::system::runtime::{ThreadSession, TurnStartParams};

use super::TurnExecutor;

pub(crate) struct TurnContext {
    pub(crate) params: TurnStartParams,
    pub(crate) turn_id: TurnId,
    pub(crate) thread: Arc<Mutex<ThreadSession>>,
    pub(crate) items: Vec<ConversationItem>,
    pub(crate) cancel_token: CancellationToken,
    pub(crate) cwd: String,
    pub(crate) permission_profile: PermissionProfile,
    pub(crate) listing: ToolListingContext,
    pub(crate) effective_model_settings: ModelSettings,
}

impl TurnContext {
    pub(crate) async fn build(
        executor: &TurnExecutor,
        params: &TurnStartParams,
        turn_id: &TurnId,
        thread: Arc<Mutex<ThreadSession>>,
    ) -> Result<Self> {
        let (thread_metadata, items, cancel_token) = {
            let thread_guard = thread.lock().await;
            let turn = thread_guard
                .turns
                .iter()
                .find(|t| t.metadata.turn_id == *turn_id)
                .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;

            let mut items = thread_guard.prefix_items.clone();
            for lt in &thread_guard.turns {
                items.extend(lt.items.clone());
            }

            (
                thread_guard.metadata.clone(),
                items,
                turn.cancel_token.clone(),
            )
        };

        let cwd = params
            .cwd
            .clone()
            .or(thread_metadata.cwd.clone())
            .unwrap_or_else(|| {
                executor
                    .runtime
                    .inner
                    .effective_config
                    .working_directory
                    .clone()
            });
        let permission_profile = params.permission_profile.clone().unwrap_or_else(|| {
            executor
                .runtime
                .inner
                .permission_manager
                .resolve_permission_profile(&executor.runtime.inner.effective_config, &cwd)
        });
        let listing = ToolListingContext {
            thread_id: params.thread_id.clone(),
            cwd: cwd.clone(),
            features: executor.runtime.inner.effective_config.features.clone(),
        };
        let effective_model_settings =
            params
                .model_settings
                .clone()
                .unwrap_or_else(|| ModelSettings {
                    model_id: thread_metadata.model_id.clone(),
                    provider_id: thread_metadata.provider_id.clone(),
                    reasoning_effort: executor
                        .runtime
                        .inner
                        .effective_config
                        .model_settings
                        .reasoning_effort,
                    summary_mode: None,
                    service_tier: None,
                    web_search_enabled: false,
                    context_window: executor
                        .runtime
                        .inner
                        .effective_config
                        .model_settings
                        .context_window,
                });

        Ok(Self {
            params: params.clone(),
            turn_id: turn_id.clone(),
            thread,
            items,
            cancel_token,
            cwd,
            permission_profile,
            listing,
            effective_model_settings,
        })
    }

    pub(crate) fn model_request(
        &self,
        system_instructions: Vec<String>,
        tool_definitions: Vec<ToolDefinition>,
    ) -> ModelRequest {
        ModelRequest {
            system_instructions,
            model_settings: self.effective_model_settings.clone(),
            conversation_items: self.items.clone(),
            tool_definitions,
            output_schema: self.params.output_schema.clone(),
        }
    }
}
