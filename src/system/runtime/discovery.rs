//! Read-only listing/metadata methods on [`ConversationRuntime`].
//!
//! These are thin façades over the gateway/extension/persistence managers.
//! They live in their own file so the main `runtime/mod.rs` stays focused on
//! lifecycle (thread/turn) operations.

use anyhow::Result;

use super::ConversationRuntime;
use crate::system::agent::ModelDescription;
use crate::system::domain::{
    ConnectorDefinition, McpServerDefinition, PluginDefinition, SkillDefinition,
};

impl ConversationRuntime {
    pub fn list_models(&self) -> Vec<ModelDescription> {
        self.inner
            .model_gateway
            .list_models(&self.inner.effective_config.model_settings)
    }

    pub async fn list_skills(&self) -> Vec<SkillDefinition> {
        self.inner
            .extension_manager
            .list_skills(&self.inner.effective_config.working_directory)
            .await
    }

    pub async fn list_plugins(&self) -> Vec<PluginDefinition> {
        self.inner.extension_manager.list_plugins().await
    }

    pub async fn list_connectors(&self) -> Vec<ConnectorDefinition> {
        self.inner.extension_manager.list_connectors().await
    }

    pub async fn list_mcp_servers(&self) -> Vec<McpServerDefinition> {
        self.inner.extension_manager.list_mcp_servers().await
    }

    pub fn reset_memories(&self) -> Result<()> {
        self.inner.persistence_manager.reset_memories()
    }
}
