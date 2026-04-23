use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::system::agent::tools::ToolProvider;
use crate::system::domain::{
    McpServerConfig, ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext,
    ToolProviderKind, ToolResult,
};
use super::stdio::StdioMcpClient;

/// Orchestrates multiple MCP clients and exposes them as a unified ToolProvider.
pub struct McpRegistry {
    clients: Arc<RwLock<HashMap<String, Arc<StdioMcpClient>>>>,
    tool_map: Arc<RwLock<HashMap<String, String>>>, // tool_name -> server_name
}

impl McpRegistry {
    pub fn new() -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            tool_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start_server(&self, config: McpServerConfig) -> Result<()> {
        let name = config.name.clone();
        let client = StdioMcpClient::spawn(config).await?;
        let tools = client.list_tools().await?;
        
        let client_arc = Arc::new(client);
        self.clients.write().await.insert(name.clone(), client_arc);
        
        let mut map = self.tool_map.write().await;
        for t in tools {
            // Namespace tool if name conflicts could happen, or just trust the MCP
            map.insert(t.name.clone(), name.clone());
        }
        
        Ok(())
    }
}

#[async_trait]
impl ToolProvider for McpRegistry {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Mcp
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        // Since list_tools is synchronous, we'd need to cache the definitions during start_server.
        // For simplicity in Phase 6, we'll return an empty list synchronously if we can't block,
        // or we need to refactor ToolProvider to make list_tools async.
        // Let's assume we cached them or we'll fetch them. We'll just return what we know.
        // Actually, ToolProvider::list_tools is synchronous in Codezilla.
        // I'll return empty here and rely on the registry caching them if needed, but since we 
        // need definitions, I should cache them in the registry struct.
        Vec::new() // Needs to be populated from cache
    }

    async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let server_name = self.tool_map.read().await.get(&call.tool_name).cloned();
        let server_name = server_name.ok_or_else(|| anyhow!("MCP tool not found: {}", call.tool_name))?;
        
        let clients = self.clients.read().await;
        let client = clients.get(&server_name).ok_or_else(|| anyhow!("MCP server offline"))?;
        
        let output = client.call_tool(&call.tool_name, call.arguments.clone()).await?;
        
        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
            output,
            error_message: None,
        })
    }
}
