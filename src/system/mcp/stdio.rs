use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::system::domain::{McpServerConfig, ToolDefinition, ToolProviderKind};

/// A client connecting to an MCP server via stdio using JSON-RPC 2.0.
pub struct StdioMcpClient {
    #[allow(dead_code)]
    config: McpServerConfig,
    rpc_tx: mpsc::Sender<(Value, tokio::sync::oneshot::Sender<Result<Value>>)>,
}

impl StdioMcpClient {
    pub async fn spawn(config: McpServerConfig) -> Result<Self> {
        let cmd = config
            .command
            .clone()
            .ok_or_else(|| anyhow!("MCP server {} has no command", config.name))?;
        if cmd.is_empty() {
            bail!("MCP server command is empty");
        }

        let mut child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Let stderr flow to the app log
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        let (rpc_tx, mut rpc_rx) =
            mpsc::channel::<(Value, tokio::sync::oneshot::Sender<Result<Value>>)>(32);

        // RPC event loop task
        tokio::spawn(async move {
            let mut stdin = stdin;
            let mut reader = BufReader::new(stdout).lines();
            let mut pending_requests =
                HashMap::<String, tokio::sync::oneshot::Sender<Result<Value>>>::new();

            loop {
                tokio::select! {
                    Some((mut req, reply_tx)) = rpc_rx.recv() => {
                        let id = Uuid::new_v4().to_string();
                        req["id"] = json!(id.clone());
                        req["jsonrpc"] = json!("2.0");

                        let msg = format!("{}\n", serde_json::to_string(&req).unwrap());
                        if stdin.write_all(msg.as_bytes()).await.is_err() {
                            let _ = reply_tx.send(Err(anyhow!("Failed to write to MCP stdin")));
                            break;
                        }
                        pending_requests.insert(id, reply_tx);
                    }
                    line_res = reader.next_line() => {
                        match line_res {
                            Ok(Some(line)) => {
                                if let Ok(resp) = serde_json::from_str::<Value>(&line) {
                                    if let Some(id) = resp.get("id").and_then(Value::as_str) {
                                        if let Some(tx) = pending_requests.remove(id) {
                                            if let Some(error) = resp.get("error") {
                                                let _ = tx.send(Err(anyhow!("MCP error: {}", error)));
                                            } else {
                                                let _ = tx.send(Ok(resp.get("result").cloned().unwrap_or(json!({}))));
                                            }
                                        }
                                    }
                                }
                            }
                            Ok(None) | Err(_) => break, // EOF or read error
                        }
                    }
                }
            }
        });

        // Initialize connection
        let client = Self { config, rpc_tx };
        client
            .call(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "codezilla",
                        "version": "2.0.0"
                    }
                }),
            )
            .await?;
        client.call("notifications/initialized", json!({})).await?;

        Ok(client)
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.rpc_tx
            .send((
                json!({
                    "method": method,
                    "params": params
                }),
                tx,
            ))
            .await?;
        rx.await?
    }

    pub async fn list_tools(&self) -> Result<Vec<ToolDefinition>> {
        let result = self.call("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("invalid tools/list response"))?;

        let mut defs = Vec::new();
        for t in tools {
            defs.push(ToolDefinition {
                name: t["name"].as_str().unwrap_or("").to_string(),
                description: t["description"].as_str().unwrap_or("").to_string(),
                input_schema: t["inputSchema"].clone(),
                provider_kind: ToolProviderKind::Mcp,
                requires_approval: true,
                supports_parallel_calls: false,
            });
        }
        Ok(defs)
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        let result = self
            .call(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments
                }),
            )
            .await?;

        // MCP tools/call returns {"content": [{"type": "text", "text": "..."}]}
        // We simplify this back down to a string/JSON for Codezilla's ToolResult.output
        let content = result.get("content").and_then(Value::as_array);
        if let Some(content) = content {
            if let Some(first) = content.first() {
                if first["type"] == "text" {
                    return Ok(json!(first["text"].as_str().unwrap_or("")));
                }
            }
        }
        Ok(result)
    }
}
