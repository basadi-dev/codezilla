//! Conversation-memory tools exposed to the agent.
//!
//! These tools make memory explicit and inspectable. The automatic turn
//! capture/retrieval path still exists, but the model can now deliberately
//! search, store, update, and forget memory records.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::memory_ops::{apply_memory_plan, MemoryControllerPlan};
use super::tools::ToolProvider;
use crate::system::domain::{
    MemoryMode, ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext,
    ToolProviderKind, ToolResult,
};
use crate::system::persistence::PersistenceManager;

const MAX_LIMIT: usize = 50;

pub struct MemoryToolProvider {
    persistence: Arc<PersistenceManager>,
}

impl MemoryToolProvider {
    pub fn new(persistence: Arc<PersistenceManager>) -> Self {
        Self { persistence }
    }
}

#[async_trait]
impl ToolProvider for MemoryToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "run_memory_plan".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Apply a validated memory-controller JSON plan containing search/store/update/forget operations. This is the preferred interface for a post-trained memory controller model.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "search": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "limit": {"type": "integer"}
                                },
                                "required": ["query"]
                            }
                        },
                        "store": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string"},
                                    "scope": {"type": "string", "enum": ["thread", "global"]},
                                    "content": {"type": "string"},
                                    "importance": {"type": "number"}
                                },
                                "required": ["kind", "content"]
                            }
                        },
                        "update": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memoryId": {"type": "string"},
                                    "kind": {"type": "string"},
                                    "scope": {"type": "string", "enum": ["thread", "global"]},
                                    "content": {"type": "string"},
                                    "importance": {"type": "number"}
                                },
                                "required": ["memoryId"]
                            }
                        },
                        "forget": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "memoryId": {"type": "string"}
                                },
                                "required": ["memoryId"]
                            }
                        },
                        "answerStrategy": {"type": "string"}
                    }
                }),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "search_memory".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Search durable conversation memory for relevant context. Use before answering when prior user preferences, decisions, or project history may matter.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language memory search query."},
                        "limit": {"type": "integer", "description": "Maximum results, default 6, max 50."}
                    },
                    "required": ["query"]
                }),
                requires_approval: false,
                supports_parallel_calls: true,
            },
            ToolDefinition {
                name: "save_memory".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Store a durable memory when the user states a reusable preference, fact, decision, project constraint, or long-lived task context.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "description": "preference, fact, decision, project_context, task, summary, or episodic."},
                        "scope": {"type": "string", "enum": ["thread", "global"], "description": "thread for this conversation/project, global for durable user-level memory."},
                        "content": {"type": "string", "description": "Concise, self-contained memory text."},
                        "importance": {"type": "number", "description": "0.0 to 1.0 importance score."}
                    },
                    "required": ["kind", "content"]
                }),
                requires_approval: true,
                supports_parallel_calls: true,
            },
            ToolDefinition {
                name: "update_memory".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Update an existing memory when newer information refines, scopes, or corrects it.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "kind": {"type": "string"},
                        "scope": {"type": "string", "enum": ["thread", "global"]},
                        "content": {"type": "string"},
                        "importance": {"type": "number"}
                    },
                    "required": ["memory_id"]
                }),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "forget_memory".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Delete a memory that is stale, wrong, sensitive, or explicitly requested to be forgotten.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"}
                    },
                    "required": ["memory_id"]
                }),
                requires_approval: true,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "list_memory".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Inspect recent stored memories for this thread plus global memories. Use for debugging memory state.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum results, default 20, max 50."}
                    }
                }),
                requires_approval: false,
                supports_parallel_calls: true,
            },
        ]
    }

    async fn execute(&self, call: &ToolCall, ctx: &ToolExecutionContext) -> Result<ToolResult> {
        self.ensure_memory_enabled(&ctx.thread_id)?;
        let output = match call.tool_name.as_str() {
            "search_memory" => {
                let query = required_str(&call.arguments, "query")?;
                let limit = int_arg(&call.arguments, "limit", 6, MAX_LIMIT);
                let hits =
                    self.persistence
                        .search_conversation_memories(&ctx.thread_id, query, limit)?;
                json!({ "memories": hits })
            }
            "save_memory" => {
                let kind = required_str(&call.arguments, "kind")?;
                let content = required_str(&call.arguments, "content")?;
                let scope = call
                    .arguments
                    .get("scope")
                    .and_then(Value::as_str)
                    .unwrap_or("thread");
                validate_scope(scope)?;
                let importance = call
                    .arguments
                    .get("importance")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.7)
                    .clamp(0.0, 1.0) as f32;
                let memory_id = self.persistence.append_conversation_memory(
                    &ctx.thread_id,
                    &ctx.turn_id,
                    kind,
                    scope,
                    content,
                    importance,
                )?;
                json!({ "memoryId": memory_id, "stored": true })
            }
            "update_memory" => {
                let memory_id = required_str(&call.arguments, "memory_id")?;
                let scope = call.arguments.get("scope").and_then(Value::as_str);
                if let Some(scope) = scope {
                    validate_scope(scope)?;
                }
                let updated = self.persistence.update_conversation_memory(
                    memory_id,
                    &ctx.thread_id,
                    call.arguments.get("kind").and_then(Value::as_str),
                    scope,
                    call.arguments.get("content").and_then(Value::as_str),
                    call.arguments
                        .get("importance")
                        .and_then(Value::as_f64)
                        .map(|v| v.clamp(0.0, 1.0) as f32),
                )?;
                json!({ "memoryId": memory_id, "updated": updated })
            }
            "forget_memory" => {
                let memory_id = required_str(&call.arguments, "memory_id")?;
                let deleted = self
                    .persistence
                    .delete_conversation_memory(memory_id, &ctx.thread_id)?;
                json!({ "memoryId": memory_id, "deleted": deleted })
            }
            "list_memory" => {
                let limit = int_arg(&call.arguments, "limit", 20, MAX_LIMIT);
                let memories = self
                    .persistence
                    .list_conversation_memories(&ctx.thread_id, limit)?;
                json!({ "memories": memories })
            }
            "run_memory_plan" => {
                let plan: MemoryControllerPlan = serde_json::from_value(call.arguments.clone())?;
                let applied =
                    apply_memory_plan(&self.persistence, &ctx.thread_id, &ctx.turn_id, &plan)?;
                json!({ "applied": applied })
            }
            other => return Err(anyhow!("unknown memory tool: {other}")),
        };

        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
            output,
            error_message: None,
        })
    }
}

impl MemoryToolProvider {
    fn ensure_memory_enabled(&self, thread_id: &str) -> Result<()> {
        let thread = self.persistence.read_thread(thread_id)?;
        if matches!(thread.metadata.memory_mode, MemoryMode::Enabled) {
            Ok(())
        } else {
            Err(anyhow!("memory_disabled: thread memory mode is disabled"))
        }
    }
}

fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("tool_invalid_arguments: missing {key}"))
}

fn int_arg(args: &Value, key: &str, default: usize, cap: usize) -> usize {
    args.get(key)
        .and_then(Value::as_i64)
        .map(|n| n.max(1) as usize)
        .unwrap_or(default)
        .min(cap)
}

fn validate_scope(scope: &str) -> Result<()> {
    if matches!(scope, "thread" | "global") {
        Ok(())
    } else {
        Err(anyhow!(
            "tool_invalid_arguments: scope must be \"thread\" or \"global\""
        ))
    }
}
