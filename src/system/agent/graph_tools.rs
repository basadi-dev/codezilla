//! Codebase-memory graph tools — exposes the SQLite knowledge graph to the agent.
//!
//! Implements [`ToolProvider`] with four tools:
//! - `index_graph`   — (re)build the graph for the current cwd.
//! - `search_graph`  — BM25 search over symbols.
//! - `trace_path`    — BFS over CALLS edges (inbound/outbound).
//! - `find_impact`   — reverse BFS blast-radius from a file.
//!
//! The provider owns a [`GraphStore`] opened lazily on first use and reused
//! while the cwd stays the same. The DB is persistent, so a graph indexed in
//! an earlier session is queryable immediately. The DB lives at
//! `<app_home>/state/codebase-graph/<hash(cwd)>.sqlite3`.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;

use super::tools::ToolProvider;
use crate::system::domain::{
    ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext, ToolProviderKind,
    ToolResult,
};
use crate::system::intel::graph::{db_path_for, GraphStore};

/// Upper bounds on tool arguments so a bad call can't stall the agent.
const MAX_FILES_CAP: usize = 50_000;
const MAX_DEPTH_CAP: usize = 16;
const MAX_LIMIT_CAP: usize = 500;

/// Tool provider exposing the codebase knowledge graph to the agent.
pub struct GraphToolProvider {
    /// Store for the most recently used cwd, opened lazily on first use and
    /// replaced when the cwd changes (so switching projects is supported).
    store: Mutex<Option<(String, GraphStore)>>,
    /// App home dir (where DBs live).
    app_home: PathBuf,
}

impl GraphToolProvider {
    pub fn new(app_home: PathBuf) -> Self {
        Self {
            store: Mutex::new(None),
            app_home,
        }
    }

    /// Run a closure on the store for `cwd`, opening/reopening it if needed.
    fn with_store<R>(&self, cwd: &str, f: impl FnOnce(&GraphStore) -> Result<R>) -> Result<R> {
        let mut guard = self.store.lock().unwrap();
        let needs_reopen = !matches!(guard.as_ref(), Some((open_cwd, _)) if open_cwd == cwd);
        if needs_reopen {
            let db_path = db_path_for(&self.app_home, std::path::Path::new(cwd));
            *guard = Some((cwd.to_string(), GraphStore::open(&db_path)?));
        }
        f(&guard.as_ref().expect("store just opened").1)
    }

    /// Like [`with_store`](Self::with_store), but fails with a hint to run
    /// `index_graph` first when the graph is empty.
    fn with_indexed_store<R>(
        &self,
        cwd: &str,
        f: impl FnOnce(&GraphStore) -> Result<R>,
    ) -> Result<R> {
        self.with_store(cwd, |store| {
            if store.node_count()? == 0 {
                return Err(anyhow!(
                    "the codebase graph for {cwd} is empty — call `index_graph` first"
                ));
            }
            f(store)
        })
    }
}

#[async_trait]
impl ToolProvider for GraphToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        ToolProviderKind::Builtin
    }

    fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "index_graph".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Index the current repository into a persistent SQLite knowledge graph \
                 (functions, classes, files, CALLS edges). Must be called before search_graph / \
                 trace_path / find_impact. Re-indexing is cheap and replaces the \
                 previous index. Returns a summary with counts."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "max_files": {
                            "type": "integer",
                            "description": "Max files to walk (default 2000)."
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max walk depth from cwd (default 6)."
                        }
                    }
                }),
                requires_approval: false,
                supports_parallel_calls: false,
            },
            ToolDefinition {
                name: "search_graph".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "BM25 search over the indexed symbol graph (function/class/struct \
                 names + signatures). Returns ranked matches with file + line. Call \
                 `index_graph` first."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Free-text query (tokens ANDed)."},
                        "limit": {"type": "integer", "description": "Max results (default 20)."}
                    },
                    "required": ["query"]
                }),
                requires_approval: false,
                supports_parallel_calls: true,
            },
            ToolDefinition {
                name: "trace_path".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Traverse the CALLS graph from a named symbol. `outbound` = what does \
                 X call; `inbound` = what calls X. BFS up to max_depth hops. Returns visited \
                 nodes with depth + file + line. Call `index_graph` first."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "description": "Symbol name to trace from."},
                        "direction": {"type": "string", "enum": ["inbound", "outbound"], "description": "inbound = callers, outbound = callees (default outbound)."},
                        "max_depth": {"type": "integer", "description": "Max BFS hops (default 3)."}
                    },
                    "required": ["function_name"]
                }),
                requires_approval: false,
                supports_parallel_calls: true,
            },
            ToolDefinition {
                name: "find_impact".into(),
                provider_kind: ToolProviderKind::Builtin,
                description: "Reverse impact analysis: given a file path, find all symbols that \
                 transitively call into symbols defined in that file (blast radius). Call \
                 `index_graph` first."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Repo-relative file path."},
                        "max_depth": {"type": "integer", "description": "Max BFS hops (default 4)."}
                    },
                    "required": ["file_path"]
                }),
                requires_approval: false,
                supports_parallel_calls: true,
            },
        ]
    }

    async fn execute(&self, call: &ToolCall, ctx: &ToolExecutionContext) -> Result<ToolResult> {
        let cwd = ctx.cwd.clone();
        let str_arg = |key: &str| -> Result<String> {
            call.arguments
                .get(key)
                .and_then(Value::as_str)
                .map(String::from)
                .ok_or_else(|| anyhow!("tool_invalid_arguments: missing {key}"))
        };
        let int_arg = |key: &str, default: usize, cap: usize| -> usize {
            call.arguments
                .get(key)
                .and_then(Value::as_i64)
                .map(|n| n.max(0) as usize)
                .unwrap_or(default)
                .min(cap)
        };

        let result: Value = match call.tool_name.as_str() {
            "index_graph" => {
                let max_files = int_arg("max_files", 2000, MAX_FILES_CAP);
                let max_depth = int_arg("max_depth", 6, MAX_DEPTH_CAP);
                self.with_store(&cwd, |s| {
                    s.index(std::path::Path::new(&cwd), max_files, max_depth)
                })?
            }
            "search_graph" => {
                let query = str_arg("query")?;
                let limit = int_arg("limit", 20, MAX_LIMIT_CAP);
                self.with_indexed_store(&cwd, |s| s.search(&query, limit))?
                    .into()
            }
            "trace_path" => {
                let name = str_arg("function_name")?;
                let direction = call
                    .arguments
                    .get("direction")
                    .and_then(Value::as_str)
                    .unwrap_or("outbound");
                if !matches!(direction, "inbound" | "outbound") {
                    return Err(anyhow!(
                        "tool_invalid_arguments: direction must be \"inbound\" or \"outbound\", got {direction:?}"
                    ));
                }
                let max_depth = int_arg("max_depth", 3, MAX_DEPTH_CAP);
                self.with_indexed_store(&cwd, |s| s.trace_path(&name, direction, max_depth))?
                    .into()
            }
            "find_impact" => {
                let file_path = str_arg("file_path")?;
                let max_depth = int_arg("max_depth", 4, MAX_DEPTH_CAP);
                self.with_indexed_store(&cwd, |s| s.find_impact(&file_path, max_depth))?
                    .into()
            }
            other => {
                return Ok(ToolResult {
                    tool_call_id: call.tool_call_id.clone(),
                    ok: false,
                    output: json!({"error": format!("unknown graph tool: {other}")}),
                    error_message: Some(format!("unknown graph tool: {other}")),
                });
            }
        };

        Ok(ToolResult {
            tool_call_id: call.tool_call_id.clone(),
            ok: true,
            output: result,
            error_message: None,
        })
    }
}