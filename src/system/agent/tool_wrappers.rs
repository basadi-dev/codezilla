//! Composite (decorator) tool providers — Phase 2 of the agent architecture.
#![allow(dead_code)] // Public API; not yet wired into the default runtime build.
//!
//! Each wrapper implements `ToolProvider` by delegating to an inner provider
//! while adding a cross-cutting concern:
//!
//! | Wrapper                 | Concern                                         |
//! |-------------------------|-------------------------------------------------|
//! | `LoggingToolProvider`   | Structured timing + status trace logging        |
//! | `CachingToolProvider`   | Dedup identical read-only calls within a turn   |
//! | `RateLimitToolProvider` | Throttle external-API tools (web_fetch, MCP)    |
//!
//! # Composition example
//! ```ignore
//! let file_provider = Arc::new(FileToolProvider::new(sandbox, perms));
//! let logged   = Arc::new(LoggingToolProvider::new(file_provider));
//! let cached   = Arc::new(CachingToolProvider::new(logged));
//! orchestrator.register_provider(cached);
//!
//! let web = Arc::new(WebToolProvider::new());
//! let limited = Arc::new(RateLimitToolProvider::new(web));
//! orchestrator.register_provider(limited);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use async_trait::async_trait;

use crate::system::domain::{
    ToolCall, ToolDefinition, ToolExecutionContext, ToolListingContext, ToolProviderKind,
    ToolResult,
};

use super::tools::ToolProvider;

// ─── LoggingToolProvider ──────────────────────────────────────────────────────

/// Wraps any `ToolProvider` and emits a structured `tracing::debug!` span on
/// every `execute` call recording tool name, argument size, elapsed time, and
/// success/failure status.
pub struct LoggingToolProvider {
    inner: Arc<dyn ToolProvider>,
}

impl LoggingToolProvider {
    pub fn new(inner: Arc<dyn ToolProvider>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl ToolProvider for LoggingToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        self.inner.get_kind()
    }

    fn list_tools(&self, context: &ToolListingContext) -> Vec<ToolDefinition> {
        self.inner.list_tools(context)
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        let t0 = Instant::now();
        let arg_bytes = call.arguments.to_string().len();
        let result = self.inner.execute(call, context).await;
        let elapsed_ms = t0.elapsed().as_millis();
        match &result {
            Ok(r) => tracing::debug!(
                tool = %call.tool_name,
                tool_call_id = %call.tool_call_id,
                arg_bytes,
                elapsed_ms,
                ok = r.ok,
                "tool_call: completed"
            ),
            Err(e) => tracing::warn!(
                tool = %call.tool_name,
                tool_call_id = %call.tool_call_id,
                arg_bytes,
                elapsed_ms,
                error = %e,
                "tool_call: provider error"
            ),
        }
        result
    }
}

// ─── CachingToolProvider ──────────────────────────────────────────────────────

type CacheKey = String;

/// Cached entry keyed by `(turn_id, tool_name, arguments_json)`.
/// Entries from older turns are automatically excluded by including `turn_id`
/// in the key — no explicit eviction is needed.
#[derive(Clone)]
struct CacheEntry {
    ok: bool,
    output: serde_json::Value,
    error_message: Option<String>,
}

/// Wraps any `ToolProvider` and short-circuits identical read-only calls within
/// the same turn. Non-cacheable tools (writes, commands) bypass the cache.
///
/// The cache is **not** per-turn; instead the turn ID is part of the cache key
/// so entries from prior turns never collide. Call `trim_old_turns()` after a
/// turn completes if you want to reclaim memory.
pub struct CachingToolProvider {
    inner: Arc<dyn ToolProvider>,
    cache: Arc<Mutex<HashMap<CacheKey, CacheEntry>>>,
    /// Tools whose results may be cached (all others are passed through).
    cacheable_tools: Vec<String>,
}

impl CachingToolProvider {
    pub fn new(inner: Arc<dyn ToolProvider>) -> Self {
        Self {
            inner,
            cache: Arc::new(Mutex::new(HashMap::new())),
            cacheable_tools: vec![
                "read_file".into(),
                "list_dir".into(),
                "grep_search".into(),
                "image_metadata".into(),
            ],
        }
    }

    /// Override the set of tool names whose results will be cached.
    #[allow(dead_code)]
    pub fn with_cacheable_tools(mut self, tools: Vec<String>) -> Self {
        self.cacheable_tools = tools;
        self
    }

    /// Drop all cache entries whose key contains `turn_id` as a prefix.
    /// Call this at turn end to reclaim memory.
    #[allow(dead_code)]
    pub fn trim_turn(&self, turn_id: &str) {
        let prefix = format!("{turn_id}:");
        let mut guard = self.cache.lock().unwrap();
        guard.retain(|k, _| !k.starts_with(&prefix));
    }

    fn is_cacheable(&self, tool_name: &str) -> bool {
        self.cacheable_tools.iter().any(|t| t == tool_name)
    }

    fn cache_key(turn_id: &str, call: &ToolCall) -> CacheKey {
        format!("{}:{}:{}", turn_id, call.tool_name, call.arguments)
    }
}

#[async_trait]
impl ToolProvider for CachingToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        self.inner.get_kind()
    }

    fn list_tools(&self, context: &ToolListingContext) -> Vec<ToolDefinition> {
        self.inner.list_tools(context)
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        if !self.is_cacheable(&call.tool_name) {
            return self.inner.execute(call, context).await;
        }

        let key = Self::cache_key(&context.turn_id, call);

        // Fast-path: cache hit
        if let Some(entry) = self.cache.lock().unwrap().get(&key).cloned() {
            tracing::debug!(
                tool = %call.tool_name,
                turn_id = %context.turn_id,
                "tool_cache: hit — skipping redundant call"
            );
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: entry.ok,
                output: entry.output,
                error_message: entry.error_message,
            });
        }

        let result = self.inner.execute(call, context).await?;

        // Only cache successful results — failed reads should be retried.
        if result.ok {
            self.cache.lock().unwrap().insert(
                key,
                CacheEntry {
                    ok: result.ok,
                    output: result.output.clone(),
                    error_message: result.error_message.clone(),
                },
            );
        }

        Ok(result)
    }
}

// ─── RateLimitToolProvider ────────────────────────────────────────────────────

struct BucketState {
    window_start: u64,
    calls_in_window: usize,
}

/// Wraps any `ToolProvider` and enforces a per-tool sliding-window rate limit.
/// Rejected calls return a structured `rate_limit_exceeded` error so the model
/// can back off gracefully rather than seeing a hard crash.
///
/// Default limits: `web_fetch` → 10 calls per 60 s.
pub struct RateLimitToolProvider {
    inner: Arc<dyn ToolProvider>,
    /// tool_name → (max_calls_per_window, window_seconds)
    limits: HashMap<String, (usize, u64)>,
    state: Arc<Mutex<HashMap<String, BucketState>>>,
}

impl RateLimitToolProvider {
    pub fn new(inner: Arc<dyn ToolProvider>) -> Self {
        let mut limits = HashMap::new();
        limits.insert("web_fetch".into(), (10, 60));
        Self {
            inner,
            limits,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add or override a rate limit for a specific tool.
    pub fn with_limit(
        mut self,
        tool_name: impl Into<String>,
        max_calls: usize,
        window_secs: u64,
    ) -> Self {
        self.limits.insert(tool_name.into(), (max_calls, window_secs));
        self
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Returns `true` if the call is within quota and increments the counter.
    fn check_and_record(&self, tool_name: &str) -> bool {
        let Some(&(max_calls, window_secs)) = self.limits.get(tool_name) else {
            return true; // no limit configured for this tool
        };
        let now = Self::now_secs();
        let mut state = self.state.lock().unwrap();
        let bucket = state
            .entry(tool_name.to_string())
            .or_insert_with(|| BucketState { window_start: now, calls_in_window: 0 });

        if now >= bucket.window_start + window_secs {
            bucket.window_start = now;
            bucket.calls_in_window = 0;
        }
        if bucket.calls_in_window >= max_calls {
            return false;
        }
        bucket.calls_in_window += 1;
        true
    }
}

#[async_trait]
impl ToolProvider for RateLimitToolProvider {
    fn get_kind(&self) -> ToolProviderKind {
        self.inner.get_kind()
    }

    fn list_tools(&self, context: &ToolListingContext) -> Vec<ToolDefinition> {
        self.inner.list_tools(context)
    }

    async fn execute(&self, call: &ToolCall, context: &ToolExecutionContext) -> Result<ToolResult> {
        if !self.check_and_record(&call.tool_name) {
            tracing::warn!(
                tool = %call.tool_name,
                tool_call_id = %call.tool_call_id,
                "rate_limit: call rejected — quota exceeded"
            );
            let limit = self
                .limits
                .get(&call.tool_name)
                .map(|(max, win)| format!("{max} calls per {win}s"))
                .unwrap_or_default();
            return Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: false,
                output: serde_json::json!({
                    "error": "rate_limit_exceeded",
                    "tool": call.tool_name,
                    "limit": limit,
                    "message": format!(
                        "Tool '{}' has been called too many times within the rate limit window \
                         ({}). Wait before retrying.",
                        call.tool_name, limit
                    )
                }),
                error_message: Some(format!(
                    "rate_limit_exceeded: '{}' — {}",
                    call.tool_name, limit
                )),
            });
        }
        self.inner.execute(call, context).await
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::{ApprovalPolicy, PermissionProfile};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EchoProvider {
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ToolProvider for EchoProvider {
        fn get_kind(&self) -> ToolProviderKind {
            ToolProviderKind::Builtin
        }
        fn list_tools(&self, _ctx: &ToolListingContext) -> Vec<ToolDefinition> {
            vec![]
        }
        async fn execute(&self, call: &ToolCall, _ctx: &ToolExecutionContext) -> Result<ToolResult> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult {
                tool_call_id: call.tool_call_id.clone(),
                ok: true,
                output: json!({ "echo": call.tool_name }),
                error_message: None,
            })
        }
    }

    fn exec_ctx(turn_id: &str) -> ToolExecutionContext {
        ToolExecutionContext {
            thread_id: "thread1".into(),
            turn_id: turn_id.into(),
            cwd: ".".into(),
            permission_profile: PermissionProfile::default(),
            approval_policy: ApprovalPolicy::default(),
            agent_depth: 0,
        }
    }

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            tool_call_id: "c1".into(),
            provider_kind: ToolProviderKind::Builtin,
            tool_name: name.into(),
            arguments: args,
        }
    }

    #[tokio::test]
    async fn logging_passes_through_result() {
        let inner = Arc::new(EchoProvider { call_count: Arc::new(AtomicUsize::new(0)) });
        let logged = LoggingToolProvider::new(inner);
        let result = logged
            .execute(&call("read_file", json!({"path": "x.rs"})), &exec_ctx("t1"))
            .await
            .unwrap();
        assert!(result.ok);
    }

    #[tokio::test]
    async fn caching_deduplicates_read_file_within_turn() {
        let count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(EchoProvider { call_count: count.clone() });
        let cached = CachingToolProvider::new(inner);
        let c = call("read_file", json!({"path": "src/main.rs"}));
        let ctx = exec_ctx("turn1");

        cached.execute(&c, &ctx).await.unwrap();
        cached.execute(&c, &ctx).await.unwrap();
        cached.execute(&c, &ctx).await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 1, "inner called once; rest served from cache");
    }

    #[tokio::test]
    async fn caching_does_not_cross_turns() {
        let count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(EchoProvider { call_count: count.clone() });
        let cached = CachingToolProvider::new(inner);
        let c = call("read_file", json!({"path": "src/lib.rs"}));

        cached.execute(&c, &exec_ctx("turn_a")).await.unwrap();
        cached.execute(&c, &exec_ctx("turn_b")).await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 2, "different turns → two real calls");
    }

    #[tokio::test]
    async fn caching_does_not_cache_writes() {
        let count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(EchoProvider { call_count: count.clone() });
        let cached = CachingToolProvider::new(inner);
        let c = call("write_file", json!({"path": "out.rs", "content": "x"}));
        let ctx = exec_ctx("t1");

        cached.execute(&c, &ctx).await.unwrap();
        cached.execute(&c, &ctx).await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 2, "write_file must never be cached");
    }

    #[tokio::test]
    async fn rate_limit_allows_up_to_max() {
        let inner = Arc::new(EchoProvider { call_count: Arc::new(AtomicUsize::new(0)) });
        let limited = RateLimitToolProvider::new(inner).with_limit("web_fetch", 2, 60);
        let c = call("web_fetch", json!({"url": "https://example.com"}));
        let ctx = exec_ctx("t1");

        let r1 = limited.execute(&c, &ctx).await.unwrap();
        let r2 = limited.execute(&c, &ctx).await.unwrap();
        let r3 = limited.execute(&c, &ctx).await.unwrap();

        assert!(r1.ok, "first call allowed");
        assert!(r2.ok, "second call allowed");
        assert!(!r3.ok, "third call must be rejected");
        assert!(
            r3.error_message
                .as_deref()
                .unwrap_or("")
                .contains("rate_limit_exceeded"),
            "error message must mention rate_limit_exceeded"
        );
    }

    #[tokio::test]
    async fn rate_limit_passes_unlimited_tools() {
        let inner = Arc::new(EchoProvider { call_count: Arc::new(AtomicUsize::new(0)) });
        let limited = RateLimitToolProvider::new(inner).with_limit("web_fetch", 1, 60);
        let c = call("read_file", json!({"path": "x"})); // no limit configured
        let ctx = exec_ctx("t1");

        for _ in 0..5 {
            assert!(limited.execute(&c, &ctx).await.unwrap().ok);
        }
    }
}
