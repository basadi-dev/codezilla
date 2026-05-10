//! Pluggable middleware for the agent loop.
// pre_turn / post_response hooks and FailTurn / CompleteTurn actions are public
// API reserved for future middleware implementations.
#![allow(dead_code)]
//!
//! The `TurnMiddleware` trait defines three hook points that run at natural
//! pauses in the executor loop:
//!
//! 1. **`pre_turn`** — before the first model call. Modify system instructions,
//!    inject context, or short-circuit the turn.
//! 2. **`post_response`** — after each model response, before tool dispatch.
//!    Inspect/filter tool calls, inject instructions.
//! 3. **`post_tools`** — after each tool-dispatch round completes. Review
//!    results, inject feedback, or terminate the turn.
//!
//! Middlewares are composed in a `MiddlewareChain` and executed in registration
//! order. Any middleware returning `MiddlewareAction::FailTurn` or
//! `MiddlewareAction::CompleteTurn` short-circuits the chain.

pub mod checkpoint_review;
pub mod loop_guard;
pub use checkpoint_review::CheckpointReviewMiddleware;
// GuardVerdict and LoopGuardState are public API used by tests and future executor refactors.
#[allow(unused_imports)]
pub use loop_guard::{ConstrainedMode, GuardVerdict, LoopGuardState};

use anyhow::Result;
use async_trait::async_trait;

use crate::system::config::AgentConfig;
use crate::system::domain::{FileChangeSummary, ThreadId, ToolCall, ToolResult, TurnId};
use crate::system::runtime::ConversationRuntime;

// ─── MiddlewareAction ─────────────────────────────────────────────────────────

/// Result of a middleware hook. Determines what the executor does next.
#[derive(Debug, Clone)]
pub enum MiddlewareAction {
    /// Continue to the next middleware / the normal executor flow.
    Continue,
    /// Inject a system-level instruction into the next model call.
    /// The executor persists this as a `SystemMessage` conversation item.
    InjectInstruction(String),
    /// Terminate the turn as a failure with the given (kind, message).
    FailTurn { kind: String, message: String },
    /// Terminate the turn successfully (early completion).
    CompleteTurn,
}

// ─── TurnMiddlewareContext ────────────────────────────────────────────────────

/// Shared context passed to every middleware hook. Contains read/write access
/// to the turn's evolving state.
pub struct TurnMiddlewareContext {
    pub runtime: Option<ConversationRuntime>,
    pub thread_id: ThreadId,
    pub turn_id: TurnId,
    pub agent_config: AgentConfig,
    pub agent_depth: u32,
    pub cwd: String,
    pub user_task_text: String,

    // ── Evolving state (read/write by middlewares) ─────────────────────────
    /// System instructions to prepend to the model call.
    pub system_instructions: Vec<String>,
    /// Accumulated file changes across the turn.
    pub file_changes: Vec<FileChangeSummary>,
    /// Plan steps extracted from model narration.
    pub plan_steps: Vec<String>,
    /// Completed actions for progress tracking.
    pub completed_actions: Vec<String>,
    /// Current agent loop iteration count.
    pub agent_iterations: usize,
    /// Completed tool-dispatch rounds.
    pub completed_tool_rounds: usize,
}

// ─── TurnMiddleware trait ─────────────────────────────────────────────────────

/// Extension point for the agent loop. Implement this trait to add custom
/// behaviour at natural pause points in the executor.
///
/// All methods have default no-op implementations so you only need to
/// override the hooks you care about.
#[async_trait]
pub trait TurnMiddleware: Send + Sync {
    /// A unique name used for logging and observability.
    fn name(&self) -> &str;

    /// Called once before the first model call. Can modify system instructions,
    /// inject pre-computed context, or short-circuit the turn entirely.
    async fn pre_turn(&self, _ctx: &mut TurnMiddlewareContext) -> Result<MiddlewareAction> {
        Ok(MiddlewareAction::Continue)
    }

    /// Called after each model response, before tool dispatch.
    /// Receives the assistant's text and tool calls.
    /// Can filter, modify, or reject tool calls.
    async fn post_response(
        &self,
        _ctx: &mut TurnMiddlewareContext,
        _assistant_text: &str,
        _tool_calls: &[ToolCall],
    ) -> Result<MiddlewareAction> {
        Ok(MiddlewareAction::Continue)
    }

    /// Called after each tool-dispatch round completes.
    /// Receives the tool results and any file changes produced in this round.
    async fn post_tools(
        &self,
        _ctx: &mut TurnMiddlewareContext,
        _results: &[ToolResult],
        _round_file_changes: &[FileChangeSummary],
    ) -> Result<MiddlewareAction> {
        Ok(MiddlewareAction::Continue)
    }
}

// ─── MiddlewareChain ──────────────────────────────────────────────────────────

/// Ordered collection of middlewares. Executes each in registration order;
/// short-circuits on any non-Continue action.
pub struct MiddlewareChain {
    middlewares: Vec<Box<dyn TurnMiddleware>>,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    pub fn push(&mut self, middleware: Box<dyn TurnMiddleware>) {
        self.middlewares.push(middleware);
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Run `pre_turn` on every middleware in order. Returns the first
    /// non-Continue action, or Continue if all pass.
    pub async fn run_pre_turn(&self, ctx: &mut TurnMiddlewareContext) -> Result<MiddlewareAction> {
        for mw in &self.middlewares {
            let action = mw.pre_turn(ctx).await?;
            match &action {
                MiddlewareAction::Continue => continue,
                other => {
                    tracing::debug!(
                        middleware = mw.name(),
                        action = ?other,
                        "middleware chain: pre_turn short-circuited"
                    );
                    return Ok(action);
                }
            }
        }
        Ok(MiddlewareAction::Continue)
    }

    /// Run `post_response` on every middleware in order.
    pub async fn run_post_response(
        &self,
        ctx: &mut TurnMiddlewareContext,
        assistant_text: &str,
        tool_calls: &[ToolCall],
    ) -> Result<MiddlewareAction> {
        for mw in &self.middlewares {
            let action = mw.post_response(ctx, assistant_text, tool_calls).await?;
            match &action {
                MiddlewareAction::Continue => continue,
                other => {
                    tracing::debug!(
                        middleware = mw.name(),
                        action = ?other,
                        "middleware chain: post_response short-circuited"
                    );
                    return Ok(action);
                }
            }
        }
        Ok(MiddlewareAction::Continue)
    }

    /// Run `post_tools` on every middleware in order.
    pub async fn run_post_tools(
        &self,
        ctx: &mut TurnMiddlewareContext,
        results: &[ToolResult],
        round_file_changes: &[FileChangeSummary],
    ) -> Result<MiddlewareAction> {
        for mw in &self.middlewares {
            let action = mw.post_tools(ctx, results, round_file_changes).await?;
            match &action {
                MiddlewareAction::Continue => continue,
                other => {
                    tracing::debug!(
                        middleware = mw.name(),
                        action = ?other,
                        "middleware chain: post_tools short-circuited"
                    );
                    return Ok(action);
                }
            }
        }
        Ok(MiddlewareAction::Continue)
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingMiddleware {
        name: String,
        pre_count: Arc<AtomicUsize>,
        post_resp_count: Arc<AtomicUsize>,
        post_tools_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl TurnMiddleware for CountingMiddleware {
        fn name(&self) -> &str {
            &self.name
        }

        async fn pre_turn(&self, _ctx: &mut TurnMiddlewareContext) -> Result<MiddlewareAction> {
            self.pre_count.fetch_add(1, Ordering::SeqCst);
            Ok(MiddlewareAction::Continue)
        }

        async fn post_response(
            &self,
            _ctx: &mut TurnMiddlewareContext,
            _text: &str,
            _calls: &[ToolCall],
        ) -> Result<MiddlewareAction> {
            self.post_resp_count.fetch_add(1, Ordering::SeqCst);
            Ok(MiddlewareAction::Continue)
        }

        async fn post_tools(
            &self,
            _ctx: &mut TurnMiddlewareContext,
            _results: &[ToolResult],
            _changes: &[FileChangeSummary],
        ) -> Result<MiddlewareAction> {
            self.post_tools_count.fetch_add(1, Ordering::SeqCst);
            Ok(MiddlewareAction::Continue)
        }
    }

    struct ShortCircuitMiddleware;

    #[async_trait]
    impl TurnMiddleware for ShortCircuitMiddleware {
        fn name(&self) -> &str {
            "short_circuit"
        }

        async fn pre_turn(&self, _ctx: &mut TurnMiddlewareContext) -> Result<MiddlewareAction> {
            Ok(MiddlewareAction::FailTurn {
                kind: "test".into(),
                message: "blocked by middleware".into(),
            })
        }
    }

    fn make_test_ctx() -> TurnMiddlewareContext {
        // Minimal context for unit tests — no runtime needed since the
        // counting/short-circuit middlewares don't touch it.
        TurnMiddlewareContext {
            runtime: None,
            thread_id: "test_thread".into(),
            turn_id: "test_turn".into(),
            agent_config: AgentConfig::default(),
            agent_depth: 0,
            cwd: ".".into(),
            user_task_text: "test task".into(),
            system_instructions: Vec::new(),
            file_changes: Vec::new(),
            plan_steps: Vec::new(),
            completed_actions: Vec::new(),
            agent_iterations: 0,
            completed_tool_rounds: 0,
        }
    }

    #[tokio::test]
    async fn chain_runs_all_middlewares_in_order() {
        let pre = Arc::new(AtomicUsize::new(0));
        let post_resp = Arc::new(AtomicUsize::new(0));
        let post_tools = Arc::new(AtomicUsize::new(0));

        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(CountingMiddleware {
            name: "a".into(),
            pre_count: pre.clone(),
            post_resp_count: post_resp.clone(),
            post_tools_count: post_tools.clone(),
        }));
        chain.push(Box::new(CountingMiddleware {
            name: "b".into(),
            pre_count: pre.clone(),
            post_resp_count: post_resp.clone(),
            post_tools_count: post_tools.clone(),
        }));

        let mut ctx = make_test_ctx();
        let action = chain.run_pre_turn(&mut ctx).await.unwrap();
        assert!(matches!(action, MiddlewareAction::Continue));
        assert_eq!(pre.load(Ordering::SeqCst), 2);

        let action = chain
            .run_post_response(&mut ctx, "hello", &[])
            .await
            .unwrap();
        assert!(matches!(action, MiddlewareAction::Continue));
        assert_eq!(post_resp.load(Ordering::SeqCst), 2);

        let action = chain.run_post_tools(&mut ctx, &[], &[]).await.unwrap();
        assert!(matches!(action, MiddlewareAction::Continue));
        assert_eq!(post_tools.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chain_short_circuits_on_fail_turn() {
        let post_mw_count = Arc::new(AtomicUsize::new(0));
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(ShortCircuitMiddleware));
        chain.push(Box::new(CountingMiddleware {
            name: "should_not_run".into(),
            pre_count: post_mw_count.clone(),
            post_resp_count: post_mw_count.clone(),
            post_tools_count: post_mw_count.clone(),
        }));

        let mut ctx = make_test_ctx();
        let action = chain.run_pre_turn(&mut ctx).await.unwrap();
        assert!(matches!(action, MiddlewareAction::FailTurn { .. }));
        // The second middleware should not have run.
        assert_eq!(post_mw_count.load(Ordering::SeqCst), 0);
    }
}
