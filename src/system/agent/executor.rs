use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

use super::middleware::{ConstrainedMode, GuardVerdict};
use super::model_gateway::{estimate_items_token_pct, is_context_overflow_error, ModelStreamEvent};
mod context;
mod tool_dispatch;
pub(crate) mod utils;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnPhase {
    /// Initial context gathering. Read-only tools allowed within budget.
    Orient,
    /// Model should produce a structured plan before editing.
    Plan,
    /// Executing plan steps. Edit and command tools allowed.
    Execute,
    /// Post-change verification. Command tools preferred.
    Verify,
}

impl TurnPhase {
    fn label(&self) -> &'static str {
        match self {
            TurnPhase::Orient => "ORIENT",
            TurnPhase::Plan => "PLAN",
            TurnPhase::Execute => "EXECUTE",
            TurnPhase::Verify => "VERIFY",
        }
    }
}

use self::context::TurnContext;
use self::utils::{
    already_read_directive, classify_turn_intent, derive_thread_title, extract_plan_from_response,
    find_repetition_start, initial_read_budget, intent_directive, intent_to_reasoning_effort,
    is_degenerate_repetition, is_read_only_tool, progress_summary, recently_read_paths,
    should_retry_no_tool_completion, thinking_instruction, user_requested_verification,
    wants_verbose_repo_map, ProgressState, TurnIntent,
};
use crate::system::domain::{
    now_seconds, ConversationItem, ItemKind, RuntimeEventKind, ThreadStatus,
    TokenUsage, ToolCall, ToolResult, TurnMetrics, TurnStatus, UserInput, KEY_TEXT,
};
use crate::system::error as cod_error;
use crate::system::runtime::RepoMapVerbosity;

// ─── TurnExecutor ─────────────────────────────────────────────────────────────

/// Owns the per-turn agent loop. Created per `start_turn` call and runs in a
/// dedicated tokio task. Holds only an `Arc` clone of the runtime — no shared
/// mutable state of its own.
pub struct TurnExecutor {
    pub runtime: crate::system::runtime::ConversationRuntime,
}

impl TurnExecutor {
    pub async fn run_turn(
        &self,
        params: crate::system::runtime::TurnStartParams,
        turn_id: crate::system::domain::TurnId,
    ) -> Result<()> {
        let thread = self
            .runtime
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;

        {
            let mut thread = thread.lock().await;
            let turn = thread
                .turns
                .iter_mut()
                .find(|t| t.metadata.turn_id == turn_id)
                .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
            turn.status = TurnStatus::Running;
            turn.metadata.status = TurnStatus::Running;
            turn.metadata.updated_at = now_seconds();
            self.runtime
                .inner
                .persistence_manager
                .update_turn(&turn.metadata)?;
        }

        for input in params.input.clone() {
            self.append_user_input(&params.thread_id, &turn_id, &input)
                .await?;
        }

        let turn_intent = classify_turn_intent(&params.input);
        let verification_requested = user_requested_verification(&params.input);
        let verbose_repo_map = match params.repo_map_verbosity {
            Some(RepoMapVerbosity::Verbose) => true,
            Some(RepoMapVerbosity::Lean) => false,
            None => wants_verbose_repo_map(&params.input),
        };

        // ── Codebase intelligence — build the repo map once for the entire turn ─
        // The map is expensive to build (directory walk + file reads + hashing)
        // so we compute it here and thread the result through every call to
        // system_instructions(). This avoids N-1 redundant rebuilds in a
        // multi-round turn.
        let cwd_for_persist = params.cwd.as_deref().unwrap_or(".");
        let repo_map_text: Option<String> = {
            let intel_cfg = &self.runtime.inner.effective_config.codebase_intel;
            if intel_cfg.enabled {
                // Show the user that indexing is in progress.
                self.runtime
                    .publish_event(
                        RuntimeEventKind::CompactionStatus,
                        Some(params.thread_id.clone()),
                        Some(turn_id.clone()),
                        json!({ "status": "started", "message": "Indexing codebase…" }),
                    )
                    .await?;

                let t0 = std::time::Instant::now();
                let repo_map = self.runtime.inner.repo_map.clone();
                let result = tokio::task::spawn_blocking({
                    let cwd = cwd_for_persist.to_string();
                    let mut cfg = intel_cfg.clone();
                    if verbose_repo_map {
                        cfg.include_non_indexable = true;
                        cfg.include_binary = true;
                        cfg.include_hidden = true;
                        cfg.include_excluded_paths = true;
                    }
                    move || repo_map.build_map(&cwd, &cfg)
                })
                .await
                .unwrap_or(None);

                let elapsed_ms = t0.elapsed().as_millis();
                let status_msg = match &result {
                    Some(map) => {
                        let approx_files = map.lines().filter(|l| l.contains('[')).count();
                        format!("Indexed {approx_files} files in {elapsed_ms}ms")
                    }
                    None => format!("Codebase index skipped ({elapsed_ms}ms)"),
                };
                self.runtime
                    .publish_event(
                        RuntimeEventKind::CompactionStatus,
                        Some(params.thread_id.clone()),
                        Some(turn_id.clone()),
                        json!({ "status": "completed", "message": status_msg }),
                    )
                    .await?;

                result
            } else {
                None
            }
        };

        let agent_cfg = self.runtime.inner.effective_config.agent.clone();

        // Persist the system prompt once per turn for DB-level auditing and
        // post-mortem debugging. System entries are hidden in the TUI but are
        // available for inspection in the raw conversation store.
        // We also cache the result here so the first loop iteration can reuse
        // it rather than calling system_instructions() a second time.
        let initial_system_instructions = self
            .system_instructions(
                cwd_for_persist,
                None,
                repo_map_text.as_deref(),
                turn_intent,
                &[],
            )
            .await?;
        {
            let system_text = initial_system_instructions.join("\n\n");
            if !system_text.is_empty() {
                self.persist_turn_item(ConversationItem {
                    item_id: format!("sys_{}", Uuid::new_v4().simple()),
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    created_at: now_seconds(),
                    kind: ItemKind::SystemMessage,
                    payload: json!({ "text": system_text }),
                })
                .await?;
            }
        }
        // Seed the first loop iteration with the already-computed instructions;
        // subsequent iterations will recompute (reasoning_effort may differ).
        let mut cached_system_instructions: Option<Vec<String>> = Some(initial_system_instructions);

        // ── Exploration strategy (optional) ──────────────────────────────────
        // If a registered strategy activates (e.g. speculative tournament), run
        // it before the main loop and inject the winning plan into system
        // instructions so the main loop skips exploration and executes directly.
        {
            use super::strategy::ExplorationStrategy;
            let strategy = super::strategy::SpeculativeStrategy::from_config(&agent_cfg);
            if strategy.should_activate(turn_intent, &params.input, params.agent_depth) {
                let user_task_text = params
                    .input
                    .iter()
                    .filter_map(|i| i.text.as_ref().map(|t| t.text.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim_start_matches("/speculate")
                    .trim()
                    .to_string();

                if !user_task_text.is_empty() {
                    match strategy
                        .explore(
                            &user_task_text,
                            &self.runtime,
                            &params.thread_id,
                            &turn_id,
                            cwd_for_persist,
                            params.agent_depth,
                        )
                        .await
                    {
                        Ok(result) => {
                            // Persist full exploration result for transcript history.
                            if !result.metadata.is_null() {
                                self.persist_turn_item(ConversationItem {
                                    item_id: format!("spec_{}", Uuid::new_v4().simple()),
                                    thread_id: params.thread_id.clone(),
                                    turn_id: turn_id.clone(),
                                    created_at: now_seconds(),
                                    kind: ItemKind::SpeculativeResult,
                                    payload: result.metadata,
                                })
                                .await?;
                            }

                            // Inject the plan into cached system instructions.
                            if let Some(plan_instruction) = result.plan_instruction {
                                if let Some(instructions) = cached_system_instructions.as_mut() {
                                    instructions.push(plan_instruction);
                                }
                                tracing::info!(
                                    turn_id = %turn_id,
                                    strategy = strategy.name(),
                                    "executor: exploration plan injected into context"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                turn_id = %turn_id,
                                error = %e,
                                strategy = strategy.name(),
                                "executor: exploration strategy failed, falling back to normal execution"
                            );
                            self.runtime
                                .publish_event(
                                    RuntimeEventKind::Warning,
                                    Some(params.thread_id.clone()),
                                    Some(turn_id.clone()),
                                    json!({
                                        "message": format!(
                                            "Exploration strategy '{}' failed: {e}. Falling back to normal execution.",
                                            strategy.name()
                                        )
                                    }),
                                )
                                .await?;
                        }
                    }
                }
            }
        }

        // Tracks whether we have already done one automatic context-overflow
        // trim this turn. We only retry once to avoid an infinite loop.
        let mut auto_trim_attempted = false;

        // Saved items from auto-trim, keyed by turn_id.  If the retry also
        // fails we restore these so the in-memory session is not left corrupted.
        let mut trimmed_turn_items: HashMap<String, Vec<ConversationItem>> = HashMap::new();

        // ── Loop-break guards ─────────────────────────────────────────────────
        // Consolidated into a single struct for testability and readability.
        // See middleware/loop_guard.rs for the full state machine and checks.
        let mut guards = super::middleware::LoopGuardState::new();
        let mut no_tool_nudge_instruction: Option<String> = None;
        let turn_start_time = std::time::Instant::now();
        const REPEAT_THRESHOLD: usize = 3;

        // ── Phase-aware anti-looping state ─────────────────────────────────
        let needs_planning = matches!(
            turn_intent,
            TurnIntent::Edit | TurnIntent::Debug | TurnIntent::Unknown
        );
        let mut current_phase = TurnPhase::Orient;
        let read_budget = initial_read_budget(turn_intent);
        let mut plan_steps: Vec<String> = Vec::new();
        let mut completed_actions: Vec<String> = Vec::new();

        // ── Middleware chain ───────────────────────────────────────────────
        // Build the middleware pipeline. Currently includes:
        //   - CheckpointReviewMiddleware: validates file changes between rounds.
        // Future middlewares (LoopGuard, Security, etc.) get added here.
        let middleware_chain = {
            use super::middleware::{MiddlewareChain, CheckpointReviewMiddleware};
            let mut chain = MiddlewareChain::new();
            chain.push(Box::new(CheckpointReviewMiddleware::new()));
            chain
        };

        // Extract the user task text once for the reviewer's context.
        let user_task_text_for_review: String = params
            .input
            .iter()
            .filter_map(|i| i.text.as_ref().map(|t| t.text.as_str()))
            .collect::<Vec<_>>()
            .join("\n");

        loop {
            guards.agent_iterations += 1;
            if let Some(GuardVerdict::FailTurn { kind, message }) =
                guards.check_iteration_limit(&agent_cfg)
            {
                tracing::error!(
                    turn_id = %turn_id,
                    guards.agent_iterations,
                    "executor: backstop limit reached — terminating turn"
                );
                return self.fail_turn(&params.thread_id, &turn_id, &kind, &message).await;
            }
            tracing::debug!(
                turn_id = %turn_id,
                guards.agent_iterations,
                guards.consecutive_failures,
                guards.no_tool_nudges,
                guards.completed_tool_rounds,
                intent = ?turn_intent,
                phase = current_phase.label(),
                "executor: loop iteration"
            );
            if let Some(thread) = self.runtime.load_thread(&params.thread_id).await? {
                let thread = thread.lock().await;
                let turn = thread
                    .turns
                    .iter()
                    .find(|t| t.metadata.turn_id == turn_id)
                    .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                if turn.cancel_token.is_cancelled() {
                    return self.complete_interrupted(&params.thread_id, &turn_id).await;
                }
            }

            self.drain_steering(&params.thread_id, &turn_id).await?;

            // Inject nudge as a conversation item at the recency boundary
            // instead of burying it in the system prompt where attention
            // degradation causes the model to ignore it.
            if let Some(instruction) = no_tool_nudge_instruction.take() {
                let nudge_item = ConversationItem {
                    item_id: format!("nudge_{}", Uuid::new_v4().simple()),
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    created_at: now_seconds(),
                    kind: ItemKind::SystemMessage,
                    payload: json!({ "text": instruction }),
                };
                self.persist_turn_item(nudge_item).await?;
            }

            let mut turn_ctx = TurnContext::build(self, &params, &turn_id, thread.clone()).await?;

            // ── Adaptive reasoning effort ──────────────────────────────────
            // On every iteration, adjust reasoning_effort based on intent.
            // The user's explicit setting takes priority (via Auto → override).
            let adaptive_effort = intent_to_reasoning_effort(
                turn_intent,
                turn_ctx.effective_model_settings.reasoning_effort,
            );
            if adaptive_effort != turn_ctx.effective_model_settings.reasoning_effort {
                tracing::debug!(
                    turn_id = %turn_id,
                    intent = ?turn_intent,
                    from = turn_ctx.effective_model_settings.reasoning_effort.as_str(),
                    to = adaptive_effort.as_str(),
                    "executor: adaptive reasoning effort override"
                );
                turn_ctx.effective_model_settings.reasoning_effort = adaptive_effort;
            }

            // Use the pre-computed instructions on the first iteration to avoid
            // a redundant async call. Recompute on subsequent iterations so
            // reasoning_effort from turn_ctx is correctly applied.
            let mut system_instructions = if let Some(cached) = cached_system_instructions.take() {
                cached
            } else {
                let already_read = recently_read_paths(&turn_ctx.items, 8);
                self.system_instructions(
                    &turn_ctx.cwd,
                    Some(turn_ctx.effective_model_settings.reasoning_effort.as_str()),
                    repo_map_text.as_deref(),
                    turn_intent,
                    &already_read,
                )
                .await?
            };

            // ── Inject structured progress summary from iteration 2+ ──────
            // Gives the model compact structured state so it doesn't have to
            // infer progress from scattered transcript. Reduces looping.
            if guards.agent_iterations > 1 {
                let changed_paths: Vec<String> =
                    guards.file_changes.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
                system_instructions.push(progress_summary(&ProgressState {
                    phase: current_phase.label(),
                    iteration: guards.agent_iterations,
                    reads_used: guards.total_read_only_rounds,
                    reads_budget: read_budget,
                    plan: &plan_steps,
                    completed_actions: &completed_actions,
                    changed_files: &changed_paths,
                    verified: guards.command_attempted_after_last_file_change,
                }));
            }

            let tools = self
                .runtime
                .inner
                .tool_orchestrator
                .list_available_tools(&turn_ctx.listing);
            let request = turn_ctx.model_request(system_instructions, tools);

            let mut response = self
                .runtime
                .inner
                .model_gateway
                .start_response(request, turn_ctx.cancel_token.clone())
                .await?;

            let mut assistant_item_id: Option<String> = None;
            let mut reasoning_item_id: Option<String> = None;
            let mut assistant_text = String::new();
            let mut reasoning_text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut final_usage = TokenUsage::default();
            let mut estimated_usage = TokenUsage::default();
            // Set to Some(error) when the model returns Failed so we can
            // handle context-overflow recovery outside the inner loop.
            let mut pending_fail: Option<(String, String)> = None;

            // Guard 3 — streaming response length limit.
            // Some local models enter degenerate token loops (repeating the
            // same pattern forever). This cap prevents unbounded memory growth
            // and runaway streaming.

            while let Some(event) = response.recv().await {
                if turn_ctx.cancel_token.is_cancelled() {
                    return self.complete_interrupted(&params.thread_id, &turn_id).await;
                }
                match event {
                    ModelStreamEvent::AssistantDelta(delta) => {
                        if assistant_item_id.is_none() {
                            let item_id = format!("item_{}", Uuid::new_v4().simple());
                            assistant_item_id = Some(item_id.clone());
                            self.runtime.publish_event(
                                RuntimeEventKind::ItemStarted,
                                Some(params.thread_id.clone()), Some(turn_id.clone()),
                                json!({"itemId": item_id, "kind": ItemKind::AgentMessage, "payload": {"text": ""}}),
                            ).await?;
                        }
                        assistant_text.push_str(&delta);

                        // ── Degeneration guards ───────────────────────────────
                        if assistant_text.len() > agent_cfg.max_response_chars {
                            tracing::warn!(
                                turn_id,
                                chars = assistant_text.len(),
                                max_response_chars = agent_cfg.max_response_chars,
                                "streaming guard: response exceeded configured char limit — truncating"
                            );
                            self.runtime.publish_event(
                                RuntimeEventKind::Warning,
                                Some(params.thread_id.clone()), Some(turn_id.clone()),
                                json!({"message": "Model response was too long and has been truncated."}),
                            ).await?;
                            assistant_text.truncate(agent_cfg.max_response_chars);
                            break;
                        }
                        if is_degenerate_repetition(&assistant_text) {
                            tracing::warn!(
                                turn_id,
                                chars = assistant_text.len(),
                                "streaming guard: detected repetitive model output — stopping"
                            );
                            self.runtime.publish_event(
                                RuntimeEventKind::Warning,
                                Some(params.thread_id.clone()), Some(turn_id.clone()),
                                json!({"message": "Model entered a repetitive output loop and has been stopped."}),
                            ).await?;
                            // Trim back to the non-repeating prefix.
                            if let Some(pos) = find_repetition_start(&assistant_text) {
                                assistant_text.truncate(pos);
                            }
                            break;
                        }

                        self.runtime.publish_event(
                            RuntimeEventKind::ItemUpdated,
                            Some(params.thread_id.clone()), Some(turn_id.clone()),
                            json!({"itemId": assistant_item_id, "delta": delta, "mode": "append"}),
                        ).await?;
                    }
                    ModelStreamEvent::ReasoningDelta(delta) => {
                        // Accumulate for persistence (same pattern as assistant_text).
                        reasoning_text.push_str(&delta);
                        if reasoning_item_id.is_none() {
                            reasoning_item_id =
                                Some(format!("reasoning_{}", Uuid::new_v4().simple()));
                        }
                        let reasoning_id = reasoning_item_id
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(|| format!("reasoning_{}", Uuid::new_v4().simple()));
                        // Also publish to the event bus so the TUI live-renders it.
                        self.runtime.publish_event(
                            RuntimeEventKind::ItemUpdated,
                            Some(params.thread_id.clone()), Some(turn_id.clone()),
                            json!({"itemId": reasoning_id, "kind": ItemKind::ReasoningText, "delta": delta, "mode": "append"}),
                        ).await?;
                    }
                    ModelStreamEvent::ToolCalls(calls) => {
                        tool_calls.extend(calls);
                    }
                    ModelStreamEvent::StreamingUsage(usage) => {
                        estimated_usage = usage.clone();
                        // Forward live token usage to the TUI during streaming.
                        let _ = self
                            .runtime
                            .publish_event(
                                crate::system::domain::RuntimeEventKind::TokenUsageUpdate,
                                Some(params.thread_id.clone()),
                                Some(turn_id.clone()),
                                serde_json::json!({
                                    "inputTokens": usage.input_tokens,
                                    "outputTokens": usage.output_tokens,
                                    "cachedTokens": usage.cached_tokens,
                                }),
                            )
                            .await;
                    }
                    ModelStreamEvent::Completed(usage) => {
                        final_usage = usage;
                    }
                    ModelStreamEvent::Failed { display, raw } => {
                        pending_fail = Some((display, raw));
                        break;
                    }
                }
            }
            if turn_ctx.cancel_token.is_cancelled() {
                return self.complete_interrupted(&params.thread_id, &turn_id).await;
            }

            // ── Context-overflow auto-recovery ────────────────────────────────
            // On the first context-overflow error, trim all older turns from
            // the in-memory session (keeps the current turn intact) and retry.
            // A second overflow is not retried — it surfaces as a normal error.
            // Items are saved before trimming so they can be restored if the
            // retry also fails, preventing data loss.
            if let Some((display_error, raw_error)) = pending_fail.as_ref() {
                if !auto_trim_attempted
                    && (is_context_overflow_error(raw_error)
                        || is_context_overflow_error(display_error))
                {
                    auto_trim_attempted = true;

                    tracing::warn!(
                        turn_id = %turn_id,
                        raw_error = %&raw_error[..raw_error.len().min(200)],
                        "executor: context overflow detected — auto-trimming older turns"
                    );

                    // Notify the user via a Warning event (TUI renders this).
                    self.runtime.publish_event(
                        RuntimeEventKind::Warning,
                        Some(params.thread_id.clone()),
                        Some(turn_id.clone()),
                        json!({
                            "message": "Context limit reached — trimming history and retrying automatically…"
                        }),
                    ).await?;

                    // Save items from every completed turn before clearing, so
                    // we can restore them if the retry also fails.
                    {
                        let mut guard = turn_ctx.thread.lock().await;
                        for lt in guard.turns.iter_mut() {
                            if lt.metadata.turn_id != turn_id {
                                let saved = std::mem::take(&mut lt.items);
                                trimmed_turn_items.insert(lt.metadata.turn_id.clone(), saved);
                            }
                        }
                    }

                    continue; // retry the outer agent loop
                }

                // If we auto-trimmed but still failed, restore the saved items
                // so the in-memory session is not left corrupted.
                if !trimmed_turn_items.is_empty() {
                    tracing::warn!(
                        turn_id = %turn_id,
                        "executor: auto-trim retry also failed — restoring trimmed items"
                    );
                    let mut guard = turn_ctx.thread.lock().await;
                    for lt in guard.turns.iter_mut() {
                        if let Some(saved) = trimmed_turn_items.remove(&lt.metadata.turn_id) {
                            lt.items = saved;
                        }
                    }
                }

                // Second overflow or a different error — humanize and fail the turn.
                let err_display = cod_error::from_raw(raw_error);
                return self
                    .fail_turn(
                        &params.thread_id,
                        &turn_id,
                        err_display.kind.label(),
                        raw_error,
                    )
                    .await;
            }

            // Pre-compute whether an intent nudge will fire so we can skip
            // persisting the narration. Persisted narration poisons the context
            // by reinforcing the model's plan-without-acting pattern.
            let will_nudge_intent = tool_calls.is_empty()
                && !assistant_text.is_empty()
                && guards.no_tool_nudges < agent_cfg.max_no_tool_nudges
                && should_retry_no_tool_completion(
                    &assistant_text,
                    &turn_ctx.items,
                    guards.completed_tool_rounds,
                );
            let will_nudge_verification = tool_calls.is_empty()
                && !assistant_text.is_empty()
                && verification_requested
                && !guards.file_changes.is_empty()
                && !guards.command_attempted_after_last_file_change
                && guards.verification_nudges == 0;

            // Persist reasoning/thinking text if the model produced any.
            // ItemKind::ReasoningText is defined in the domain but was previously
            // never written — this wires it up so the full turn is reconstructable.
            if !reasoning_text.is_empty() {
                let item_id = reasoning_item_id
                    .take()
                    .unwrap_or_else(|| format!("reasoning_{}", Uuid::new_v4().simple()));
                let reasoning_item = ConversationItem {
                    item_id,
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    created_at: now_seconds(),
                    kind: ItemKind::ReasoningText,
                    payload: json!({ "text": reasoning_text }),
                };
                self.persist_turn_item(reasoning_item).await?;
                reasoning_text = String::new();
            }

            if let Some(item_id) = assistant_item_id {
                if !will_nudge_intent && !will_nudge_verification {
                    let item = ConversationItem {
                        item_id,
                        thread_id: params.thread_id.clone(),
                        turn_id: turn_id.clone(),
                        created_at: now_seconds(),
                        kind: ItemKind::AgentMessage,
                        payload: json!({ "text": assistant_text }),
                    };
                    self.persist_turn_item(item).await?;
                } else {
                    tracing::debug!(
                        turn_id = %turn_id,
                        "executor: suppressing narration persistence — corrective nudge will fire"
                    );
                }
            }

            if tool_calls.is_empty() {
                tracing::debug!(
                    turn_id = %turn_id,
                    guards.agent_iterations,
                    guards.completed_tool_rounds,
                    guards.no_tool_nudges,
                    guards.consecutive_empty_responses,
                    assistant_text_len = assistant_text.len(),
                    assistant_text_preview = %assistant_text.chars().take(200).collect::<String>(),
                    "executor: no tool calls in model response"
                );

                // ── Empty-response guard (Guard 4) ────────────────────────────
                // A completely blank response (no text AND no tools) is a
                // different failure mode from intent-narration-without-a-call.
                // It means the model is confused or context-saturated — not
                // that the task is done. Retry once with an explicit recovery
                // prompt; on the second consecutive blank, fail the turn.
                if assistant_text.is_empty() {
                    guards.consecutive_empty_responses += 1;
                    guards.total_nudges += 1;
                    tracing::warn!(
                        turn_id = %turn_id,
                        guards.consecutive_empty_responses,
                        max_empty_responses = agent_cfg.max_empty_responses,
                        "executor: model returned empty response (no text, no tool calls)"
                    );
                    if let Some(GuardVerdict::FailTurn { kind, message }) =
                        guards.check_empty_responses(&agent_cfg)
                    {
                        return self.fail_turn(&params.thread_id, &turn_id, &kind, &message).await;
                    }
                    no_tool_nudge_instruction = Some(
                        "Your previous response was completely empty — no text and no tool call \
                         was emitted. If the task is not yet complete, you must emit at least \
                         one tool call now. If the task is complete, provide a brief summary \
                         of what was accomplished."
                            .into(),
                    );
                    self.runtime
                        .publish_event(
                            RuntimeEventKind::Warning,
                            Some(params.thread_id.clone()),
                            Some(turn_id.clone()),
                            json!({
                                "message": "Model returned an empty response — retrying with an explicit recovery prompt."
                            }),
                        )
                        .await?;
                    continue;
                }

                // Model produced text — not a blank response, reset the counter.
                guards.consecutive_empty_responses = 0;

                // ── Intent-narration guard ────────────────────────────────────
                if guards.no_tool_nudges < agent_cfg.max_no_tool_nudges
                    && should_retry_no_tool_completion(
                        &assistant_text,
                        &turn_ctx.items,
                        guards.completed_tool_rounds,
                    )
                {
                    // ── Plan extraction: if the model narrated a plan,
                    // capture it even though we're nudging for a tool call.
                    if plan_steps.is_empty() {
                        let extracted = extract_plan_from_response(&assistant_text);
                        if !extracted.is_empty() {
                            tracing::info!(
                                turn_id = %turn_id,
                                steps = extracted.len(),
                                "executor: extracted plan from narration"
                            );
                            plan_steps = extracted;
                            if current_phase == TurnPhase::Orient
                                || current_phase == TurnPhase::Plan
                            {
                                current_phase = TurnPhase::Execute;
                            }
                        }
                    }
                    guards.no_tool_nudges += 1;
                    guards.total_nudges += 1;
                    // Escalate to constrained mode on the 2nd nudge to make
                    // looping structurally harder, not just recoverable.
                    if guards.no_tool_nudges >= 2 && guards.constrained_mode.is_none() {
                        guards.constrained_mode = Some(ConstrainedMode::FinalOrActionOnly);
                        tracing::warn!(
                            turn_id = %turn_id,
                            guards.no_tool_nudges,
                            "executor: escalating to constrained mode after repeated narration"
                        );
                    }
                    if let Some(GuardVerdict::FailTurn { kind, message }) =
                        guards.check_nudge_budget(&agent_cfg)
                    {
                        tracing::error!(
                            turn_id = %turn_id,
                            guards.total_nudges,
                            "executor: cumulative nudge limit reached — terminating turn"
                        );
                        return self.fail_turn(&params.thread_id, &turn_id, &kind, &message).await;
                    }
                    tracing::warn!(
                        turn_id = %turn_id,
                        guards.no_tool_nudges,
                        max_no_tool_nudges = agent_cfg.max_no_tool_nudges,
                        "executor: nudging model to emit tool call (described intent but no call)"
                    );
                    let recent_files = recently_read_paths(&turn_ctx.items, 3);
                    let file_hint = if recent_files.is_empty() {
                        String::new()
                    } else {
                        let paths = recent_files.join(", ");
                        format!(
                            " You recently read: {paths}. \
                             Use `patch_file` with start_line, end_line, and content to \
                             edit the lines you need to change."
                        )
                    };
                    no_tool_nudge_instruction = Some(format!(
                        "You described a plan but emitted no tool call — nothing ran. \
                         Do not narrate. Emit exactly one tool call now.{file_hint} \
                         Retry {}/{}.",
                        guards.no_tool_nudges, agent_cfg.max_no_tool_nudges
                    ));
                    self.runtime
                        .publish_event(
                            RuntimeEventKind::Warning,
                            Some(params.thread_id.clone()),
                            Some(turn_id.clone()),
                            json!({
                                "message": "Model described an action but emitted no tool call — retrying with a tool-use nudge."
                            }),
                        )
                        .await?;
                    continue;
                }

                // ── Requested-verification guard ─────────────────────────────
                // If the user explicitly asked to test/verify, do not let the
                // agent finish immediately after editing. Require at least one
                // command attempt after the last file change, then let the model
                // report pass/fail normally.
                if verification_requested
                    && !guards.file_changes.is_empty()
                    && !guards.command_attempted_after_last_file_change
                    && guards.verification_nudges == 0
                {
                    guards.verification_nudges += 1;
                    guards.total_nudges += 1;
                    if let Some(GuardVerdict::FailTurn { kind, message }) =
                        guards.check_nudge_budget(&agent_cfg)
                    {
                        tracing::error!(
                            turn_id = %turn_id,
                            guards.total_nudges,
                            "executor: cumulative nudge limit reached — terminating turn"
                        );
                        return self.fail_turn(&params.thread_id, &turn_id, &kind, &message).await;
                    }
                    tracing::warn!(
                        turn_id = %turn_id,
                        "executor: nudging model to verify file changes"
                    );
                    no_tool_nudge_instruction = Some(
                        "You changed files, and the user explicitly asked to test or verify the \
                         result. Do not finish yet. Emit exactly one `bash_exec` command now to \
                         run the narrowest relevant test, build, or check command. If no direct \
                         test exists, run the closest available validation command."
                            .into(),
                    );
                    self.runtime
                        .publish_event(
                            RuntimeEventKind::Warning,
                            Some(params.thread_id.clone()),
                            Some(turn_id.clone()),
                            json!({
                                "message": "Agent changed files but has not run verification — retrying with a test/check nudge."
                            }),
                        )
                        .await?;
                    continue;
                }
                tracing::info!(
                    turn_id = %turn_id,
                    guards.agent_iterations,
                    guards.completed_tool_rounds,
                    "executor: turn completing — no tool calls, no nudge required"
                );
                let metrics = TurnMetrics {
                    agent_iterations: guards.agent_iterations,
                    tool_call_count: guards.total_tool_calls,
                    elapsed_ms: turn_start_time.elapsed().as_millis() as u64,
                    file_changes: guards.file_changes.clone(),
                };
                return self
                    .complete_turn(
                        &params.thread_id,
                        &turn_id,
                        final_usage,
                        estimated_usage,
                        metrics,
                    )
                    .await;
            }
            guards.no_tool_nudges = 0;
            guards.consecutive_empty_responses = 0;

            // Snapshot whether this round is pure read-only *before* tool_calls is moved.
            let round_is_read_only = tool_calls.iter().all(|c| is_read_only_tool(&c.tool_name));
            let round_has_command = tool_calls
                .iter()
                .any(|c| matches!(c.tool_name.as_str(), "bash_exec" | "shell_exec"));
            let round_call_count = tool_calls.len();

            let (
                had_any_success,
                round_file_changes,
                invalid_tool_calls,
                blocked_read_only_calls,
                max_repeat_count,
                non_retryable_stop_reason,
            ) = tokio::select! {
                _ = turn_ctx.cancel_token.cancelled() => {
                    return self.complete_interrupted(&params.thread_id, &turn_id).await;
                }
                result = self.execute_tool_round(
                    &turn_ctx,
                    tool_calls,
                    &mut guards.seen_read_signatures,
                    &mut guards.cross_round_counts,
                    guards.constrained_mode.is_some(),
                ) => result?,
            };
            guards.record_tool_round(had_any_success, round_call_count);
            if !round_file_changes.is_empty() {
                guards.command_attempted_after_last_file_change = false;
                // Track completed actions for progress summary
                for fc in &round_file_changes {
                    completed_actions.push(format!("edited {}", fc.path));
                }
            }
            if round_has_command {
                guards.command_attempted_after_last_file_change = true;
                completed_actions.push("ran command".into());
            }
            let round_file_changes_count = round_file_changes.len();
            guards.file_changes.extend(round_file_changes.iter().cloned());
            // ── Middleware: post-tool hooks ──────────────────────────────────
            // Run all registered middlewares after tool dispatch. Currently
            // this fires the checkpoint review; future middlewares hook in here.
            {
                use super::middleware::{MiddlewareAction, TurnMiddlewareContext};
                let mut mw_ctx = TurnMiddlewareContext {
                    runtime: Some(self.runtime.clone()),
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    agent_config: agent_cfg.clone(),
                    agent_depth: params.agent_depth,
                    cwd: cwd_for_persist.to_string(),
                    user_task_text: user_task_text_for_review.clone(),
                    system_instructions: Vec::new(),
                    file_changes: guards.file_changes.clone(),
                    plan_steps: plan_steps.clone(),
                    completed_actions: completed_actions.clone(),
                    agent_iterations: guards.agent_iterations,
                    completed_tool_rounds: guards.completed_tool_rounds,
                };
                // round_file_changes was consumed by extend above, so pass the
                // tail of guards.file_changes that corresponds to this round.
                let round_start = guards.file_changes.len().saturating_sub(round_file_changes_count);
                let round_changes_slice = &guards.file_changes[round_start..];
                match middleware_chain
                    .run_post_tools(&mut mw_ctx, &[], round_changes_slice)
                    .await?
                {
                    MiddlewareAction::Continue => {}
                    MiddlewareAction::InjectInstruction(instruction) => {
                        no_tool_nudge_instruction = Some(instruction);
                    }
                    MiddlewareAction::FailTurn { kind, message } => {
                        return self
                            .fail_turn(&params.thread_id, &turn_id, &kind, &message)
                            .await;
                    }
                    MiddlewareAction::CompleteTurn => {
                        tracing::info!(
                            turn_id = %turn_id,
                            "executor: middleware requested early turn completion"
                        );
                        let metrics = TurnMetrics {
                            agent_iterations: guards.agent_iterations,
                            tool_call_count: guards.total_tool_calls,
                            elapsed_ms: turn_start_time.elapsed().as_millis() as u64,
                            file_changes: guards.file_changes.clone(),
                        };
                        return self
                            .complete_turn(
                                &params.thread_id,
                                &turn_id,
                                TokenUsage::default(),
                                TokenUsage::default(),
                                metrics,
                            )
                            .await;
                    }
                }
            }

            if let Some(stop_reason) = non_retryable_stop_reason {
                tracing::warn!(
                    turn_id = %turn_id,
                    reason = %stop_reason,
                    "executor: non-retryable tool failure detected — terminating turn"
                );
                return self
                    .fail_turn(
                        &params.thread_id,
                        &turn_id,
                        "non_retryable_tool_failure",
                        &stop_reason,
                    )
                    .await;
            }

            // ── Phase transitions after tool execution ────────────────────
            if current_phase == TurnPhase::Orient && !round_is_read_only {
                // First action tool → transition to Execute (skip Plan)
                current_phase = TurnPhase::Execute;
                tracing::debug!(
                    turn_id = %turn_id,
                    "executor: phase Orient → Execute (first action tool)"
                );
            } else if current_phase == TurnPhase::Plan && !round_is_read_only {
                current_phase = TurnPhase::Execute;
                tracing::debug!(
                    turn_id = %turn_id,
                    "executor: phase Plan → Execute"
                );
            } else if current_phase == TurnPhase::Execute
                && !guards.file_changes.is_empty()
                && guards.command_attempted_after_last_file_change
            {
                current_phase = TurnPhase::Verify;
                tracing::debug!(
                    turn_id = %turn_id,
                    "executor: phase Execute → Verify (files changed + command ran)"
                );
            }
            guards.total_invalid_tool_calls += invalid_tool_calls;

            if invalid_tool_calls > 0 {
                tracing::warn!(
                    turn_id = %turn_id,
                    invalid_tool_calls,
                    guards.total_invalid_tool_calls,
                    "executor: invalid tool calls detected this round"
                );
                if let Some(GuardVerdict::FailTurn { kind, message }) = guards.check_invalid_tool_calls() {
                    return self.fail_turn(&params.thread_id, &turn_id, &kind, &message).await;
                }
            }
            if blocked_read_only_calls > 0 {
                guards.constrained_mode_violations += 1;
                tracing::warn!(
                    turn_id = %turn_id,
                    blocked_read_only_calls,
                    guards.constrained_mode_violations,
                    "executor: constrained mode violation (read-only calls while blocked)"
                );
                if guards.constrained_mode_violations >= 2 {
                    return self
                        .fail_turn(
                            &params.thread_id,
                            &turn_id,
                            "constrained_mode_violation",
                            "The model kept issuing read-only tools after being instructed to act or provide a final answer.",
                        )
                        .await;
                }
                no_tool_nudge_instruction = Some(
                    "Read-only tools are now blocked. Your next response must either call `patch_file`, call `bash_exec`, or provide the final answer."
                        .into(),
                );
                continue;
            }

            // ── Cross-round repetition guard ──────────────────────────────────
            // If the model has emitted the same tool call (any tool, any args)
            // REPEAT_THRESHOLD times across this turn, it's stuck. Switch to
            // constrained mode and nudge once. Only fires once per turn.
            if !guards.repetition_handled
                && max_repeat_count >= REPEAT_THRESHOLD
                && guards.constrained_mode.is_none()
            {
                guards.repetition_handled = true;
                guards.total_nudges += 1;
                guards.constrained_mode = Some(ConstrainedMode::FinalOrActionOnly);
                tracing::warn!(
                    turn_id = %turn_id,
                    max_repeat_count,
                    "executor: cross-round repetition detected — entering constrained mode"
                );
                self.runtime
                    .publish_event(
                        RuntimeEventKind::Warning,
                        Some(params.thread_id.clone()),
                        Some(turn_id.clone()),
                        json!({
                            "message": format!(
                                "Same tool call repeated {max_repeat_count} times — nudging the agent to take a different approach."
                            )
                        }),
                    )
                    .await?;
                no_tool_nudge_instruction = Some(
                    "You have emitted the same tool call repeatedly. The result will not change. \
                     Either take a different action (`patch_file` or `bash_exec`), or provide \
                     the final answer now. Read-only tools are blocked."
                        .into(),
                );
                continue;
            }

            // ── Read-only exploration guard ───────────────────────────────────
            // Reset when any action tool ran this round; bump when every call
            // was read-only. On hitting the threshold inject a single nudge and
            // reset so a follow-up nudge fires again if the model ignores it.
            if guards.constrained_mode.is_none() && round_is_read_only {
                guards.consecutive_read_only_rounds += 1;
                guards.total_read_only_rounds += 1;
                // Use intent-aware budget instead of flat config value.
                // Effective threshold still shrinks with each nudge.
                let effective_read_only_limit = (read_budget >> guards.total_nudges).max(1);
                tracing::debug!(
                    turn_id = %turn_id,
                    guards.consecutive_read_only_rounds,
                    effective_read_only_limit,
                    "executor: read-only round ({}/{})", guards.consecutive_read_only_rounds, effective_read_only_limit
                );
                if guards.consecutive_read_only_rounds >= effective_read_only_limit {
                    guards.consecutive_read_only_rounds = 0;
                    guards.total_nudges += 1;
                    guards.constrained_mode = Some(ConstrainedMode::FinalOrActionOnly);
                    // Phase transition: Orient → Plan for edit/debug tasks
                    if needs_planning && current_phase == TurnPhase::Orient {
                        current_phase = TurnPhase::Plan;
                        tracing::info!(
                            turn_id = %turn_id,
                            "executor: phase Orient → Plan (read budget exhausted)"
                        );
                    }
                    let recent_files = recently_read_paths(&turn_ctx.items, 3);
                    let file_hint = if recent_files.is_empty() {
                        String::new()
                    } else {
                        let paths = recent_files.join(", ");
                        format!(
                            " You recently read: {paths}. \
                             If you need to edit any of these files, use `patch_file` with \
                             start_line/end_line/content to surgically replace the lines you \
                             want to change. This is much easier than rewriting the entire file."
                        )
                    };
                    let msg = match turn_intent {
                        TurnIntent::Answer | TurnIntent::Review | TurnIntent::Inventory => format!(
                            "You have enough context. Your next response must provide the final answer. \
                             Do not call `read_file`, `grep_search`, or `list_dir` again.{file_hint}"
                        ),
                        _ => format!(
                            "You have enough context. Your next response must either call `patch_file`, \
                             call `bash_exec`, or provide the final answer. Do not call `read_file`, \
                             `grep_search`, or `list_dir` again.{file_hint}"
                        ),
                    };
                    tracing::warn!(
                        turn_id = %turn_id,
                        "executor: read-only saturation nudge fired"
                    );
                    self.runtime
                        .publish_event(
                            RuntimeEventKind::Warning,
                            Some(params.thread_id.clone()),
                            Some(turn_id.clone()),
                            json!({
                                "message": format!(
                                    "Agent has been exploring for {effective_read_only_limit} rounds \
                                     — nudging it to take an action."
                                )
                            }),
                        )
                        .await?;
                    no_tool_nudge_instruction = Some(msg);
                }
            } else if guards.constrained_mode.is_none() {
                if guards.consecutive_read_only_rounds > 0 {
                    tracing::debug!(
                        turn_id = %turn_id,
                        guards.consecutive_read_only_rounds,
                        "executor: action tool executed — resetting read-only counter"
                    );
                }
                guards.consecutive_read_only_rounds = 0;
            }

            if guards.constrained_mode.is_some() && !round_is_read_only && had_any_success {
                guards.constrained_mode = None;
                guards.constrained_mode_violations = 0;
                tracing::debug!(
                    turn_id = %turn_id,
                    "executor: constrained mode cleared after successful action"
                );
            }

            // ── Consecutive-failure guard ─────────────────────────────────────
            // reset/increment is now handled by record_tool_round above.
            if !had_any_success {
                tracing::warn!(
                    turn_id = %turn_id,
                    guards.consecutive_failures,
                    max_consecutive_failures = agent_cfg.max_consecutive_failures,
                    guards.completed_tool_rounds,
                    "executor: all tools in this round FAILED"
                );
            }
            if let Some(GuardVerdict::FailTurn { kind, message }) =
                guards.check_consecutive_failures(&agent_cfg)
            {
                tracing::error!(
                    turn_id = %turn_id,
                    guards.consecutive_failures,
                    "executor: consecutive-failure limit reached — terminating turn"
                );
                return self.fail_turn(&params.thread_id, &turn_id, &kind, &message).await;
            }
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    async fn append_user_input(
        &self,
        thread_id: &str,
        turn_id: &str,
        input: &UserInput,
    ) -> Result<()> {
        let text = input.text.as_ref().map(|t| t.text.clone());
        let image_paths: Vec<String> = input.images.iter().map(|i| i.path.clone()).collect();

        if text.is_none() && image_paths.is_empty() {
            return Ok(());
        }

        // Persist text + (zero or more) images as a single UserMessage so they
        // become one multimodal LLM message. If there is no text, persist as a
        // UserAttachment for back-compat with surfaces that filter on item kind.
        let item = if text.is_none() && !image_paths.is_empty() {
            ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
                created_at: now_seconds(),
                kind: ItemKind::UserAttachment,
                payload: json!({ "imagePaths": image_paths }),
            }
        } else {
            let mut payload = json!({ "text": text.clone().unwrap_or_default() });
            if !image_paths.is_empty() {
                payload["imagePaths"] = json!(image_paths);
            }
            ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
                created_at: now_seconds(),
                kind: ItemKind::UserMessage,
                payload,
            }
        };
        self.persist_turn_item(item).await?;

        // ── Auto-title: set thread title from first user message ──────────
        if let Some(text_str) = text.as_deref() {
            if !text_str.is_empty() {
                if let Some(thread) = self.runtime.load_thread(thread_id).await? {
                    let mut guard = thread.lock().await;
                    if guard.metadata.title.is_none() {
                        let title = derive_thread_title(text_str);
                        guard.metadata.title = Some(title);
                        guard.metadata.updated_at = now_seconds();
                        let _ = self
                            .runtime
                            .inner
                            .persistence_manager
                            .update_thread(&guard.metadata);
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn persist_turn_item(&self, item: ConversationItem) -> Result<()> {
        self.runtime.inner.persistence_manager.append_item(&item)?;
        if let Some(thread) = self.runtime.load_thread(&item.thread_id).await? {
            let mut thread = thread.lock().await;
            if let Some(turn) = thread
                .turns
                .iter_mut()
                .find(|t| t.metadata.turn_id == item.turn_id)
            {
                turn.items.push(item.clone());
            }
        }
        self.runtime
            .publish_event(
                crate::system::domain::RuntimeEventKind::ItemCompleted,
                Some(item.thread_id.clone()),
                Some(item.turn_id.clone()),
                serde_json::to_value(&item)?,
            )
            .await?;
        Ok(())
    }

    async fn persist_tool_result(
        &self,
        thread_id: &str,
        turn_id: &str,
        result: &ToolResult,
    ) -> Result<()> {
        let item = ConversationItem {
            item_id: format!("item_{}", Uuid::new_v4().simple()),
            thread_id: thread_id.into(),
            turn_id: turn_id.into(),
            created_at: now_seconds(),
            kind: ItemKind::ToolResult,
            payload: serde_json::to_value(result)?,
        };
        self.persist_turn_item(item).await
    }

    pub async fn complete_turn(
        &self,
        thread_id: &str,
        turn_id: &str,
        actual_usage: TokenUsage,
        estimated_usage: TokenUsage,
        metrics: TurnMetrics,
    ) -> Result<()> {
        let thread = self
            .runtime
            .load_thread(thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {thread_id}"))?;
        let metadata = {
            let mut thread = thread.lock().await;
            let metadata = {
                let turn = thread
                    .turns
                    .iter_mut()
                    .find(|t| t.metadata.turn_id == turn_id)
                    .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                turn.status = TurnStatus::Completed;
                turn.metadata.status = TurnStatus::Completed;
                turn.metadata.updated_at = now_seconds();
                turn.metadata.token_usage = actual_usage;
                turn.metadata.estimated_token_usage = estimated_usage;
                self.runtime
                    .inner
                    .persistence_manager
                    .update_turn(&turn.metadata)?;
                turn.metadata.clone()
            };
            thread.active_turn_id = None;
            thread.metadata.status = ThreadStatus::Idle;
            thread.metadata.updated_at = now_seconds();
            self.runtime
                .inner
                .persistence_manager
                .update_thread(&thread.metadata)?;
            metadata
        };
        self.runtime
            .publish_event(
                crate::system::domain::RuntimeEventKind::TurnCompleted,
                Some(thread_id.into()),
                Some(turn_id.into()),
                json!({
                    "turnId": metadata.turn_id,
                    "threadId": metadata.thread_id,
                    "status": metadata.status,
                    "estimatedTokenUsage": metadata.estimated_token_usage,
                    "tokenUsage": metadata.token_usage,
                    "metrics": metrics,
                }),
            )
            .await?;

        self.maybe_auto_compact(thread_id).await;

        Ok(())
    }

    /// Check context usage and trigger compaction in the background if over threshold.
    async fn maybe_auto_compact(&self, thread_id: &str) {
        let cfg = self.runtime.inner.effective_config.auto_compaction.clone();
        if !cfg.enabled {
            return;
        }

        // Estimate current token usage from persisted items.
        let persisted = match self
            .runtime
            .inner
            .persistence_manager
            .read_thread(thread_id)
        {
            Ok(p) => p,
            Err(_) => return,
        };

        // Per-model threshold, falling back to the global default.
        let model_id = persisted.metadata.model_id.clone();
        let threshold = cfg
            .model_thresholds
            .get(&model_id)
            .copied()
            .unwrap_or(cfg.threshold_pct);

        let context_window = self
            .runtime
            .inner
            .effective_config
            .models
            .iter()
            .find(|m| m.model_id == model_id)
            .and_then(|m| m.context_window)
            .or(self
                .runtime
                .inner
                .effective_config
                .model_settings
                .context_window);
        let prompt_budget = super::model_gateway::calculate_prompt_budget(context_window);

        let used_pct = estimate_items_token_pct(&persisted.items, prompt_budget);
        tracing::debug!(
            thread_id,
            used_pct = format!("{used_pct:.1}"),
            threshold,
            items_count = persisted.items.len(),
            "maybe_auto_compact: context usage check"
        );
        if used_pct < threshold as f64 {
            return;
        }

        tracing::info!(
            thread_id,
            used_pct = format!("{used_pct:.1}"),
            threshold,
            "auto-compaction triggered"
        );

        let runtime = self.runtime.clone();
        let thread_id = thread_id.to_string();
        tokio::spawn(async move {
            let _ = runtime
                .publish_event(
                    crate::system::domain::RuntimeEventKind::CompactionStatus,
                    Some(thread_id.clone()),
                    None,
                    json!({ "status": "started", "message": "Auto-compacting context…" }),
                )
                .await;

            let result = runtime
                .compact_thread(crate::system::runtime::ThreadCompactParams {
                    thread_id: thread_id.clone(),
                    strategy: crate::system::domain::CompactionStrategy::SummarizePrefix,
                })
                .await;

            match result {
                Ok(r) => {
                    let mut summary_chars: Option<usize> = None;
                    let mut summary_tokens_est: Option<usize> = None;
                    if let Some(summary_item_id) = r.summary_item_id.as_ref() {
                        if let Ok(persisted) =
                            runtime.inner.persistence_manager.read_thread(&r.thread_id)
                        {
                            if let Some(item) = persisted
                                .items
                                .iter()
                                .find(|i| &i.item_id == summary_item_id)
                            {
                                if let Some(text) =
                                    item.payload.get(KEY_TEXT).and_then(|v| v.as_str())
                                {
                                    let chars = text.chars().count();
                                    summary_chars = Some(chars);
                                    summary_tokens_est =
                                        Some(super::model_gateway::estimate_text_tokens(text));
                                }
                            }
                        }
                    }
                    tracing::info!(
                        thread_id = %r.thread_id,
                        strategy = "SUMMARIZE_PREFIX",
                        items_removed = r.items_removed,
                        summary_item_id = ?r.summary_item_id,
                        summary_chars = summary_chars.unwrap_or(0),
                        summary_tokens_est = summary_tokens_est.unwrap_or(0),
                        "auto-compaction completed"
                    );
                    let _ = runtime
                        .publish_event(
                            crate::system::domain::RuntimeEventKind::CompactionStatus,
                            Some(thread_id),
                            None,
                            json!({
                                "status": "completed",
                                "message": format!("✓ Auto-compacted ({} items summarised)", r.items_removed),
                                "items_removed": r.items_removed,
                            }),
                        )
                        .await;
                }
                Err(e) => {
                    tracing::warn!("auto_compaction_failed: {e}");
                }
            }
        });
    }

    pub async fn fail_turn(
        &self,
        thread_id: &str,
        turn_id: &str,
        kind_label: &str,
        message: &str,
    ) -> Result<()> {
        let raw_message = message.to_string();
        let normalized_message = cod_error::from_raw(message).message;
        let api_request_id = cod_error::extract_api_request_id(&raw_message);
        let _ = self.runtime.inner.persistence_manager.append_log(
            "error",
            &format!(
                "turn_failed thread_id={thread_id} turn_id={turn_id} kind={kind_label} message={normalized_message} raw={raw_message} api_request_id={}",
                api_request_id.clone().unwrap_or_else(|| "<none>".into())
            ),
        );

        // Persist a visible Error item so the failure reason appears in the
        // transcript and survives thread reloads.
        let mut error_payload = json!({
            "kind": kind_label,
            "message": normalized_message,
            "rawMessage": raw_message,
        });
        if let Some(request_id) = &api_request_id {
            error_payload["apiRequestId"] = json!(request_id);
        }
        let error_item = ConversationItem {
            item_id: format!("item_{}", Uuid::new_v4().simple()),
            thread_id: thread_id.into(),
            turn_id: turn_id.into(),
            created_at: now_seconds(),
            kind: ItemKind::Error,
            payload: error_payload,
        };
        self.persist_turn_item(error_item).await?;

        let thread = self
            .runtime
            .load_thread(thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {thread_id}"))?;
        {
            let mut thread = thread.lock().await;
            {
                let turn = thread
                    .turns
                    .iter_mut()
                    .find(|t| t.metadata.turn_id == turn_id)
                    .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                turn.status = TurnStatus::Failed;
                turn.metadata.status = TurnStatus::Failed;
                turn.metadata.updated_at = now_seconds();
                self.runtime
                    .inner
                    .persistence_manager
                    .update_turn(&turn.metadata)?;
            }
            thread.active_turn_id = None;
            thread.metadata.status = ThreadStatus::Idle;
            self.runtime
                .inner
                .persistence_manager
                .update_thread(&thread.metadata)?;
        }
        // Evict the hot-cache entry so the next turn cold-reloads from
        // persistence. This ensures prefix_items (e.g. compaction summaries)
        // are correctly reconstructed even when a prior turn failed mid-stream
        // without triggering a normal cache eviction.
        self.runtime
            .inner
            .loaded_threads
            .write()
            .await
            .remove(thread_id);

        let mut event_payload = json!({
            "kind": kind_label,
            "reason": normalized_message,
            "rawReason": raw_message,
        });
        if let Some(request_id) = api_request_id {
            event_payload["apiRequestId"] = json!(request_id);
        }

        self.runtime
            .publish_event(
                crate::system::domain::RuntimeEventKind::TurnFailed,
                Some(thread_id.into()),
                Some(turn_id.into()),
                event_payload,
            )
            .await?;
        Ok(())
    }

    async fn complete_interrupted(&self, thread_id: &str, turn_id: &str) -> Result<()> {
        let thread = self
            .runtime
            .load_thread(thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {thread_id}"))?;
        let metadata = {
            let mut thread = thread.lock().await;
            let metadata = {
                let turn = thread
                    .turns
                    .iter_mut()
                    .find(|t| t.metadata.turn_id == turn_id)
                    .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                turn.status = TurnStatus::Interrupted;
                turn.metadata.status = TurnStatus::Interrupted;
                turn.metadata.updated_at = now_seconds();
                self.runtime
                    .inner
                    .persistence_manager
                    .update_turn(&turn.metadata)?;
                turn.metadata.clone()
            };
            thread.active_turn_id = None;
            thread.metadata.status = ThreadStatus::Interrupted;
            self.runtime
                .inner
                .persistence_manager
                .update_thread(&thread.metadata)?;
            metadata
        };
        self.runtime
            .publish_event(
                crate::system::domain::RuntimeEventKind::TurnCompleted,
                Some(thread_id.into()),
                Some(turn_id.into()),
                serde_json::to_value(&metadata)?,
            )
            .await?;
        Ok(())
    }

    async fn drain_steering(&self, thread_id: &str, turn_id: &str) -> Result<()> {
        let thread = self
            .runtime
            .load_thread(thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {thread_id}"))?;
        let steering = {
            let mut t = thread.lock().await;
            std::mem::take(&mut t.pending_steering)
        };
        for input in steering {
            self.append_user_input(thread_id, turn_id, &input).await?;
        }
        Ok(())
    }

    async fn system_instructions(
        &self,
        cwd: &str,
        reasoning_effort: Option<&str>,
        repo_map_text: Option<&str>,
        intent: TurnIntent,
        already_read: &[String],
    ) -> Result<Vec<String>> {
        let mut instructions = vec![self.runtime.inner.effective_config.system_prompt.clone()];
        if let Some(instruction) = thinking_instruction(reasoning_effort) {
            instructions.push(instruction);
        }
        if let Some(directive) = intent_directive(intent) {
            instructions.push(directive);
        }
        if let Some(directive) = already_read_directive(already_read) {
            instructions.push(directive);
        }
        let skills = self.runtime.inner.extension_manager.list_skills(cwd).await;
        for skill in skills {
            if skill.enabled {
                instructions.push(format!("Skill {}: {}", skill.name, skill.description));
            }
        }

        // ── Codebase intelligence — repo map injection ────────────────────────
        // The map is pre-built once per turn in run_turn() and passed in here.
        // This avoids redundant directory walks + file reads on every iteration.
        if let Some(map_text) = repo_map_text {
            instructions.push(format!(
                "The following is an automatically generated structural map of the \
                 repository. Use it to locate files and symbols before resorting to \
                 `read_file` or `list_dir` calls. It is current as of this turn start \
                 and reflects the latest on-disk state.\n\n{map_text}"
            ));
        }

        Ok(instructions)
    }

}
