use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::HashSet;
use uuid::Uuid;

use super::model_gateway::{estimate_items_token_pct, is_context_overflow_error, ModelStreamEvent};
mod context;
mod tool_dispatch;
mod utils;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstrainedMode {
    FinalOrActionOnly,
}

use self::context::TurnContext;
use self::utils::{
    classify_turn_intent, derive_thread_title, find_repetition_start, is_degenerate_repetition,
    is_read_only_tool, recently_read_paths, should_retry_no_tool_completion, thinking_instruction,
    user_requested_verification, TurnIntent,
};
use crate::system::domain::{
    now_seconds, ConversationItem, FileChangeSummary, ItemKind, RuntimeEventKind, ThreadStatus,
    TokenUsage, ToolCall, ToolResult, TurnMetrics, TurnStatus, UserInput,
};
use crate::system::error as cod_error;

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
                    let cfg = intel_cfg.clone();
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

        // Persist the system prompt once per turn for DB-level auditing and
        // post-mortem debugging. System entries are hidden in the TUI but are
        // available for inspection in the raw conversation store.
        // We also cache the result here so the first loop iteration can reuse
        // it rather than calling system_instructions() a second time.
        let initial_system_instructions = self
            .system_instructions(cwd_for_persist, None, repo_map_text.as_deref())
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

        // Tracks whether we have already done one automatic context-overflow
        // trim this turn. We only retry once to avoid an infinite loop.
        let mut auto_trim_attempted = false;

        let agent_cfg = self.runtime.inner.effective_config.agent.clone();
        let turn_intent = classify_turn_intent(&params.input);
        let verification_requested = user_requested_verification(&params.input);

        // ── Loop-break guards ─────────────────────────────────────────────────
        //
        // Guard 1 — consecutive failures: incremented each iteration where
        // every tool call in that round returned ok:false; reset to 0 as soon
        // as any tool succeeds. Catches stuck loops (e.g. bad args that keep
        // failing) without harming legitimate long tasks.
        let mut consecutive_failures: usize = 0;

        // Guard 2 — absolute backstop: a very high ceiling that only fires
        // when the model loops *successfully* without ever finishing. Legitimate
        // agentic tasks rarely need more than ~100 rounds.
        let mut agent_iterations: usize = 0;
        let mut no_tool_nudges: usize = 0;
        let mut no_tool_nudge_instruction: Option<String> = None;
        let mut completed_tool_rounds: usize = 0;
        let mut total_tool_calls: usize = 0;
        let mut file_changes: Vec<FileChangeSummary> = Vec::new();
        let turn_start_time = std::time::Instant::now();

        // Guard 3 — read-only exploration saturation: counts consecutive rounds
        // where *every* tool call was a pure read (read_file, list_dir, grep …).
        // If the model keeps exploring without ever writing or executing, it gets
        // a single nudge telling it to act. Resets to 0 as soon as one action
        // tool (write_file, bash_exec, …) appears in any round.
        // Threshold is deliberately low (4) — the repo map is now injected at
        // turn start, so the model already knows the file tree and top-level
        // symbols.  Excessive reading beyond 4 rounds is a sign of context bloat
        // or confusion, not legitimate exploration.
        let mut consecutive_read_only_rounds: usize = 0;

        // Guard 4 — empty-response circuit breaker: a model response that
        // contains *neither* text nor tool calls is a distinct failure mode from
        // "intent narration without a call". It usually signals context
        // saturation, reasoning-only output that wasn't surfaced, or a confused
        // model state. We retry once with an explicit prompt then fail the turn
        // so the user sees a clear error rather than a silent non-completion.
        let mut consecutive_empty_responses: usize = 0;

        // Guard 5 — cumulative nudge escalation: counts ALL nudges (intent,
        // read-only, empty-response). If the model keeps needing correction
        // it won't self-correct. Fail fast rather than burning tokens.
        let mut total_nudges: usize = 0;
        let mut total_invalid_tool_calls: usize = 0;
        let mut seen_read_signatures: HashSet<String> = HashSet::new();
        let mut constrained_mode: Option<ConstrainedMode> = None;
        let mut constrained_mode_violations: usize = 0;
        let mut verification_nudges: usize = 0;
        let mut command_attempted_after_last_file_change = false;

        loop {
            agent_iterations += 1;
            if agent_iterations > agent_cfg.max_iterations {
                tracing::error!(
                    turn_id = %turn_id,
                    agent_iterations,
                    "executor: backstop limit reached — terminating turn"
                );
                return self
                    .fail_turn(
                        &params.thread_id,
                        &turn_id,
                        "loop_limit",
                        &format!(
                            "Agent exceeded {} tool-call iterations without \
                             finishing. The turn has been stopped to prevent an infinite loop.",
                            agent_cfg.max_iterations
                        ),
                    )
                    .await;
            }
            tracing::debug!(
                turn_id = %turn_id,
                agent_iterations,
                consecutive_failures,
                no_tool_nudges,
                completed_tool_rounds,
                intent = ?turn_intent,
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

            let turn_ctx = TurnContext::build(self, &params, &turn_id, thread.clone()).await?;

            // Use the pre-computed instructions on the first iteration to avoid
            // a redundant async call. Recompute on subsequent iterations so
            // reasoning_effort from turn_ctx is correctly applied.
            let system_instructions = if let Some(cached) = cached_system_instructions.take() {
                cached
            } else {
                self.system_instructions(
                    &turn_ctx.cwd,
                    turn_ctx
                        .effective_model_settings
                        .reasoning_effort
                        .as_deref(),
                    repo_map_text.as_deref(),
                )
                .await?
            };
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
            let mut assistant_text = String::new();
            let mut reasoning_text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut final_usage = TokenUsage::default();
            // Set to Some(error) when the model returns Failed so we can
            // handle context-overflow recovery outside the inner loop.
            let mut pending_fail: Option<String> = None;

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
                        // Also publish to the event bus so the TUI live-renders it.
                        self.runtime.publish_event(
                            RuntimeEventKind::ItemUpdated,
                            Some(params.thread_id.clone()), Some(turn_id.clone()),
                            json!({"itemId": format!("reasoning_{turn_id}"), "delta": delta, "mode": "append"}),
                        ).await?;
                    }
                    ModelStreamEvent::ToolCalls(calls) => {
                        tool_calls.extend(calls);
                    }
                    ModelStreamEvent::Completed(usage) => {
                        final_usage = usage;
                    }
                    ModelStreamEvent::Failed(error) => {
                        pending_fail = Some(error);
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
            if let Some(ref error) = pending_fail {
                if !auto_trim_attempted && is_context_overflow_error(error) {
                    auto_trim_attempted = true;

                    // Notify the user via a Warning event (TUI renders this).
                    self.runtime.publish_event(
                        RuntimeEventKind::Warning,
                        Some(params.thread_id.clone()),
                        Some(turn_id.clone()),
                        json!({
                            "message": "Context limit reached — trimming history and retrying automatically…"
                        }),
                    ).await?;

                    // Drop items from every completed turn so only the current
                    // turn's items remain in memory. Layer 1's token-budget guard
                    // will then fit the history safely on the next iteration.
                    {
                        let mut guard = turn_ctx.thread.lock().await;
                        for lt in guard.turns.iter_mut() {
                            if lt.metadata.turn_id != turn_id {
                                lt.items.clear();
                            }
                        }
                    }

                    continue; // retry the outer agent loop
                }

                // Second overflow or a different error — humanize and fail the turn.
                let err_display = cod_error::from_raw(error);
                return self
                    .fail_turn(
                        &params.thread_id,
                        &turn_id,
                        err_display.kind.label(),
                        &err_display.message,
                    )
                    .await;
            }

            // Pre-compute whether an intent nudge will fire so we can skip
            // persisting the narration. Persisted narration poisons the context
            // by reinforcing the model's plan-without-acting pattern.
            let will_nudge_intent = tool_calls.is_empty()
                && !assistant_text.is_empty()
                && no_tool_nudges < agent_cfg.max_no_tool_nudges
                && should_retry_no_tool_completion(
                    &assistant_text,
                    &turn_ctx.items,
                    completed_tool_rounds,
                );
            let will_nudge_verification = tool_calls.is_empty()
                && !assistant_text.is_empty()
                && verification_requested
                && !file_changes.is_empty()
                && !command_attempted_after_last_file_change
                && verification_nudges == 0;

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

            // Persist reasoning/thinking text if the model produced any.
            // ItemKind::ReasoningText is defined in the domain but was previously
            // never written — this wires it up so the full turn is reconstructable.
            if !reasoning_text.is_empty() {
                let reasoning_item = ConversationItem {
                    item_id: format!("item_{}", Uuid::new_v4().simple()),
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    created_at: now_seconds(),
                    kind: ItemKind::ReasoningText,
                    payload: json!({ "text": reasoning_text }),
                };
                self.persist_turn_item(reasoning_item).await?;
                reasoning_text = String::new();
            }

            if tool_calls.is_empty() {
                tracing::debug!(
                    turn_id = %turn_id,
                    agent_iterations,
                    completed_tool_rounds,
                    no_tool_nudges,
                    consecutive_empty_responses,
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
                    consecutive_empty_responses += 1;
                    total_nudges += 1;
                    tracing::warn!(
                        turn_id = %turn_id,
                        consecutive_empty_responses,
                        max_empty_responses = agent_cfg.max_empty_responses,
                        "executor: model returned empty response (no text, no tool calls)"
                    );
                    if consecutive_empty_responses >= agent_cfg.max_empty_responses {
                        return self
                            .fail_turn(
                                &params.thread_id,
                                &turn_id,
                                "empty_response",
                                &format!(
                                    "The model returned {consecutive_empty_responses} consecutive \
                                     empty responses (no text and no tool calls). The turn has \
                                     been stopped to prevent a silent incomplete result."
                                ),
                            )
                            .await;
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
                consecutive_empty_responses = 0;

                // ── Intent-narration guard ────────────────────────────────────
                if no_tool_nudges < agent_cfg.max_no_tool_nudges
                    && should_retry_no_tool_completion(
                        &assistant_text,
                        &turn_ctx.items,
                        completed_tool_rounds,
                    )
                {
                    no_tool_nudges += 1;
                    total_nudges += 1;
                    if total_nudges >= agent_cfg.max_total_nudges {
                        tracing::error!(
                            turn_id = %turn_id,
                            total_nudges,
                            "executor: cumulative nudge limit reached — terminating turn"
                        );
                        return self
                            .fail_turn(
                                &params.thread_id,
                                &turn_id,
                                "nudge_limit",
                                &format!(
                                    "Agent required {total_nudges} corrections without making \
                                     progress. The turn has been stopped."
                                ),
                            )
                            .await;
                    }
                    tracing::warn!(
                        turn_id = %turn_id,
                        no_tool_nudges,
                        max_no_tool_nudges = agent_cfg.max_no_tool_nudges,
                        "executor: nudging model to emit tool call (described intent but no call)"
                    );
                    let recent_files = recently_read_paths(&turn_ctx.items);
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
                        no_tool_nudges, agent_cfg.max_no_tool_nudges
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
                    && !file_changes.is_empty()
                    && !command_attempted_after_last_file_change
                    && verification_nudges == 0
                {
                    verification_nudges += 1;
                    total_nudges += 1;
                    if total_nudges >= agent_cfg.max_total_nudges {
                        tracing::error!(
                            turn_id = %turn_id,
                            total_nudges,
                            "executor: cumulative nudge limit reached — terminating turn"
                        );
                        return self
                            .fail_turn(
                                &params.thread_id,
                                &turn_id,
                                "nudge_limit",
                                &format!(
                                    "Agent required {total_nudges} corrections without making \
                                     progress. The turn has been stopped."
                                ),
                            )
                            .await;
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
                    agent_iterations,
                    completed_tool_rounds,
                    "executor: turn completing — no tool calls, no nudge required"
                );
                let metrics = TurnMetrics {
                    agent_iterations,
                    tool_call_count: total_tool_calls,
                    elapsed_ms: turn_start_time.elapsed().as_millis() as u64,
                    file_changes: file_changes.clone(),
                };
                return self
                    .complete_turn(&params.thread_id, &turn_id, final_usage, metrics)
                    .await;
            }
            no_tool_nudges = 0;
            consecutive_empty_responses = 0;

            // Snapshot whether this round is pure read-only *before* tool_calls is moved.
            let round_is_read_only = tool_calls.iter().all(|c| is_read_only_tool(&c.tool_name));
            let round_has_command = tool_calls
                .iter()
                .any(|c| matches!(c.tool_name.as_str(), "bash_exec" | "shell_exec"));
            let round_call_count = tool_calls.len();

            let (had_any_success, round_file_changes, invalid_tool_calls, blocked_read_only_calls) = tokio::select! {
                _ = turn_ctx.cancel_token.cancelled() => {
                    return self.complete_interrupted(&params.thread_id, &turn_id).await;
                }
                result = self.execute_tool_round(
                    &turn_ctx,
                    tool_calls,
                    &mut seen_read_signatures,
                    constrained_mode.is_some(),
                ) => result?,
            };
            completed_tool_rounds += 1;
            total_tool_calls += round_call_count;
            if !round_file_changes.is_empty() {
                command_attempted_after_last_file_change = false;
            }
            if round_has_command {
                command_attempted_after_last_file_change = true;
            }
            file_changes.extend(round_file_changes);
            total_invalid_tool_calls += invalid_tool_calls;

            if invalid_tool_calls > 0 {
                tracing::warn!(
                    turn_id = %turn_id,
                    invalid_tool_calls,
                    total_invalid_tool_calls,
                    "executor: invalid tool calls detected this round"
                );
                if total_invalid_tool_calls >= 2 {
                    return self
                        .fail_turn(
                            &params.thread_id,
                            &turn_id,
                            "invalid_tool_limit",
                            "The model repeatedly emitted invalid tool arguments. The turn has been stopped to prevent a failing loop.",
                        )
                        .await;
                }
            }
            if blocked_read_only_calls > 0 {
                constrained_mode_violations += 1;
                tracing::warn!(
                    turn_id = %turn_id,
                    blocked_read_only_calls,
                    constrained_mode_violations,
                    "executor: constrained mode violation (read-only calls while blocked)"
                );
                if constrained_mode_violations >= 2 {
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

            // ── Read-only exploration guard ───────────────────────────────────
            // Reset when any action tool ran this round; bump when every call
            // was read-only. On hitting the threshold inject a single nudge and
            // reset so a follow-up nudge fires again if the model ignores it.
            if constrained_mode.is_none() && round_is_read_only {
                consecutive_read_only_rounds += 1;
                // Effective threshold shrinks with each nudge.
                let effective_read_only_limit =
                    (agent_cfg.max_consecutive_read_only_rounds >> total_nudges).max(1);
                tracing::debug!(
                    turn_id = %turn_id,
                    consecutive_read_only_rounds,
                    effective_read_only_limit,
                    "executor: read-only round ({consecutive_read_only_rounds}/{effective_read_only_limit})"
                );
                if consecutive_read_only_rounds >= effective_read_only_limit {
                    consecutive_read_only_rounds = 0;
                    total_nudges += 1;
                    constrained_mode = Some(ConstrainedMode::FinalOrActionOnly);
                    let recent_files = recently_read_paths(&turn_ctx.items);
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
            } else if constrained_mode.is_none() {
                if consecutive_read_only_rounds > 0 {
                    tracing::debug!(
                        turn_id = %turn_id,
                        consecutive_read_only_rounds,
                        "executor: action tool executed — resetting read-only counter"
                    );
                }
                consecutive_read_only_rounds = 0;
            }

            if constrained_mode.is_some() && !round_is_read_only && had_any_success {
                constrained_mode = None;
                constrained_mode_violations = 0;
                tracing::debug!(
                    turn_id = %turn_id,
                    "executor: constrained mode cleared after successful action"
                );
            }

            // ── Consecutive-failure guard ─────────────────────────────────────
            // Reset counter if any tool succeeded this round; bump it if every
            // tool failed. Trips the break after the configured all-fail
            // rounds in a row — catches stuck loops without harming legitimate
            // long tasks where tools normally succeed.
            if had_any_success {
                consecutive_failures = 0;
            } else {
                consecutive_failures += 1;
                tracing::warn!(
                    turn_id = %turn_id,
                    consecutive_failures,
                    max_consecutive_failures = agent_cfg.max_consecutive_failures,
                    completed_tool_rounds,
                    "executor: all tools in this round FAILED"
                );
                if consecutive_failures > agent_cfg.max_consecutive_failures {
                    tracing::error!(
                        turn_id = %turn_id,
                        consecutive_failures,
                        "executor: consecutive-failure limit reached — terminating turn"
                    );
                    return self
                        .fail_turn(
                            &params.thread_id,
                            &turn_id,
                            "loop_limit",
                            &format!(
                                "Agent made {consecutive_failures} consecutive rounds where every \
                                 tool call failed. The turn has been stopped. Check that tool \
                                 arguments are correct and the requested paths/commands exist."
                            ),
                        )
                        .await;
                }
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
        if let Some(text) = &input.text {
            self.persist_turn_item(ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
                created_at: now_seconds(),
                kind: ItemKind::UserMessage,
                payload: json!({ "text": text.text }),
            })
            .await?;

            // ── Auto-title: set thread title from first user message ──────────
            // Only fires once — when the thread still has no title.
            if let Some(thread) = self.runtime.load_thread(thread_id).await? {
                let mut guard = thread.lock().await;
                if guard.metadata.title.is_none() {
                    let title = derive_thread_title(&text.text);
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
        if let Some(image) = &input.image {
            self.persist_turn_item(ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: thread_id.into(),
                turn_id: turn_id.into(),
                created_at: now_seconds(),
                kind: ItemKind::UserAttachment,
                payload: json!({ "path": image.path }),
            })
            .await?;
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
        usage: TokenUsage,
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
                turn.metadata.token_usage = usage;
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

        let model_settings = self
            .runtime
            .inner
            .effective_config
            .models
            .iter()
            .find(|m| m.model_id == model_id)
            .cloned()
            .unwrap_or_else(|| self.runtime.inner.effective_config.model_settings.clone());
        let prompt_budget =
            super::model_gateway::calculate_prompt_budget(model_settings.context_window);

        let used_pct = estimate_items_token_pct(&persisted.items, prompt_budget);
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
        // Persist a visible Error item so the failure reason appears in the
        // transcript and survives thread reloads.
        let error_item = ConversationItem {
            item_id: format!("item_{}", Uuid::new_v4().simple()),
            thread_id: thread_id.into(),
            turn_id: turn_id.into(),
            created_at: now_seconds(),
            kind: ItemKind::Error,
            payload: json!({ "kind": kind_label, "message": message }),
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

        self.runtime
            .publish_event(
                crate::system::domain::RuntimeEventKind::TurnFailed,
                Some(thread_id.into()),
                Some(turn_id.into()),
                json!({ "kind": kind_label, "reason": message }),
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
    ) -> Result<Vec<String>> {
        let mut instructions = vec![self.runtime.inner.effective_config.system_prompt.clone()];
        if let Some(instruction) = thinking_instruction(reasoning_effort) {
            instructions.push(instruction);
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
