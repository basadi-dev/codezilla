use anyhow::{anyhow, Result};
use serde_json::json;
use uuid::Uuid;

use super::model_gateway::{estimate_items_token_pct, is_context_overflow_error, ModelStreamEvent};
mod context;
mod tool_dispatch;
mod utils;

use self::context::TurnContext;
use self::utils::{
    derive_thread_title, find_repetition_start, is_degenerate_repetition, is_read_only_tool,
    should_retry_no_tool_completion, thinking_instruction,
};
use crate::system::domain::{
    now_seconds, ConversationItem, ItemKind, RuntimeEventKind, ThreadStatus, TokenUsage, ToolCall,
    ToolResult, TurnStatus, UserInput,
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

        // Tracks whether we have already done one automatic context-overflow
        // trim this turn. We only retry once to avoid an infinite loop.
        let mut auto_trim_attempted = false;

        // ── Loop-break guards ─────────────────────────────────────────────────
        //
        // Guard 1 — consecutive failures: incremented each iteration where
        // every tool call in that round returned ok:false; reset to 0 as soon
        // as any tool succeeds. Catches stuck loops (e.g. bad args that keep
        // failing) without harming legitimate long tasks.
        const MAX_CONSECUTIVE_FAILURES: usize = 5;
        let mut consecutive_failures: usize = 0;

        // Guard 2 — absolute backstop: a very high ceiling that only fires
        // when the model loops *successfully* without ever finishing. Legitimate
        // agentic tasks rarely need more than ~100 rounds.
        const MAX_AGENT_ITERATIONS: usize = 150;
        let mut agent_iterations: usize = 0;
        const MAX_NO_TOOL_NUDGES: usize = 2;
        let mut no_tool_nudges: usize = 0;
        let mut no_tool_nudge_instruction: Option<String> = None;
        let mut completed_tool_rounds: usize = 0;

        // Guard 3 — read-only exploration saturation: counts consecutive rounds
        // where *every* tool call was a pure read (read_file, list_dir, grep …).
        // If the model keeps exploring without ever writing or executing, it gets
        // a single nudge telling it to act. Resets to 0 as soon as one action
        // tool (write_file, bash_exec, …) appears in any round.
        const MAX_CONSECUTIVE_READ_ONLY_ROUNDS: usize = 15;
        let mut consecutive_read_only_rounds: usize = 0;

        loop {
            agent_iterations += 1;
            if agent_iterations > MAX_AGENT_ITERATIONS {
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
                            "Agent exceeded {MAX_AGENT_ITERATIONS} tool-call iterations without \
                             finishing. The turn has been stopped to prevent an infinite loop."
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

            let turn_ctx = TurnContext::build(self, &params, &turn_id, thread.clone()).await?;

            let mut system_instructions = self
                .system_instructions(
                    &turn_ctx.cwd,
                    turn_ctx
                        .effective_model_settings
                        .reasoning_effort
                        .as_deref(),
                )
                .await?;
            if let Some(instruction) = no_tool_nudge_instruction.take() {
                system_instructions.push(instruction);
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
                .start_response(request)
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
            const MAX_RESPONSE_CHARS: usize = 256_000; // ~64k tokens

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
                        if assistant_text.len() > MAX_RESPONSE_CHARS {
                            tracing::warn!(
                                turn_id,
                                chars = assistant_text.len(),
                                "streaming guard: response exceeded {MAX_RESPONSE_CHARS} chars — truncating"
                            );
                            self.runtime.publish_event(
                                RuntimeEventKind::Warning,
                                Some(params.thread_id.clone()), Some(turn_id.clone()),
                                json!({"message": "Model response was too long and has been truncated."}),
                            ).await?;
                            assistant_text.truncate(MAX_RESPONSE_CHARS);
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

            if let Some(item_id) = assistant_item_id {
                let item = ConversationItem {
                    item_id,
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    created_at: now_seconds(),
                    kind: ItemKind::AgentMessage,
                    payload: json!({ "text": assistant_text }),
                };
                self.persist_turn_item(item).await?;
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
                    assistant_text_len = assistant_text.len(),
                    assistant_text_preview = %assistant_text.chars().take(200).collect::<String>(),
                    "executor: no tool calls in model response"
                );
                if no_tool_nudges < MAX_NO_TOOL_NUDGES
                    && should_retry_no_tool_completion(
                        &assistant_text,
                        &turn_ctx.items,
                        completed_tool_rounds,
                    )
                {
                    no_tool_nudges += 1;
                    tracing::warn!(
                        turn_id = %turn_id,
                        no_tool_nudges,
                        MAX_NO_TOOL_NUDGES,
                        "executor: nudging model to emit tool call (described intent but no call)"
                    );
                    no_tool_nudge_instruction = Some(format!(
                        "The previous assistant response described using a tool but emitted no \
                         tool call, so no action ran. Do not continue narrating. If the task is \
                         not complete, your next response must contain exactly one tool call and \
                         only the minimal text needed before it. Retry {no_tool_nudges}/{MAX_NO_TOOL_NUDGES}."
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
                tracing::info!(
                    turn_id = %turn_id,
                    agent_iterations,
                    completed_tool_rounds,
                    "executor: turn completing — no tool calls, no nudge required"
                );
                return self
                    .complete_turn(&params.thread_id, &turn_id, final_usage)
                    .await;
            }
            no_tool_nudges = 0;

            // Snapshot whether this round is pure read-only *before* tool_calls is moved.
            let round_is_read_only = tool_calls.iter().all(|c| is_read_only_tool(&c.tool_name));

            let had_any_success = self.execute_tool_round(&turn_ctx, tool_calls).await?;
            completed_tool_rounds += 1;

            // ── Read-only exploration guard ───────────────────────────────────
            // Reset when any action tool ran this round; bump when every call
            // was read-only. On hitting the threshold inject a single nudge and
            // reset so a follow-up nudge fires again if the model ignores it.
            if round_is_read_only {
                consecutive_read_only_rounds += 1;
                tracing::debug!(
                    turn_id = %turn_id,
                    consecutive_read_only_rounds,
                    MAX_CONSECUTIVE_READ_ONLY_ROUNDS,
                    "executor: read-only round ({consecutive_read_only_rounds}/{MAX_CONSECUTIVE_READ_ONLY_ROUNDS})"
                );
                if consecutive_read_only_rounds >= MAX_CONSECUTIVE_READ_ONLY_ROUNDS {
                    consecutive_read_only_rounds = 0;
                    let msg = format!(
                        "You have spent {MAX_CONSECUTIVE_READ_ONLY_ROUNDS} consecutive rounds \
                         reading files or exploring without taking any action. If you have \
                         gathered enough context, you must now either write a file, run a \
                         command, or give a final answer. Do not read more files unless \
                         absolutely necessary."
                    );
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
                                    "Agent has been exploring for {MAX_CONSECUTIVE_READ_ONLY_ROUNDS} rounds \
                                     — nudging it to take an action."
                                )
                            }),
                        )
                        .await?;
                    no_tool_nudge_instruction = Some(msg);
                }
            } else {
                if consecutive_read_only_rounds > 0 {
                    tracing::debug!(
                        turn_id = %turn_id,
                        consecutive_read_only_rounds,
                        "executor: action tool executed — resetting read-only counter"
                    );
                }
                consecutive_read_only_rounds = 0;
            }

            // ── Consecutive-failure guard ─────────────────────────────────────
            // Reset counter if any tool succeeded this round; bump it if every
            // tool failed. Trips the break after MAX_CONSECUTIVE_FAILURES all-fail
            // rounds in a row — catches stuck loops without harming legitimate
            // long tasks where tools normally succeed.
            if had_any_success {
                consecutive_failures = 0;
            } else {
                consecutive_failures += 1;
                tracing::warn!(
                    turn_id = %turn_id,
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                    completed_tool_rounds,
                    "executor: all tools in this round FAILED (consecutive_failures={consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                );
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES {
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
                serde_json::to_value(&metadata)?,
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
        Ok(instructions)
    }
}
