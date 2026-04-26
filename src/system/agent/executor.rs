use anyhow::{anyhow, Result};
use futures::future::join_all;
use serde_json::{json, Value};
use uuid::Uuid;

use super::model_gateway::{estimate_items_token_pct, is_context_overflow_error, ModelStreamEvent};
use crate::system::domain::{
    now_seconds, ActionDescriptor, ApprovalCategory, ApprovalDecision, ApprovalPolicy,
    ApprovalRequest,
    ConversationItem, ItemKind, ModelSettings, RuntimeEventKind, ThreadStatus, TokenUsage,
    ToolCall, ToolExecutionContext, ToolListingContext, ToolResult, TurnStatus, UserInput,
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

        loop {
            agent_iterations += 1;
            if agent_iterations > MAX_AGENT_ITERATIONS {
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
            if self.is_cancelled(&params.thread_id, &turn_id).await? {
                return self.complete_interrupted(&params.thread_id, &turn_id).await;
            }

            self.drain_steering(&params.thread_id, &turn_id).await?;

            let (thread_metadata, items, cancel_token) = {
                let thread = thread.lock().await;
                let turn = thread
                    .turns
                    .iter()
                    .find(|t| t.metadata.turn_id == turn_id)
                    .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                // Prefix items come first (e.g. ReasoningSummary from auto-compaction
                // which is stored under the synthetic "compaction" turn_id and therefore
                // not assigned to any real turn during session reload).
                let mut items = thread.prefix_items.clone();
                for lt in &thread.turns {
                    items.extend(lt.items.clone());
                }
                (thread.metadata.clone(), items, turn.cancel_token.clone())
            };

            let cwd = params
                .cwd
                .clone()
                .or(thread_metadata.cwd.clone())
                .unwrap_or_else(|| {
                    self.runtime
                        .inner
                        .effective_config
                        .working_directory
                        .clone()
                });
            let permission_profile = params.permission_profile.clone().unwrap_or_else(|| {
                self.runtime
                    .inner
                    .permission_manager
                    .resolve_permission_profile(&self.runtime.inner.effective_config, &cwd)
            });
            let listing = ToolListingContext {
                thread_id: params.thread_id.clone(),
                cwd: cwd.clone(),
                features: self.runtime.inner.effective_config.features.clone(),
            };
            let tools = self
                .runtime
                .inner
                .tool_orchestrator
                .list_available_tools(&listing);
            let effective_model_settings =
                params
                    .model_settings
                    .clone()
                    .unwrap_or_else(|| ModelSettings {
                        model_id: thread_metadata.model_id.clone(),
                        provider_id: thread_metadata.provider_id.clone(),
                        reasoning_effort: self
                            .runtime
                            .inner
                            .effective_config
                            .model_settings
                            .reasoning_effort
                            .clone(),
                        summary_mode: None,
                        service_tier: None,
                        web_search_enabled: false,
                    });
            let request = super::model_gateway::ModelRequest {
                system_instructions: self
                    .system_instructions(&cwd, effective_model_settings.reasoning_effort.as_deref())
                    .await?,
                model_settings: effective_model_settings,
                conversation_items: items.clone(),
                tool_definitions: tools,
                output_schema: params.output_schema.clone(),
            };

            let mut response = self
                .runtime
                .inner
                .model_gateway
                .start_response(request)
                .await?;

            let mut assistant_item_id: Option<String> = None;
            let mut assistant_text = String::new();
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
                if cancel_token.is_cancelled() {
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
                        let mut guard = thread.lock().await;
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

            if tool_calls.is_empty() {
                return self
                    .complete_turn(&params.thread_id, &turn_id, final_usage)
                    .await;
            }

            let had_any_success = self
                .execute_tool_round(
                    tool_calls,
                    &params,
                    &turn_id,
                    &thread,
                    &cwd,
                    &permission_profile,
                    &listing,
                )
                .await?;

            // ── Consecutive-failure guard ─────────────────────────────────────
            // Reset counter if any tool succeeded this round; bump it if every
            // tool failed. Trips the break after MAX_CONSECUTIVE_FAILURES all-fail
            // rounds in a row — catches stuck loops without harming legitimate
            // long tasks where tools normally succeed.
            if had_any_success {
                consecutive_failures = 0;
            } else {
                consecutive_failures += 1;
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES {
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

        let used_pct = estimate_items_token_pct(&persisted.items);
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

    async fn is_cancelled(&self, thread_id: &str, turn_id: &str) -> Result<bool> {
        let thread = self
            .runtime
            .load_thread(thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {thread_id}"))?;
        let thread = thread.lock().await;
        let turn = thread
            .turns
            .iter()
            .find(|t| t.metadata.turn_id == turn_id)
            .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
        Ok(turn.cancel_token.is_cancelled())
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

    // ── Tool round dispatcher ──────────────────────────────────────────────────

    /// Dispatch one round of tool calls (everything the model emitted in a
    /// single turn response). Calls that advertise `supports_parallel_calls`
    /// are grouped into batches and executed concurrently with `join_all`.
    /// Non-parallel-safe calls get their own sequential batch.
    ///
    /// Returns `true` if any tool in the round succeeded (used by the
    /// consecutive-failure guard in `run_turn`).
    async fn execute_tool_round(
        &self,
        tool_calls: Vec<ToolCall>,
        params: &crate::system::runtime::TurnStartParams,
        turn_id: &str,
        thread: &tokio::sync::Mutex<crate::system::runtime::ThreadSession>,
        cwd: &str,
        permission_profile: &crate::system::domain::PermissionProfile,
        listing_ctx: &ToolListingContext,
    ) -> Result<bool> {
        // 1. Normalise: rewrite shell_exec with shell operators → bash_exec.
        let calls: Vec<ToolCall> = tool_calls
            .into_iter()
            .map(promote_to_bash_if_needed)
            .collect();

        // 2. Persist all ToolCall items in original order *before* executing
        //    anything, so the transcript always shows calls before results.
        for call in &calls {
            let call_item = ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: params.thread_id.clone(),
                turn_id: turn_id.into(),
                created_at: now_seconds(),
                kind: ItemKind::ToolCall,
                payload: serde_json::to_value(call)?,
            };
            self.persist_turn_item(call_item).await?;
        }

        // 3. Partition into batches (consecutive parallel-safe calls grouped;
        //    each non-parallel-safe call is a solo sequential batch).
        let batches = partition_into_batches(&calls, |name| {
            self.runtime
                .inner
                .tool_orchestrator
                .is_parallel_safe(name, listing_ctx)
        });

        tracing::debug!(
            turn_id,
            total_calls = calls.len(),
            batch_count = batches.len(),
            "tool_round: dispatching"
        );

        // 4. Dispatch each batch, collecting results in original-call order.
        let mut had_any_success = false;
        for batch in batches {
            if batch.len() > 1 {
                tracing::debug!("tool_round: parallel batch of {}", batch.len());
            }
            let results = self
                .dispatch_batch(batch, params, turn_id, thread, cwd, permission_profile)
                .await?;
            for result in results {
                if result.ok {
                    had_any_success = true;
                }
                self.persist_tool_result(&params.thread_id, turn_id, &result)
                    .await?;
            }
        }

        Ok(had_any_success)
    }

    /// Execute a single batch of tool calls.
    ///
    /// Phase A — sequential approval: each call in the batch is approved
    ///   one at a time (preserves the existing single-prompt UX).
    /// Phase B — parallel execution: all approved calls are dispatched
    ///   concurrently with `join_all`.
    /// Phase C — stable ordering: results are sorted back to original
    ///   call-index order before being returned.
    async fn dispatch_batch(
        &self,
        batch: Vec<(usize, ToolCall)>,
        params: &crate::system::runtime::TurnStartParams,
        turn_id: &str,
        thread: &tokio::sync::Mutex<crate::system::runtime::ThreadSession>,
        cwd: &str,
        permission_profile: &crate::system::domain::PermissionProfile,
    ) -> Result<Vec<ToolResult>> {
        // ── Phase A: sequential approval ──────────────────────────────────────
        let mut ready: Vec<(usize, ToolCall, ApprovalPolicy)> = Vec::new();
        let mut results: Vec<(usize, ToolResult)> = Vec::new();

        for (idx, call) in batch {
            // Re-read policy live so a Ctrl+A toggle between batches takes
            // effect immediately.
            let approval_policy = {
                let t = thread.lock().await;
                t.approval_policy_override.clone()
            }
            .or_else(|| params.approval_policy.clone())
            .unwrap_or_else(|| self.runtime.inner.effective_config.approval_policy.clone());

            let action = action_for_tool_call(&call, cwd);
            if self.runtime.inner.permission_manager.requires_approval(
                &action,
                &approval_policy,
                cwd,
            ) {
                let approval = self
                    .runtime
                    .inner
                    .approval_manager
                    .create_approval(ApprovalRequest {
                        approval_id: format!("approval_{}", Uuid::new_v4().simple()),
                        thread_id: params.thread_id.clone(),
                        turn_id: turn_id.into(),
                        category: action.category,
                        title: format!("Approve {}", call.tool_name),
                        justification: format!("The assistant requested {}", call.tool_name),
                        action: serde_json::to_value(&action)?,
                    })
                    .await;

                {
                    let mut t = thread.lock().await;
                    let turn = t
                        .turns
                        .iter_mut()
                        .find(|t| t.metadata.turn_id == turn_id)
                        .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                    turn.status = TurnStatus::WaitingForApproval;
                    turn.metadata.status = TurnStatus::WaitingForApproval;
                    self.runtime
                        .inner
                        .persistence_manager
                        .update_turn(&turn.metadata)?;
                }

                self.runtime
                    .publish_event(
                        RuntimeEventKind::ApprovalRequested,
                        Some(params.thread_id.clone()),
                        Some(turn_id.into()),
                        serde_json::to_value(&approval)?,
                    )
                    .await?;

                let transcript = self
                    .runtime
                    .inner
                    .persistence_manager
                    .read_thread(&params.thread_id)?
                    .items;
                let resolution = self
                    .runtime
                    .inner
                    .approval_manager
                    .wait_for_approval(&approval.request.approval_id, 300, &transcript)
                    .await?;
                self.runtime
                    .publish_event(
                        RuntimeEventKind::ApprovalResolved,
                        Some(params.thread_id.clone()),
                        Some(turn_id.into()),
                        serde_json::to_value(&resolution)?,
                    )
                    .await?;

                if resolution.decision != ApprovalDecision::Approved {
                    results.push((
                        idx,
                        ToolResult {
                            tool_call_id: call.tool_call_id.clone(),
                            ok: false,
                            output: json!({ "approved": false }),
                            error_message: Some(format!("approval {:?}", resolution.decision)),
                        },
                    ));
                    continue;
                }
            }

            ready.push((idx, call, approval_policy));
        }

        // ── Phase B: parallel execution ───────────────────────────────────────
        if !ready.is_empty() {
            let futures: Vec<_> = ready
                .into_iter()
                .map(|(idx, call, approval_policy)| {
                    let orchestrator = &self.runtime.inner.tool_orchestrator;
                    let ctx = ToolExecutionContext {
                        thread_id: params.thread_id.clone(),
                        turn_id: turn_id.into(),
                        cwd: cwd.into(),
                        permission_profile: permission_profile.clone(),
                        approval_policy,
                        agent_depth: params.agent_depth,
                    };
                    async move {
                        let result =
                            orchestrator
                                .execute(&call, ctx)
                                .await
                                .unwrap_or_else(|e| ToolResult {
                                    tool_call_id: call.tool_call_id.clone(),
                                    ok: false,
                                    output: json!({ "error": e.to_string() }),
                                    error_message: Some(e.to_string()),
                                });
                        (idx, result)
                    }
                })
                .collect();

            results.extend(join_all(futures).await);
        }

        // ── Phase C: sort by original index before returning ──────────────────
        results.sort_by_key(|(idx, _)| *idx);
        Ok(results.into_iter().map(|(_, r)| r).collect())
    }
}

// ─── partition_into_batches ───────────────────────────────────────────────────

/// Split an ordered slice of `ToolCall`s into sequential execution batches.
///
/// Consecutive calls that are all parallel-safe are grouped into a single batch
/// so they can be executed with `join_all`. Any call that is *not* parallel-safe
/// gets its own single-element batch and acts as a serialisation barrier.
///
/// Examples:
///   [read, read, read]          → [(read, read, read)]
///   [read, write, read]         → [(read), (write), (read)]
///   [read, read, write, read]   → [(read, read), (write), (read)]
///   [bash, bash]                → [(bash), (bash)]
fn partition_into_batches<F>(calls: &[ToolCall], is_parallel: F) -> Vec<Vec<(usize, ToolCall)>>
where
    F: Fn(&str) -> bool,
{
    let mut batches: Vec<Vec<(usize, ToolCall)>> = Vec::new();
    let mut current: Vec<(usize, ToolCall)> = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        if is_parallel(&call.tool_name) {
            current.push((i, call.clone()));
        } else {
            if !current.is_empty() {
                batches.push(std::mem::take(&mut current));
            }
            batches.push(vec![(i, call.clone())]);
        }
    }

    if !current.is_empty() {
        batches.push(current);
    }

    batches
}

fn thinking_instruction(reasoning_effort: Option<&str>) -> Option<String> {
    match reasoning_effort {
        None | Some("off") => None,
        Some("low") => Some(
            "Think briefly before responding. A short internal reasoning pass is enough.".into(),
        ),
        Some("medium") => Some(
            "Think through this carefully, step by step, before giving your final answer.".into(),
        ),
        Some("high") => Some(
            "Think extra hard. Reason deeply and thoroughly, considering multiple angles and edge \
             cases, before providing your answer."
                .into(),
        ),
        Some(other) => Some(format!(
            "Reasoning effort: {other}. Think carefully before responding."
        )),
    }
}

// ─── action_for_tool_call (module-private helper) ────────────────────────────

fn action_for_tool_call(call: &ToolCall, cwd: &str) -> ActionDescriptor {
    let category = match call.tool_name.as_str() {
        "bash_exec" | "shell_exec" => ApprovalCategory::SandboxEscalation,
        "write_file" | "create_directory" | "remove_path" | "copy_path" => {
            ApprovalCategory::FileChange
        }
        _ => ApprovalCategory::Other,
    };
    let paths = match call.tool_name.as_str() {
        "write_file" | "create_directory" | "remove_path" => call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .map(|p| vec![p.to_string()])
            .unwrap_or_else(|| vec![cwd.into()]),
        "copy_path" => vec![
            call.arguments
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or(cwd)
                .to_string(),
            call.arguments
                .get("target")
                .and_then(Value::as_str)
                .unwrap_or(cwd)
                .to_string(),
        ],
        _ => vec![cwd.into()],
    };
    let command = match call.tool_name.as_str() {
        "bash_exec" => call
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .map(|command| vec!["bash".to_string(), "-c".to_string(), command.to_string()]),
        "shell_exec" => call
            .arguments
            .get("argv")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .or_else(|| {
                call.arguments
                    .get("argv")
                    .and_then(Value::as_str)
                    .map(|argv| vec![argv.to_string()])
            }),
        _ => None,
    };
    ActionDescriptor {
        action_type: call.tool_name.clone(),
        command,
        paths,
        domains: Vec::new(),
        category,
    }
}

// ─── derive_thread_title ──────────────────────────────────────────────────────

/// Build a short, human-readable title from the first user message.
/// Takes the first non-empty line, strips leading `/` commands, and
/// truncates to 72 chars with an ellipsis when needed.
fn derive_thread_title(text: &str) -> String {
    const MAX: usize = 72;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('/'))
        .unwrap_or_else(|| text.lines().next().unwrap_or("").trim());

    if line.is_empty() {
        return "Untitled thread".into();
    }

    let chars: Vec<char> = line.chars().collect();
    if chars.len() <= MAX {
        line.to_string()
    } else {
        chars[..MAX].iter().collect::<String>() + "…"
    }
}

// ─── Shell-operator safety net ────────────────────────────────────────────────

/// Shell operators and patterns that only make sense inside a shell.
/// Any of these appearing as a standalone argv token is a dead giveaway that
/// the model intended shell semantics but called `shell_exec` by mistake.
const SHELL_OPERATOR_TOKENS: &[&str] = &[
    "|",
    "||",
    "&&",
    ";",
    "&",
    ">",
    ">>",
    "<",
    "<<",
    "2>&1",
    "2>/dev/null",
    "1>/dev/null",
    "1>&2",
    "2>>",
    "1>>",
];

/// Inspect a `shell_exec` ToolCall for shell operators in its argv.
/// If any are found, rewrite the call as a `bash_exec` command string so that
/// the operators are interpreted correctly by bash.
///
/// This is the runtime safety net — it catches model mistakes that slipped
/// past the system prompt and schema guidance.
pub(crate) fn promote_to_bash_if_needed(call: ToolCall) -> ToolCall {
    if call.tool_name != "shell_exec" {
        return call;
    }

    // Only inspect array argv; string argv goes through simple_tokenize in
    // ShellToolProvider which also won't support shell operators, so promote
    // string argv too if it looks like a shell command.
    let needs_promotion = if let Some(arr) = call.arguments.get("argv").and_then(|v| v.as_array()) {
        arr.iter().filter_map(|v| v.as_str()).any(|tok| {
            SHELL_OPERATOR_TOKENS.contains(&tok)
                || (tok.contains('*') && !tok.starts_with("--")) // glob (not a flag)
                || tok.starts_with("2>")
                || tok.starts_with("1>")
                || tok == "?"
        })
    } else if let Some(s) = call.arguments.get("argv").and_then(|v| v.as_str()) {
        // String argv — check if it looks like it has shell operators
        SHELL_OPERATOR_TOKENS.iter().any(|op| {
            // Match operator as a whole word, not a substring of a flag
            s.split_whitespace().any(|tok| tok == *op)
        }) || s.contains("2>&1")
            || s.contains("| ")
            || s.contains(" |")
    } else {
        false
    };

    if !needs_promotion {
        return call;
    }

    // Build the shell command string by joining argv tokens
    let shell_cmd = if let Some(arr) = call.arguments.get("argv").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else if let Some(s) = call.arguments.get("argv").and_then(|v| v.as_str()) {
        s.to_string()
    } else {
        return call; // nothing to do
    };

    tracing::warn!(
        tool_call_id = %call.tool_call_id,
        shell_cmd = %shell_cmd,
        "shell_exec contained shell operators — auto-promoting to bash_exec"
    );

    // Build a new arguments object: replace argv with command, keep cwd/env
    let mut new_args = call.arguments.clone();
    if let Some(obj) = new_args.as_object_mut() {
        obj.remove("argv");
        obj.insert("command".to_string(), serde_json::Value::String(shell_cmd));
    }

    ToolCall {
        tool_name: "bash_exec".into(),
        tool_call_id: call.tool_call_id,
        provider_kind: call.provider_kind,
        arguments: new_args,
    }
}

// ─── Degenerate-output detection ──────────────────────────────────────────────
//
// Some local/quantized models enter token-generation loops, repeating the same
// pattern indefinitely. These helpers detect that condition early.
//
// All comparisons are done on **bytes** (`as_bytes()`) to avoid UTF-8 boundary
// panics. If the same Unicode text repeats, the same bytes also repeat, so the
// detection is equally correct at the byte level.

/// Returns `true` when the *tail* of the text contains a byte run of length
/// ≥ `MIN_PATTERN_LEN` that repeats at least `MIN_REPEATS` times consecutively.
///
/// Only examines the last `WINDOW` bytes to stay O(1) per streaming delta.
fn is_degenerate_repetition(text: &str) -> bool {
    const MIN_PATTERN_LEN: usize = 40;
    const MIN_REPEATS: usize = 5;
    const WINDOW: usize = 4_000;

    let bytes = text.as_bytes();
    if bytes.len() < MIN_PATTERN_LEN * MIN_REPEATS {
        return false;
    }

    let tail = if bytes.len() > WINDOW {
        &bytes[bytes.len() - WINDOW..]
    } else {
        bytes
    };

    // Try candidate pattern lengths 40, 50, 60 … 200 bytes.
    let max_pat = 200.min(tail.len() / MIN_REPEATS);
    for pat_len in (MIN_PATTERN_LEN..=max_pat).step_by(10) {
        let pattern = &tail[tail.len() - pat_len..];
        let mut count = 0usize;
        let mut pos = tail.len() - pat_len;
        while pos >= pat_len {
            pos -= pat_len;
            if &tail[pos..pos + pat_len] == pattern {
                count += 1;
            } else {
                break;
            }
        }
        if count >= MIN_REPEATS {
            return true;
        }
    }
    false
}

/// Find the **byte** offset (into `text`) where the repetitive pattern starts,
/// so we can `truncate()` to the clean prefix. Returns `None` if no repetition
/// is detected. The returned offset is always on a UTF-8 char boundary because
/// we walk backward to the nearest boundary before returning.
fn find_repetition_start(text: &str) -> Option<usize> {
    const MIN_PATTERN_LEN: usize = 40;
    const MIN_REPEATS: usize = 5;
    const WINDOW: usize = 4_000;

    let bytes = text.as_bytes();
    if bytes.len() < MIN_PATTERN_LEN * MIN_REPEATS {
        return None;
    }

    let search_start = bytes.len().saturating_sub(WINDOW);
    let tail = &bytes[search_start..];

    let max_pat = 200.min(tail.len() / MIN_REPEATS);
    for pat_len in (MIN_PATTERN_LEN..=max_pat).step_by(10) {
        let pattern = &tail[tail.len() - pat_len..];
        let mut earliest = tail.len() - pat_len;
        let mut count = 0usize;
        let mut pos = tail.len() - pat_len;
        while pos >= pat_len {
            pos -= pat_len;
            if &tail[pos..pos + pat_len] == pattern {
                count += 1;
                earliest = pos;
            } else {
                break;
            }
        }
        if count >= MIN_REPEATS {
            // Walk the raw offset back to the nearest valid UTF-8 char boundary.
            let raw = search_start + earliest;
            let boundary = (0..=raw)
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0);
            return Some(boundary);
        }
    }
    None
}


