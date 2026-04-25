use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use uuid::Uuid;

use super::model_gateway::{estimate_items_token_pct, is_context_overflow_error, ModelStreamEvent};
use crate::system::domain::{
    now_seconds, ActionDescriptor, ApprovalCategory, ApprovalDecision, ApprovalRequest,
    ConversationItem, ItemKind, ModelSettings, RuntimeEventKind, ThreadStatus, TokenUsage,
    ToolCall, ToolExecutionContext, ToolListingContext, ToolResult, TurnStatus, UserInput,
};

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

        loop {
            if self.is_cancelled(&params.thread_id, &turn_id).await? {
                return self.complete_interrupted(&params.thread_id, &turn_id).await;
            }

            self.drain_steering(&params.thread_id, &turn_id).await?;

            let (thread_metadata, items, cancel_token, live_approval_policy) = {
                let thread = thread.lock().await;
                let turn = thread
                    .turns
                    .iter()
                    .find(|t| t.metadata.turn_id == turn_id)
                    .ok_or_else(|| anyhow!("turn_not_found: {turn_id}"))?;
                let mut items = Vec::new();
                for lt in &thread.turns {
                    items.extend(lt.items.clone());
                }
                (
                    thread.metadata.clone(),
                    items,
                    turn.cancel_token.clone(),
                    // Read the live policy set on the session — this can change
                    // mid-turn (e.g. user toggles auto-approve) without restarting.
                    thread.approval_policy_override.clone(),
                )
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
            let approval_policy = live_approval_policy
                .or_else(|| params.approval_policy.clone())
                .unwrap_or_else(|| self.runtime.inner.effective_config.approval_policy.clone());
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
            let effective_model_settings = params
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

                // Second overflow or a different error — fail the turn.
                return self.fail_turn(&params.thread_id, &turn_id, error).await;
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

            for call in tool_calls {
                let call_item = ConversationItem {
                    item_id: format!("item_{}", Uuid::new_v4().simple()),
                    thread_id: params.thread_id.clone(),
                    turn_id: turn_id.clone(),
                    created_at: now_seconds(),
                    kind: ItemKind::ToolCall,
                    payload: serde_json::to_value(&call)?,
                };
                self.persist_turn_item(call_item).await?;

                // Re-read the live policy per tool call so a Ctrl+A toggle during
                // an LLM response takes effect on the very next tool, not the
                // next full loop iteration.
                let approval_policy = {
                    let t = thread.lock().await;
                    t.approval_policy_override.clone()
                }
                .or_else(|| params.approval_policy.clone())
                .unwrap_or_else(|| {
                    self.runtime.inner.effective_config.approval_policy.clone()
                });

                let action = action_for_tool_call(&call, &cwd);
                if self.runtime.inner.permission_manager.requires_approval(
                    &action,
                    &approval_policy,
                    &cwd,
                ) {
                    let approval = self
                        .runtime
                        .inner
                        .approval_manager
                        .create_approval(ApprovalRequest {
                            approval_id: format!("approval_{}", Uuid::new_v4().simple()),
                            thread_id: params.thread_id.clone(),
                            turn_id: turn_id.clone(),
                            category: action.category,
                            title: format!("Approve {}", call.tool_name),
                            justification: format!("The assistant requested {}", call.tool_name),
                            action: serde_json::to_value(&action)?,
                        })
                        .await;

                    {
                        let mut thread = thread.lock().await;
                        let turn = thread
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
                            crate::system::domain::RuntimeEventKind::ApprovalRequested,
                            Some(params.thread_id.clone()),
                            Some(turn_id.clone()),
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
                            crate::system::domain::RuntimeEventKind::ApprovalResolved,
                            Some(params.thread_id.clone()),
                            Some(turn_id.clone()),
                            serde_json::to_value(&resolution)?,
                        )
                        .await?;

                    if resolution.decision != ApprovalDecision::Approved {
                        let denial = ToolResult {
                            tool_call_id: call.tool_call_id.clone(),
                            ok: false,
                            output: json!({ "approved": false }),
                            error_message: Some(format!("approval {:?}", resolution.decision)),
                        };
                        self.persist_tool_result(&params.thread_id, &turn_id, &denial)
                            .await?;
                        continue;
                    }
                }

                let result = self
                    .runtime
                    .inner
                    .tool_orchestrator
                    .execute(
                        &call,
                        ToolExecutionContext {
                            thread_id: params.thread_id.clone(),
                            turn_id: turn_id.clone(),
                            cwd: cwd.clone(),
                            permission_profile: permission_profile.clone(),
                            approval_policy: approval_policy.clone(),
                        },
                    )
                    .await
                    .unwrap_or_else(|e| ToolResult {
                        tool_call_id: call.tool_call_id.clone(),
                        ok: false,
                        output: json!({ "error": e.to_string() }),
                        error_message: Some(e.to_string()),
                    });
                self.persist_tool_result(&params.thread_id, &turn_id, &result)
                    .await?;
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
        let persisted = match self.runtime.inner.persistence_manager.read_thread(thread_id) {
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

    pub async fn fail_turn(&self, thread_id: &str, turn_id: &str, reason: &str) -> Result<()> {
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
        self.runtime
            .publish_event(
                crate::system::domain::RuntimeEventKind::TurnFailed,
                Some(thread_id.into()),
                Some(turn_id.into()),
                json!({ "reason": reason }),
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

    async fn system_instructions(&self, cwd: &str, reasoning_effort: Option<&str>) -> Result<Vec<String>> {
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

// ─── thinking instruction helper ─────────────────────────────────────────────

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
        Some(other) => Some(format!("Reasoning effort: {other}. Think carefully before responding.")),
    }
}

// ─── action_for_tool_call (module-private helper) ────────────────────────────

fn action_for_tool_call(call: &ToolCall, cwd: &str) -> ActionDescriptor {
    let category = match call.tool_name.as_str() {
        "shell_exec" => ApprovalCategory::SandboxEscalation,
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
    ActionDescriptor {
        action_type: call.tool_name.clone(),
        command: call
            .arguments
            .get("argv")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            }),
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
