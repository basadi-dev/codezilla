//! Thread lifecycle methods on [`ConversationRuntime`].
//!
//! Covers create / resume / fork / read / list / compact / rollback /
//! archive / delete plus the in-memory session loader used by every other
//! runtime entry point. Pulled out of `mod.rs` so that file can shrink toward
//! "constructor + accessors + cross-cutting helpers" only.

use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    ConversationRuntime, LoadedTurn, ThreadCompactParams, ThreadCompactResult, ThreadForkParams,
    ThreadForkResult, ThreadListParams, ThreadListResult, ThreadMemoryModeParams,
    ThreadModelSettingsParams, ThreadReadParams, ThreadReadResult, ThreadResumeParams,
    ThreadResumeResult, ThreadRollbackParams, ThreadRollbackResult, ThreadSession,
    ThreadStartParams, ThreadStartResult,
};
use crate::system::agent::model_gateway::build_compaction_messages;
use crate::system::domain::{
    now_seconds, ApprovalPolicy, ConversationItem, ItemKind, MemoryMode, ModelSettings,
    PersistedThread, RuntimeEventKind, ThreadFilter, ThreadMetadata, ThreadStatus, TurnMetadata,
};

impl ConversationRuntime {
    pub async fn start_thread(&self, params: ThreadStartParams) -> Result<ThreadStartResult> {
        let model = params
            .model_settings
            .unwrap_or_else(|| self.inner.effective_config.model_settings.clone());
        let cwd = params
            .cwd
            .unwrap_or_else(|| self.inner.effective_config.working_directory.clone());
        let metadata = ThreadMetadata {
            thread_id: format!("thread_{}", Uuid::new_v4().simple()),
            title: None,
            created_at: now_seconds(),
            updated_at: now_seconds(),
            cwd: Some(cwd.clone()),
            model_id: model.model_id.clone(),
            provider_id: model.provider_id.clone(),
            status: ThreadStatus::Idle,
            forked_from_id: None,
            archived: false,
            ephemeral: params.ephemeral,
            memory_mode: MemoryMode::Enabled,
        };
        self.inner.persistence_manager.create_thread(&metadata)?;

        let thread = Arc::new(Mutex::new(ThreadSession {
            metadata: metadata.clone(),
            turns: Vec::new(),
            active_turn_id: None,
            subscribed_clients: HashSet::new(),
            owner_connection_id: None,
            pending_steering: Vec::new(),
            approval_policy_override: params.approval_policy.clone(),
            prefix_items: Vec::new(),
        }));
        self.inner
            .loaded_threads
            .write()
            .await
            .insert(metadata.thread_id.clone(), thread);

        self.publish_event(
            RuntimeEventKind::ThreadStarted,
            Some(metadata.thread_id.clone()),
            None,
            serde_json::to_value(&metadata)?,
        )
        .await?;

        Ok(ThreadStartResult { metadata })
    }

    pub async fn resume_thread(&self, params: ThreadResumeParams) -> Result<ThreadResumeResult> {
        let persisted = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;
        self.load_persisted_thread(&persisted).await?;
        Ok(ThreadResumeResult {
            metadata: persisted.metadata,
            turns: persisted.turns,
        })
    }

    pub async fn fork_thread(&self, params: ThreadForkParams) -> Result<ThreadForkResult> {
        let source = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;
        let new_id = format!("thread_{}", Uuid::new_v4().simple());
        let metadata = ThreadMetadata {
            thread_id: new_id.clone(),
            title: source.metadata.title.clone(),
            created_at: now_seconds(),
            updated_at: now_seconds(),
            cwd: source.metadata.cwd.clone(),
            model_id: source.metadata.model_id.clone(),
            provider_id: source.metadata.provider_id.clone(),
            status: ThreadStatus::Idle,
            forked_from_id: Some(source.metadata.thread_id.clone()),
            archived: false,
            ephemeral: params.ephemeral,
            memory_mode: source.metadata.memory_mode,
        };
        self.inner.persistence_manager.create_thread(&metadata)?;

        let mut turn_map = HashMap::new();
        for turn in source.turns {
            let new_turn = TurnMetadata {
                turn_id: format!("turn_{}", Uuid::new_v4().simple()),
                thread_id: new_id.clone(),
                created_at: turn.created_at,
                updated_at: turn.updated_at,
                status: turn.status,
                started_by_surface: turn.started_by_surface,
                token_usage: turn.token_usage,
            };
            turn_map.insert(turn.turn_id, new_turn.clone());
            self.inner.persistence_manager.create_turn(&new_turn)?;
            self.inner.persistence_manager.update_turn(&new_turn)?;
        }
        for item in source.items {
            let new_item = ConversationItem {
                item_id: format!("item_{}", Uuid::new_v4().simple()),
                thread_id: new_id.clone(),
                turn_id: turn_map
                    .get(&item.turn_id)
                    .map(|t| t.turn_id.clone())
                    .unwrap_or_else(|| item.turn_id.clone()),
                created_at: item.created_at,
                kind: item.kind,
                payload: item.payload,
            };
            self.inner.persistence_manager.append_item(&new_item)?;
        }

        self.resume_thread(ThreadResumeParams {
            thread_id: new_id.clone(),
        })
        .await?;
        Ok(ThreadForkResult { metadata })
    }

    pub async fn read_thread(&self, params: ThreadReadParams) -> Result<ThreadReadResult> {
        Ok(ThreadReadResult {
            thread: self
                .inner
                .persistence_manager
                .read_thread(&params.thread_id)?,
        })
    }

    pub async fn list_threads(&self, params: ThreadListParams) -> Result<ThreadListResult> {
        Ok(ThreadListResult {
            threads: self.inner.persistence_manager.list_threads(ThreadFilter {
                cwd: params.cwd,
                archived: params.archived,
                search_term: params.search_term,
                limit: params.limit.unwrap_or(20),
                cursor: params.cursor,
            })?,
        })
    }

    pub async fn compact_thread(&self, params: ThreadCompactParams) -> Result<ThreadCompactResult> {
        // ── 1. Read the full thread from persistence ──────────────────────────
        let persisted = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;

        let item_count_before = persisted.items.len();
        if item_count_before == 0 {
            return Ok(ThreadCompactResult {
                thread_id: params.thread_id,
                summary_item_id: None,
                items_removed: 0,
            });
        }

        // ── 2. Build system instructions (same as a normal turn) ───────────────
        let cwd = persisted
            .metadata
            .cwd
            .clone()
            .unwrap_or_else(|| self.inner.effective_config.working_directory.clone());
        let skills = self.inner.extension_manager.list_skills(&cwd).await;
        let mut system_instructions = vec![self.inner.effective_config.system_prompt.clone()];
        for skill in skills {
            if skill.enabled {
                system_instructions.push(format!("Skill {}: {}", skill.name, skill.description));
            }
        }

        // ── 3. Ask the LLM to summarise ────────────────────────────────────────
        // Use the thread's configured model for compaction.
        let model_settings = ModelSettings {
            model_id: persisted.metadata.model_id.clone(),
            provider_id: persisted.metadata.provider_id.clone(),
            reasoning_effort: None, // no extended thinking needed for summaries
            summary_mode: None,
            service_tier: None,
            web_search_enabled: false,
            context_window: self.inner.effective_config.model_settings.context_window,
        };

        let compaction_messages = build_compaction_messages(&system_instructions, &persisted.items);

        let llm_response = self
            .inner
            .model_gateway
            .inner_client()
            .complete(
                &model_settings.provider_id,
                &compaction_messages,
                &[], // no tools needed for summarisation
                &model_settings.model_id,
                0.2,
                None,
                2048,
            )
            .await
            .map_err(|e| anyhow!("compact_thread_llm_failed: {e}"))?;

        let summary_text = if llm_response.content.trim().is_empty() {
            "[Compaction summary unavailable]".to_string()
        } else {
            llm_response.content.trim().to_string()
        };

        // ── 4. Tombstone all existing items ────────────────────────────────────
        let items_removed = self
            .inner
            .persistence_manager
            .tombstone_all_items(&params.thread_id)? as i32;

        // ── 5. Insert the summary as a single ReasoningSummary item ────────────
        let summary_item_id = format!("item_{}", Uuid::new_v4().simple());
        let summary_item = ConversationItem {
            item_id: summary_item_id.clone(),
            thread_id: params.thread_id.clone(),
            turn_id: "compaction".into(),
            created_at: now_seconds(),
            kind: ItemKind::ReasoningSummary,
            payload: json!({
                "text": summary_text,
                "strategy": params.strategy,
                "items_compacted": items_removed,
            }),
        };
        self.inner.persistence_manager.append_item(&summary_item)?;

        // ── 6. Reload the in-memory session so the live thread sees the summary ─
        {
            let mut loaded = self.inner.loaded_threads.write().await;
            loaded.remove(&params.thread_id);
        }
        // Warm the cache again so the next turn picks up the new state.
        let _ = self.load_thread(&params.thread_id).await;

        self.publish_event(
            RuntimeEventKind::ItemCompleted,
            Some(params.thread_id.clone()),
            Some(summary_item.turn_id.clone()),
            serde_json::to_value(&summary_item)?,
        )
        .await?;

        Ok(ThreadCompactResult {
            thread_id: params.thread_id,
            summary_item_id: Some(summary_item_id),
            items_removed,
        })
    }

    pub async fn rollback_thread(
        &self,
        params: ThreadRollbackParams,
    ) -> Result<ThreadRollbackResult> {
        let persisted = self
            .inner
            .persistence_manager
            .read_thread(&params.thread_id)?;
        let turns_removed = persisted
            .turns
            .iter()
            .filter(|turn| {
                turn.created_at
                    > persisted
                        .turns
                        .iter()
                        .find(|target| target.turn_id == params.to_turn_id)
                        .map(|t| t.created_at)
                        .unwrap_or(i64::MAX)
            })
            .count() as i32;
        Ok(ThreadRollbackResult {
            thread_id: params.thread_id,
            turns_removed,
        })
    }

    pub async fn set_thread_memory_mode(&self, params: ThreadMemoryModeParams) -> Result<()> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        {
            let mut thread = thread.lock().await;
            thread.metadata.memory_mode = params.memory_mode;
            thread.metadata.updated_at = now_seconds();
            self.inner
                .persistence_manager
                .update_thread(&thread.metadata)?;
        }
        Ok(())
    }

    pub async fn set_thread_model_settings(&self, params: ThreadModelSettingsParams) -> Result<()> {
        let thread = self
            .load_thread(&params.thread_id)
            .await?
            .ok_or_else(|| anyhow!("thread_not_found: {}", params.thread_id))?;
        {
            let mut thread = thread.lock().await;
            thread.metadata.model_id = params.model_id;
            thread.metadata.provider_id = params.provider_id;
            thread.metadata.updated_at = now_seconds();
            self.inner
                .persistence_manager
                .update_thread(&thread.metadata)?;
        }
        Ok(())
    }

    /// Update the live approval-policy override for a running thread so that
    /// mid-turn policy changes (e.g. toggling auto-approve) take effect on the
    /// next tool-call evaluation without restarting the turn.
    pub async fn set_thread_approval_policy(
        &self,
        thread_id: &str,
        policy: Option<ApprovalPolicy>,
    ) -> Result<()> {
        if let Some(thread) = self.load_thread(thread_id).await? {
            let mut guard = thread.lock().await;
            guard.approval_policy_override = policy;
        }
        Ok(())
    }

    pub async fn archive_thread(&self, thread_id: &str) -> Result<()> {
        self.inner.persistence_manager.archive_thread(thread_id)?;
        if let Some(thread) = self.load_thread(thread_id).await? {
            let mut thread = thread.lock().await;
            thread.metadata.archived = true;
            thread.metadata.status = ThreadStatus::Archived;
            thread.metadata.updated_at = now_seconds();
        }
        Ok(())
    }

    pub async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.inner.loaded_threads.write().await.remove(thread_id);
        self.inner.persistence_manager.delete_thread(thread_id)?;
        Ok(())
    }

    pub(crate) async fn load_thread(
        &self,
        thread_id: &str,
    ) -> Result<Option<Arc<Mutex<ThreadSession>>>> {
        if let Some(thread) = self
            .inner
            .loaded_threads
            .read()
            .await
            .get(thread_id)
            .cloned()
        {
            return Ok(Some(thread));
        }
        if let Ok(persisted) = self.inner.persistence_manager.read_thread(thread_id) {
            let thread = self.load_persisted_thread(&persisted).await?;
            return Ok(Some(thread));
        }
        Ok(None)
    }

    async fn load_persisted_thread(
        &self,
        persisted: &PersistedThread,
    ) -> Result<Arc<Mutex<ThreadSession>>> {
        if let Some(thread) = self
            .inner
            .loaded_threads
            .read()
            .await
            .get(&persisted.metadata.thread_id)
            .cloned()
        {
            return Ok(thread);
        }

        let mut turns_map: HashMap<String, LoadedTurn> = persisted
            .turns
            .iter()
            .map(|turn| {
                (
                    turn.turn_id.clone(),
                    LoadedTurn {
                        metadata: turn.clone(),
                        items: Vec::new(),
                        status: turn.status,
                        pending_tool_calls: HashMap::new(),
                        stream_buffer: Vec::new(),
                        cancel_token: CancellationToken::new(),
                    },
                )
            })
            .collect();

        // Items whose turn_id does not match any real turn (e.g. the ReasoningSummary
        // written by compact_thread with turn_id "compaction") are collected into
        // prefix_items so the executor can prepend them to the model context.
        let mut prefix_items: Vec<ConversationItem> = Vec::new();
        for item in &persisted.items {
            if let Some(turn) = turns_map.get_mut(&item.turn_id) {
                turn.items.push(item.clone());
            } else {
                prefix_items.push(item.clone());
            }
        }

        let thread = Arc::new(Mutex::new(ThreadSession {
            metadata: persisted.metadata.clone(),
            turns: turns_map.into_values().collect(),
            active_turn_id: None,
            subscribed_clients: HashSet::new(),
            owner_connection_id: None,
            pending_steering: Vec::new(),
            approval_policy_override: None,
            prefix_items,
        }));
        self.inner
            .loaded_threads
            .write()
            .await
            .insert(persisted.metadata.thread_id.clone(), thread.clone());
        Ok(thread)
    }
}
