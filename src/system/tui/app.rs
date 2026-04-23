use anyhow::Result;
use std::collections::HashMap;

use super::super::domain::{
    ApprovalDecision, ApprovalResolution, ConversationItem, PendingApproval,
    RuntimeEvent, RuntimeEventKind, ThreadMetadata, TurnStatus, UserInput,
};
use super::super::runtime::{
    ConversationRuntime, ThreadForkParams, ThreadListParams, ThreadReadParams,
    ThreadResumeParams, ThreadStartParams, TurnInterruptParams, TurnStartParams, TurnSteerParams,
};
use super::types::{
    entry_from_item, thread_label, short_turn_id, ComposerState, EntryKind, FocusPane,
    PendingApprovalView, TranscriptEntry, THREAD_LIMIT,
};

/// Full application state for the interactive TUI.
pub struct InteractiveApp {
    pub runtime: ConversationRuntime,
    pub current_thread_id: String,
    pub current_thread_meta: Option<ThreadMetadata>,
    pub threads: Vec<ThreadMetadata>,
    pub selected_thread_id: Option<String>,
    pub transcript: Vec<TranscriptEntry>,
    pub transcript_index: HashMap<String, usize>,
    pub transcript_scroll: u16,
    pub auto_scroll: bool,
    pub focus: FocusPane,
    pub composer: ComposerState,
    pub pending_approval: Option<PendingApprovalView>,
    pub active_turn_id: Option<String>,
    pub status_message: String,
    pub error_message: Option<String>,
    pub should_quit: bool,
}

impl InteractiveApp {
    pub async fn bootstrap(runtime: ConversationRuntime, initial_thread_id: String) -> Result<Self> {
        let mut app = Self {
            runtime,
            current_thread_id: initial_thread_id.clone(),
            current_thread_meta: None,
            threads: Vec::new(),
            selected_thread_id: Some(initial_thread_id),
            transcript: Vec::new(),
            transcript_index: HashMap::new(),
            transcript_scroll: 0,
            auto_scroll: true,
            focus: FocusPane::Composer,
            composer: ComposerState::default(),
            pending_approval: None,
            active_turn_id: None,
            status_message: "Ready".into(),
            error_message: None,
            should_quit: false,
        };
        app.refresh_threads().await?;
        let current = app.current_thread_id.clone();
        app.load_thread(&current).await?;
        Ok(app)
    }

    pub async fn refresh_threads(&mut self) -> Result<()> {
        self.threads = self
            .runtime
            .list_threads(ThreadListParams {
                cwd: None,
                archived: Some(false),
                search_term: None,
                limit: Some(THREAD_LIMIT),
                cursor: None,
            })
            .await?
            .threads;

        if self.threads.is_empty() {
            self.selected_thread_id = None;
            return Ok(());
        }

        let fallback = self.current_thread_id.clone();
        let desired = self
            .selected_thread_id
            .clone()
            .unwrap_or_else(|| fallback.clone());
        if self.threads.iter().any(|t| t.thread_id == desired) {
            self.selected_thread_id = Some(desired);
        } else {
            self.selected_thread_id = Some(fallback);
        }
        Ok(())
    }

    pub async fn load_thread(&mut self, thread_id: &str) -> Result<()> {
        self.runtime
            .resume_thread(ThreadResumeParams {
                thread_id: thread_id.to_string(),
            })
            .await?;
        let persisted = self
            .runtime
            .read_thread(ThreadReadParams {
                thread_id: thread_id.to_string(),
            })
            .await?
            .thread;

        self.current_thread_id = thread_id.to_string();
        self.current_thread_meta = Some(persisted.metadata.clone());
        self.selected_thread_id = Some(thread_id.to_string());
        self.transcript.clear();
        self.transcript_index.clear();
        for item in &persisted.items {
            self.upsert_transcript_entry(entry_from_item(item));
        }
        self.active_turn_id = persisted
            .turns
            .iter()
            .rev()
            .find(|t| {
                matches!(
                    t.status,
                    TurnStatus::Created | TurnStatus::Running | TurnStatus::WaitingForApproval
                )
            })
            .map(|t| t.turn_id.clone());
        self.pending_approval = None;
        self.auto_scroll = true;
        self.error_message = None;
        self.status_message = format!("Opened {}", thread_label(&persisted.metadata));
        Ok(())
    }

    pub async fn create_new_thread(&mut self) -> Result<()> {
        let cwd = self
            .current_thread_meta
            .as_ref()
            .and_then(|t| t.cwd.clone())
            .or_else(|| {
                Some(
                    self.runtime
                        .effective_config()
                        .working_directory
                        .clone(),
                )
            });

        let created = self
            .runtime
            .start_thread(ThreadStartParams {
                cwd,
                model_settings: None,
                approval_policy: None,
                permission_profile: None,
                ephemeral: false,
            })
            .await?;
        self.refresh_threads().await?;
        self.load_thread(&created.metadata.thread_id).await?;
        self.status_message = format!("Created {}", thread_label(&created.metadata));
        self.error_message = None;
        Ok(())
    }

    pub async fn fork_current_thread(&mut self) -> Result<()> {
        let forked = self
            .runtime
            .fork_thread(ThreadForkParams {
                thread_id: self.current_thread_id.clone(),
                ephemeral: false,
            })
            .await?;
        self.refresh_threads().await?;
        self.load_thread(&forked.metadata.thread_id).await?;
        self.status_message = format!("Forked into {}", thread_label(&forked.metadata));
        self.error_message = None;
        Ok(())
    }

    pub async fn open_selected_thread(&mut self) -> Result<()> {
        if let Some(thread_id) = self.selected_thread_id.clone() {
            self.load_thread(&thread_id).await?;
        }
        Ok(())
    }

    pub fn next_focus(&mut self) {
        self.focus = match self.focus {
            FocusPane::Threads => FocusPane::Transcript,
            FocusPane::Transcript => FocusPane::Composer,
            FocusPane::Composer => FocusPane::Threads,
        };
    }

    pub fn previous_focus(&mut self) {
        self.focus = match self.focus {
            FocusPane::Threads => FocusPane::Composer,
            FocusPane::Transcript => FocusPane::Threads,
            FocusPane::Composer => FocusPane::Transcript,
        };
    }

    pub fn select_thread_delta(&mut self, delta: isize) {
        if self.threads.is_empty() {
            return;
        }
        let current_index = self
            .selected_thread_id
            .as_ref()
            .and_then(|selected| {
                self.threads
                    .iter()
                    .position(|t| &t.thread_id == selected)
            })
            .unwrap_or(0) as isize;
        let next = (current_index + delta).clamp(0, self.threads.len() as isize - 1) as usize;
        self.selected_thread_id = Some(self.threads[next].thread_id.clone());
    }

    pub async fn submit_composer(&mut self) -> Result<()> {
        let trimmed = self.composer.trimmed_text();
        if trimmed.is_empty() {
            return Ok(());
        }

        let raw = self.composer.take_text();
        if self.try_handle_slash_command(raw.trim()).await? {
            return Ok(());
        }

        if self.pending_approval.is_some() {
            self.error_message =
                Some("Resolve the approval request before sending more input".into());
            return Ok(());
        }

        self.error_message = None;
        if let Some(turn_id) = self.active_turn_id.clone() {
            self.runtime
                .steer_turn(TurnSteerParams {
                    thread_id: self.current_thread_id.clone(),
                    expected_turn_id: turn_id.clone(),
                    input: vec![UserInput::from_text(raw)],
                })
                .await?;
            self.status_message = format!("Queued input for {}", short_turn_id(&turn_id));
        } else {
            let turn = self
                .runtime
                .start_turn(
                    TurnStartParams {
                        thread_id: self.current_thread_id.clone(),
                        input: vec![UserInput::from_text(raw)],
                        cwd: None,
                        model_settings: None,
                        approval_policy: None,
                        permission_profile: None,
                        output_schema: None,
                    },
                    super::super::domain::SurfaceKind::Interactive,
                )
                .await?;
            self.active_turn_id = Some(turn.turn.turn_id.clone());
            self.status_message = format!("Started {}", short_turn_id(&turn.turn.turn_id));
        }
        self.auto_scroll = true;
        Ok(())
    }

    pub async fn try_handle_slash_command(&mut self, command: &str) -> Result<bool> {
        if !command.starts_with('/') {
            return Ok(false);
        }

        let handled = if matches!(command, "/quit" | "/exit") {
            self.should_quit = true;
            true
        } else if matches!(command, "/new") {
            self.create_new_thread().await?;
            true
        } else if matches!(command, "/fork") {
            self.fork_current_thread().await?;
            true
        } else if matches!(command, "/threads" | "/reload") {
            self.refresh_threads().await?;
            self.status_message = format!("Loaded {} threads", self.threads.len());
            self.error_message = None;
            true
        } else if matches!(command, "/interrupt") {
            self.interrupt_active_turn().await?;
            true
        } else if let Some(rest) = command.strip_prefix("/open ") {
            self.load_thread(rest.trim()).await?;
            true
        } else if let Some(rest) = command.strip_prefix("/resume ") {
            self.load_thread(rest.trim()).await?;
            true
        } else if matches!(command, "/help") {
            self.status_message =
                "Keys: Tab switch pane, Ctrl+N new, Ctrl+F fork, Ctrl+C interrupt, Ctrl+Q quit"
                    .into();
            self.error_message = None;
            true
        } else {
            self.error_message = Some(format!("Unknown command: {command}"));
            true
        };

        Ok(handled)
    }

    pub async fn interrupt_active_turn(&mut self) -> Result<()> {
        let Some(turn_id) = self.active_turn_id.clone() else {
            self.status_message = "No active turn to interrupt".into();
            self.error_message = None;
            return Ok(());
        };
        self.runtime
            .interrupt_turn(TurnInterruptParams {
                thread_id: self.current_thread_id.clone(),
                turn_id: turn_id.clone(),
            })
            .await?;
        self.status_message = format!("Interrupt requested for {}", short_turn_id(&turn_id));
        self.error_message = None;
        Ok(())
    }

    pub async fn resolve_pending_approval(&mut self, decision: ApprovalDecision) -> Result<()> {
        let Some(approval) = self.pending_approval.clone() else {
            return Ok(());
        };
        self.runtime
            .resolve_approval(&approval.approval.request.approval_id, decision, None)
            .await?;
        self.status_message = format!(
            "{decision:?} {}",
            approval.approval.request.title.to_lowercase()
        );
        self.error_message = None;
        self.pending_approval = None;
        Ok(())
    }

    pub async fn handle_runtime_event(&mut self, event: RuntimeEvent) -> Result<()> {
        match event.kind {
            RuntimeEventKind::ThreadStarted
            | RuntimeEventKind::TurnStarted
            | RuntimeEventKind::TurnCompleted
            | RuntimeEventKind::TurnFailed => {
                self.refresh_threads().await?;
            }
            _ => {}
        }

        if event.thread_id.as_deref() != Some(self.current_thread_id.as_str()) {
            return Ok(());
        }

        match event.kind {
            RuntimeEventKind::ThreadStarted => {
                if let Ok(metadata) =
                    serde_json::from_value::<ThreadMetadata>(event.payload.clone())
                {
                    self.current_thread_meta = Some(metadata.clone());
                    self.status_message = format!("Started {}", thread_label(&metadata));
                    self.error_message = None;
                }
            }
            RuntimeEventKind::TurnStarted => {
                self.active_turn_id = event.turn_id.clone();
                self.status_message = "Thinking…".into();
                self.error_message = None;
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Turn",
                    "Assistant is preparing a response",
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::ItemStarted => self.handle_item_started(&event)?,
            RuntimeEventKind::ItemUpdated => self.handle_item_updated(&event),
            RuntimeEventKind::ItemCompleted => self.handle_item_completed(&event)?,
            RuntimeEventKind::ApprovalRequested => {
                let pending = serde_json::from_value::<PendingApproval>(event.payload.clone())?;
                let preview = serde_json::to_string_pretty(&pending.request.action)
                    .unwrap_or_else(|_| pending.request.action.to_string());
                self.pending_approval = Some(PendingApprovalView {
                    approval: pending.clone(),
                    action_preview: preview,
                });
                self.status_message = format!("Approval required: {}", pending.request.title);
                self.error_message = None;
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Approval",
                    &pending.request.justification,
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::ApprovalResolved => {
                let resolution =
                    serde_json::from_value::<ApprovalResolution>(event.payload.clone())?;
                self.pending_approval = None;
                self.status_message = format!("Approval {:?}", resolution.decision);
                self.error_message = None;
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Status,
                    "Approval",
                    &format!("Decision: {:?}", resolution.decision),
                    Some(event.emitted_at / 1000),
                );
            }
            RuntimeEventKind::TurnCompleted => {
                self.active_turn_id = None;
                self.pending_approval = None;
                if let Some(thread) = self.current_thread_meta.as_mut() {
                    thread.status = super::super::domain::ThreadStatus::Idle;
                }
                self.status_message = "Ready".into();
                self.error_message = None;
            }
            RuntimeEventKind::TurnFailed => {
                self.active_turn_id = None;
                self.pending_approval = None;
                let reason = event
                    .payload
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("turn failed");
                self.error_message = Some(reason.to_string());
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Error,
                    "Error",
                    reason,
                    Some(event.emitted_at / 1000),
                );
                if let Some(thread) = self.current_thread_meta.as_mut() {
                    thread.status = super::super::domain::ThreadStatus::Idle;
                }
            }
            RuntimeEventKind::Warning | RuntimeEventKind::Disconnected => {
                let text = serde_json::to_string(&event.payload).unwrap_or_else(|_| "{}".into());
                self.push_status_entry(
                    event.event_id,
                    EntryKind::Error,
                    "Runtime",
                    &text,
                    Some(event.emitted_at / 1000),
                );
            }
        }

        Ok(())
    }

    fn handle_item_started(&mut self, event: &RuntimeEvent) -> Result<()> {
        use anyhow::anyhow;
        let item_id = event
            .payload
            .get("itemId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing itemId"))?;
        let kind = event
            .payload
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("AGENT_MESSAGE");

        let entry_kind = match kind {
            "AGENT_MESSAGE" => EntryKind::Assistant,
            "REASONING_TEXT" | "REASONING_SUMMARY" => EntryKind::Reasoning,
            _ => EntryKind::Status,
        };
        let title = match entry_kind {
            EntryKind::Assistant => "Codezilla",
            EntryKind::Reasoning => "Reasoning",
            _ => "Runtime",
        };
        self.upsert_transcript_entry(TranscriptEntry {
            item_id: item_id.to_string(),
            kind: entry_kind,
            title: title.into(),
            body: String::new(),
            timestamp: Some(event.emitted_at / 1000),
            pending: true,
        });
        self.status_message = "Streaming response…".into();
        self.auto_scroll = true;
        Ok(())
    }

    fn handle_item_updated(&mut self, event: &RuntimeEvent) {
        let Some(item_id) = event.payload.get("itemId").and_then(|v| v.as_str()) else {
            return;
        };
        let delta = event
            .payload
            .get("delta")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if let Some(index) = self.transcript_index.get(item_id).copied() {
            self.transcript[index].body.push_str(delta);
        } else {
            self.upsert_transcript_entry(TranscriptEntry {
                item_id: item_id.to_string(),
                kind: EntryKind::Assistant,
                title: "Codezilla".into(),
                body: delta.to_string(),
                timestamp: Some(event.emitted_at / 1000),
                pending: true,
            });
        }
        self.status_message = "Streaming response…".into();
        self.auto_scroll = true;
    }

    fn handle_item_completed(&mut self, event: &RuntimeEvent) -> Result<()> {
        let item = serde_json::from_value::<ConversationItem>(event.payload.clone())?;
        self.upsert_transcript_entry(entry_from_item(&item));
        self.auto_scroll = true;
        Ok(())
    }

    pub fn upsert_transcript_entry(&mut self, entry: TranscriptEntry) {
        if let Some(index) = self.transcript_index.get(&entry.item_id).copied() {
            self.transcript[index] = entry;
        } else {
            let index = self.transcript.len();
            self.transcript_index.insert(entry.item_id.clone(), index);
            self.transcript.push(entry);
        }
    }

    pub fn push_status_entry(
        &mut self,
        item_id: String,
        kind: EntryKind,
        title: &str,
        body: &str,
        timestamp: Option<i64>,
    ) {
        self.upsert_transcript_entry(TranscriptEntry {
            item_id,
            kind,
            title: title.into(),
            body: body.into(),
            timestamp,
            pending: false,
        });
        self.auto_scroll = true;
    }
}
