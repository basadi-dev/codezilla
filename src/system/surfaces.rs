use anyhow::{bail, Result};
use std::io::{self, Write};

use super::domain::{OutputMode, RuntimeEventKind, SurfaceKind, UserInput};
use super::runtime::{
    ConversationRuntime, EventFilter, ThreadForkParams, ThreadResumeParams, ThreadStartParams,
    TurnStartParams,
};
use super::tui::run_interactive_tui;

#[derive(Debug, Clone)]
pub struct InteractiveInvocation {
    pub prompt: Option<String>,
    pub resume_thread_id: Option<String>,
    pub fork_thread_id: Option<String>,
    pub cwd: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExecInvocation {
    pub prompt: Option<String>,
    pub stdin_suffix: Option<String>,
    #[allow(dead_code)]
    pub output_mode: OutputMode,
    #[allow(dead_code)]
    pub output_last_message_path: Option<String>,
    pub ephemeral: bool,
    pub cwd: Option<String>,
    pub thread_id: Option<String>,
}

pub struct InteractiveSurface {
    runtime: ConversationRuntime,
}

impl InteractiveSurface {
    pub fn new(runtime: ConversationRuntime) -> Self {
        Self { runtime }
    }

    pub async fn start(&self, invocation: InteractiveInvocation) -> Result<i32> {
        let thread_id = if let Some(thread_id) = invocation.resume_thread_id {
            self.runtime
                .resume_thread(ThreadResumeParams {
                    thread_id: thread_id.clone(),
                })
                .await?;
            thread_id
        } else if let Some(thread_id) = invocation.fork_thread_id {
            self.runtime
                .fork_thread(ThreadForkParams {
                    thread_id,
                    ephemeral: false,
                })
                .await?
                .metadata
                .thread_id
        } else {
            self.runtime
                .start_thread(ThreadStartParams {
                    cwd: invocation.cwd.clone(),
                    model_settings: None,
                    approval_policy: None,
                    permission_profile: None,
                    ephemeral: false,
                })
                .await?
                .metadata
                .thread_id
        };

        run_interactive_tui(self.runtime.clone(), thread_id, invocation.prompt).await
    }
}

pub struct ExecSurface {
    runtime: ConversationRuntime,
    output_mode: OutputMode,
    write_last_message_path: Option<String>,
}

impl ExecSurface {
    pub fn new(
        runtime: ConversationRuntime,
        output_mode: OutputMode,
        write_last_message_path: Option<String>,
    ) -> Self {
        Self {
            runtime,
            output_mode,
            write_last_message_path,
        }
    }

    pub async fn start(&self, invocation: ExecInvocation) -> Result<i32> {
        let prompt = resolve_exec_prompt(invocation.prompt, invocation.stdin_suffix)?;
        let thread_id = if let Some(thread_id) = invocation.thread_id {
            self.runtime
                .resume_thread(ThreadResumeParams {
                    thread_id: thread_id.clone(),
                })
                .await?;
            thread_id
        } else {
            self.runtime
                .start_thread(ThreadStartParams {
                    cwd: invocation.cwd,
                    model_settings: None,
                    approval_policy: None,
                    permission_profile: None,
                    ephemeral: invocation.ephemeral,
                })
                .await?
                .metadata
                .thread_id
        };

        let subscriber_id = format!("exec_{}", uuid::Uuid::new_v4().simple());
        let mut subscription = self.runtime.event_bus().subscribe(
            subscriber_id.clone(),
            EventFilter {
                thread_id: Some(thread_id.clone()),
            },
        );

        let turn = self
            .runtime
            .start_turn(
                TurnStartParams {
                    thread_id: thread_id.clone(),
                    input: vec![UserInput::from_text(prompt)],
                    cwd: None,
                    model_settings: None,
                    approval_policy: None,
                    permission_profile: None,
                    output_schema: None,
                },
                SurfaceKind::Exec,
            )
            .await?;

        let mut last_message = String::new();
        let exit_code = loop {
            match subscription.receiver.recv().await {
                Ok(event) => {
                    if event.thread_id.as_deref() != Some(thread_id.as_str()) {
                        continue;
                    }
                    match self.output_mode {
                        OutputMode::Jsonl => {
                            println!("{}", serde_json::to_string(&event)?);
                        }
                        OutputMode::Human => match event.kind {
                            RuntimeEventKind::ItemUpdated => {
                                if let Some(delta) =
                                    event.payload.get("delta").and_then(|v| v.as_str())
                                {
                                    print!("{delta}");
                                    io::stdout().flush()?;
                                    last_message.push_str(delta);
                                }
                            }
                            RuntimeEventKind::ApprovalRequested => {
                                bail!(
                                    "approval required in exec mode for turn {}",
                                    turn.turn.turn_id
                                );
                            }
                            RuntimeEventKind::TurnCompleted => {
                                if !matches!(self.output_mode, OutputMode::Jsonl) {
                                    println!();
                                }
                                break 0;
                            }
                            RuntimeEventKind::TurnFailed => {
                                if let Some(reason) =
                                    event.payload.get("reason").and_then(|v| v.as_str())
                                {
                                    eprintln!("Error: {reason}");
                                }
                                break 1;
                            }
                            _ => {}
                        },
                    }
                }
                Err(_) => break 1,
            }
        };

        if let Some(path) = &self.write_last_message_path {
            std::fs::write(path, last_message)?;
        }

        self.runtime.event_bus().unsubscribe(&subscriber_id);
        Ok(exit_code)
    }
}

fn resolve_exec_prompt(prompt: Option<String>, stdin_suffix: Option<String>) -> Result<String> {
    match (prompt, stdin_suffix) {
        (Some(prompt), Some(stdin)) if !stdin.trim().is_empty() => {
            Ok(format!("{prompt}\n\n{stdin}"))
        }
        (Some(prompt), _) => Ok(prompt),
        (None, Some(stdin)) if !stdin.trim().is_empty() => Ok(stdin),
        _ => bail!("exec requires a prompt via argument or stdin"),
    }
}
