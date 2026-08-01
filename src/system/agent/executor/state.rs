use std::collections::HashSet;

use crate::system::domain::{FileChangeSummary, ToolCall};

use super::utils::is_read_only_tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CommandPurpose {
    Explore,
    Build,
    Test,
    Lint,
    Format,
    Other,
}

impl CommandPurpose {
    pub(super) fn verifies(self) -> bool {
        matches!(self, Self::Build | Self::Test | Self::Lint | Self::Format)
    }
}

pub(super) fn command_purpose(call: &ToolCall) -> CommandPurpose {
    if !matches!(call.tool_name.as_str(), "bash_exec" | "shell_exec") {
        return CommandPurpose::Other;
    }
    let command = call
        .arguments
        .get("command")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .or_else(|| {
            call.arguments.get("argv").and_then(|value| {
                value.as_array().map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            })
        })
        .unwrap_or_default()
        .to_ascii_lowercase();

    if contains_command(
        &command,
        &[" test", "pytest", "cargo test", "go test", "swift test"],
    ) {
        CommandPurpose::Test
    } else if contains_command(
        &command,
        &[
            "clippy",
            " lint",
            "eslint",
            "ruff",
            "mypy",
            "golangci-lint",
            "diff --check",
        ],
    ) {
        CommandPurpose::Lint
    } else if contains_command(&command, &[" fmt", "format", "rustfmt", "prettier"]) {
        CommandPurpose::Format
    } else if contains_command(
        &command,
        &[
            " build",
            " compile",
            "cargo check",
            "xcodebuild",
            "tsc",
            "make check",
        ],
    ) {
        CommandPurpose::Build
    } else if contains_command(
        &command,
        &["git status", "git diff", " ls", "find ", "rg ", "grep "],
    ) {
        CommandPurpose::Explore
    } else {
        CommandPurpose::Other
    }
}

fn contains_command(command: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| command.contains(pattern))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnPhase {
    Orient,
    Plan,
    Execute,
    Verify,
}

impl TurnPhase {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Orient => "ORIENT",
            Self::Plan => "PLAN",
            Self::Execute => "EXECUTE",
            Self::Verify => "VERIFY",
        }
    }

    pub(super) fn after_tool_round(
        self,
        round_is_read_only: bool,
        verification: VerificationStatus,
    ) -> Self {
        match (self, round_is_read_only, verification) {
            (Self::Orient | Self::Plan, false, _) => Self::Execute,
            (Self::Execute, _, VerificationStatus::Passed) => Self::Verify,
            (Self::Verify, _, VerificationStatus::Pending | VerificationStatus::Failed) => {
                Self::Execute
            }
            _ => self,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum VerificationStatus {
    #[default]
    NotRequired,
    Pending,
    Passed,
    Failed,
}

#[derive(Debug, Default)]
pub(super) struct TurnEvidence {
    pub(super) files_read: HashSet<String>,
    pub(super) files_changed: HashSet<String>,
    pub(super) successful_commands: usize,
    pub(super) failed_commands: usize,
    pub(super) successful_verifications: usize,
    pub(super) failed_verifications: usize,
    pub(super) completed_tool_rounds: usize,
    pub(super) failed_tool_rounds: usize,
    pub(super) verification: VerificationStatus,
}

impl TurnEvidence {
    pub(super) fn observe_calls(&mut self, calls: &[ToolCall]) {
        for call in calls {
            if is_read_only_tool(&call.tool_name) {
                if let Some(path) = call.arguments.get("path").and_then(|v| v.as_str()) {
                    self.files_read.insert(path.to_string());
                }
            }
        }
    }

    pub(super) fn observe_round(
        &mut self,
        had_any_success: bool,
        file_changes: &[FileChangeSummary],
        successful_commands: usize,
        failed_commands: usize,
        successful_verifications: usize,
        failed_verifications: usize,
    ) {
        self.completed_tool_rounds += 1;
        if !had_any_success {
            self.failed_tool_rounds += 1;
        }

        if !file_changes.is_empty() {
            self.files_changed
                .extend(file_changes.iter().map(|change| change.path.clone()));
            self.verification = VerificationStatus::Pending;
        }

        self.successful_commands += successful_commands;
        self.failed_commands += failed_commands;
        self.successful_verifications += successful_verifications;
        self.failed_verifications += failed_verifications;
        if !self.files_changed.is_empty() {
            if successful_verifications > 0 {
                self.verification = VerificationStatus::Passed;
                // A successful check resolves earlier failure pressure so an
                // automatic profile can return from high to its task baseline.
                self.failed_commands = 0;
                self.failed_tool_rounds = 0;
            } else if failed_verifications > 0 {
                self.verification = VerificationStatus::Failed;
            }
        }
    }
}
