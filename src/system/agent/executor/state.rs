use std::collections::HashSet;

use crate::system::domain::{FileChangeSummary, ToolCall};

use super::utils::is_read_only_tool;

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
        if !self.files_changed.is_empty() {
            if successful_commands > 0 {
                self.verification = VerificationStatus::Passed;
            } else if failed_commands > 0 {
                self.verification = VerificationStatus::Failed;
            }
        }
    }
}
