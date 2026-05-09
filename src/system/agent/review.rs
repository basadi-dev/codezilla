//! Checkpoint review: lightweight sequential review at natural boundaries.
//!
//! After each "logical unit of work" (a tool round that produces file changes),
//! the executor can optionally fire a checkpoint review — a single non-agentic
//! model call that validates the changes against the plan/task. Unlike a parallel
//! coder+reviewer architecture, this avoids synchronization overhead entirely:
//! the review runs only at natural pauses between tool rounds.
//!
//! The reviewer produces a structured `CheckpointReviewVerdict` containing:
//!   - `approved`: whether the change looks correct
//!   - `issues`: concrete problems found (if any)
//!   - `suggestions`: optional improvements
//!
//! If `approved` is false, the verdict is injected as a system message into the
//! executor's next iteration, giving the coder agent immediate, actionable
//! feedback without breaking its context or flow.

use anyhow::Result;
use serde_json::json;

use crate::system::config::AgentConfig;
use crate::system::domain::{
    CheckpointReviewVerdict, FileChangeSummary, ReviewIssue, RuntimeEventKind, ThreadId, TurnId,
};
use crate::system::runtime::ConversationRuntime;

// ─── CheckpointReviewer ───────────────────────────────────────────────────────

pub(crate) struct CheckpointReviewer {
    runtime: ConversationRuntime,
    /// Maximum combined diff chars to include in the review prompt.
    /// Larger diffs are summarized to stay within a reasonable token budget.
    max_diff_chars: usize,
}

impl CheckpointReviewer {
    pub fn new(runtime: ConversationRuntime, config: &AgentConfig) -> Self {
        Self {
            runtime,
            max_diff_chars: config.checkpoint_review_max_diff_chars,
        }
    }

    /// Run a checkpoint review on the file changes produced in the latest
    /// tool round. This is a single non-agentic model call — no tool access,
    /// no multi-turn conversation.
    ///
    /// Returns `None` if the changes are approved (no issues), or
    /// `Some(verdict)` if the reviewer found problems.
    pub async fn review(
        &self,
        task_context: &str,
        plan_steps: &[String],
        completed_actions: &[String],
        new_file_changes: &[FileChangeSummary],
        thread_id: &ThreadId,
        turn_id: &TurnId,
    ) -> Result<CheckpointReviewVerdict> {
        // Publish lifecycle event so the TUI shows review status.
        let _ = self
            .runtime
            .publish_event(
                RuntimeEventKind::CheckpointReviewStarted,
                Some(thread_id.clone()),
                Some(turn_id.clone()),
                json!({
                    "filesReviewed": new_file_changes.iter().map(|f| &f.path).collect::<Vec<_>>(),
                }),
            )
            .await;

        let prompt = self.build_review_prompt(
            task_context,
            plan_steps,
            completed_actions,
            new_file_changes,
        );

        // Use a single non-streaming complete() call for speed. The reviewer
        // doesn't need tool access or multi-turn — just one shot.
        let model_settings = self.runtime.inner.effective_config.model_settings.clone();
        let messages = vec![crate::llm::Message::user(prompt)];

        let response = self
            .runtime
            .inner
            .model_gateway
            .inner_client()
            .complete(
                &model_settings.provider_id,
                &messages,
                &[], // No tools needed for review
                &model_settings.model_id,
                0.1,         // Low temperature for consistent review output
                Some("low"), // Low reasoning effort — fast review
                2048,
            )
            .await?;

        let verdict = parse_review_verdict(&response.content, new_file_changes);

        let _ = self
            .runtime
            .publish_event(
                RuntimeEventKind::CheckpointReviewCompleted,
                Some(thread_id.clone()),
                Some(turn_id.clone()),
                json!({
                    "approved": verdict.approved,
                    "issueCount": verdict.issues.len(),
                    "suggestionCount": verdict.suggestions.len(),
                }),
            )
            .await;

        tracing::info!(
            thread_id = %thread_id,
            turn_id = %turn_id,
            approved = verdict.approved,
            issues = verdict.issues.len(),
            suggestions = verdict.suggestions.len(),
            "checkpoint_review: completed"
        );

        Ok(verdict)
    }

    fn build_review_prompt(
        &self,
        task_context: &str,
        plan_steps: &[String],
        completed_actions: &[String],
        file_changes: &[FileChangeSummary],
    ) -> String {
        let mut prompt = String::with_capacity(4096);

        prompt.push_str(
            "You are a code reviewer performing a checkpoint review during an agentic coding session.\n\n",
        );

        // Task context
        prompt.push_str("## Original Task\n");
        prompt.push_str(task_context);
        prompt.push_str("\n\n");

        // Plan (if available)
        if !plan_steps.is_empty() {
            prompt.push_str("## Current Plan\n");
            for (i, step) in plan_steps.iter().enumerate() {
                prompt.push_str(&format!("{}. {}\n", i + 1, step));
            }
            prompt.push('\n');
        }

        // Progress
        if !completed_actions.is_empty() {
            prompt.push_str("## Actions Completed So Far\n");
            for action in completed_actions {
                prompt.push_str(&format!("- {action}\n"));
            }
            prompt.push('\n');
        }

        // File changes (the meat of the review)
        prompt.push_str("## Changes to Review\n\n");
        let mut total_diff_chars = 0;
        for fc in file_changes {
            prompt.push_str(&format!(
                "### `{}` ({}, +{} -{} lines)\n",
                fc.path, fc.kind, fc.lines_added, fc.lines_removed
            ));

            if !fc.diff.is_empty() {
                let remaining = self.max_diff_chars.saturating_sub(total_diff_chars);
                if remaining > 0 {
                    let diff_to_show = if fc.diff.len() > remaining {
                        let truncated: String = fc.diff.chars().take(remaining).collect();
                        format!(
                            "{truncated}\n... [diff truncated, {} more chars]",
                            fc.diff.len() - remaining
                        )
                    } else {
                        fc.diff.clone()
                    };
                    total_diff_chars += diff_to_show.len();
                    prompt.push_str("```diff\n");
                    prompt.push_str(&diff_to_show);
                    prompt.push_str("\n```\n\n");
                } else {
                    prompt.push_str("[diff omitted — review budget exceeded]\n\n");
                }
            }
        }

        // Instructions
        prompt.push_str(
            "## Your Task\n\
             Review the above changes and report:\n\
             1. Are there any **bugs** introduced by these changes?\n\
             2. Are there any **logic errors** or **missing edge cases**?\n\
             3. Do the changes **align with the task** described above?\n\
             4. Are there any **style issues** that could cause problems?\n\n\
             Respond in this exact format:\n\n\
             APPROVED: [yes/no]\n\n\
             ISSUES:\n\
             - [issue description] (severity: [critical/warning/info]) (file: [path])\n\
             - ...\n\n\
             SUGGESTIONS:\n\
             - [optional improvement that is not blocking]\n\
             - ...\n\n\
             If there are no issues, write ISSUES: none\n\
             If there are no suggestions, write SUGGESTIONS: none\n\n\
             Be concise. Focus on correctness, not style preferences. \
             Only flag issues that would cause actual bugs or broken behavior.",
        );

        prompt
    }
}

// ─── Verdict parsing ──────────────────────────────────────────────────────────

fn parse_review_verdict(text: &str, file_changes: &[FileChangeSummary]) -> CheckpointReviewVerdict {
    let lower = text.to_ascii_lowercase();

    // Extract approved status
    let approved = extract_approved(&lower);

    // Extract issues
    let issues = extract_issues(text, file_changes);

    // Extract suggestions
    let suggestions = extract_suggestions(text);

    CheckpointReviewVerdict {
        approved,
        issues,
        suggestions,
    }
}

fn extract_approved(lower: &str) -> bool {
    for line in lower.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("approved") {
            if let Some(rest) = trimmed.split_once(':') {
                let value = rest.1.trim();
                return value.starts_with("yes")
                    || value.starts_with("true")
                    || value.starts_with("✓")
                    || value.starts_with("✅");
            }
        }
    }
    // Default to approved if we can't parse — fail open, not closed
    true
}

fn extract_issues(text: &str, file_changes: &[FileChangeSummary]) -> Vec<ReviewIssue> {
    let mut issues = Vec::new();
    let mut in_issues_section = false;

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();

        // Section detection
        if lower.starts_with("issues") {
            in_issues_section = true;
            if lower.contains("none") {
                return Vec::new();
            }
            continue;
        }
        if lower.starts_with("suggestions") || lower.starts_with("##") {
            in_issues_section = false;
            continue;
        }

        if in_issues_section && trimmed.starts_with('-') {
            let issue_text = trimmed.trim_start_matches('-').trim();
            if issue_text.is_empty() {
                continue;
            }

            let severity = if lower.contains("critical") {
                "critical"
            } else if lower.contains("warning") {
                "warning"
            } else {
                "info"
            }
            .to_string();

            // Try to extract file path from the issue
            let file = file_changes
                .iter()
                .find(|fc| lower.contains(&fc.path.to_ascii_lowercase()))
                .map(|fc| fc.path.clone())
                .unwrap_or_default();

            // Clean up the issue text — strip severity/file markers
            let description = issue_text
                .split("(severity:")
                .next()
                .unwrap_or(issue_text)
                .split("(file:")
                .next()
                .unwrap_or(issue_text)
                .trim()
                .to_string();

            issues.push(ReviewIssue {
                description,
                severity,
                file,
            });
        }
    }

    issues
}

fn extract_suggestions(text: &str) -> Vec<String> {
    let mut suggestions = Vec::new();
    let mut in_suggestions_section = false;

    for line in text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();

        if lower.starts_with("suggestions") {
            in_suggestions_section = true;
            if lower.contains("none") {
                return Vec::new();
            }
            continue;
        }
        if lower.starts_with("##") || (lower.starts_with("issues") && in_suggestions_section) {
            break;
        }

        if in_suggestions_section && trimmed.starts_with('-') {
            let suggestion = trimmed.trim_start_matches('-').trim();
            if !suggestion.is_empty() {
                suggestions.push(suggestion.to_string());
            }
        }
    }

    suggestions
}

/// Build a system message to inject the review feedback into the executor's
/// next iteration. Only called when the review found issues.
pub(crate) fn build_review_feedback_instruction(verdict: &CheckpointReviewVerdict) -> String {
    let mut instruction = String::with_capacity(1024);

    instruction.push_str(
        "## ⚠️ Checkpoint Review Feedback\n\
         A review of your recent file changes found issues that need to be addressed \
         before continuing.\n\n",
    );

    if !verdict.issues.is_empty() {
        instruction.push_str("### Issues Found\n");
        for (i, issue) in verdict.issues.iter().enumerate() {
            let file_ref = if issue.file.is_empty() {
                String::new()
            } else {
                format!(" in `{}`", issue.file)
            };
            instruction.push_str(&format!(
                "{}. **[{}]**{}: {}\n",
                i + 1,
                issue.severity.to_uppercase(),
                file_ref,
                issue.description,
            ));
        }
        instruction.push('\n');
    }

    if !verdict.suggestions.is_empty() {
        instruction.push_str("### Suggestions (optional)\n");
        for suggestion in &verdict.suggestions {
            instruction.push_str(&format!("- {suggestion}\n"));
        }
        instruction.push('\n');
    }

    let critical_count = verdict
        .issues
        .iter()
        .filter(|i| i.severity == "critical")
        .count();

    if critical_count > 0 {
        instruction.push_str(&format!(
            "**{critical_count} critical issue(s) found.** Fix these before proceeding \
             to the next step of the plan. Address them with targeted edits now.\n"
        ));
    } else {
        instruction.push_str(
            "Address the issues above with targeted edits, then continue with the plan.\n",
        );
    }

    instruction
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_approved_yes() {
        let text = "APPROVED: yes\n\nISSUES: none\n\nSUGGESTIONS: none";
        let verdict = parse_review_verdict(text, &[]);
        assert!(verdict.approved);
        assert!(verdict.issues.is_empty());
        assert!(verdict.suggestions.is_empty());
    }

    #[test]
    fn parse_approved_no_with_issues() {
        let text = "APPROVED: no\n\nISSUES:\n\
                    - Missing null check (severity: critical) (file: src/main.rs)\n\
                    - Unused variable (severity: warning) (file: src/lib.rs)\n\n\
                    SUGGESTIONS:\n\
                    - Consider adding a doc comment\n";

        let file_changes = vec![
            FileChangeSummary {
                path: "src/main.rs".into(),
                kind: "modify".into(),
                lines_added: 5,
                lines_removed: 2,
                diff: String::new(),
            },
            FileChangeSummary {
                path: "src/lib.rs".into(),
                kind: "modify".into(),
                lines_added: 3,
                lines_removed: 1,
                diff: String::new(),
            },
        ];

        let verdict = parse_review_verdict(text, &file_changes);
        assert!(!verdict.approved);
        assert_eq!(verdict.issues.len(), 2);
        assert_eq!(verdict.issues[0].severity, "critical");
        assert_eq!(verdict.issues[0].file, "src/main.rs");
        assert!(verdict.issues[0].description.contains("null check"));
        assert_eq!(verdict.issues[1].severity, "warning");
        assert_eq!(verdict.suggestions.len(), 1);
    }

    #[test]
    fn parse_defaults_to_approved_on_unparseable() {
        let text = "This response doesn't follow the format at all.";
        let verdict = parse_review_verdict(text, &[]);
        assert!(verdict.approved);
    }

    #[test]
    fn build_feedback_instruction_includes_critical_count() {
        let verdict = CheckpointReviewVerdict {
            approved: false,
            issues: vec![
                ReviewIssue {
                    description: "Missing bounds check".into(),
                    severity: "critical".into(),
                    file: "src/parser.rs".into(),
                },
                ReviewIssue {
                    description: "Unused import".into(),
                    severity: "info".into(),
                    file: "src/lib.rs".into(),
                },
            ],
            suggestions: vec!["Add tests".into()],
        };

        let instruction = build_review_feedback_instruction(&verdict);
        assert!(instruction.contains("1 critical issue(s) found"));
        assert!(instruction.contains("Missing bounds check"));
        assert!(instruction.contains("src/parser.rs"));
        assert!(instruction.contains("Add tests"));
    }

    #[test]
    fn build_feedback_instruction_no_critical() {
        let verdict = CheckpointReviewVerdict {
            approved: false,
            issues: vec![ReviewIssue {
                description: "Minor style issue".into(),
                severity: "warning".into(),
                file: String::new(),
            }],
            suggestions: vec![],
        };

        let instruction = build_review_feedback_instruction(&verdict);
        assert!(instruction.contains("Address the issues above"));
        assert!(!instruction.contains("critical issue(s) found"));
    }
}
