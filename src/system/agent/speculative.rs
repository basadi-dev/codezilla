//! Speculative execution: launch N parallel read-only candidate sub-agents,
//! then use a judge model pass to select the best approach for implementation.
//!
//! This module implements the "tournament" pattern for agentic coding.
//! Candidates are read-only scouts that explore the codebase and produce
//! structured implementation plans. The judge is a single non-agentic model
//! call that compares plans and picks the winner.

use anyhow::{anyhow, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
use uuid::Uuid;

use super::supervisor::{AgentSupervisor, ChildAgentRequest, TurnCompletionOutcome};
use crate::system::config::AgentConfig;
use crate::system::domain::{
    ApprovalPolicy, ApprovalPolicyKind, CandidateSolution, JudgeRanking, JudgeVerdict,
    PermissionProfile, RuntimeEventKind, SandboxMode, SpeculativeResult, ThreadId, TurnId,
};
use crate::system::runtime::ConversationRuntime;

// ─── SpeculativeOrchestrator ──────────────────────────────────────────────────

pub(crate) struct SpeculativeOrchestrator {
    runtime: ConversationRuntime,
    supervisor: AgentSupervisor,
    candidates_count: usize,
    candidate_timeout_secs: u64,
}

impl SpeculativeOrchestrator {
    pub fn new(runtime: ConversationRuntime, config: &AgentConfig) -> Self {
        let supervisor = AgentSupervisor::new(
            runtime.clone(),
            config.max_concurrent_child_agents(),
        );
        Self {
            runtime,
            supervisor,
            candidates_count: config.speculative_candidates,
            candidate_timeout_secs: config.speculative_candidate_timeout_secs,
        }
    }

    /// Run the full speculative execution pipeline:
    /// 1. Spawn N candidate read-only exploration agents in parallel
    /// 2. Collect their structured plans
    /// 3. Run a judge model pass to select the best approach
    ///
    /// Returns a boxed future to break the recursive opaque type cycle:
    /// `run_turn -> speculative::run -> supervisor::run_child -> start_turn -> run_turn`
    pub fn run(
        &self,
        task: &str,
        parent_thread_id: &ThreadId,
        parent_turn_id: &TurnId,
        cwd: &str,
        agent_depth: u32,
    ) -> Pin<Box<dyn Future<Output = Result<SpeculativeResult>> + Send + '_>> {
        // Clone everything we need into owned values for the async block.
        let task = task.to_string();
        let parent_thread_id = parent_thread_id.clone();
        let parent_turn_id = parent_turn_id.clone();
        let cwd = cwd.to_string();

        Box::pin(async move {
            self.run_inner(&task, &parent_thread_id, &parent_turn_id, &cwd, agent_depth)
                .await
        })
    }

    async fn run_inner(
        &self,
        task: &str,
        parent_thread_id: &ThreadId,
        parent_turn_id: &TurnId,
        cwd: &str,
        agent_depth: u32,
    ) -> Result<SpeculativeResult> {
        let n = self.candidates_count;

        tracing::info!(
            parent_thread_id = %parent_thread_id,
            parent_turn_id = %parent_turn_id,
            candidates = n,
            timeout_secs = self.candidate_timeout_secs,
            "speculative: starting parallel candidate exploration"
        );

        // Publish lifecycle event so the TUI can show speculative phase status.
        let _ = self
            .runtime
            .publish_event(
                RuntimeEventKind::SpeculativeJudgeStarted,
                Some(parent_thread_id.clone()),
                Some(parent_turn_id.clone()),
                json!({
                    "candidates": n,
                    "status": "spawning_candidates",
                }),
            )
            .await;

        // ── 1. Spawn N candidates in parallel ─────────────────────────────────
        let mut handles = Vec::with_capacity(n);
        for i in 0..n {
            let prompt = candidate_agent_prompt(task, i + 1, n);

            // Announce each candidate spawn.
            let _ = self
                .runtime
                .publish_event(
                    RuntimeEventKind::SpeculativeCandidateStarted,
                    Some(parent_thread_id.clone()),
                    Some(parent_turn_id.clone()),
                    json!({
                        "candidateIndex": i,
                        "totalCandidates": n,
                    }),
                )
                .await;

            let supervisor = self.supervisor.clone();
            let cwd = cwd.to_string();
            let parent_thread_id = parent_thread_id.clone();
            let parent_turn_id = parent_turn_id.clone();
            let timeout = self.candidate_timeout_secs;
            let depth = agent_depth;

            let child_future = tokio::spawn(async move {
                supervisor.run_child(ChildAgentRequest {
                    prompt,
                    cwd,
                    // Read-only: no approvals needed, sandbox locked to ReadOnly.
                    approval_policy: ApprovalPolicy {
                        kind: ApprovalPolicyKind::Never,
                        granular: None,
                    },
                    permission_profile: PermissionProfile {
                        sandbox_mode: SandboxMode::ReadOnly,
                        writable_roots: Vec::new(),
                        network_enabled: false,
                        allowed_domains: Vec::new(),
                        allowed_unix_sockets: Vec::new(),
                    },
                    timeout_secs: timeout,
                    agent_depth: depth + 1,
                    parent_thread_id,
                    parent_turn_id,
                    parent_tool_call_id: format!("spec_candidate_{i}"),
                }).await
            });

            handles.push((i, child_future));
        }

        // ── 2. Await all candidates ───────────────────────────────────────────
        let mut candidates: Vec<CandidateSolution> = Vec::with_capacity(n);
        for (i, handle) in handles {
            let t0 = std::time::Instant::now();
            match handle.await {
                Ok(Ok(run)) => {
                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                    let approach_label = extract_approach_label(&run.result_text);
                    let files_examined = extract_files_examined(&run.result_text);
                    let estimated_complexity = extract_complexity(&run.result_text);

                    let outcome_label = match &run.outcome {
                        TurnCompletionOutcome::Completed => "completed",
                        TurnCompletionOutcome::TimedOut => "timed_out",
                        TurnCompletionOutcome::Failed(_) => "failed",
                        TurnCompletionOutcome::Interrupted => "interrupted",
                    };

                    tracing::info!(
                        candidate = i,
                        approach = %approach_label,
                        outcome = outcome_label,
                        elapsed_ms,
                        "speculative: candidate finished"
                    );

                    // Only include candidates that produced meaningful output.
                    let plan_text = run.result_text.trim().to_string();
                    if !plan_text.is_empty()
                        && !matches!(
                            run.outcome,
                            TurnCompletionOutcome::Failed(_) | TurnCompletionOutcome::Interrupted
                        )
                    {
                        let candidate = CandidateSolution {
                            candidate_id: format!("candidate_{i}"),
                            agent_thread_id: run.child_thread_id,
                            approach_label,
                            plan_text,
                            files_examined,
                            estimated_complexity,
                            elapsed_ms,
                        };

                        let _ = self
                            .runtime
                            .publish_event(
                                RuntimeEventKind::SpeculativeCandidateCompleted,
                                Some(parent_thread_id.clone()),
                                Some(parent_turn_id.clone()),
                                json!({
                                    "candidateIndex": i,
                                    "approachLabel": &candidate.approach_label,
                                    "elapsedMs": elapsed_ms,
                                    "status": outcome_label,
                                }),
                            )
                            .await;

                        candidates.push(candidate);
                    } else {
                        tracing::warn!(
                            candidate = i,
                            outcome = outcome_label,
                            "speculative: candidate produced no usable plan, discarding"
                        );
                    }
                }
                Ok(Err(e)) => {
                    tracing::warn!(candidate = i, error = %e, "speculative: candidate agent error");
                }
                Err(e) => {
                    tracing::warn!(candidate = i, error = %e, "speculative: candidate task panicked");
                }
            }
        }

        if candidates.is_empty() {
            return Err(anyhow!(
                "speculative execution: all {n} candidate agents failed to produce a plan"
            ));
        }

        // If only one candidate survived, skip the judge — it wins by default.
        if candidates.len() == 1 {
            tracing::info!("speculative: only 1 candidate survived, skipping judge");
            let winner = &candidates[0];
            let verdict = JudgeVerdict {
                selected_candidate_id: winner.candidate_id.clone(),
                rationale: "Only one candidate produced a usable plan.".into(),
                ranking: vec![JudgeRanking {
                    candidate_id: winner.candidate_id.clone(),
                    score: 1.0,
                    strengths: "Only viable approach.".into(),
                    weaknesses: String::new(),
                }],
            };
            return Ok(SpeculativeResult {
                candidates,
                verdict,
            });
        }

        // ── 3. Run judge evaluation ───────────────────────────────────────────
        let _ = self
            .runtime
            .publish_event(
                RuntimeEventKind::SpeculativeJudgeStarted,
                Some(parent_thread_id.clone()),
                Some(parent_turn_id.clone()),
                json!({
                    "status": "judging",
                    "candidateCount": candidates.len(),
                }),
            )
            .await;

        let verdict = self.run_judge(task, &candidates, parent_thread_id, parent_turn_id).await?;

        let _ = self
            .runtime
            .publish_event(
                RuntimeEventKind::SpeculativeJudgeCompleted,
                Some(parent_thread_id.clone()),
                Some(parent_turn_id.clone()),
                json!({
                    "selectedCandidateId": &verdict.selected_candidate_id,
                    "rationale": &verdict.rationale,
                    "candidateCount": candidates.len(),
                }),
            )
            .await;

        tracing::info!(
            selected = %verdict.selected_candidate_id,
            rationale = %verdict.rationale,
            "speculative: judge selected winner"
        );

        Ok(SpeculativeResult {
            candidates,
            verdict,
        })
    }

    /// Run the judge as a sub-agent that evaluates candidate plans and returns
    /// a structured selection. The judge itself is a read-only agent that only
    /// needs to process text — no tool access needed.
    async fn run_judge(
        &self,
        task: &str,
        candidates: &[CandidateSolution],
        parent_thread_id: &ThreadId,
        parent_turn_id: &TurnId,
    ) -> Result<JudgeVerdict> {
        let prompt = judge_prompt(task, candidates);

        let run = self
            .supervisor
            .run_child(ChildAgentRequest {
                prompt,
                cwd: ".".to_string(),
                approval_policy: ApprovalPolicy {
                    kind: ApprovalPolicyKind::Never,
                    granular: None,
                },
                permission_profile: PermissionProfile {
                    sandbox_mode: SandboxMode::ReadOnly,
                    writable_roots: Vec::new(),
                    network_enabled: false,
                    allowed_domains: Vec::new(),
                    allowed_unix_sockets: Vec::new(),
                },
                timeout_secs: 60, // Judge should be fast — just comparing text plans
                agent_depth: 1,   // Judge doesn't need to spawn children
                parent_thread_id: parent_thread_id.clone(),
                parent_turn_id: parent_turn_id.clone(),
                parent_tool_call_id: format!("spec_judge_{}", Uuid::new_v4().simple()),
            })
            .await?;

        parse_judge_verdict(&run.result_text, candidates)
    }
}

// ─── Prompt templates ─────────────────────────────────────────────────────────

fn candidate_agent_prompt(task: &str, candidate_index: usize, total: usize) -> String {
    format!(
        "You are candidate {candidate_index} of {total} in a solution exploration phase.\n\n\
         ## Task\n{task}\n\n\
         ## Your Role\n\
         Explore the codebase and produce a DETAILED IMPLEMENTATION PLAN. \
         Do NOT make any file edits — this is a read-only exploration phase. \
         Your plan will be evaluated against other candidates' plans by an independent judge.\n\n\
         ## Required Output Format\n\
         Structure your response exactly like this:\n\n\
         **Approach Label**: [2-5 word name for your approach]\n\n\
         **Summary**: [2-3 sentence overview]\n\n\
         **Files to Change**:\n\
         - `path/to/file.rs` — [what changes are needed, be specific about line numbers]\n\
         - ...\n\n\
         **Implementation Steps**:\n\
         1. [Concrete step with specific code changes]\n\
         2. ...\n\n\
         **Risk Assessment**: [What could go wrong, edge cases to watch]\n\n\
         **Estimated Complexity**: [low / medium / high]\n\n\
         Be as concrete as possible — reference specific functions, line numbers, and exact code \
         changes. The more concrete your plan, the higher it will score."
    )
}

fn judge_prompt(task: &str, candidates: &[CandidateSolution]) -> String {
    let mut prompt = format!(
        "You are an expert code reviewer judging {} candidate approaches to a coding task.\n\n\
         ## Original Task\n{task}\n\n\
         ## Candidate Solutions\n\n",
        candidates.len()
    );

    for (i, c) in candidates.iter().enumerate() {
        prompt.push_str(&format!(
            "### Candidate {} — \"{}\"\n\
             - Files examined: {}\n\
             - Self-assessed complexity: {}\n\
             - Time taken: {}ms\n\n\
             {}\n\n---\n\n",
            i + 1,
            c.approach_label,
            if c.files_examined.is_empty() {
                "(none reported)".to_string()
            } else {
                c.files_examined.join(", ")
            },
            c.estimated_complexity,
            c.elapsed_ms,
            c.plan_text,
        ));
    }

    prompt.push_str(
        "## Your Task\n\
         Evaluate each candidate on these criteria:\n\
         1. **Correctness** — Will this approach actually solve the problem?\n\
         2. **Completeness** — Does it cover edge cases and error handling?\n\
         3. **Simplicity** — Is it the least complex path that still works?\n\
         4. **Risk** — What's the chance of introducing regressions?\n\n\
         Return your verdict in this exact format:\n\n\
         SELECTED: [candidate number, e.g. 1]\n\
         RATIONALE: [2-3 sentences explaining why this approach is best]\n\
         \n\
         RANKING:\n\
         - Candidate [N]: score=[0.0-1.0] strengths=[...] weaknesses=[...]\n\
         - Candidate [N]: score=[0.0-1.0] strengths=[...] weaknesses=[...]\n\n\
         Be decisive. Pick the approach most likely to succeed with minimal risk.",
    );

    prompt
}

// ─── Output parsing ───────────────────────────────────────────────────────────

/// Extract the approach label from candidate output.
fn extract_approach_label(text: &str) -> String {
    // Look for "**Approach Label**: ..." or "Approach Label: ..."
    for line in text.lines() {
        let trimmed = line.trim();
        let stripped = trimmed
            .trim_start_matches("**")
            .trim_start_matches('*');
        if let Some(rest) = stripped
            .strip_prefix("Approach Label")
            .or_else(|| stripped.strip_prefix("approach label"))
        {
            let label = rest
                .trim_start_matches("**")
                .trim_start_matches(':')
                .trim_start_matches("**:")
                .trim();
            if !label.is_empty() {
                return label.chars().take(60).collect();
            }
        }
    }
    "Unnamed approach".to_string()
}

/// Extract file paths mentioned in the candidate's plan.
fn extract_files_examined(text: &str) -> Vec<String> {
    let mut files = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Match backtick-quoted paths like `src/foo.rs`
        let mut rest = trimmed;
        while let Some(start) = rest.find('`') {
            let after = &rest[start + 1..];
            if let Some(end) = after.find('`') {
                let path = &after[..end];
                if path.contains('/')
                    && !path.contains(' ')
                    && path.len() < 200
                    && !files.contains(&path.to_string())
                {
                    files.push(path.to_string());
                }
                rest = &after[end + 1..];
            } else {
                break;
            }
        }
    }
    files
}

/// Extract the self-assessed complexity from candidate output.
fn extract_complexity(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    for line in lower.lines() {
        if line.contains("complexity") || line.contains("estimated complexity") {
            if line.contains("high") {
                return "high".to_string();
            }
            if line.contains("medium") {
                return "medium".to_string();
            }
            if line.contains("low") {
                return "low".to_string();
            }
        }
    }
    "medium".to_string() // default
}

/// Parse the judge's text output into a structured `JudgeVerdict`.
/// Designed to be lenient — models don't always follow the exact format.
fn parse_judge_verdict(
    text: &str,
    candidates: &[CandidateSolution],
) -> Result<JudgeVerdict> {
    let lower = text.to_ascii_lowercase();

    // Extract selected candidate number
    let selected_num = extract_selected_number(&lower).unwrap_or(1);
    let selected_idx = (selected_num as usize).saturating_sub(1).min(candidates.len() - 1);
    let selected_candidate_id = candidates[selected_idx].candidate_id.clone();

    // Extract rationale
    let rationale = extract_rationale(text).unwrap_or_else(|| {
        format!(
            "Candidate {} was selected as the best approach.",
            selected_num
        )
    });

    // Build ranking (best-effort parsing)
    let mut ranking: Vec<JudgeRanking> = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let score = if i == selected_idx {
            0.9
        } else {
            extract_candidate_score(&lower, i + 1).unwrap_or(0.5)
        };
        ranking.push(JudgeRanking {
            candidate_id: c.candidate_id.clone(),
            score,
            strengths: String::new(),
            weaknesses: String::new(),
        });
    }

    Ok(JudgeVerdict {
        selected_candidate_id,
        rationale,
        ranking,
    })
}

fn extract_selected_number(lower: &str) -> Option<u32> {
    // Try "SELECTED: 2" or "selected: 2" or "I select candidate 2"
    for line in lower.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("selected") {
            // Extract first digit after the colon
            if let Some(rest) = trimmed.split_once(':') {
                let digits: String = rest
                    .1
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(n) = digits.parse::<u32>() {
                    if n > 0 {
                        return Some(n);
                    }
                }
            }
        }
        // Also try "candidate N is the best" or "I recommend candidate N"
        if (trimmed.contains("candidate") && trimmed.contains("best"))
            || trimmed.contains("recommend candidate")
            || trimmed.contains("select candidate")
        {
            let digits: String = trimmed
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = digits.parse::<u32>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }
    }
    None
}

fn extract_rationale(text: &str) -> Option<String> {
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("rationale") {
            let rest = trimmed.split_once(':').map(|(_, v)| v.trim())?;
            if !rest.is_empty() {
                // Collect multi-line rationale (up to next section header)
                let mut rationale = rest.to_string();
                for next_line in text.lines().skip(i + 1) {
                    let next_trimmed = next_line.trim();
                    if next_trimmed.is_empty()
                        || next_trimmed.starts_with("RANKING")
                        || next_trimmed.starts_with("ranking")
                        || next_trimmed.starts_with("- Candidate")
                        || next_trimmed.starts_with("- candidate")
                    {
                        break;
                    }
                    rationale.push(' ');
                    rationale.push_str(next_trimmed);
                }
                return Some(rationale);
            }
        }
    }
    None
}

fn extract_candidate_score(lower: &str, candidate_num: usize) -> Option<f32> {
    let marker = format!("candidate {candidate_num}");
    for line in lower.lines() {
        if line.contains(&marker) && line.contains("score") {
            // Try to extract "score=0.85" or "score: 0.85"
            if let Some(pos) = line.find("score") {
                let after = &line[pos + 5..];
                let digits: String = after
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                if let Ok(score) = digits.parse::<f32>() {
                    return Some(score.clamp(0.0, 1.0));
                }
            }
        }
    }
    None
}

/// Build the system instruction text to inject the winning plan into the
/// main executor's context. This replaces the normal Orient → Plan phase
/// with a pre-evaluated plan.
pub(crate) fn build_speculative_plan_instruction(result: &SpeculativeResult) -> String {
    let winner = result
        .candidates
        .iter()
        .find(|c| c.candidate_id == result.verdict.selected_candidate_id);

    let plan_text = winner
        .map(|w| w.plan_text.as_str())
        .unwrap_or("[plan text unavailable]");

    let approach = winner
        .map(|w| w.approach_label.as_str())
        .unwrap_or("Unknown");

    format!(
        "## Pre-evaluated Implementation Plan\n\
         The following approach was selected from {} candidates by an independent \
         evaluation pass. Execute this plan directly — do NOT re-explore alternatives \
         or second-guess the approach. Focus on precise implementation.\n\n\
         **Selected approach**: \"{approach}\" (score: {:.0}%)\n\
         **Judge rationale**: {}\n\n\
         ---\n\n\
         {plan_text}\n\n\
         ---\n\n\
         Execute the above plan step by step. Start with the first file change immediately.",
        result.candidates.len(),
        winner
            .and_then(|w| result.verdict.ranking.iter().find(|r| r.candidate_id == w.candidate_id))
            .map(|r| r.score * 100.0)
            .unwrap_or(90.0),
        result.verdict.rationale,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_approach_label_from_markdown() {
        let text = "**Approach Label**: Caching with TTL\n\n**Summary**: blah blah";
        assert_eq!(extract_approach_label(text), "Caching with TTL");
    }

    #[test]
    fn extract_approach_label_missing_falls_back() {
        assert_eq!(
            extract_approach_label("Just some random text"),
            "Unnamed approach"
        );
    }

    #[test]
    fn extract_files_from_backticks() {
        let text = "Change `src/main.rs` and `src/lib.rs` to add the feature.";
        let files = extract_files_examined(text);
        assert_eq!(files, vec!["src/main.rs", "src/lib.rs"]);
    }

    #[test]
    fn extract_complexity_high() {
        let text = "**Estimated Complexity**: high\nThis is risky.";
        assert_eq!(extract_complexity(text), "high");
    }

    #[test]
    fn parse_judge_selects_correct_candidate() {
        let text = "SELECTED: 2\nRATIONALE: Candidate 2 is more thorough.\n\nRANKING:\n- Candidate 1: score=0.6 strengths=simple weaknesses=incomplete\n- Candidate 2: score=0.9 strengths=thorough weaknesses=complex";
        let candidates = vec![
            CandidateSolution {
                candidate_id: "candidate_0".into(),
                agent_thread_id: "t0".into(),
                approach_label: "Simple".into(),
                plan_text: "plan a".into(),
                files_examined: vec![],
                estimated_complexity: "low".into(),
                elapsed_ms: 100,
            },
            CandidateSolution {
                candidate_id: "candidate_1".into(),
                agent_thread_id: "t1".into(),
                approach_label: "Thorough".into(),
                plan_text: "plan b".into(),
                files_examined: vec![],
                estimated_complexity: "medium".into(),
                elapsed_ms: 200,
            },
        ];
        let verdict = parse_judge_verdict(text, &candidates).unwrap();
        assert_eq!(verdict.selected_candidate_id, "candidate_1");
        assert!(verdict.rationale.contains("thorough"));
    }

    #[test]
    fn single_candidate_skips_judge() {
        // Verified at the orchestrator level, but the plan builder should handle
        // a single-candidate result cleanly.
        let result = SpeculativeResult {
            candidates: vec![CandidateSolution {
                candidate_id: "candidate_0".into(),
                agent_thread_id: "t0".into(),
                approach_label: "Only option".into(),
                plan_text: "Do the thing".into(),
                files_examined: vec!["src/main.rs".into()],
                estimated_complexity: "low".into(),
                elapsed_ms: 50,
            }],
            verdict: JudgeVerdict {
                selected_candidate_id: "candidate_0".into(),
                rationale: "Only one candidate.".into(),
                ranking: vec![JudgeRanking {
                    candidate_id: "candidate_0".into(),
                    score: 1.0,
                    strengths: "Only option".into(),
                    weaknesses: String::new(),
                }],
            },
        };
        let instruction = build_speculative_plan_instruction(&result);
        assert!(instruction.contains("Do the thing"));
        assert!(instruction.contains("1 candidates"));
    }
}
