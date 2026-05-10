//! Strategy pattern for exploration/planning.
// GreedyStrategy is public API; SpeculativeStrategy is wired in executor.rs.
#![allow(dead_code)]
//!
//! The `ExplorationStrategy` trait generalizes how the agent decides on an
//! approach before entering the main tool-execution loop. The current
//! speculative execution (tournament of candidates + judge) becomes one
//! strategy implementation; a trivial "greedy" pass-through is the default.
//!
//! Adding a new strategy (Tree-of-Thought, iterative refinement, etc.)
//! requires only implementing this trait — no changes to the executor loop.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::system::domain::{ThreadId, TurnId, UserInput};
use crate::system::runtime::ConversationRuntime;

use super::executor::utils::TurnIntent;

// ─── ExplorationResult ────────────────────────────────────────────────────────

/// Returned by an exploration strategy after it finishes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationResult {
    /// Instruction to inject into the system prompt for the main loop.
    /// If `None`, the executor proceeds with normal Orient→Plan phases.
    pub plan_instruction: Option<String>,
    /// Optional metadata for observability (persisted as a conversation item).
    #[serde(default)]
    pub metadata: Value,
}

// ─── ExplorationStrategy trait ────────────────────────────────────────────────

/// Defines how the agent explores solution space before committing to a plan.
///
/// Strategies run once per turn, before the main executor loop. They receive
/// the user's task and return an optional plan that the executor injects into
/// the model's context.
#[async_trait]
pub trait ExplorationStrategy: Send + Sync {
    /// A unique name for logging and observability.
    fn name(&self) -> &str;

    /// Whether this strategy should activate for the given turn.
    ///
    /// Called early in `run_turn` — if no strategy activates, the executor
    /// falls through to the default greedy behaviour (Orient → Plan → Execute).
    fn should_activate(&self, intent: TurnIntent, input: &[UserInput], agent_depth: u32) -> bool;

    /// Run the exploration/planning phase.
    ///
    /// Returns an `ExplorationResult` whose `plan_instruction` is injected
    /// into the system prompt if present. The `metadata` field is persisted
    /// for observability.
    async fn explore(
        &self,
        task: &str,
        runtime: &ConversationRuntime,
        thread_id: &ThreadId,
        turn_id: &TurnId,
        cwd: &str,
        agent_depth: u32,
    ) -> Result<ExplorationResult>;
}

// ─── GreedyStrategy ───────────────────────────────────────────────────────────

/// The default "do nothing" strategy — passes through to the executor's
/// built-in Orient → Plan → Execute → Verify phase machine.
pub struct GreedyStrategy;

#[async_trait]
impl ExplorationStrategy for GreedyStrategy {
    fn name(&self) -> &str {
        "greedy"
    }

    fn should_activate(&self, _intent: TurnIntent, _input: &[UserInput], _depth: u32) -> bool {
        // Never activates — the executor's default behaviour IS the greedy strategy.
        false
    }

    async fn explore(
        &self,
        _task: &str,
        _runtime: &ConversationRuntime,
        _thread_id: &ThreadId,
        _turn_id: &TurnId,
        _cwd: &str,
        _agent_depth: u32,
    ) -> Result<ExplorationResult> {
        Ok(ExplorationResult {
            plan_instruction: None,
            metadata: Value::Null,
        })
    }
}

// ─── SpeculativeStrategy ──────────────────────────────────────────────────────

/// Wraps the existing `SpeculativeOrchestrator` as an `ExplorationStrategy`.
///
/// This is the tournament-style approach: spawn N parallel single-shot
/// candidates, have a judge evaluate them, and inject the winning plan
/// into the executor's context.
pub struct SpeculativeStrategy {
    pub candidates: usize,
    pub auto_for_edit_debug: bool,
}

impl SpeculativeStrategy {
    pub fn from_config(config: &crate::system::config::AgentConfig) -> Self {
        Self {
            candidates: config.speculative_candidates,
            auto_for_edit_debug: config.speculative_auto,
        }
    }
}

#[async_trait]
impl ExplorationStrategy for SpeculativeStrategy {
    fn name(&self) -> &str {
        "speculative"
    }

    fn should_activate(&self, intent: TurnIntent, input: &[UserInput], agent_depth: u32) -> bool {
        // Disabled if fewer than 2 candidates configured
        if self.candidates < 2 {
            return false;
        }

        // Don't speculate inside child agents (avoid recursive speculation storms)
        if agent_depth > 0 {
            return false;
        }

        // Explicit trigger: /speculate prefix in user input
        let explicit = input.iter().any(|i| {
            i.text
                .as_ref()
                .map(|t| t.text.trim_start().starts_with("/speculate"))
                .unwrap_or(false)
        });
        if explicit {
            return true;
        }

        // Auto-mode: only for complex intents
        if self.auto_for_edit_debug {
            return matches!(intent, TurnIntent::Edit | TurnIntent::Debug);
        }

        false
    }

    async fn explore(
        &self,
        task: &str,
        runtime: &ConversationRuntime,
        thread_id: &ThreadId,
        turn_id: &TurnId,
        cwd: &str,
        agent_depth: u32,
    ) -> Result<ExplorationResult> {
        let agent_cfg = &runtime.inner.effective_config.agent;
        let orchestrator =
            super::speculative::SpeculativeOrchestrator::new(runtime.clone(), agent_cfg);

        let result = orchestrator
            .run(task, thread_id, turn_id, cwd, agent_depth)
            .await?;

        let plan_instruction = super::speculative::build_speculative_plan_instruction(&result);

        let metadata = serde_json::json!({
            "summary": format!(
                "[Speculative execution: {} candidates explored, \
                 \"{}\" selected (score {:.0}%)]\\n\\n{}",
                result.candidates.len(),
                result.verdict.selected_candidate_id,
                result.verdict.ranking.iter()
                    .find(|r| r.candidate_id == result.verdict.selected_candidate_id)
                    .map(|r| r.score * 100.0)
                    .unwrap_or(90.0),
                result.verdict.rationale,
            ),
            "candidates": result.candidates,
            "verdict": result.verdict,
        });

        Ok(ExplorationResult {
            plan_instruction: Some(plan_instruction),
            metadata,
        })
    }
}

// ─── TreeOfThoughtStrategy ────────────────────────────────────────────────────

/// Structured multi-step exploration strategy.
///
/// Differs from `SpeculativeStrategy` in three ways:
/// 1. **Branching prompt** asks the model to generate N distinct approaches
///    with explicit trade-off analysis (pros, cons, risks, file impact).
/// 2. **Pruning step** filters out approaches that look clearly infeasible
///    before expanding them — avoids wasting inference on dead ends.
/// 3. **Deepening step** expands surviving branches with concrete first
///    implementation steps before the final judge call.
///
/// Best used for complex multi-file refactors or ambiguous tasks where
/// the right approach is not obvious upfront.
pub struct TreeOfThoughtStrategy {
    /// Number of branches to generate in the first pass (default 3).
    pub branches: usize,
    /// Number of surviving branches after pruning (default 2).
    pub survivors: usize,
    /// Auto-activate for complex intents at depth 0.
    pub auto_for_complex: bool,
}

impl TreeOfThoughtStrategy {
    pub fn from_config(config: &crate::system::config::AgentConfig) -> Self {
        Self {
            branches: config.speculative_candidates.max(2),
            survivors: 2,
            auto_for_complex: config.speculative_auto,
        }
    }
}

#[async_trait]
impl ExplorationStrategy for TreeOfThoughtStrategy {
    fn name(&self) -> &str {
        "tree_of_thought"
    }

    fn should_activate(&self, intent: TurnIntent, input: &[UserInput], agent_depth: u32) -> bool {
        if agent_depth > 0 {
            return false;
        }
        // Explicit trigger via /tot prefix
        let explicit = input.iter().any(|i| {
            i.text
                .as_ref()
                .map(|t| t.text.trim_start().starts_with("/tot"))
                .unwrap_or(false)
        });
        if explicit {
            return true;
        }
        if self.auto_for_complex {
            return matches!(intent, TurnIntent::Edit | TurnIntent::Debug);
        }
        false
    }

    async fn explore(
        &self,
        task: &str,
        runtime: &ConversationRuntime,
        _thread_id: &ThreadId,
        _turn_id: &TurnId,
        _cwd: &str,
        _agent_depth: u32,
    ) -> Result<ExplorationResult> {
        let orchestrator =
            TreeOfThoughtOrchestrator::new(runtime.clone(), self.branches, self.survivors);
        let result = orchestrator.run(task).await?;

        let metadata = serde_json::json!({
            "summary": format!(
                "[Tree-of-Thought: {} branches explored, {} survived pruning, \
                 \"{}\" selected]\n\n{}",
                result.branches_explored,
                result.survivors_after_pruning,
                result.winner_label,
                result.rationale,
            ),
            "branches_explored": result.branches_explored,
            "survivors_after_pruning": result.survivors_after_pruning,
            "winner_label": &result.winner_label,
        });

        Ok(ExplorationResult {
            plan_instruction: Some(result.plan_instruction),
            metadata,
        })
    }
}

// ─── TreeOfThoughtOrchestrator ────────────────────────────────────────────────

struct TotResult {
    branches_explored: usize,
    survivors_after_pruning: usize,
    winner_label: String,
    plan_instruction: String,
    rationale: String,
}

struct TreeOfThoughtOrchestrator {
    runtime: ConversationRuntime,
    branches: usize,
    survivors: usize,
}

impl TreeOfThoughtOrchestrator {
    fn new(runtime: ConversationRuntime, branches: usize, survivors: usize) -> Self {
        Self {
            runtime,
            branches,
            survivors,
        }
    }

    async fn run(&self, task: &str) -> Result<TotResult> {
        let client = self.runtime.inner.model_gateway.inner_client().clone();
        let settings = &self.runtime.inner.effective_config.model_settings;
        let provider = settings.provider_id.clone();
        let model = settings.model_id.clone();

        // ── Step 1: Generate N branches ───────────────────────────────────────
        tracing::info!(
            branches = self.branches,
            "tree_of_thought: generating branches"
        );
        let branch_prompt = tot_branch_prompt(task, self.branches);
        let branch_response = client
            .complete(
                &provider,
                &[crate::llm::Message::user(branch_prompt)],
                &[],
                &model,
                0.5,
                Some("medium"),
                4096,
            )
            .await
            .map_err(|e| anyhow::anyhow!("tot_branch_failed: {e}"))?;

        let branches_text = branch_response.content.trim().to_string();
        if branches_text.is_empty() {
            return Err(anyhow::anyhow!(
                "tot: branch generation produced empty response"
            ));
        }

        // ── Step 2: Prune — keep the top K survivors ──────────────────────────
        tracing::info!(
            survivors = self.survivors,
            "tree_of_thought: pruning branches"
        );
        let prune_prompt = tot_prune_prompt(task, &branches_text, self.survivors);
        let prune_response = client
            .complete(
                &provider,
                &[crate::llm::Message::user(prune_prompt)],
                &[],
                &model,
                0.3,
                Some("medium"),
                3072,
            )
            .await
            .map_err(|e| anyhow::anyhow!("tot_prune_failed: {e}"))?;
        let pruned_text = prune_response.content.trim().to_string();

        // ── Step 3: Deepen — expand each survivor with concrete first steps ───
        tracing::info!("tree_of_thought: deepening survivors");
        let deepen_prompt = tot_deepen_prompt(task, &pruned_text);
        let deepen_response = client
            .complete(
                &provider,
                &[crate::llm::Message::user(deepen_prompt)],
                &[],
                &model,
                0.3,
                Some("high"),
                5120,
            )
            .await
            .map_err(|e| anyhow::anyhow!("tot_deepen_failed: {e}"))?;
        let deepened_text = deepen_response.content.trim().to_string();

        // ── Step 4: Judge — select the best approach ──────────────────────────
        tracing::info!("tree_of_thought: judge selecting winner");
        let judge_prompt = tot_judge_prompt(task, &deepened_text);
        let judge_response = client
            .complete(
                &provider,
                &[crate::llm::Message::user(judge_prompt)],
                &[],
                &model,
                0.2,
                Some("medium"),
                2048,
            )
            .await
            .map_err(|e| anyhow::anyhow!("tot_judge_failed: {e}"))?;
        let judgment = judge_response.content.trim().to_string();

        let winner_label = extract_tot_winner_label(&judgment);
        let rationale = extract_tot_rationale(&judgment);
        let plan_instruction =
            build_tot_plan_instruction(task, &deepened_text, &winner_label, &rationale);

        Ok(TotResult {
            branches_explored: self.branches,
            survivors_after_pruning: self.survivors,
            winner_label,
            plan_instruction,
            rationale,
        })
    }
}

// ─── Prompt helpers ───────────────────────────────────────────────────────────

fn tot_branch_prompt(task: &str, n: usize) -> String {
    format!(
        "You are a senior software architect. Analyse this coding task and generate exactly \
         {n} DISTINCT implementation approaches.\n\n\
         TASK:\n{task}\n\n\
         For each approach write:\n\
         APPROACH <N>: <one-line label>\n\
         PROS: <2-3 bullet points>\n\
         CONS: <2-3 bullet points>\n\
         RISKS: <1-2 bullet points>\n\
         FILES AFFECTED: <comma-separated list of likely files/modules>\n\
         COMPLEXITY: low | medium | high\n\n\
         Separate each approach with '---'. \
         Make the approaches genuinely distinct — different algorithms, different \
         refactoring strategies, or different scopes."
    )
}

fn tot_prune_prompt(task: &str, branches: &str, keep: usize) -> String {
    format!(
        "You are reviewing implementation approaches for this task:\n\
         TASK:\n{task}\n\n\
         CANDIDATE APPROACHES:\n{branches}\n\n\
         Select the {keep} most promising approaches. Discard any that are:\n\
         - Clearly over-engineered for the task scope\n\
         - Likely to introduce regressions or break existing tests\n\
         - Dependent on infrastructure not present in the codebase\n\n\
         Output ONLY the text of the {keep} surviving approaches, in the same \
         format as the input. Do not add commentary outside the approach blocks."
    )
}

fn tot_deepen_prompt(task: &str, survivors: &str) -> String {
    format!(
        "You are expanding implementation approaches into concrete action plans.\n\n\
         TASK:\n{task}\n\n\
         SURVIVING APPROACHES:\n{survivors}\n\n\
         For each approach, add a FIRST STEPS section:\n\
         FIRST STEPS:\n\
         1. <concrete first action, e.g. 'Read src/foo.rs lines 1-50 to understand X'>\n\
         2. <second action>\n\
         3. <third action>\n\n\
         Keep first steps concrete and immediately actionable. \
         Each step should be something a developer can do right now."
    )
}

fn tot_judge_prompt(task: &str, deepened: &str) -> String {
    format!(
        "You are a technical lead choosing the best implementation plan.\n\n\
         TASK:\n{task}\n\n\
         EXPANDED APPROACHES:\n{deepened}\n\n\
         Select the single best approach. Your response must contain:\n\
         SELECTED: <exact approach label from above>\n\
         RATIONALE: <2-3 sentences explaining why this approach is best>\n\n\
         Consider: correctness, minimal blast radius, testability, and how \
         directly the first steps address the task."
    )
}

fn extract_tot_winner_label(judgment: &str) -> String {
    judgment
        .lines()
        .find(|l| l.trim_start().starts_with("SELECTED:"))
        .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().to_string()))
        .unwrap_or_else(|| "Approach 1".to_string())
}

fn extract_tot_rationale(judgment: &str) -> String {
    let mut in_rationale = false;
    let mut lines = Vec::new();
    for line in judgment.lines() {
        if line.trim_start().starts_with("RATIONALE:") {
            in_rationale = true;
            let after = line.split_once(':').map(|(_, v)| v.trim()).unwrap_or("");
            if !after.is_empty() {
                lines.push(after.to_string());
            }
        } else if in_rationale {
            lines.push(line.to_string());
        }
    }
    lines.join(" ").trim().to_string()
}

fn build_tot_plan_instruction(task: &str, deepened: &str, winner: &str, rationale: &str) -> String {
    // Extract the winning approach block from the deepened text
    let winner_block = deepened
        .split("---")
        .find(|block| block.contains(winner))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| deepened.to_string());

    format!(
        "## Tree-of-Thought Selected Approach\n\n\
         **Task**: {task}\n\n\
         **Selected**: {winner}\n\
         **Rationale**: {rationale}\n\n\
         **Full plan**:\n{winner_block}\n\n\
         Execute the FIRST STEPS in order. Do not deviate from this plan \
         unless a step reveals a critical blocker."
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::UserInput;

    #[test]
    fn greedy_never_activates() {
        let strategy = GreedyStrategy;
        assert!(!strategy.should_activate(TurnIntent::Edit, &[], 0));
        assert!(!strategy.should_activate(TurnIntent::Answer, &[], 0));
        assert!(!strategy.should_activate(TurnIntent::Debug, &[], 0));
    }

    #[test]
    fn speculative_activates_for_edit_when_auto() {
        let strategy = SpeculativeStrategy {
            candidates: 3,
            auto_for_edit_debug: true,
        };
        assert!(strategy.should_activate(TurnIntent::Edit, &[], 0));
        assert!(strategy.should_activate(TurnIntent::Debug, &[], 0));
        assert!(!strategy.should_activate(TurnIntent::Answer, &[], 0));
    }

    #[test]
    fn speculative_disabled_at_depth() {
        let strategy = SpeculativeStrategy {
            candidates: 3,
            auto_for_edit_debug: true,
        };
        assert!(!strategy.should_activate(TurnIntent::Edit, &[], 1));
    }

    #[test]
    fn speculative_disabled_with_few_candidates() {
        let strategy = SpeculativeStrategy {
            candidates: 1,
            auto_for_edit_debug: true,
        };
        assert!(!strategy.should_activate(TurnIntent::Edit, &[], 0));
    }

    #[test]
    fn speculative_explicit_trigger() {
        let strategy = SpeculativeStrategy {
            candidates: 3,
            auto_for_edit_debug: false,
        };
        let input = vec![UserInput::from_text("/speculate fix the bug")];
        assert!(strategy.should_activate(TurnIntent::Answer, &input, 0));
    }

    #[test]
    fn tot_activates_with_explicit_trigger() {
        let strategy = TreeOfThoughtStrategy {
            branches: 3,
            survivors: 2,
            auto_for_complex: false,
        };
        let input = vec![UserInput::from_text("/tot refactor the auth module")];
        assert!(strategy.should_activate(TurnIntent::Edit, &input, 0));
    }

    #[test]
    fn tot_activates_for_edit_when_auto() {
        let strategy = TreeOfThoughtStrategy {
            branches: 3,
            survivors: 2,
            auto_for_complex: true,
        };
        assert!(strategy.should_activate(TurnIntent::Edit, &[], 0));
        assert!(strategy.should_activate(TurnIntent::Debug, &[], 0));
        assert!(!strategy.should_activate(TurnIntent::Answer, &[], 0));
    }

    #[test]
    fn tot_disabled_at_depth() {
        let strategy = TreeOfThoughtStrategy {
            branches: 3,
            survivors: 2,
            auto_for_complex: true,
        };
        assert!(!strategy.should_activate(TurnIntent::Edit, &[], 1));
    }
}
