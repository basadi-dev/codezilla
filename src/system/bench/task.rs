use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// A single benchmark task definition, loaded from a `task.yaml` file.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BenchTask {
    /// Unique identifier for this task.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Difficulty level: easy, medium, hard.
    #[serde(default = "default_difficulty")]
    pub difficulty: String,
    /// Category: bugfix, feature, refactor, test, etc.
    #[serde(default = "default_category")]
    pub category: String,
    /// The prompt that will be sent to the agent.
    pub prompt: String,
    /// How to set up the workspace before the agent runs.
    pub setup: TaskSetup,
    /// How to validate the agent's output.
    pub validate: TaskValidation,
    /// Maximum time the agent is allowed to run (seconds).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Optional cost cap (not enforced, but recorded in results).
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// Optional list of tags for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Describes how to prepare the workspace before the agent runs.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TaskSetup {
    /// Path to a directory that will be copied into the workspace.
    /// Relative to the task directory.
    #[serde(default)]
    pub fixtures: Option<String>,
    /// Shell commands to run after copying fixtures (in order).
    #[serde(default)]
    pub commands: Vec<String>,
    /// A shell command that MUST fail (non-zero exit) after setup.
    /// Used for bugfix tasks to verify the fixtures are actually broken
    /// before the agent starts. Prevents false-positive passes.
    #[serde(default)]
    pub assert_fails: Option<String>,
}

/// How to validate the agent's work product.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TaskValidation {
    /// A shell command to run in the workspace. Exit 0 = pass.
    #[serde(default)]
    pub command: Option<String>,
    /// Path to a file containing the expected unified diff.
    /// If provided, the workspace diff is compared against it.
    #[serde(default)]
    pub expected_files: Vec<ExpectedFile>,
    /// If true, check that the workspace has no uncommitted changes
    /// beyond the expected files (i.e. no collateral damage).
    #[serde(default)]
    pub no_extra_changes: bool,
}

/// An expected file state after the agent finishes.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExpectedFile {
    /// Path relative to workspace root.
    pub path: String,
    /// If present, the file content must contain all of these substrings.
    #[serde(default)]
    pub contains: Vec<String>,
    /// If present, the file content must NOT contain any of these substrings.
    #[serde(default)]
    pub not_contains: Vec<String>,
    /// If true, the file must exist.
    #[serde(default = "default_true")]
    pub exists: bool,
    /// Minimum number of occurrences of a pattern in the file.
    /// For example, `min_test_count: 15` with a `contains: ["def test_"]`
    /// ensures at least 15 test functions exist.
    #[serde(default)]
    pub min_test_count: Option<usize>,
}

fn default_difficulty() -> String {
    "medium".into()
}
fn default_category() -> String {
    "general".into()
}
fn default_timeout() -> u64 {
    120
}
fn default_true() -> bool {
    true
}

/// Load all task definitions from a directory.
///
/// Scans `tasks_dir` for subdirectories containing a `task.yaml` file.
/// Tasks are returned sorted by ID for deterministic ordering.
pub fn load_tasks(tasks_dir: &Path) -> Result<Vec<(PathBuf, BenchTask)>> {
    let mut tasks = Vec::new();

    if !tasks_dir.exists() {
        anyhow::bail!("Tasks directory does not exist: {}", tasks_dir.display());
    }

    for entry in std::fs::read_dir(tasks_dir)
        .with_context(|| format!("reading tasks directory: {}", tasks_dir.display()))?
    {
        let entry = entry?;
        let task_dir = entry.path();
        if !task_dir.is_dir() {
            continue;
        }

        let task_file = task_dir.join("task.yaml");
        if !task_file.exists() {
            continue;
        }

        let content = std::fs::read_to_string(&task_file)
            .with_context(|| format!("reading task file: {}", task_file.display()))?;
        let task: BenchTask = serde_yaml::from_str(&content)
            .with_context(|| format!("parsing task file: {}", task_file.display()))?;
        tasks.push((task_dir, task));
    }

    tasks.sort_by(|a, b| a.1.id.cmp(&b.1.id));
    Ok(tasks)
}
