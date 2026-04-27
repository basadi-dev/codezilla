use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::task::{load_tasks, BenchTask, TaskValidation};

// ── Public interface ─────────────────────────────────────────────────────────

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Path to the directory containing task subdirectories.
    pub tasks_dir: PathBuf,
    /// Filter: only run tasks whose ID matches this glob pattern.
    pub filter: Option<String>,
    /// Model ID override (passed to `codezilla exec --model`).
    pub model: Option<String>,
    /// Path to the codezilla binary.
    pub codezilla_bin: PathBuf,
    /// Path to the codezilla config file to use.
    pub config_path: Option<PathBuf>,
    /// Output directory for results.
    pub output_dir: PathBuf,
}

/// Result of a single task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskResult {
    pub task_id: String,
    pub title: String,
    pub difficulty: String,
    pub category: String,
    pub passed: bool,
    pub exit_code: i32,
    pub elapsed_ms: u64,
    pub agent_iterations: usize,
    pub tool_call_count: usize,
    pub token_usage: TokenUsageSummary,
    #[serde(default)]
    pub file_changes: Vec<FileChangeSummaryResult>,
    pub validation_output: String,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsageSummary {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileChangeSummaryResult {
    pub path: String,
    pub kind: String,
    pub lines_added: usize,
    pub lines_removed: usize,
}

/// Summary of a complete benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
    pub pass_rate: f64,
    pub total_elapsed_ms: u64,
    pub avg_iterations: f64,
    pub avg_tool_calls: f64,
    pub results: Vec<TaskResult>,
}

// ── Runner ───────────────────────────────────────────────────────────────────

/// Execute a full benchmark run.
pub fn run_bench(config: &BenchConfig) -> Result<BenchSummary> {
    let tasks = load_tasks(&config.tasks_dir)?;
    if tasks.is_empty() {
        anyhow::bail!("No benchmark tasks found in {}", config.tasks_dir.display());
    }

    // Apply filter if provided.
    let tasks: Vec<_> = if let Some(ref filter) = config.filter {
        let pattern = glob::Pattern::new(filter)
            .with_context(|| format!("invalid filter pattern: {filter}"))?;
        tasks
            .into_iter()
            .filter(|(_, t)| pattern.matches(&t.id))
            .collect()
    } else {
        tasks
    };

    eprintln!("╭─ Codezilla Benchmark ────────────────────────────────────╮");
    eprintln!(
        "│ Tasks: {:<4}  Model: {:<36} │",
        tasks.len(),
        config.model.as_deref().unwrap_or("(config default)")
    );
    eprintln!("╰──────────────────────────────────────────────────────────╯");

    let run_start = Instant::now();
    let mut results = Vec::new();

    for (i, (task_dir, task)) in tasks.iter().enumerate() {
        eprint!(
            "  [{}/{}] {} (timeout {}s) ... ",
            i + 1,
            tasks.len(),
            task.id,
            task.timeout_secs
        );

        let result = run_single_task(config, task_dir, task);
        match &result {
            Ok(r) if r.passed => eprintln!("✅ PASS ({:.1}s)", r.elapsed_ms as f64 / 1000.0),
            Ok(r) => eprintln!("❌ FAIL ({:.1}s)", r.elapsed_ms as f64 / 1000.0),
            Err(e) => eprintln!("💥 ERROR: {e}"),
        }

        results.push(result.unwrap_or_else(|e| TaskResult {
            task_id: task.id.clone(),
            title: task.title.clone(),
            difficulty: task.difficulty.clone(),
            category: task.category.clone(),
            passed: false,
            exit_code: -1,
            elapsed_ms: 0,
            agent_iterations: 0,
            tool_call_count: 0,
            token_usage: TokenUsageSummary::default(),
            file_changes: Vec::new(),
            validation_output: String::new(),
            error: Some(e.to_string()),
        }));
    }

    let total_elapsed = run_start.elapsed().as_millis() as u64;
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results
        .iter()
        .filter(|r| !r.passed && r.error.is_none())
        .count();
    let errors = results.iter().filter(|r| r.error.is_some()).count();
    let total = results.len();

    let avg_iterations = if total > 0 {
        results
            .iter()
            .map(|r| r.agent_iterations as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };
    let avg_tool_calls = if total > 0 {
        results
            .iter()
            .map(|r| r.tool_call_count as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };

    let summary = BenchSummary {
        total,
        passed,
        failed,
        errors,
        pass_rate: if total > 0 {
            passed as f64 / total as f64 * 100.0
        } else {
            0.0
        },
        total_elapsed_ms: total_elapsed,
        avg_iterations,
        avg_tool_calls,
        results,
    };

    // Print summary table.
    eprintln!();
    eprintln!("╭─ Results ────────────────────────────────────────────────╮");
    eprintln!(
        "│ Pass: {:<4}  Fail: {:<4}  Error: {:<4}  Rate: {:>5.1}%        │",
        passed, failed, errors, summary.pass_rate
    );
    let line2 = format!(
        "Time: {:.1}s  Avg iter: {:.1}  Avg tools: {:.1}",
        total_elapsed as f64 / 1000.0,
        avg_iterations,
        avg_tool_calls
    );
    eprintln!("│ {:<56} │", line2);
    eprintln!("╰──────────────────────────────────────────────────────────╯");

    // Write results JSONL.
    std::fs::create_dir_all(&config.output_dir)?;
    let results_path = config.output_dir.join("results.jsonl");
    let mut results_file = std::fs::File::create(&results_path)?;
    for result in &summary.results {
        serde_json::to_writer(&mut results_file, result)?;
        std::io::Write::write_all(&mut results_file, b"\n")?;
    }

    // Write summary JSON.
    let summary_path = config.output_dir.join("summary.json");
    let summary_file = std::fs::File::create(&summary_path)?;
    serde_json::to_writer_pretty(summary_file, &summary)?;

    eprintln!("  Results: {}", results_path.display());
    eprintln!("  Summary: {}", summary_path.display());

    Ok(summary)
}

// ── Single task execution ────────────────────────────────────────────────────

fn run_single_task(config: &BenchConfig, task_dir: &Path, task: &BenchTask) -> Result<TaskResult> {
    // 1. Create a temporary workspace.
    let workdir = tempdir_in_project(config)?;

    // 2. Setup: copy fixtures and run setup commands.
    setup_workspace(task_dir, task, &workdir)?;

    // 3. Initialise a git repo so we can diff later.
    run_shell(
        &workdir,
        "git init -q && git add -A && git commit -q --allow-empty -m 'initial'",
    )?;

    // 4. Run codezilla exec.
    let start = Instant::now();
    let (exit_code, events) = run_codezilla(config, task, &workdir)?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    // 5. Extract metrics from the event stream.
    let metrics = extract_metrics(&events);

    if exit_code != 0 {
        return Ok(TaskResult {
            task_id: task.id.clone(),
            title: task.title.clone(),
            difficulty: task.difficulty.clone(),
            category: task.category.clone(),
            passed: false,
            exit_code,
            elapsed_ms,
            agent_iterations: metrics.agent_iterations,
            tool_call_count: metrics.tool_call_count,
            token_usage: metrics.token_usage,
            file_changes: metrics.file_changes,
            validation_output: String::new(),
            error: Some(
                extract_terminal_error(&events)
                    .unwrap_or_else(|| format!("codezilla exec failed with exit code {exit_code}")),
            ),
        });
    }

    // 6. Run validation.
    let (passed, validation_output) = run_validation(&task.validate, &workdir)?;

    Ok(TaskResult {
        task_id: task.id.clone(),
        title: task.title.clone(),
        difficulty: task.difficulty.clone(),
        category: task.category.clone(),
        passed,
        exit_code,
        elapsed_ms,
        agent_iterations: metrics.agent_iterations,
        tool_call_count: metrics.tool_call_count,
        token_usage: metrics.token_usage,
        file_changes: metrics.file_changes,
        validation_output,
        error: None,
    })
}

fn tempdir_in_project(config: &BenchConfig) -> Result<PathBuf> {
    let base = config.output_dir.join("workspaces");
    std::fs::create_dir_all(&base)?;
    let dir = base.join(format!("bench_{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn setup_workspace(task_dir: &Path, task: &BenchTask, workdir: &Path) -> Result<()> {
    // Copy fixtures if specified.
    if let Some(ref fixtures_rel) = task.setup.fixtures {
        let fixtures_dir = task_dir.join(fixtures_rel);
        if fixtures_dir.exists() {
            copy_dir_recursive(&fixtures_dir, workdir)?;
        }
    }

    // Run setup commands.
    for cmd in &task.setup.commands {
        run_shell(workdir, cmd).with_context(|| format!("setup command failed: {cmd}"))?;
    }

    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    for entry in walkdir::WalkDir::new(src) {
        let entry = entry?;
        let relative = entry.path().strip_prefix(src)?;
        let target = dst.join(relative);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target)?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

fn run_shell(cwd: &Path, command: &str) -> Result<String> {
    let output = Command::new("bash")
        .args(["-c", command])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to spawn: {command}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        anyhow::bail!(
            "command failed (exit {}): {command}\nstdout: {stdout}\nstderr: {stderr}",
            output.status.code().unwrap_or(-1)
        );
    }

    Ok(format!("{stdout}{stderr}"))
}

fn run_codezilla(
    config: &BenchConfig,
    task: &BenchTask,
    workdir: &Path,
) -> Result<(i32, Vec<Value>)> {
    let mut cmd = Command::new(&config.codezilla_bin);
    cmd.arg("--cd")
        .arg(workdir.to_string_lossy().as_ref())
        .arg("--sandbox")
        .arg("danger-full-access")
        .arg("--dangerously-bypass-approvals-and-sandbox");

    if let Some(ref model) = config.model {
        cmd.arg("--model").arg(model);
    }

    if let Some(ref config_path) = config.config_path {
        cmd.arg("--config")
            .arg(config_path.to_string_lossy().as_ref());
    }

    cmd.arg("exec")
        .arg("--json")
        .arg("--ephemeral")
        .arg(&task.prompt);

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().with_context(|| {
        format!(
            "failed to spawn codezilla: {}",
            config.codezilla_bin.display()
        )
    })?;

    // Drain stdout on a background thread and wait for explicit terminal events.
    // A plain agent message after tool use is not enough: the executor may still
    // need another loop to patch files or run validation commands.
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
    let events_shared: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = Arc::clone(&events_shared);
    let done_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_flag_clone = Arc::clone(&done_flag);

    let reader_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(event) => {
                    let kind = event
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Hard stop on terminal events.
                    let is_terminal = kind == "TURN_COMPLETED" || kind == "TURN_FAILED";
                    events_clone.lock().unwrap().push(event);
                    if is_terminal {
                        done_flag_clone.store(true, std::sync::atomic::Ordering::Release);
                        break;
                    }
                }
                Err(_) => {
                    tracing::debug!("bench: non-JSON line from codezilla: {line}");
                }
            }
        }
    });

    // Drain stderr as well so verbose logging cannot block the child process.
    let stderr_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if !line.trim().is_empty() {
                tracing::debug!("bench: codezilla stderr: {line}");
            }
        }
    });

    // Poll for completion: child exit, done signal, or timeout.
    let timeout = Duration::from_secs(if task.timeout_secs == 0 {
        300
    } else {
        task.timeout_secs
    });
    let deadline = Instant::now() + timeout;
    let exit_code = loop {
        // Terminal events mean the child can exit normally.
        if done_flag.load(std::sync::atomic::Ordering::Acquire) {
            let status = child.wait()?;
            break status.code().unwrap_or(-1);
        }
        match child.try_wait()? {
            Some(status) => break status.code().unwrap_or(-1),
            None => {
                if Instant::now() >= deadline {
                    tracing::warn!(
                        "bench: task '{}' timed out after {}s — killing subprocess",
                        task.id,
                        timeout.as_secs()
                    );
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("task timed out after {}s", timeout.as_secs());
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    };

    let _ = reader_thread.join();
    let _ = stderr_thread.join();
    let events = Arc::try_unwrap(events_shared)
        .unwrap_or_default()
        .into_inner()
        .unwrap_or_default();

    Ok((exit_code, events))
}

// ── Metrics extraction ───────────────────────────────────────────────────────

fn extract_terminal_error(events: &[Value]) -> Option<String> {
    for event in events.iter().rev() {
        let kind = event.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "TURN_FAILED" {
            continue;
        }
        let payload = event.get("payload")?;
        if let Some(reason) = payload.get("reason").and_then(|v| v.as_str()) {
            return Some(reason.to_string());
        }
        if let Some(message) = payload.get("message").and_then(|v| v.as_str()) {
            return Some(message.to_string());
        }
    }
    None
}

struct ExtractedMetrics {
    agent_iterations: usize,
    tool_call_count: usize,
    token_usage: TokenUsageSummary,
    file_changes: Vec<FileChangeSummaryResult>,
}

fn extract_metrics(events: &[Value]) -> ExtractedMetrics {
    // Look for the TurnCompleted event which carries the metrics payload.
    for event in events.iter().rev() {
        let kind = event.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "TURN_COMPLETED" {
            continue;
        }

        let payload = match event.get("payload") {
            Some(p) => p,
            None => continue,
        };

        let metrics = payload.get("metrics");
        let token_usage = payload.get("tokenUsage");

        let agent_iterations = metrics
            .and_then(|m| m.get("agentIterations"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let tool_call_count = metrics
            .and_then(|m| m.get("toolCallCount"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let input_tokens = token_usage
            .and_then(|u| u.get("inputTokens"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = token_usage
            .and_then(|u| u.get("outputTokens"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let cached_tokens = token_usage
            .and_then(|u| u.get("cachedTokens"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let file_changes = metrics
            .and_then(|m| m.get("fileChanges"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|fc| {
                        Some(FileChangeSummaryResult {
                            path: fc.get("path")?.as_str()?.to_string(),
                            kind: fc.get("kind")?.as_str()?.to_string(),
                            lines_added: fc.get("linesAdded").and_then(|v| v.as_u64()).unwrap_or(0)
                                as usize,
                            lines_removed: fc
                                .get("linesRemoved")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        return ExtractedMetrics {
            agent_iterations,
            tool_call_count,
            token_usage: TokenUsageSummary {
                input_tokens,
                output_tokens,
                cached_tokens,
            },
            file_changes,
        };
    }

    // Fallback: count tool calls from events.
    let tool_call_count = events
        .iter()
        .filter(|e| {
            e.get("kind")
                .and_then(|v| v.as_str())
                .map(|k| k == "ITEM_COMPLETED")
                .unwrap_or(false)
                && e.get("payload")
                    .and_then(|p| p.get("kind"))
                    .and_then(|v| v.as_str())
                    .map(|k| k == "TOOL_CALL")
                    .unwrap_or(false)
        })
        .count();

    ExtractedMetrics {
        agent_iterations: 0,
        tool_call_count,
        token_usage: TokenUsageSummary::default(),
        file_changes: Vec::new(),
    }
}

// ── Validation ───────────────────────────────────────────────────────────────

fn run_validation(validation: &TaskValidation, workdir: &Path) -> Result<(bool, String)> {
    let mut output_parts = Vec::new();
    let mut all_passed = true;

    // A) Run command-based validation.
    if let Some(ref command) = validation.command {
        match run_shell(workdir, command) {
            Ok(out) => {
                output_parts.push(format!("Command validation PASSED:\n{out}"));
            }
            Err(e) => {
                all_passed = false;
                output_parts.push(format!("Command validation FAILED:\n{e}"));
            }
        }
    }

    // B) Check expected files.
    for expected in &validation.expected_files {
        let file_path = workdir.join(&expected.path);

        if expected.exists {
            if !file_path.exists() {
                all_passed = false;
                output_parts.push(format!("Expected file missing: {}", expected.path));
                continue;
            }

            let content = std::fs::read_to_string(&file_path)
                .with_context(|| format!("reading expected file: {}", expected.path))?;

            for needle in &expected.contains {
                if !content.contains(needle) {
                    all_passed = false;
                    output_parts.push(format!(
                        "File {} missing expected content: {needle}",
                        expected.path
                    ));
                }
            }
            for needle in &expected.not_contains {
                if content.contains(needle) {
                    all_passed = false;
                    output_parts.push(format!(
                        "File {} contains forbidden content: {needle}",
                        expected.path
                    ));
                }
            }
        } else if file_path.exists() {
            all_passed = false;
            output_parts.push(format!("File should not exist but does: {}", expected.path));
        }
    }

    // If no validation criteria were specified, consider it a pass
    // if the agent completed without error (exit code already checked).
    if validation.command.is_none() && validation.expected_files.is_empty() {
        output_parts.push("No validation criteria — pass by completion.".into());
    }

    let combined_output = output_parts.join("\n");
    Ok((all_passed, combined_output))
}
