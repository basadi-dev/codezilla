use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
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
    /// How many times to run each task (≥1). Used for flake detection.
    pub runs: usize,
    /// Maximum number of tasks to run concurrently (1 = sequential).
    pub parallelism: usize,
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
    /// Estimated USD cost based on token usage and model pricing.
    pub estimated_cost_usd: f64,
    #[serde(default)]
    pub file_changes: Vec<FileChangeSummaryResult>,
    pub validation_output: String,
    #[serde(default)]
    pub error: Option<String>,
    /// How many times this task was attempted (only >1 when --runs N is used).
    #[serde(default = "default_one")]
    pub run_count: usize,
    /// How many of those attempts passed.
    #[serde(default)]
    pub pass_count: usize,
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
    /// Total estimated USD cost across all task runs.
    pub total_cost_usd: f64,
    pub results: Vec<TaskResult>,
}

fn default_one() -> usize {
    1
}

// ── Cost estimation ──────────────────────────────────────────────────────────

/// Pricing table: (input $/M, output $/M, cache-read multiplier of input).
/// Matched by substring of the model ID (lower-cased).
const PRICING: &[(&str, f64, f64, f64)] = &[
    // Claude 4 family
    ("claude-opus-4",     15.0,  75.0, 0.10),
    ("claude-sonnet-4",    3.0,  15.0, 0.10),
    // Claude 3.7 / 3.5 family
    ("claude-opus-3",     15.0,  75.0, 0.10),
    ("claude-sonnet-3-7",  3.0,  15.0, 0.10),
    ("claude-sonnet-3-5",  3.0,  15.0, 0.10),
    ("claude-haiku-3",     0.25,  1.25, 0.10),
    // GPT-4o family
    ("gpt-4o-mini",        0.15,  0.60, 0.10),
    ("gpt-4o",             2.50, 10.00, 0.10),
    ("gpt-4-turbo",       10.00, 30.00, 0.10),
    // Gemini family
    ("gemini-2.5-pro",     1.25, 10.00, 0.25),
    ("gemini-2.0-flash",   0.10,  0.40, 0.25),
    ("gemini-1.5-pro",     3.50, 10.50, 0.25),
    ("gemini-1.5-flash",   0.075, 0.30, 0.25),
];

/// Returns estimated cost in USD for a completed task run.
fn estimate_cost_usd(usage: &TokenUsageSummary, model: Option<&str>) -> f64 {
    let model_lc = model.unwrap_or("").to_lowercase();
    let (input_rate, output_rate, cache_multiplier) = PRICING
        .iter()
        .find(|(fragment, ..)| model_lc.contains(fragment))
        .map(|(_, i, o, c)| (*i, *o, *c))
        // Default: claude-sonnet-4 pricing
        .unwrap_or((3.0, 15.0, 0.10));

    let non_cached = (usage.input_tokens - usage.cached_tokens).max(0) as f64;
    let cached = usage.cached_tokens as f64;
    let output = usage.output_tokens as f64;

    (non_cached * input_rate + cached * input_rate * cache_multiplier + output * output_rate)
        / 1_000_000.0
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

    let runs = config.runs.max(1);
    let parallelism = config.parallelism.max(1);
    let total_attempts = tasks.len() * runs;

    eprintln!("╭─ Codezilla Benchmark ────────────────────────────────────╮");
    eprintln!(
        "│ Tasks: {:<4}  Model: {:<36} │",
        tasks.len(),
        config.model.as_deref().unwrap_or("(config default)")
    );
    if runs > 1 || parallelism > 1 {
        eprintln!(
            "│ Runs/task: {:<3}  Parallelism: {:<27} │",
            runs, parallelism
        );
    }
    eprintln!("╰──────────────────────────────────────────────────────────╯");

    let run_start = Instant::now();

    // Build full work queue: (task_index, run_index).
    let tasks = Arc::new(tasks);
    let work: Vec<(usize, usize)> = (0..tasks.len())
        .flat_map(|ti| (0..runs).map(move |ri| (ti, ri)))
        .collect();

    // Shared result accumulator + progress counter.
    let results_shared: Arc<Mutex<Vec<(usize, TaskResult)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(total_attempts)));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let print_mu = Arc::new(Mutex::new(())); // serialise progress lines

    // Fixed-size thread pool pulling from a shared work queue.
    let work_queue = Arc::new(Mutex::new(work.into_iter()));

    let mut handles = Vec::new();
    for _ in 0..parallelism {
        let config = config.clone();
        let tasks = Arc::clone(&tasks);
        let wq = Arc::clone(&work_queue);
        let acc = Arc::clone(&results_shared);
        let done_ctr = Arc::clone(&completed);
        let pmx = Arc::clone(&print_mu);

        let handle = std::thread::spawn(move || loop {
            let item = { wq.lock().unwrap().next() };
            let (ti, ri) = match item {
                Some(v) => v,
                None => break,
            };
            let (task_dir, task) = &tasks[ti];
            let label = if runs > 1 {
                format!("{} (run {}/{})", task.id, ri + 1, runs)
            } else {
                task.id.clone()
            };

            let result = run_single_task(&config, task_dir, task);
            let done = done_ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

            {
                let _lk = pmx.lock().unwrap();
                match &result {
                    Ok(r) if r.passed => eprintln!(
                        "  [{}/{}] {} … ✅ PASS ({:.1}s, ${:.4})",
                        done, total_attempts, label,
                        r.elapsed_ms as f64 / 1000.0,
                        r.estimated_cost_usd
                    ),
                    Ok(r) => eprintln!(
                        "  [{}/{}] {} … ❌ FAIL ({:.1}s)",
                        done, total_attempts, label,
                        r.elapsed_ms as f64 / 1000.0
                    ),
                    Err(e) => eprintln!(
                        "  [{}/{}] {} … 💥 ERROR: {e}",
                        done, total_attempts, label
                    ),
                }
            }

            let task_result = result.unwrap_or_else(|e| TaskResult {
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
                estimated_cost_usd: 0.0,
                file_changes: Vec::new(),
                validation_output: String::new(),
                run_count: 1,
                pass_count: 0,
                error: Some(e.to_string()),
            });
            acc.lock().unwrap().push((ti, task_result));
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let total_elapsed = run_start.elapsed().as_millis() as u64;

    // Aggregate per-task: group all run results, merge into one canonical TaskResult.
    let raw = Arc::try_unwrap(results_shared)
        .unwrap_or_default()
        .into_inner()
        .unwrap_or_default();

    let mut per_task: std::collections::BTreeMap<usize, Vec<TaskResult>> = Default::default();
    for (ti, r) in raw {
        per_task.entry(ti).or_default().push(r);
    }

    let mut results: Vec<TaskResult> = per_task
        .into_values()
        .map(|mut run_results| {
            let n_runs = run_results.len();
            let n_pass = run_results.iter().filter(|r| r.passed).count();
            let total_cost: f64 = run_results.iter().map(|r| r.estimated_cost_usd).sum();
            let total_in: i64 = run_results.iter().map(|r| r.token_usage.input_tokens).sum();
            let total_out: i64 = run_results.iter().map(|r| r.token_usage.output_tokens).sum();
            let total_cached: i64 = run_results.iter().map(|r| r.token_usage.cached_tokens).sum();

            // Canonical = last result; overwrite aggregated fields.
            let mut c = run_results.pop().unwrap();
            c.run_count = n_runs;
            c.pass_count = n_pass;
            c.passed = n_pass == n_runs; // passes only if ALL runs pass
            c.estimated_cost_usd = total_cost;
            c.token_usage = TokenUsageSummary {
                input_tokens: total_in,
                output_tokens: total_out,
                cached_tokens: total_cached,
            };
            c
        })
        .collect();

    results.sort_by(|a, b| a.task_id.cmp(&b.task_id));

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed && r.error.is_none()).count();
    let errors = results.iter().filter(|r| r.error.is_some()).count();
    let total_cost_usd: f64 = results.iter().map(|r| r.estimated_cost_usd).sum();

    let avg = |f: fn(&TaskResult) -> f64| -> f64 {
        if total > 0 { results.iter().map(f).sum::<f64>() / total as f64 } else { 0.0 }
    };
    let avg_iterations = avg(|r| r.agent_iterations as f64);
    let avg_tool_calls = avg(|r| r.tool_call_count as f64);

    let summary = BenchSummary {
        total,
        passed,
        failed,
        errors,
        pass_rate: if total > 0 { passed as f64 / total as f64 * 100.0 } else { 0.0 },
        total_elapsed_ms: total_elapsed,
        avg_iterations,
        avg_tool_calls,
        total_cost_usd,
        results,
    };

    // ── Print summary table ──────────────────────────────────────────────────
    eprintln!();
    eprintln!("╭─ Results ────────────────────────────────────────────────╮");
    eprintln!(
        "│ Pass: {:<4}  Fail: {:<4}  Error: {:<4}  Rate: {:>5.1}%        │",
        passed, failed, errors, summary.pass_rate
    );
    let line2 = format!(
        "Time: {:.1}s  Avg iter: {:.1}  Avg tools: {:.1}",
        total_elapsed as f64 / 1000.0, avg_iterations, avg_tool_calls
    );
    eprintln!("│ {:<56} │", line2);
    let line3 = format!(
        "Est. cost: ${:.4}  ({} task(s) × {} run(s))",
        total_cost_usd, total, runs
    );
    eprintln!("│ {:<56} │", line3);
    eprintln!("╰──────────────────────────────────────────────────────────╯");

    // ── Persist results ──────────────────────────────────────────────────────
    std::fs::create_dir_all(&config.output_dir)?;
    let results_path = config.output_dir.join("results.jsonl");
    let mut results_file = std::fs::File::create(&results_path)?;
    for result in &summary.results {
        serde_json::to_writer(&mut results_file, result)?;
        std::io::Write::write_all(&mut results_file, b"\n")?;
    }

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

    // 2b. Pre-flight sanity check: if the task declares `assert_fails`,
    //     verify the fixtures are actually broken before the agent runs.
    //     This prevents false-positive passes on bugfix tasks.
    if let Some(ref assert_cmd) = task.setup.assert_fails {
        if run_shell(&workdir, assert_cmd).is_ok() {
            anyhow::bail!(
                "task '{}': pre-flight assertion violated — fixture code should fail \
                 but the command succeeded: {}",
                task.id,
                assert_cmd
            );
        }
    }

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
        let cost = estimate_cost_usd(&metrics.token_usage, config.model.as_deref());
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
            estimated_cost_usd: cost,
            file_changes: metrics.file_changes,
            validation_output: String::new(),
            run_count: 1,
            pass_count: 0,
            error: Some(
                extract_terminal_error(&events)
                    .unwrap_or_else(|| format!("codezilla exec failed with exit code {exit_code}")),
            ),
        });
    }

    // 6. Run validation.
    let (passed, validation_output) = run_validation(&task.validate, &workdir)?;

    // 7. Clean up workspace for passing tasks to prevent disk bloat.
    if passed {
        let _ = std::fs::remove_dir_all(&workdir);
    }

    let cost = estimate_cost_usd(&metrics.token_usage, config.model.as_deref());

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
        estimated_cost_usd: cost,
        file_changes: metrics.file_changes,
        validation_output,
        run_count: 1,
        pass_count: if passed { 1 } else { 0 },
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

            // C) Check minimum test count (pattern occurrence count).
            if let Some(min_count) = expected.min_test_count {
                // Count occurrences of the first `contains` pattern (typically "def test_").
                if let Some(pattern) = expected.contains.first() {
                    let actual_count = content.matches(pattern).count();
                    if actual_count < min_count {
                        all_passed = false;
                        output_parts.push(format!(
                            "File {} has {} occurrences of '{}', expected at least {}",
                            expected.path, actual_count, pattern, min_count
                        ));
                    }
                }
            }
        } else if file_path.exists() {
            all_passed = false;
            output_parts.push(format!("File should not exist but does: {}", expected.path));
        }
    }

    // D) Check for collateral file changes (no_extra_changes).
    if validation.no_extra_changes {
        // Stage all changes so `git diff --name-only HEAD` picks up new files too.
        let _ = run_shell(workdir, "git add -A");
        let diff_output = run_shell(workdir, "git diff --name-only --cached HEAD")
            .unwrap_or_default();
        let changed_files: HashSet<&str> = diff_output
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        // Build the set of files that are *expected* to change.
        let expected_paths: HashSet<&str> = validation
            .expected_files
            .iter()
            .map(|f| f.path.as_str())
            .collect();

        let extra: Vec<&&str> = changed_files
            .iter()
            .filter(|p| !expected_paths.contains(**p))
            .collect();

        if !extra.is_empty() {
            all_passed = false;
            output_parts.push(format!(
                "Unexpected file changes (no_extra_changes violated): {}",
                extra
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
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

// ── Regression comparison ────────────────────────────────────────────────────

/// Compare two bench result directories and print a regression diff table.
///
/// `base_dir` and `head_dir` must each contain a `summary.json` produced by
/// `run_bench`. If `base_dir`/`head_dir` is a bare directory (e.g. `bench/results`),
/// the function automatically picks the *latest* timestamped sub-directory.
pub fn run_compare(base_dir: &str, head_dir: &str) -> Result<()> {
    let base_summary = load_summary(base_dir)?;
    let head_summary = load_summary(head_dir)?;

    // Index by task_id.
    let base_map: std::collections::HashMap<&str, &TaskResult> = base_summary
        .results
        .iter()
        .map(|r| (r.task_id.as_str(), r))
        .collect();
    let head_map: std::collections::HashMap<&str, &TaskResult> = head_summary
        .results
        .iter()
        .map(|r| (r.task_id.as_str(), r))
        .collect();

    let mut all_ids: Vec<&str> = base_map
        .keys()
        .chain(head_map.keys())
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    all_ids.sort_unstable();

    let mut regressions = 0usize;
    let mut improvements = 0usize;

    eprintln!("╭─ Bench Compare ──────────────────────────────────────────╮");
    eprintln!("│ BASE: {:<52} │", base_dir);
    eprintln!("│ HEAD: {:<52} │", head_dir);
    eprintln!("├──────────────────────────────────────┬────────┬──────────┤");
    eprintln!("│ Task                                 │ Base   │ Head     │");
    eprintln!("├──────────────────────────────────────┼────────┼──────────┤");

    for id in &all_ids {
        let base_r = base_map.get(id);
        let head_r = head_map.get(id);

        let (base_label, head_label, marker) = match (base_r, head_r) {
            (Some(b), Some(h)) => {
                let bl = if b.passed { "PASS" } else { "FAIL" };
                let hl = if h.passed { "PASS" } else { "FAIL" };
                let m = match (b.passed, h.passed) {
                    (true, false) => { regressions += 1; "🔴 REGRESSED" }
                    (false, true) => { improvements += 1; "🟢 IMPROVED " }
                    (true, true)  => "✅ stable    ",
                    (false, false)=> "❌ still-fail",
                };
                (bl.to_string(), hl.to_string(), m)
            }
            (None, Some(h)) => {
                let hl = if h.passed { "PASS" } else { "FAIL" };
                (String::from("N/A "), hl.to_string(), "🆕 NEW       ")
            }
            (Some(b), None) => {
                let bl = if b.passed { "PASS" } else { "FAIL" };
                (bl.to_string(), String::from("N/A "), "🗑️  REMOVED   ")
            }
            (None, None) => continue,
        };
        eprintln!(
            "│ {:<36} │ {:<6} │ {:<8} │  {}",
            &id[..id.len().min(36)], base_label, head_label, marker
        );
    }

    eprintln!("╰──────────────────────────────────────┴────────┴──────────╯");
    eprintln!();

    // Summary line.
    let base_cost = base_summary.total_cost_usd;
    let head_cost = head_summary.total_cost_usd;
    let cost_delta = head_cost - base_cost;
    let cost_sign = if cost_delta >= 0.0 { "+" } else { "" };

    eprintln!(
        "  Pass rate:  {:.1}% → {:.1}%",
        base_summary.pass_rate, head_summary.pass_rate
    );
    eprintln!(
        "  Regressions: {}   Improvements: {}",
        regressions, improvements
    );
    eprintln!(
        "  Est. cost:  ${:.4} → ${:.4}  ({}{:.4})",
        base_cost, head_cost, cost_sign, cost_delta
    );

    if regressions > 0 {
        eprintln!("\n  ⚠️  {} task(s) regressed.", regressions);
    } else {
        eprintln!("\n  ✅ No regressions.");
    }

    Ok(())
}

fn load_summary(dir: &str) -> Result<BenchSummary> {
    let path = PathBuf::from(dir);

    // If the given path contains a summary.json directly, use it.
    let direct = path.join("summary.json");
    if direct.exists() {
        let content = std::fs::read_to_string(&direct)
            .with_context(|| format!("reading {}", direct.display()))?;
        return serde_json::from_str(&content)
            .with_context(|| format!("parsing {}", direct.display()));
    }

    // Otherwise, look for timestamped sub-directories and pick the latest.
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&path)
        .with_context(|| format!("opening directory: {dir}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("summary.json").exists())
        .collect();
    entries.sort();

    let latest = entries
        .last()
        .ok_or_else(|| anyhow!("no bench result directories found in {dir}"))?;

    let summary_file = latest.join("summary.json");
    let content = std::fs::read_to_string(&summary_file)
        .with_context(|| format!("reading {}", summary_file.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("parsing {}", summary_file.display()))
}
