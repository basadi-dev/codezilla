use anyhow::Result;
use std::path::Path;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::system::config::EffectiveConfig;

/// Initialises the global tracing subscriber.
/// Returns a `WorkerGuard` that must be held for the life of the process
/// (dropping it flushes the background writer thread).
pub fn init(cfg: &EffectiveConfig) -> Result<WorkerGuard> {
    let level = match cfg.log_level.as_str() {
        "trace" => "trace",
        "debug" => "debug",
        "warn" | "warning" => "warn",
        "error" => "error",
        _ => "info",
    };

    // Build the file appender
    let log_path = Path::new(&cfg.log_file);
    if let Some(dir) = log_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file_name = log_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("codezilla.log");
    let dir = log_path.parent().unwrap_or(Path::new("logs"));

    let file_appender = tracing_appender::rolling::never(dir, file_name);
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    // JSON file layer
    let file_layer = fmt::layer()
        .with_writer(non_blocking)
        .with_ansi(false)
        .json();

    // Always use file-only logging when running TUI.
    // Writing to stderr corrupts the ratatui alternate screen.
    // The inline_mode flag is not available here, so we default to file-only.
    // Users can check logs/codezilla.log for warnings/errors.
    tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .init();

    Ok(guard)
}
