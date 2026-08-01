use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;
use uuid::Uuid;

use crate::system::domain::{CommandExecutionRecord, SandboxMode, SandboxRequest};

pub struct SandboxManager;

impl SandboxManager {
    pub fn new() -> Self {
        Self
    }

    pub async fn run_command(
        &self,
        argv: &[String],
        cwd: &str,
        env: &HashMap<String, String>,
        sandbox: &SandboxRequest,
    ) -> Result<CommandExecutionRecord> {
        if argv.is_empty() {
            bail!("tool_execution_failed: command argv cannot be empty");
        }
        let command_argv = sandboxed_command(argv, sandbox)?;

        let mut command = Command::new(&command_argv[0]);
        if command_argv.len() > 1 {
            command.args(&command_argv[1..]);
        }
        command.current_dir(cwd);
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        for (key, value) in env {
            command.env(key, value);
        }

        let output = command.output().await?;
        Ok(CommandExecutionRecord {
            process_id: format!("proc_{}", Uuid::new_v4().simple()),
            command: argv.to_vec(),
            cwd: cwd.into(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            truncated: false,
        })
    }

    pub async fn read_file(&self, path: &str, sandbox: &SandboxRequest) -> Result<Vec<u8>> {
        self.ensure_path_allowed(path, sandbox, false)?;
        Ok(tokio::fs::read(path).await?)
    }

    pub async fn write_file(
        &self,
        path: &str,
        data: &[u8],
        sandbox: &SandboxRequest,
    ) -> Result<()> {
        self.ensure_path_allowed(path, sandbox, true)?;
        if let Some(parent) = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, data).await?;
        Ok(())
    }

    pub async fn create_directory(
        &self,
        path: &str,
        recursive: bool,
        sandbox: &SandboxRequest,
    ) -> Result<()> {
        self.ensure_path_allowed(path, sandbox, true)?;
        if recursive {
            tokio::fs::create_dir_all(path).await?;
        } else {
            tokio::fs::create_dir(path).await?;
        }
        Ok(())
    }

    pub async fn remove_path(
        &self,
        path: &str,
        recursive: bool,
        _force: bool,
        sandbox: &SandboxRequest,
    ) -> Result<()> {
        self.ensure_path_allowed(path, sandbox, true)?;
        let meta = tokio::fs::metadata(path).await?;
        if meta.is_dir() {
            if recursive {
                tokio::fs::remove_dir_all(path).await?;
            } else {
                tokio::fs::remove_dir(path).await?;
            }
        } else {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }

    pub async fn copy_path(
        &self,
        source: &str,
        target: &str,
        _recursive: bool,
        sandbox: &SandboxRequest,
    ) -> Result<()> {
        self.ensure_path_allowed(source, sandbox, false)?;
        self.ensure_path_allowed(target, sandbox, true)?;
        if let Some(parent) = Path::new(target).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::copy(source, target).await?;
        Ok(())
    }

    fn ensure_path_allowed(&self, path: &str, sandbox: &SandboxRequest, write: bool) -> Result<()> {
        match sandbox.sandbox_mode.unwrap_or(SandboxMode::WorkspaceWrite) {
            SandboxMode::DangerFullAccess | SandboxMode::External => Ok(()),
            SandboxMode::ReadOnly if write => bail!("path_not_writable: read-only sandbox"),
            SandboxMode::ReadOnly | SandboxMode::WorkspaceWrite => {
                self.ensure_writable_root(path, sandbox, write)
            }
        }
    }

    fn ensure_writable_root(
        &self,
        path: &str,
        sandbox: &SandboxRequest,
        write: bool,
    ) -> Result<()> {
        if !write {
            return Ok(());
        }
        let path = resolve_path(path)?;
        let allowed = sandbox
            .writable_roots
            .iter()
            .filter_map(|root| resolve_path(root).ok())
            .any(|root| path.starts_with(root));
        if allowed {
            Ok(())
        } else {
            bail!("path_not_writable: {path:?} is outside writable roots");
        }
    }
}

/// Wrap a command with an OS-enforced sandbox for profiles that promise
/// restrictions. Unsupported platforms fail closed instead of silently
/// treating a policy label as a security boundary.
fn sandboxed_command(argv: &[String], sandbox: &SandboxRequest) -> Result<Vec<String>> {
    match sandbox.sandbox_mode.unwrap_or(SandboxMode::WorkspaceWrite) {
        SandboxMode::DangerFullAccess | SandboxMode::External => Ok(argv.to_vec()),
        SandboxMode::ReadOnly | SandboxMode::WorkspaceWrite => restricted_command(argv, sandbox),
    }
}

#[cfg(target_os = "macos")]
fn restricted_command(argv: &[String], sandbox: &SandboxRequest) -> Result<Vec<String>> {
    const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
    if !Path::new(SANDBOX_EXEC).is_file() {
        bail!("sandbox_unavailable: macOS sandbox-exec is required for restricted profiles");
    }

    let mut profile = vec![
        "(version 1)".to_string(),
        "(deny default)".to_string(),
        // Command tools need to execute binaries and read their libraries and
        // project inputs. Writes and network remain opt-in below.
        "(allow process*)".to_string(),
        "(allow file-read*)".to_string(),
        "(allow sysctl-read)".to_string(),
    ];

    if matches!(sandbox.sandbox_mode, Some(SandboxMode::WorkspaceWrite)) {
        for root in &sandbox.writable_roots {
            let root = resolve_path(root)?;
            profile.push(format!(
                "(allow file-write* (subpath \"{}\"))",
                escape_sandbox_string(&root.to_string_lossy())
            ));
        }
    }

    // The macOS profile language cannot safely express a hostname allowlist
    // for arbitrary child processes. When a list is configured, deny command
    // networking; web_fetch/web_search enforce that list at the URL layer.
    if sandbox.network_enabled && sandbox.allowed_domains.is_empty() {
        profile.push("(allow network*)".to_string());
    }

    let mut wrapped = vec![SANDBOX_EXEC.into(), "-p".into(), profile.join("\n")];
    wrapped.extend(argv.iter().cloned());
    Ok(wrapped)
}

#[cfg(target_os = "linux")]
fn restricted_command(argv: &[String], sandbox: &SandboxRequest) -> Result<Vec<String>> {
    let bwrap = std::env::var("CODEZILLA_BWRAP_PATH").unwrap_or_else(|_| "bwrap".into());
    let mut wrapped = vec![
        bwrap,
        "--die-with-parent".into(),
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
    ];
    if !sandbox.network_enabled || !sandbox.allowed_domains.is_empty() {
        wrapped.push("--unshare-net".into());
    }
    if matches!(sandbox.sandbox_mode, Some(SandboxMode::WorkspaceWrite)) {
        for root in &sandbox.writable_roots {
            let root = resolve_path(root)?;
            let root = root.to_string_lossy().into_owned();
            wrapped.extend(["--bind".into(), root.clone(), root]);
        }
    }
    wrapped.push("--".into());
    wrapped.extend(argv.iter().cloned());
    Ok(wrapped)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn restricted_command(_argv: &[String], _sandbox: &SandboxRequest) -> Result<Vec<String>> {
    bail!("sandbox_unavailable: restricted profiles require a supported OS sandbox")
}

#[cfg(target_os = "macos")]
fn escape_sandbox_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Resolve the longest existing prefix before comparing a path with its
/// allowed roots. This catches `..` traversal and symlinks in a parent of a
/// new target, while still supporting files that have not been created yet.
fn resolve_path(path: &str) -> Result<PathBuf> {
    let input = Path::new(path);
    let absolute = if input.is_absolute() {
        input.to_path_buf()
    } else {
        std::env::current_dir()?.join(input)
    };

    let mut existing = absolute.as_path();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("invalid path: {path}"))?;
        suffix.push(name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| anyhow::anyhow!("invalid path: {path}"))?;
    }

    let mut resolved = existing.canonicalize()?;
    for component in suffix.iter().rev() {
        resolved.push(component);
    }

    // Existing components may include a lexical `.` or `..` sequence on a
    // platform where it was not normalised by canonicalize above.
    let mut normalized = PathBuf::new();
    for component in resolved.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}
