use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
        self.ensure_command_allowed(argv, sandbox)?;
        self.ensure_writable_root(cwd, sandbox, false)?;

        let mut command = Command::new(&argv[0]);
        if argv.len() > 1 {
            command.args(&argv[1..]);
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
        if let Some(parent) = Path::new(path).parent() {
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

    fn ensure_command_allowed(&self, argv: &[String], sandbox: &SandboxRequest) -> Result<()> {
        match sandbox.sandbox_mode.unwrap_or(SandboxMode::WorkspaceWrite) {
            SandboxMode::DangerFullAccess | SandboxMode::External => Ok(()),
            SandboxMode::ReadOnly | SandboxMode::WorkspaceWrite => {
                let joined = argv.join(" ").to_lowercase();
                if !sandbox.network_enabled
                    && [
                        "curl ",
                        "wget ",
                        "ssh ",
                        "scp ",
                        "git clone",
                        "npm install",
                        "cargo install",
                    ]
                    .iter()
                    .any(|needle| joined.contains(needle))
                {
                    bail!("network_blocked: command appears to require network access");
                }
                Ok(())
            }
        }
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
        let path = Path::new(path)
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(path));
        let allowed = sandbox
            .writable_roots
            .iter()
            .map(PathBuf::from)
            .any(|root| path.starts_with(root));
        if allowed {
            Ok(())
        } else {
            bail!("path_not_writable: {path:?} is outside writable roots");
        }
    }
}
