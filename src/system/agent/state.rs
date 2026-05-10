//! StateManager trait — Phase 3 of the agent architecture.
#![allow(dead_code)] // Public API; not yet wired into the default runtime build.
//!
//! Abstracts how the agent saves and restores workspace state. Two main uses:
//!
//! 1. **Turn-level rollback**: snapshot before a risky edit, restore if the
//!    result fails review. Builds on the file-level `CheckpointStore` for
//!    individual tool calls; `StateManager` covers the whole turn at once.
//!
//! 2. **Tree-of-Thought forking**: `branch()` duplicates the workspace into
//!    an isolated temp directory so multiple plan candidates can execute
//!    concurrently without stepping on each other's edits. The winning branch
//!    is merged back; losers are discarded.
//!
//! # Implementations
//! * `InMemoryStateManager` — captures file bytes in a `HashMap`. Fast,
//!   zero-dependency, limited to workspaces < memory budget. Good for most
//!   agent scenarios.
//!
//! A `GitStateManager` (git stash / worktree) can be added later without
//! changing the call sites — all callers use the trait.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use walkdir::WalkDir;

// ─── Public value types ───────────────────────────────────────────────────────

/// Metadata for a captured workspace snapshot.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Unique snapshot identifier (e.g. `snap_<uuid>`).
    pub id: String,
    /// Human-readable label set by the caller.
    pub label: String,
    /// Unix timestamp when the snapshot was taken.
    pub created_at: u64,
    /// Number of files captured.
    pub file_count: usize,
}

/// Handle to an isolated workspace fork produced by `StateManager::branch`.
#[derive(Debug, Clone)]
pub struct BranchHandle {
    /// Unique branch identifier.
    pub branch_id: String,
    /// Path to the isolated workspace root.
    pub workspace_path: PathBuf,
    /// Snapshot this branch was forked from.
    pub source_snapshot_id: String,
}

// ─── StateManager trait ───────────────────────────────────────────────────────

/// Manages workspace snapshots and branches for speculative execution.
///
/// All methods are `async` to allow implementations that call git or other
/// blocking I/O. The trait is object-safe (`dyn StateManager` works fine).
#[async_trait]
pub trait StateManager: Send + Sync {
    /// A unique identifier used in logs.
    fn name(&self) -> &str;

    /// Capture the current state of `cwd` and return a snapshot handle.
    async fn snapshot(&self, cwd: &str, label: &str) -> Result<StateSnapshot>;

    /// Restore `cwd` to the state captured in `snapshot_id`.
    async fn restore(&self, snapshot_id: &str, cwd: &str) -> Result<()>;

    /// Fork `snapshot_id` into an isolated temp workspace for parallel
    /// exploration. Returns a handle whose `workspace_path` is ready to use.
    async fn branch(&self, snapshot_id: &str, branch_label: &str) -> Result<BranchHandle>;

    /// Copy all files from `branch.workspace_path` back into `target_cwd`,
    /// completing a successful exploration.
    async fn merge_branch(&self, branch: &BranchHandle, target_cwd: &str) -> Result<()>;

    /// List every snapshot in creation order (oldest first).
    fn list_snapshots(&self) -> Vec<StateSnapshot>;

    /// Remove a snapshot and free associated memory.
    /// Returns `true` if the snapshot existed, `false` if it was already gone.
    fn drop_snapshot(&self, snapshot_id: &str) -> bool;
}

// ─── InMemoryStateManager ─────────────────────────────────────────────────────

/// Per-file bytes captured at snapshot time.
#[derive(Debug, Clone)]
struct FileEntry {
    /// `None` means the file did not exist — restoring should delete it.
    content: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct InMemorySnapshot {
    meta: StateSnapshot,
    /// Relative-to-cwd path → content
    files: HashMap<String, FileEntry>,
}

/// In-process implementation of `StateManager`. Suitable for agent workspaces
/// up to a few hundred MB. No external dependencies.
pub struct InMemoryStateManager {
    /// snapshot_id → data
    snapshots: Arc<Mutex<HashMap<String, InMemorySnapshot>>>,
    /// Insertion order for `list_snapshots()`.
    order: Arc<Mutex<Vec<String>>>,
    /// Per-file size cap. Files larger than this are skipped during snapshots.
    max_file_bytes: usize,
}

impl Default for InMemoryStateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStateManager {
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(Mutex::new(HashMap::new())),
            order: Arc::new(Mutex::new(Vec::new())),
            max_file_bytes: 10 * 1024 * 1024, // 10 MB per file
        }
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Walk `cwd` and capture all files up to `max_file_bytes` each.
    fn scan_directory(cwd: &str, max_file_bytes: usize) -> HashMap<String, FileEntry> {
        let mut files = HashMap::new();
        let base = Path::new(cwd);
        for entry in WalkDir::new(cwd)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if entry
                .metadata()
                .map(|m| m.len() as usize > max_file_bytes)
                .unwrap_or(false)
            {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(base)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| entry.path().to_string_lossy().to_string());
            let content = std::fs::read(entry.path()).ok();
            files.insert(relative, FileEntry { content });
        }
        files
    }

    fn new_id(prefix: &str) -> String {
        format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
    }
}

#[async_trait]
impl StateManager for InMemoryStateManager {
    fn name(&self) -> &str {
        "in_memory"
    }

    async fn snapshot(&self, cwd: &str, label: &str) -> Result<StateSnapshot> {
        let id = Self::new_id("snap");
        let max_bytes = self.max_file_bytes;
        let files = tokio::task::spawn_blocking({
            let cwd = cwd.to_string();
            move || Self::scan_directory(&cwd, max_bytes)
        })
        .await?;

        let meta = StateSnapshot {
            id: id.clone(),
            label: label.to_string(),
            created_at: Self::now_secs(),
            file_count: files.len(),
        };
        {
            let mut snaps = self.snapshots.lock().unwrap();
            snaps.insert(
                id.clone(),
                InMemorySnapshot {
                    meta: meta.clone(),
                    files,
                },
            );
        }
        self.order.lock().unwrap().push(id.clone());

        tracing::debug!(
            snapshot_id = %id,
            label,
            file_count = meta.file_count,
            "state_manager: snapshot captured"
        );
        Ok(meta)
    }

    async fn restore(&self, snapshot_id: &str, cwd: &str) -> Result<()> {
        let snap = {
            self.snapshots
                .lock()
                .unwrap()
                .get(snapshot_id)
                .cloned()
                .ok_or_else(|| anyhow!("snapshot_not_found: {snapshot_id}"))?
        };
        let base = Path::new(cwd);
        let mut restored = 0usize;
        for (relative, entry) in &snap.files {
            let full = base.join(relative);
            match &entry.content {
                Some(bytes) => {
                    if let Some(parent) = full.parent() {
                        tokio::fs::create_dir_all(parent).await?;
                    }
                    tokio::fs::write(&full, bytes).await?;
                    restored += 1;
                }
                None => {
                    let _ = tokio::fs::remove_file(&full).await;
                }
            }
        }
        tracing::info!(
            snapshot_id,
            cwd,
            restored,
            "state_manager: workspace restored"
        );
        Ok(())
    }

    async fn branch(&self, snapshot_id: &str, branch_label: &str) -> Result<BranchHandle> {
        let snap = {
            self.snapshots
                .lock()
                .unwrap()
                .get(snapshot_id)
                .cloned()
                .ok_or_else(|| anyhow!("snapshot_not_found: {snapshot_id}"))?
        };
        let branch_dir = std::env::temp_dir().join(Self::new_id("cz_branch"));
        tokio::fs::create_dir_all(&branch_dir).await?;
        for (relative, entry) in &snap.files {
            if let Some(bytes) = &entry.content {
                let dest = branch_dir.join(relative);
                if let Some(parent) = dest.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                tokio::fs::write(&dest, bytes).await?;
            }
        }
        let handle = BranchHandle {
            branch_id: Self::new_id("branch"),
            workspace_path: branch_dir.clone(),
            source_snapshot_id: snapshot_id.to_string(),
        };
        tracing::info!(
            branch_id = %handle.branch_id,
            label = branch_label,
            source = snapshot_id,
            path = %branch_dir.display(),
            "state_manager: branch created"
        );
        Ok(handle)
    }

    async fn merge_branch(&self, branch: &BranchHandle, target_cwd: &str) -> Result<()> {
        let source = branch.workspace_path.clone();
        let target = PathBuf::from(target_cwd);
        let paths: Vec<PathBuf> = tokio::task::spawn_blocking({
            let source = source.clone();
            move || {
                WalkDir::new(&source)
                    .follow_links(false)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
                    .map(|e| e.path().to_path_buf())
                    .collect()
            }
        })
        .await?;

        let mut merged = 0usize;
        for file_path in &paths {
            let relative = file_path.strip_prefix(&source)?;
            let dest = target.join(relative);
            if let Some(parent) = dest.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::copy(file_path, &dest).await?;
            merged += 1;
        }
        tracing::info!(
            branch_id = %branch.branch_id,
            target_cwd,
            merged,
            "state_manager: branch merged"
        );
        Ok(())
    }

    fn list_snapshots(&self) -> Vec<StateSnapshot> {
        let order = self.order.lock().unwrap().clone();
        let snaps = self.snapshots.lock().unwrap();
        order
            .iter()
            .filter_map(|id| snaps.get(id).map(|s| s.meta.clone()))
            .collect()
    }

    fn drop_snapshot(&self, snapshot_id: &str) -> bool {
        let removed = self.snapshots.lock().unwrap().remove(snapshot_id).is_some();
        if removed {
            self.order.lock().unwrap().retain(|id| id != snapshot_id);
        }
        removed
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn snapshot_creates_entry_with_correct_label() {
        let mgr = InMemoryStateManager::new();
        assert!(mgr.list_snapshots().is_empty());

        let tmp = TempDir::new().unwrap();
        let snap = mgr
            .snapshot(tmp.path().to_str().unwrap(), "before_edit")
            .await
            .unwrap();

        assert!(!snap.id.is_empty());
        assert_eq!(snap.label, "before_edit");

        let listed = mgr.list_snapshots();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, snap.id);
    }

    #[tokio::test]
    async fn multiple_snapshots_listed_in_creation_order() {
        let mgr = InMemoryStateManager::new();
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().to_str().unwrap();

        let s1 = mgr.snapshot(cwd, "first").await.unwrap();
        let s2 = mgr.snapshot(cwd, "second").await.unwrap();
        let s3 = mgr.snapshot(cwd, "third").await.unwrap();

        let listed = mgr.list_snapshots();
        assert_eq!(listed[0].id, s1.id);
        assert_eq!(listed[1].id, s2.id);
        assert_eq!(listed[2].id, s3.id);
    }

    #[tokio::test]
    async fn drop_snapshot_removes_entry() {
        let mgr = InMemoryStateManager::new();
        let tmp = TempDir::new().unwrap();
        let snap = mgr
            .snapshot(tmp.path().to_str().unwrap(), "test")
            .await
            .unwrap();

        assert!(mgr.drop_snapshot(&snap.id));
        assert!(mgr.list_snapshots().is_empty());
        assert!(
            !mgr.drop_snapshot(&snap.id),
            "second drop must return false"
        );
    }

    #[tokio::test]
    async fn restore_unknown_snapshot_returns_error() {
        let mgr = InMemoryStateManager::new();
        let tmp = TempDir::new().unwrap();
        let err = mgr
            .restore("nonexistent_id", tmp.path().to_str().unwrap())
            .await;
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("snapshot_not_found"));
    }

    #[tokio::test]
    async fn restore_puts_files_back() {
        let mgr = InMemoryStateManager::new();
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();

        // Write a file and take a snapshot.
        let file_path = cwd.join("hello.txt");
        fs::write(&file_path, b"original content").unwrap();
        let snap = mgr.snapshot(cwd.to_str().unwrap(), "v1").await.unwrap();

        // Overwrite the file.
        fs::write(&file_path, b"modified content").unwrap();

        // Restore.
        mgr.restore(&snap.id, cwd.to_str().unwrap()).await.unwrap();
        let restored = fs::read(&file_path).unwrap();
        assert_eq!(restored, b"original content");
    }

    #[tokio::test]
    async fn branch_creates_isolated_workspace() {
        let mgr = InMemoryStateManager::new();
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();
        fs::write(cwd.join("main.rs"), b"fn main() {}").unwrap();

        let snap = mgr.snapshot(cwd.to_str().unwrap(), "base").await.unwrap();
        let branch = mgr.branch(&snap.id, "explore_a").await.unwrap();

        assert!(branch.workspace_path.exists());
        assert_eq!(branch.source_snapshot_id, snap.id);
        assert!(
            branch.workspace_path.join("main.rs").exists(),
            "branch should contain the file"
        );

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&branch.workspace_path).await;
    }

    #[tokio::test]
    async fn merge_branch_copies_files_to_target() {
        let mgr = InMemoryStateManager::new();
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path();

        // Create initial workspace.
        fs::write(cwd.join("lib.rs"), b"pub fn foo() {}").unwrap();
        let snap = mgr.snapshot(cwd.to_str().unwrap(), "base").await.unwrap();

        // Fork a branch and write a new file there.
        let branch = mgr.branch(&snap.id, "exploration").await.unwrap();
        fs::write(
            branch.workspace_path.join("new_feature.rs"),
            b"// new feature",
        )
        .unwrap();

        // Merge back.
        mgr.merge_branch(&branch, cwd.to_str().unwrap())
            .await
            .unwrap();
        assert!(
            cwd.join("new_feature.rs").exists(),
            "merged file should appear in target"
        );

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&branch.workspace_path).await;
    }
}
