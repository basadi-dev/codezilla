//! Codebase intelligence — repo map, symbol index, and file cache.
//!
//! The primary entry point is [`RepoMap`], which builds a compact structural
//! summary of the repository from a working directory path.  This summary is
//! injected into the system prompt at the start of every agent turn so the
//! model can navigate the code without issuing sequential `read_file` /
//! `list_dir` calls.
//!
//! # Design decisions
//! - **No tree-sitter**: symbols are extracted with `Regex` patterns compiled
//!   once via `once_cell::sync::Lazy`. Pattern quality is "good enough" for
//!   navigation — not perfect AST accuracy.
//! - **No external processes**: the `ignore` crate (same engine as ripgrep)
//!   handles `.gitignore` traversal in pure Rust.
//! - **O(1) per-call overhead after first run**: a SHA2-keyed in-process cache
//!   stores symbol results so unchanged files are not re-parsed.

pub mod cache;
pub mod format;
pub mod symbols;
pub mod walker;

use std::path::Path;
use std::sync::Arc;

use cache::IntelCache;
use format::{format_repo_map, FileEntry};
use symbols::extract_symbols;
use walker::{walk_repo, FileSummary};

// ─── CodebaseIntelConfig ──────────────────────────────────────────────────────

/// Configuration knobs for the codebase intelligence layer.
/// Matched against `config.yaml` under the `codebase_intel:` key.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CodebaseIntelConfig {
    /// Enable or disable the repo map. Default: true.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Maximum number of files to walk per turn. Default: 500.
    #[serde(default = "default_max_files")]
    pub max_files: usize,
    /// Maximum walk depth from cwd. Default: 4.
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    /// Token budget for the formatted map. Default: 2000.
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
    /// Include non-indexable files (docs/config) in the repo map.
    /// Default: false.
    #[serde(default)]
    pub include_non_indexable: bool,
    /// Include binary files in the repo map.
    /// Default: false.
    #[serde(default)]
    pub include_binary: bool,
    /// Include hidden files/directories during walk.
    /// Default: false.
    #[serde(default)]
    pub include_hidden: bool,
    /// Include entries from excluded/internal directories (e.g. `.git`, `target`).
    /// Default: false.
    #[serde(default)]
    pub include_excluded_paths: bool,
}

impl Default for CodebaseIntelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_files: 500,
            max_depth: 4,
            token_budget: 2000,
            include_non_indexable: false,
            include_binary: false,
            include_hidden: false,
            include_excluded_paths: false,
        }
    }
}

fn default_enabled() -> bool {
    true
}
fn default_max_files() -> usize {
    500
}
fn default_max_depth() -> usize {
    4
}
fn default_token_budget() -> usize {
    2000
}

// ─── RepoMap ──────────────────────────────────────────────────────────────────

/// High-level handle for building a repo map for a given `cwd`.
///
/// Held as `Arc<RepoMap>` inside `RuntimeInner` so it is shared across all
/// threads and turns for the lifetime of the process.
pub struct RepoMap {
    cache: Arc<IntelCache>,
}

impl RepoMap {
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            cache: Arc::new(IntelCache::new(cache_capacity)),
        }
    }

    /// Return a reference to the underlying cache so `FileToolProvider` can
    /// call `invalidate()` after writes without holding the full `RepoMap`.
    pub fn cache(&self) -> Arc<IntelCache> {
        self.cache.clone()
    }

    /// Build a repo map string for `cwd` using the supplied config.
    ///
    /// Returns `None` when `config.enabled` is false or `cwd` does not exist.
    /// Never panics — all I/O errors are silently swallowed and result in a
    /// partial or empty map (the agent falls back to tool-call exploration).
    pub fn build_map(&self, cwd: &str, config: &CodebaseIntelConfig) -> Option<String> {
        if !config.enabled {
            return None;
        }
        let root = Path::new(cwd);
        if !root.exists() {
            tracing::debug!(cwd, "intel: cwd does not exist, skipping repo map");
            return None;
        }

        let t0 = std::time::Instant::now();

        let file_list = walk_repo(
            root,
            config.max_depth,
            config.max_files,
            config.include_hidden,
            config.include_excluded_paths,
        );
        let walk_elapsed = t0.elapsed();

        if file_list.is_empty() {
            tracing::debug!(cwd, "intel: walk returned 0 files, skipping repo map");
            return None;
        }

        let indexable_count = file_list
            .iter()
            .filter(|f| !f.is_binary && f.lang.is_indexable())
            .count();
        let mut cache_hits = 0usize;
        let mut cache_misses = 0usize;
        let mut total_symbols = 0usize;

        let entries: Vec<FileEntry<'_>> = file_list
            .iter()
            .map(|summary| {
                let symbols =
                    self.symbols_for_tracked(root, summary, &mut cache_hits, &mut cache_misses);
                total_symbols += symbols.len();
                FileEntry { summary, symbols }
            })
            .collect();

        let map = format_repo_map(
            cwd,
            &entries,
            config.token_budget,
            config.include_non_indexable,
            config.include_binary,
        );
        let total_elapsed = t0.elapsed();

        if map.trim().is_empty() {
            tracing::debug!(cwd, "intel: formatted map is empty after budget trim");
            None
        } else {
            tracing::info!(
                cwd,
                files = file_list.len(),
                indexable = indexable_count,
                symbols = total_symbols,
                cache_hits,
                cache_misses,
                walk_ms = walk_elapsed.as_millis() as u64,
                total_ms = total_elapsed.as_millis() as u64,
                map_chars = map.len(),
                "intel: repo map built"
            );
            Some(map)
        }
    }

    // ── private ───────────────────────────────────────────────────────────────

    fn symbols_for_tracked(
        &self,
        root: &Path,
        summary: &FileSummary,
        cache_hits: &mut usize,
        cache_misses: &mut usize,
    ) -> Vec<symbols::Symbol> {
        // Skip binary files and non-indexable languages immediately.
        if summary.is_binary || !summary.lang.is_indexable() {
            return Vec::new();
        }

        let abs_path = root.join(&summary.rel_path);

        // Try to read the file; silently skip on I/O error.
        let Ok(content) = std::fs::read_to_string(&abs_path) else {
            return Vec::new();
        };

        // Cache hit?
        if let Some(cached) = self.cache.get(&abs_path, &content) {
            *cache_hits += 1;
            return cached;
        }

        // Cache miss — extract and store.
        *cache_misses += 1;
        let extracted = extract_symbols(&content, &summary.lang);
        self.cache.put(&abs_path, &content, extracted.clone());
        extracted
    }
}
