//! Gitignore-aware file discovery.
//!
//! Walks `root` up to `max_depth` levels, respects `.gitignore` /
//! `.codezillaignore`, skips binary files, and caps the result to keep
//! memory usage predictable.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

// ─── Language ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    Rust,
    TypeScript,
    JavaScript,
    Python,
    Go,
    Java,
    C,
    Cpp,
    Json,
    Toml,
    Yaml,
    Markdown,
    Shell,
    Other(String),
}

impl Language {
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Language::Rust,
            "ts" | "tsx" => Language::TypeScript,
            "js" | "jsx" | "mjs" | "cjs" => Language::JavaScript,
            "py" | "pyw" => Language::Python,
            "go" => Language::Go,
            "java" => Language::Java,
            "c" | "h" => Language::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Language::Cpp,
            "json" => Language::Json,
            "toml" => Language::Toml,
            "yaml" | "yml" => Language::Yaml,
            "md" | "mdx" => Language::Markdown,
            "sh" | "bash" | "zsh" | "fish" => Language::Shell,
            other => Language::Other(other.to_string()),
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Language::Rust => "Rust",
            Language::TypeScript => "TypeScript",
            Language::JavaScript => "JavaScript",
            Language::Python => "Python",
            Language::Go => "Go",
            Language::Java => "Java",
            Language::C => "C",
            Language::Cpp => "C++",
            Language::Json => "JSON",
            Language::Toml => "TOML",
            Language::Yaml => "YAML",
            Language::Markdown => "Markdown",
            Language::Shell => "Shell",
            Language::Other(s) => s.as_str(),
        }
    }

    /// Returns true for languages we can extract symbols from.
    pub fn is_indexable(&self) -> bool {
        matches!(
            self,
            Language::Rust
                | Language::TypeScript
                | Language::JavaScript
                | Language::Python
                | Language::Go
                | Language::Java
                | Language::C
                | Language::Cpp
        )
    }
}

// ─── FileSummary ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FileSummary {
    /// Path relative to the walk root.
    pub rel_path: PathBuf,
    pub lang: Language,
    #[allow(dead_code)]
    pub size_bytes: u64,
    pub is_binary: bool,
}

// ─── Walker ───────────────────────────────────────────────────────────────────

/// Walk `root` and return a file list suitable for symbol extraction.
///
/// Respects `.gitignore`, `.ignore`, and `.codezillaignore` files.
/// Binary files are included in the listing (so the model knows they exist)
/// but flagged with `is_binary = true`.
pub fn walk_repo(root: &Path, max_depth: usize, max_files: usize) -> Vec<FileSummary> {
    let mut results = Vec::new();

    let walker = WalkBuilder::new(root)
        .max_depth(Some(max_depth))
        .hidden(false) // include dot-files; .gitignore still applies
        .ignore(true) // respect .ignore
        .git_ignore(true) // respect .gitignore
        .git_global(false) // skip global gitignore (avoids user-specific noise)
        .git_exclude(false)
        .add_custom_ignore_filename(".codezillaignore")
        .follow_links(false)
        .sort_by_file_path(|a, b| a.cmp(b))
        .build();

    for entry in walker.flatten() {
        if results.len() >= max_files {
            break;
        }

        let path = entry.path();

        // Skip directories (WalkBuilder yields them too)
        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }

        let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);

        // Skip very large files (> 1 MB) — too expensive to scan
        if size_bytes > 1_048_576 {
            continue;
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let lang = Language::from_extension(ext);

        let rel_path = path.strip_prefix(root).unwrap_or(path).to_path_buf();

        // Binary detection: read first 8 KB and check for null bytes.
        let is_binary = is_binary_file(path);

        results.push(FileSummary {
            rel_path,
            lang,
            size_bytes,
            is_binary,
        });
    }

    results
}

fn is_binary_file(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 8192];
    let n = file.read(&mut buf).unwrap_or(0);
    buf[..n].contains(&0u8)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_from_extension_roundtrip() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("TS"), Language::TypeScript);
        assert!(Language::from_extension("wasm").is_other());
    }

    impl Language {
        fn is_other(&self) -> bool {
            matches!(self, Language::Other(_))
        }
    }
}
