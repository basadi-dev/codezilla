//! Token-budgeted repo-map formatter.
//!
//! Converts the per-file symbol lists into a compact text block suitable for
//! injection into the system prompt.  The output is deliberately terse — no
//! line numbers, no type annotations, no docstrings — so it costs as few
//! tokens as possible while still giving the model a navigable map.
//!
//! ## Budget enforcement
//!
//! We approximate "tokens" as `chars / 4` (the standard rough estimate).
//! Files are emitted in lexicographic order. Once the running character count
//! exceeds the budget, remaining files are listed with a single-line stub
//! `(truncated — use list_dir for details)` and no symbols.

use std::fmt::Write;

use super::symbols::Symbol;
use super::walker::FileSummary;

// ─── FileEntry ────────────────────────────────────────────────────────────────

/// A file and its extracted symbols, ready to be formatted.
pub struct FileEntry<'a> {
    pub summary: &'a FileSummary,
    pub symbols: Vec<Symbol>,
}

// ─── format_repo_map ──────────────────────────────────────────────────────────

/// Build the repo-map string from a list of `FileEntry` values.
///
/// `token_budget` is the maximum number of *tokens* (chars ÷ 4) to emit.
/// Files with symbols are prioritised; binary / non-indexable files are listed
/// last with a single line annotation.
pub fn format_repo_map(cwd: &str, files: &[FileEntry<'_>], token_budget: usize) -> String {
    let char_budget = token_budget * 4;
    let mut out = String::new();
    let _ = writeln!(out, "## Repository map  (cwd: {cwd})");
    let mut used_chars = out.len();

    // Split into indexable (have symbols or are text) and non-indexable.
    let (mut indexable, non_indexable): (Vec<_>, Vec<_>) = files
        .iter()
        .partition(|e| !e.summary.is_binary && e.summary.lang.is_indexable());

    // Sort: most symbols first (highest information density).
    indexable.sort_by(|a, b| b.symbols.len().cmp(&a.symbols.len()));

    let mut truncated = false;

    for entry in &indexable {
        if truncated {
            break;
        }

        let path_str = entry.summary.rel_path.display().to_string();
        let lang_label = entry.summary.lang.label();

        // File header line
        let header = format!("{path_str}  [{lang_label}]\n");

        if used_chars + header.len() > char_budget {
            truncated = true;
            break;
        }
        out.push_str(&header);
        used_chars += header.len();

        // Symbol lines
        for sym in &entry.symbols {
            let sym_line = format!("  {} {}\n", sym.kind.label(), sym.name);
            if used_chars + sym_line.len() > char_budget {
                truncated = true;
                break;
            }
            out.push_str(&sym_line);
            used_chars += sym_line.len();
        }
    }

    // Append non-indexable files (config, markdown, binary…)
    let mut other_header_written = false;
    for entry in &non_indexable {
        if truncated {
            break;
        }
        let path_str = entry.summary.rel_path.display().to_string();
        let ann = if entry.summary.is_binary {
            "binary"
        } else {
            "config/docs"
        };
        let line = format!("{path_str}  [{ann}]\n");

        if used_chars + line.len() > char_budget {
            truncated = true;
            break;
        }

        if !other_header_written {
            let sep = "---\n";
            if used_chars + sep.len() <= char_budget {
                out.push_str(sep);
                used_chars += sep.len();
                other_header_written = true;
            }
        }

        out.push_str(&line);
        used_chars += line.len();
    }

    if truncated {
        out.push_str("... (map truncated — use list_dir for the full file tree)\n");
    }

    out
}

/// Convenience: render a single-line symbol for display in nudge messages.
#[allow(dead_code)]
pub fn symbol_display(sym: &Symbol) -> String {
    format!("{} {}", sym.kind.label(), sym.name)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::intel::walker::Language;
    use std::path::PathBuf;

    fn make_entry(path: &str, lang: Language) -> FileSummary {
        FileSummary {
            rel_path: PathBuf::from(path),
            lang,
            size_bytes: 100,
            is_binary: false,
        }
    }

    #[test]
    fn basic_format() {
        use crate::system::intel::symbols::SymbolKind;
        let summary = make_entry("src/main.rs", Language::Rust);
        let entry = FileEntry {
            summary: &summary,
            symbols: vec![Symbol {
                name: "main".into(),
                kind: SymbolKind::Function,
                line: 1,
            }],
        };
        let map = format_repo_map("/repo", &[entry], 2000);
        assert!(map.contains("src/main.rs"));
        assert!(map.contains("fn main"));
    }

    #[test]
    fn budget_respected() {
        use crate::system::intel::symbols::SymbolKind;
        let summaries: Vec<FileSummary> = (0..100)
            .map(|i| make_entry(&format!("src/file{i}.rs"), Language::Rust))
            .collect();
        let entries: Vec<FileEntry> = summaries
            .iter()
            .map(|s| FileEntry {
                summary: s,
                symbols: vec![Symbol {
                    name: "big_function".into(),
                    kind: SymbolKind::Function,
                    line: 1,
                }],
            })
            .collect();
        // Very small budget → must truncate
        let map = format_repo_map("/repo", &entries, 50);
        assert!(map.contains("truncated"));
    }
}
