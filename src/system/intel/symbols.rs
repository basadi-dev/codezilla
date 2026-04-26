//! Regex-based symbol extraction.
//!
//! Extracts top-level declarations from source files without tree-sitter or any
//! external binary. Regex patterns are compiled once via `once_cell::sync::Lazy`
//! and reused across calls. Extraction is capped at `MAX_SYMBOLS_PER_FILE` and
//! `MAX_LINES_PER_FILE` to keep per-file cost O(1).

use once_cell::sync::Lazy;
use regex::Regex;

use super::walker::Language;

// ─── Symbol ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    #[allow(dead_code)]
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Impl,
    Type,
    Module,
    Class,
    Interface,
    Constant,
    Variable,
    Other,
}

impl SymbolKind {
    pub fn label(&self) -> &str {
        match self {
            SymbolKind::Function => "fn",
            SymbolKind::Method => "method",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Type => "type",
            SymbolKind::Module => "mod",
            SymbolKind::Class => "class",
            SymbolKind::Interface => "interface",
            SymbolKind::Constant => "const",
            SymbolKind::Variable => "var",
            SymbolKind::Other => "decl",
        }
    }
}

// ─── Per-language compiled patterns ──────────────────────────────────────────

/// Each entry is `(Regex, visibility_group_index, kind_group_index, name_group_index)`.
/// The visibility group (index 1) determines if the symbol is public (optional).
/// The kind group determines the SymbolKind. The name group is the symbol name.
struct PatternEntry {
    re: Regex,
    /// Capture group index for visibility prefix ("pub", "export", etc.). 0 = unused.
    vis_group: usize,
    /// Capture group index for the keyword ("fn", "struct", etc.).
    kind_group: usize,
    /// Capture group index for the symbol name.
    name_group: usize,
}

// ── Rust ──────────────────────────────────────────────────────────────────────

static RUST_PATTERNS: Lazy<Vec<PatternEntry>> = Lazy::new(|| {
    vec![
        // pub(crate)? fn name / pub async fn name / pub(super) fn name
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(?:async\s+)?(fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // pub struct / struct
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // pub enum / enum
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // pub trait / trait
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(trait)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // impl SomeTrait for SomeType / impl SomeType
        PatternEntry {
            re: Regex::new(r"^()(impl)(?:<[^>]*>)?\s+(?:(?:[a-zA-Z_][a-zA-Z0-9_:]*)\s+for\s+)?([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // pub type / type
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(type)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // pub mod / mod
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(mod)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // pub const
        PatternEntry {
            re: Regex::new(r"^(pub(?:\([^)]+\))?\s+)?(const)\s+([A-Z_][A-Z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
    ]
});

// ── TypeScript / JavaScript ───────────────────────────────────────────────────

static TS_PATTERNS: Lazy<Vec<PatternEntry>> = Lazy::new(|| {
    vec![
        // export function / export async function
        PatternEntry {
            re: Regex::new(
                r"^(export\s+)?(?:default\s+)?(?:async\s+)?(function)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)",
            )
            .unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // export class
        PatternEntry {
            re: Regex::new(r"^(export\s+)?(class)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // export interface
        PatternEntry {
            re: Regex::new(r"^(export\s+)?(interface)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // export type Foo = ...
        PatternEntry {
            re: Regex::new(r"^(export\s+)?(type)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*[=<]").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // export enum
        PatternEntry {
            re: Regex::new(r"^(export\s+)?(enum)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // export const FOO / export const foo =
        PatternEntry {
            re: Regex::new(r"^(export\s+)?(const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
    ]
});

// ── Python ────────────────────────────────────────────────────────────────────

static PYTHON_PATTERNS: Lazy<Vec<PatternEntry>> = Lazy::new(|| {
    vec![
        // def foo / async def foo
        PatternEntry {
            re: Regex::new(r"^()((?:async\s+)?def)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // class Foo
        PatternEntry {
            re: Regex::new(r"^()(class)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
    ]
});

// ── Go ────────────────────────────────────────────────────────────────────────

// NOTE: Go patterns intentionally match only uppercase (exported) identifiers.
// In Go, lowercase identifiers are package-private and generally not useful for
// cross-file navigation in the repo map.
static GO_PATTERNS: Lazy<Vec<PatternEntry>> = Lazy::new(|| {
    vec![
        // func FuncName / func (recv) FuncName
        PatternEntry {
            re: Regex::new(r"^()(func)\s+(?:\([^)]*\)\s+)?([A-Z][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // type Foo struct / type Foo interface / type Foo ...
        PatternEntry {
            re: Regex::new(r"^()(type)\s+([A-Z][a-zA-Z0-9_]*)\s+(?:struct|interface)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
        // const / var at package level (exported only, starts with uppercase)
        PatternEntry {
            re: Regex::new(r"^()(const|var)\s+([A-Z][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 1,
            kind_group: 2,
            name_group: 3,
        },
    ]
});

// ── Java ──────────────────────────────────────────────────────────────────────

static JAVA_PATTERNS: Lazy<Vec<PatternEntry>> = Lazy::new(|| {
    vec![
        // public class / interface / enum / record
        PatternEntry {
            re: Regex::new(r"^\s*(public\s+(?:abstract\s+|final\s+)?)(class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap(),
            vis_group: 1, kind_group: 2, name_group: 3,
        },
        // public ... method: public [static] [final] <returnType> methodName(
        // kind_group is 0 (unused) — we hardcode Function kind via the empty
        // keyword string falling through to SymbolKind::Other, so we override
        // in keyword_to_kind below by matching "method".
        PatternEntry {
            re: Regex::new(r"^\s*(public\s+(?:static\s+|final\s+|abstract\s+)*)(?:[A-Za-z<>\[\],\s]+\s+)([a-z][a-zA-Z0-9_]*)\s*\(").unwrap(),
            vis_group: 1, kind_group: 0, name_group: 2,
        },
    ]
});

// ── C / C++ ───────────────────────────────────────────────────────────────────

static C_PATTERNS: Lazy<Vec<PatternEntry>> = Lazy::new(|| {
    vec![
        // Simple function definition: return_type funcname(
        PatternEntry {
            re: Regex::new(r"^(?:static\s+|extern\s+)?(?:inline\s+)?[a-zA-Z_][a-zA-Z0-9_ *]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]").unwrap(),
            vis_group: 0, kind_group: 0, name_group: 1,
        },
        // struct / enum / union tag names
        PatternEntry {
            re: Regex::new(r"^(?:typedef\s+)?(struct|enum|union)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap(),
            vis_group: 0, kind_group: 1, name_group: 2,
        },
    ]
});

// ─── Extraction entry point ───────────────────────────────────────────────────

const MAX_SYMBOLS_PER_FILE: usize = 40;
const MAX_LINES_PER_FILE: usize = 500;

/// Extract symbols from `content` for the given `lang`.
/// Returns up to `MAX_SYMBOLS_PER_FILE` symbols.
pub fn extract_symbols(content: &str, lang: &Language) -> Vec<Symbol> {
    let patterns: &[PatternEntry] = match lang {
        Language::Rust => &RUST_PATTERNS,
        Language::TypeScript | Language::JavaScript => &TS_PATTERNS,
        Language::Python => &PYTHON_PATTERNS,
        Language::Go => &GO_PATTERNS,
        Language::Java => &JAVA_PATTERNS,
        Language::C | Language::Cpp => &C_PATTERNS,
        _ => return Vec::new(),
    };

    let mut symbols = Vec::new();

    'outer: for (line_idx, line) in content.lines().enumerate().take(MAX_LINES_PER_FILE) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
            continue;
        }

        for pat in patterns {
            let Some(caps) = pat.re.captures(trimmed) else {
                continue;
            };

            let name = caps
                .get(pat.name_group)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            if name.is_empty() || name.starts_with('_') && name.len() == 1 {
                continue;
            }

            let kind_str = if pat.kind_group > 0 {
                caps.get(pat.kind_group).map(|m| m.as_str()).unwrap_or("")
            } else {
                ""
            };
            let kind = keyword_to_kind(kind_str);

            // Filter: skip private-looking Rust items (starts with _)
            // but keep everything for C/C++ where there's no visibility keyword.
            let is_public = if pat.vis_group > 0 {
                caps.get(pat.vis_group)
                    .map(|m| !m.as_str().is_empty())
                    .unwrap_or(false)
            } else {
                true // C/C++ — always include
            };

            // For languages where visibility matters (Rust, TS), skip non-pub items
            // unless the kind is `impl` or `class` (always useful for navigation).
            if !is_public
                && matches!(
                    lang,
                    Language::Rust | Language::TypeScript | Language::JavaScript
                )
                && !matches!(kind, SymbolKind::Impl | SymbolKind::Class)
            {
                continue;
            }

            symbols.push(Symbol {
                name,
                kind,
                line: line_idx + 1,
            });

            if symbols.len() >= MAX_SYMBOLS_PER_FILE {
                break 'outer;
            }
            break; // don't double-match the same line
        }
    }

    symbols
}

fn keyword_to_kind(kw: &str) -> SymbolKind {
    match kw {
        "fn" | "function" | "def" | "async def" | "func" => SymbolKind::Function,
        "struct" | "record" => SymbolKind::Struct,
        "enum" => SymbolKind::Enum,
        "trait" => SymbolKind::Trait,
        "impl" => SymbolKind::Impl,
        "type" => SymbolKind::Type,
        "mod" => SymbolKind::Module,
        "class" => SymbolKind::Class,
        "interface" => SymbolKind::Interface,
        "const" => SymbolKind::Constant,
        "var" => SymbolKind::Variable,
        _ => SymbolKind::Other,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_rust_pub_fn() {
        let src = "pub async fn run_turn(&self, params: Params) -> Result<()> {\n";
        let syms = extract_symbols(src, &Language::Rust);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "run_turn");
        assert_eq!(syms[0].kind, SymbolKind::Function);
    }

    #[test]
    fn skip_private_rust_fn() {
        let src = "fn private_helper() {}\n";
        let syms = extract_symbols(src, &Language::Rust);
        assert!(syms.is_empty());
    }

    #[test]
    fn extract_rust_struct() {
        let src = "pub struct TurnExecutor {\n";
        let syms = extract_symbols(src, &Language::Rust);
        assert_eq!(syms[0].name, "TurnExecutor");
        assert_eq!(syms[0].kind, SymbolKind::Struct);
    }

    #[test]
    fn extract_ts_export_fn() {
        let src = "export async function doSomething(x: number): Promise<void> {\n";
        let syms = extract_symbols(src, &Language::TypeScript);
        assert_eq!(syms[0].name, "doSomething");
    }

    #[test]
    fn extract_python_class_def() {
        let src = "class MyModel(BaseModel):\n";
        let syms = extract_symbols(src, &Language::Python);
        assert_eq!(syms[0].name, "MyModel");
        assert_eq!(syms[0].kind, SymbolKind::Class);
    }
}
