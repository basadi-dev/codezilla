//! Codebase knowledge graph — SQLite-backed symbol graph + query layer.
//!
//! This is the "codebase-memory" layer: it indexes a repository into a
//! persistent graph of nodes (functions, classes, files, …) and edges
//! (DEFINES, CALLS) stored in SQLite, then answers structural queries
//! (search, trace, impact) in sub-millisecond time.
//!
//! # Design
//! - **No tree-sitter**: reuses [`super::symbols`] (regex extraction) and
//!   [`super::walker`] (gitignore-aware discovery). This keeps the project
//!   dependency-free, matching the documented intel-layer decision.
//! - **SQLite storage**: `rusqlite` (already a dependency). One DB per project
//!   under `<app_home>/state/codebase-graph/<project-hash>.sqlite3`.
//! - **Call attribution**: call sites are attributed to the nearest preceding
//!   function/method symbol in the same file (falling back to the file node),
//!   so CALLS edges are function→function where the extractor permits.
//!   Resolution prefers same-file targets, then falls back to the first
//!   global match. No import maps, no LSP — that's a future enhancement.
//! - **Query layer**: BFS over `CALLS` edges powers `trace_path` / `find_impact`.
//!
//! # Known limitations (inherited from the regex extractor)
//! - Private Rust/TS symbols are not extracted, so calls made inside them are
//!   attributed to the nearest preceding public symbol or the file node.
//! - Same-name symbols across files resolve same-file first, then to the
//!   first global match — overloads may be misattributed.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context as _, Result};
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use super::symbols::{extract_symbols, Symbol, SymbolKind};
use super::walker::walk_repo;

// ─── Node / Edge model ────────────────────────────────────────────────────────

/// Graph node kinds. Mirrors the codebase-memory-mcp node taxonomy (subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Project,
    File,
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
    Other,
}

impl NodeKind {
    fn as_str(&self) -> &'static str {
        match self {
            NodeKind::Project => "project",
            NodeKind::File => "file",
            NodeKind::Function => "function",
            NodeKind::Method => "method",
            NodeKind::Struct => "struct",
            NodeKind::Enum => "enum",
            NodeKind::Trait => "trait",
            NodeKind::Impl => "impl",
            NodeKind::Type => "type",
            NodeKind::Module => "module",
            NodeKind::Class => "class",
            NodeKind::Interface => "interface",
            NodeKind::Constant => "constant",
            NodeKind::Other => "other",
        }
    }
}

impl From<&SymbolKind> for NodeKind {
    fn from(k: &SymbolKind) -> Self {
        match k {
            SymbolKind::Function => NodeKind::Function,
            SymbolKind::Method => NodeKind::Method,
            SymbolKind::Struct => NodeKind::Struct,
            SymbolKind::Enum => NodeKind::Enum,
            SymbolKind::Trait => NodeKind::Trait,
            SymbolKind::Impl => NodeKind::Impl,
            SymbolKind::Type => NodeKind::Type,
            SymbolKind::Module => NodeKind::Module,
            SymbolKind::Class => NodeKind::Class,
            SymbolKind::Interface => NodeKind::Interface,
            SymbolKind::Constant => NodeKind::Constant,
            SymbolKind::Variable | SymbolKind::Other => NodeKind::Other,
        }
    }
}

/// Graph edge kinds (subset of the codebase-memory-mcp edge taxonomy).
/// The `kind` column is free-form TEXT, so richer edge kinds (IMPORTS,
/// IMPLEMENTS, …) can be added without a schema migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// File DEFINES Function/Class/…
    Defines,
    /// Function CALLS Function (resolved call site).
    Calls,
}

impl EdgeKind {
    fn as_str(&self) -> &'static str {
        match self {
            EdgeKind::Defines => "DEFINES",
            EdgeKind::Calls => "CALLS",
        }
    }
}

// ─── GraphStore ──────────────────────────────────────────────────────────────

/// A SQLite-backed code knowledge graph for a single project.
///
/// Thread-safe via an internal `Mutex<Connection>`. One instance per project
/// is held by the
/// [`GraphToolProvider`](crate::system::agent::graph_tools::GraphToolProvider),
/// which pairs each store with the cwd it was opened for.
pub struct GraphStore {
    conn: Mutex<Connection>,
}

impl GraphStore {
    /// Open (or create) the graph DB at `path`. Schema is initialised idempotently.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("graph: create dir {}", parent.display()))?;
        }
        let conn =
            Connection::open(path).with_context(|| format!("graph: open {}", path.display()))?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Open an in-memory graph.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS nodes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                kind        TEXT NOT NULL,
                name        TEXT NOT NULL,
                file_id     INTEGER,
                line_start  INTEGER,
                line_end    INTEGER,
                signature   TEXT,
                FOREIGN KEY (file_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_name  ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_nodes_kind  ON nodes(kind);
            CREATE INDEX IF NOT EXISTS idx_nodes_file  ON nodes(file_id);

            CREATE TABLE IF NOT EXISTS edges (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                src_id  INTEGER NOT NULL,
                kind    TEXT NOT NULL,
                dst_id  INTEGER NOT NULL,
                FOREIGN KEY (src_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (dst_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, kind);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, kind);

            -- FTS5 virtual table for BM25 search over node names + signatures.
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                name, signature, content='nodes', content_rowid='id'
            );

            -- Triggers keep the FTS index in sync with the nodes table.
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, name, signature)
                VALUES (new.id, new.name, COALESCE(new.signature, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, name, signature)
                VALUES ('delete', old.id, old.name, COALESCE(old.signature, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, name, signature)
                VALUES ('delete', old.id, old.name, COALESCE(old.signature, ''));
                INSERT INTO nodes_fts(rowid, name, signature)
                VALUES (new.id, new.name, COALESCE(new.signature, ''));
            END;
            "#,
        )?;
        Ok(())
    }

    /// Number of symbol/file nodes currently in the graph. Zero means the
    /// project has never been indexed (or the index was cleared).
    pub fn node_count(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let count = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        Ok(count)
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /// Index `root` into the graph. Walks the repo, extracts symbols, builds
    /// nodes + DEFINES edges, then resolves CALLS edges: each call site is
    /// attributed to the nearest preceding function/method in the same file
    /// (or the file node), and resolved same-file first, then globally.
    ///
    /// The whole rebuild runs in one transaction: readers keep the previous
    /// index until the new one commits. Returns a summary JSON object.
    pub fn index(&self, root: &Path, max_files: usize, max_depth: usize) -> Result<Value> {
        let t0 = std::time::Instant::now();

        // Walk + read + extract before taking the connection lock so slow IO
        // doesn't block concurrent queries.
        struct SourceFile {
            rel: String,
            content: String,
            symbols: Vec<Symbol>,
        }
        let mut sources: Vec<SourceFile> = Vec::new();
        for summary in walk_repo(root, max_depth, max_files, false, false) {
            if summary.is_binary || !summary.lang.is_indexable() {
                continue;
            }
            let abs = root.join(&summary.rel_path);
            let Ok(content) = std::fs::read_to_string(&abs) else {
                continue;
            };
            let symbols = extract_symbols(&content, &summary.lang);
            sources.push(SourceFile {
                rel: summary.rel_path.display().to_string(),
                content,
                symbols,
            });
        }

        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        tx.execute("DELETE FROM edges", [])?;
        tx.execute("DELETE FROM nodes", [])?;
        tx.execute(
            "INSERT INTO nodes (kind, name) VALUES (?1, ?2)",
            params![NodeKind::Project.as_str(), root.display().to_string()],
        )?;
        let project_id = tx.last_insert_rowid();

        // name → [(symbol node id, file node id)] for call resolution.
        let mut symbol_targets: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
        // file node id → callable (function/method) nodes as (start line, id),
        // sorted by line, for call-site attribution.
        let mut callables_by_file: HashMap<i64, Vec<(usize, i64)>> = HashMap::new();
        let mut file_node_ids: Vec<i64> = Vec::with_capacity(sources.len());
        let mut total_symbols = 0usize;

        for source in &sources {
            tx.execute(
                "INSERT INTO nodes (kind, name, file_id) VALUES (?1, ?2, ?3)",
                params![NodeKind::File.as_str(), source.rel, project_id],
            )?;
            let file_id = tx.last_insert_rowid();
            file_node_ids.push(file_id);

            for sym in &source.symbols {
                let kind = NodeKind::from(&sym.kind);
                tx.execute(
                    "INSERT INTO nodes (kind, name, file_id, line_start) VALUES (?1, ?2, ?3, ?4)",
                    params![kind.as_str(), sym.name, file_id, sym.line as i64],
                )?;
                let sym_id = tx.last_insert_rowid();
                tx.execute(
                    "INSERT INTO edges (src_id, kind, dst_id) VALUES (?1, ?2, ?3)",
                    params![file_id, EdgeKind::Defines.as_str(), sym_id],
                )?;
                symbol_targets
                    .entry(sym.name.clone())
                    .or_default()
                    .push((sym_id, file_id));
                if matches!(kind, NodeKind::Function | NodeKind::Method) {
                    callables_by_file
                        .entry(file_id)
                        .or_default()
                        .push((sym.line, sym_id));
                }
                total_symbols += 1;
            }
        }
        for callables in callables_by_file.values_mut() {
            callables.sort_unstable();
        }

        // ── CALLS edge resolution ────────────────────────────────────────────
        // Attribute each call site to the enclosing callable (nearest preceding
        // function/method in the same file; file node if none), then resolve
        // the callee same-file first, falling back to the first global match.
        // Deduplicated per (src, dst); definition sites are excluded by
        // `scan_call_sites`.
        let mut call_edges = 0usize;
        let mut emitted: HashSet<(i64, i64)> = HashSet::new();
        for (source, &file_id) in sources.iter().zip(&file_node_ids) {
            let callables = callables_by_file.get(&file_id);
            for (call_name, line) in scan_call_sites(&source.content) {
                let Some(targets) = symbol_targets.get(&call_name) else {
                    continue;
                };
                let src = callables
                    .and_then(|c| enclosing_callable(c, line))
                    .unwrap_or(file_id);
                let dst = targets
                    .iter()
                    .find(|(_, f)| *f == file_id)
                    .or_else(|| targets.first())
                    .map(|(id, _)| *id)
                    .expect("symbol_targets entries are non-empty");
                if src == dst {
                    continue; // a definition matching itself, not a real call
                }
                if emitted.insert((src, dst)) {
                    tx.execute(
                        "INSERT INTO edges (src_id, kind, dst_id) VALUES (?1, ?2, ?3)",
                        params![src, EdgeKind::Calls.as_str(), dst],
                    )?;
                    call_edges += 1;
                }
            }
        }

        tx.commit()?;

        let elapsed_ms = t0.elapsed().as_millis() as u64;
        Ok(json!({
            "project_root": root.display().to_string(),
            "files_indexed": file_node_ids.len(),
            "symbols": total_symbols,
            "call_edges": call_edges,
            "elapsed_ms": elapsed_ms,
        }))
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// BM25 search over node names + signatures. Returns up to `limit` matches.
    /// An empty/whitespace query returns no matches (FTS5 rejects empty MATCH).
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Value>> {
        let fts = fts_query(query);
        if fts.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT n.id, n.kind, n.name, n.line_start, f.name AS file
             FROM nodes_fts
             JOIN nodes n ON n.id = nodes_fts.rowid
             LEFT JOIN nodes f ON f.id = n.file_id
             WHERE nodes_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![fts, limit as i64], |row| {
            let id: i64 = row.get(0)?;
            let kind: String = row.get(1)?;
            let name: String = row.get(2)?;
            let line: Option<i64> = row.get(3)?;
            let file: Option<String> = row.get(4)?;
            Ok(json!({
                "id": id,
                "kind": kind,
                "name": name,
                "line": line,
                "file": file,
            }))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Traverse CALLS edges from a named symbol. `direction = "outbound"` follows
    /// outgoing calls (what does X call?); `"inbound"` follows incoming calls
    /// (what calls X?). BFS up to `max_depth` hops, returning the visited nodes
    /// (the start symbol at depth 0).
    pub fn trace_path(&self, name: &str, direction: &str, max_depth: usize) -> Result<Vec<Value>> {
        let conn = self.conn.lock().unwrap();
        let start_ids: Vec<i64> = {
            let mut stmt = conn.prepare("SELECT id FROM nodes WHERE name = ?1")?;
            let rows = stmt.query_map(params![name], |r| r.get::<_, i64>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        if start_ids.is_empty() {
            return Ok(Vec::new());
        }
        let visited = bfs_calls(&conn, &start_ids, direction == "inbound", max_depth)?;
        materialize_nodes(&conn, &visited, false)
    }

    /// Reverse impact analysis: given a file path, find all symbols that
    /// transitively call into symbols defined in that file (blast radius).
    pub fn find_impact(&self, file_path: &str, max_depth: usize) -> Result<Vec<Value>> {
        let conn = self.conn.lock().unwrap();
        let seeds: Vec<i64> = {
            let mut stmt = conn.prepare(
                "SELECT n.id FROM nodes n
                 JOIN nodes f ON f.id = n.file_id
                 WHERE f.name = ?1 AND n.kind IN ('function','method','class','struct','trait')",
            )?;
            let rows = stmt.query_map(params![file_path], |r| r.get::<_, i64>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        if seeds.is_empty() {
            return Ok(Vec::new());
        }
        let visited = bfs_calls(&conn, &seeds, true, max_depth)?;
        materialize_nodes(&conn, &visited, true)
    }
}

// ─── Query helpers ────────────────────────────────────────────────────────────

/// BFS over CALLS edges from `seeds`. Forward (`reverse = false`) follows
/// src→dst (what the seeds call); reverse follows dst→src (what calls the
/// seeds). Returns node id → depth, seeds at depth 0.
fn bfs_calls(
    conn: &Connection,
    seeds: &[i64],
    reverse: bool,
    max_depth: usize,
) -> Result<HashMap<i64, usize>> {
    let sql = if reverse {
        "SELECT src_id FROM edges WHERE dst_id = ?1 AND kind = 'CALLS'"
    } else {
        "SELECT dst_id FROM edges WHERE src_id = ?1 AND kind = 'CALLS'"
    };
    let mut stmt = conn.prepare(sql)?;

    let mut visited: HashMap<i64, usize> = seeds.iter().map(|&s| (s, 0)).collect();
    let mut frontier: Vec<i64> = seeds.to_vec();
    for depth in 1..=max_depth {
        let mut next: Vec<i64> = Vec::new();
        for &node in &frontier {
            let rows = stmt.query_map(params![node], |r| r.get::<_, i64>(0))?;
            for row in rows {
                let neighbour = row?;
                if let std::collections::hash_map::Entry::Vacant(slot) = visited.entry(neighbour) {
                    slot.insert(depth);
                    next.push(neighbour);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    Ok(visited)
}

/// Materialise visited node ids as JSON objects, sorted by depth then name.
/// `skip_seeds` drops depth-0 entries (used by `find_impact`, where the seeds
/// themselves aren't part of the answer).
fn materialize_nodes(
    conn: &Connection,
    visited: &HashMap<i64, usize>,
    skip_seeds: bool,
) -> Result<Vec<Value>> {
    let mut stmt = conn.prepare(
        "SELECT n.kind, n.name, n.line_start, f.name FROM nodes n
         LEFT JOIN nodes f ON f.id = n.file_id WHERE n.id = ?1",
    )?;
    let mut out = Vec::new();
    for (&id, &depth) in visited {
        if skip_seeds && depth == 0 {
            continue;
        }
        let value = stmt.query_row(params![id], |r| {
            let kind: String = r.get(0)?;
            let name: String = r.get(1)?;
            let line: Option<i64> = r.get(2)?;
            let file: Option<String> = r.get(3)?;
            Ok(json!({
                "id": id,
                "depth": depth,
                "kind": kind,
                "name": name,
                "line": line,
                "file": file,
            }))
        })?;
        out.push(value);
    }
    out.sort_by(|a, b| {
        let key = |v: &Value| {
            (
                v.get("depth").and_then(Value::as_i64).unwrap_or(0),
                v.get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )
        };
        key(a).cmp(&key(b))
    });
    Ok(out)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Find the callable enclosing 1-based `line`: the last entry in the
/// line-sorted `callables` list starting at or before `line`.
fn enclosing_callable(callables: &[(usize, i64)], line: usize) -> Option<i64> {
    let idx = callables.partition_point(|&(start, _)| start <= line);
    idx.checked_sub(1).map(|i| callables[i].1)
}

/// Scan source `content` for call-site identifiers of the form `name(`.
/// Returns `(name, 1-based line)` pairs, deduplicated per line. Skips
/// keywords and identifiers that are part of a definition header
/// (e.g. `pub fn orphan_fn(`).
fn scan_call_sites(content: &str) -> Vec<(String, usize)> {
    use regex::Regex;
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // Match `identifier(` or `identifier.method(` — capture the final name.
        Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(").unwrap()
    });

    const KEYWORDS: &[&str] = &[
        "if", "else", "for", "while", "switch", "match", "return", "fn", "def", "class", "struct",
        "enum", "trait", "impl", "type", "let", "const", "var", "pub", "async", "await", "new",
        "delete", "sizeof", "typeof", "function", "import", "export", "from", "with", "catch",
        "try",
    ];
    // Definition keywords whose presence immediately before the identifier
    // means this is a *definition*, not a call site.
    const DEF_PREFIXES: &[&str] = &["fn ", "def ", "function ", "func ", "void "];

    // Byte offset of each line start, for offset → line-number lookup.
    let mut line_starts: Vec<usize> = vec![0];
    line_starts.extend(content.match_indices('\n').map(|(i, _)| i + 1));

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for caps in re.captures_iter(content) {
        let m = caps.get(1).unwrap();
        let name = m.as_str();
        if KEYWORDS.contains(&name) || name.starts_with('_') {
            continue;
        }
        // Look back up to 12 bytes before the identifier to detect a definition
        // header (e.g. `pub fn orphan_fn(` should not count as a call site).
        // The look-back may land inside a multi-byte character (comments often
        // contain `—` or box-drawing chars), so walk back to a char boundary.
        let start = m.start();
        let mut back_start = start.saturating_sub(12);
        while !content.is_char_boundary(back_start) {
            back_start -= 1;
        }
        let prefix = &content[back_start..start];
        if DEF_PREFIXES.iter().any(|p| prefix.ends_with(p)) {
            continue;
        }
        let line = line_starts.partition_point(|&offset| offset <= start);
        if seen.insert((name.to_string(), line)) {
            out.push((name.to_string(), line));
        }
    }
    out
}

/// Build an FTS5 MATCH query from a free-text `query` string.
/// Splits on whitespace, quotes each token, ANDs them together.
fn fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Derive the on-disk DB path for a project rooted at `cwd`.
/// `<app_home>/state/codebase-graph/<sha256(cwd)>.sqlite3`
pub fn db_path_for(app_home: &Path, cwd: &Path) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(cwd.display().to_string().as_bytes());
    let hash = hasher.finalize();
    let hex: String = hash.iter().take(8).map(|b| format!("{:02x}", b)).collect();
    app_home
        .join("state")
        .join("codebase-graph")
        .join(format!("{hex}.sqlite3"))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_repo(dir: &Path, files: &[(&str, &str)]) {
        for (name, content) in files {
            let path = dir.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }
    }

    #[test]
    fn index_and_search() {
        let tmp = tempfile::tempdir().unwrap();
        write_repo(
            tmp.path(),
            &[
                ("lib.rs", "pub fn alpha() {}\npub fn beta() { alpha(); }\n"),
                ("mod.rs", "pub struct Gamma;\npub trait Delta {}\n"),
            ],
        );
        let store = GraphStore::open_in_memory().unwrap();
        let summary = store.index(tmp.path(), 100, 4).unwrap();
        assert!(summary["symbols"].as_i64().unwrap() >= 3);
        let hits = store.search("alpha", 10).unwrap();
        assert!(hits.iter().any(|h| h["name"] == "alpha"));
    }

    #[test]
    fn search_empty_query_returns_no_matches() {
        let store = GraphStore::open_in_memory().unwrap();
        assert!(store.search("", 10).unwrap().is_empty());
        assert!(store.search("   ", 10).unwrap().is_empty());
    }

    #[test]
    fn calls_are_attributed_to_enclosing_function() {
        let tmp = tempfile::tempdir().unwrap();
        write_repo(
            tmp.path(),
            &[
                ("a.rs", "pub fn caller() {\n    callee();\n}\n"),
                ("b.rs", "pub fn callee() {\n    helper();\n}\n"),
                ("c.rs", "pub fn helper() {}\n"),
            ],
        );
        let store = GraphStore::open_in_memory().unwrap();
        store.index(tmp.path(), 100, 4).unwrap();

        // Function-level edges: caller → callee → helper.
        let out = store.trace_path("caller", "outbound", 3).unwrap();
        assert!(out.iter().any(|v| v["name"] == "callee" && v["depth"] == 1));
        assert!(out.iter().any(|v| v["name"] == "helper" && v["depth"] == 2));

        // And inbound from helper finds both ancestors.
        let inbound = store.trace_path("helper", "inbound", 3).unwrap();
        assert!(inbound.iter().any(|v| v["name"] == "callee"));
        assert!(inbound.iter().any(|v| v["name"] == "caller"));
    }

    #[test]
    fn call_resolution_prefers_same_file_target() {
        let tmp = tempfile::tempdir().unwrap();
        write_repo(
            tmp.path(),
            &[
                (
                    "a.rs",
                    "pub fn helper() {}\npub fn local_caller() {\n    helper();\n}\n",
                ),
                ("z.rs", "pub fn helper() {}\n"),
            ],
        );
        let store = GraphStore::open_in_memory().unwrap();
        store.index(tmp.path(), 100, 4).unwrap();
        let out = store.trace_path("local_caller", "outbound", 1).unwrap();
        let helper = out
            .iter()
            .find(|v| v["name"] == "helper")
            .expect("local_caller should call helper");
        assert_eq!(helper["file"], "a.rs");
    }

    #[test]
    fn find_impact_reports_transitive_callers() {
        let tmp = tempfile::tempdir().unwrap();
        write_repo(
            tmp.path(),
            &[
                ("core.rs", "pub fn core_fn() {}\n"),
                ("mid.rs", "pub fn mid_fn() {\n    core_fn();\n}\n"),
                ("top.rs", "pub fn top_fn() {\n    mid_fn();\n}\n"),
            ],
        );
        let store = GraphStore::open_in_memory().unwrap();
        store.index(tmp.path(), 100, 4).unwrap();
        let impact = store.find_impact("core.rs", 4).unwrap();
        assert!(impact.iter().any(|v| v["name"] == "mid_fn"));
        assert!(impact.iter().any(|v| v["name"] == "top_fn"));
        // Seeds themselves are excluded.
        assert!(!impact.iter().any(|v| v["name"] == "core_fn"));
    }

    #[test]
    fn index_persists_across_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        write_repo(tmp.path(), &[("lib.rs", "pub fn alpha() {}\n")]);
        let db = tmp.path().join("graph.sqlite3");

        let store = GraphStore::open(&db).unwrap();
        assert_eq!(store.node_count().unwrap(), 0);
        store.index(tmp.path(), 100, 4).unwrap();
        drop(store);

        let reopened = GraphStore::open(&db).unwrap();
        assert!(reopened.node_count().unwrap() > 0);
        let hits = reopened.search("alpha", 10).unwrap();
        assert!(hits.iter().any(|h| h["name"] == "alpha"));
    }

    #[test]
    fn fts_query_quotes_tokens() {
        assert_eq!(fts_query("alpha beta"), "\"alpha\" \"beta\"");
        assert_eq!(fts_query(""), "");
    }

    #[test]
    fn scan_call_sites_reports_lines_and_skips_definitions() {
        let src = "pub fn caller() {\n    callee();\n    callee();\n}\n";
        let sites = scan_call_sites(src);
        assert!(sites.contains(&("callee".to_string(), 2)));
        assert!(sites.contains(&("callee".to_string(), 3)));
        assert!(!sites.iter().any(|(name, _)| name == "caller"));
    }

    #[test]
    fn scan_call_sites_survives_multibyte_chars_before_call() {
        // The 12-byte definition look-back must not split a multi-byte char.
        // Sweep a range of gap widths so the look-back lands on every byte
        // of the 3-byte `—` at least once.
        for pad in 0..6 {
            let src = format!("// ——————{}\nfoo();\n", " ".repeat(pad));
            let sites = scan_call_sites(&src);
            assert!(sites.contains(&("foo".to_string(), 2)), "pad={pad}");
        }
    }
}
