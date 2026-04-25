//! Markdown → ratatui `Line` renderer — minimal style.
//!
//! Philosophy: render Markdown so it reads cleanly in a terminal stream without
//! visual chrome. No box borders, no underrules, no decorative separators.
//!
//! Supported elements:
//!   • Headings        bold + colour, one blank line before (no rules)
//!   • Paragraphs      normal body colour, word-wrapped
//!   • Bold / Italic / Strikethrough  inline modifiers
//!   • Inline code     accent colour + dim background feel
//!   • Code blocks     subtle `  ` indent + muted colour (no borders)
//!   • Block quotes    `▎ ` left bar + italic
//!   • Lists           •/◦/▪ bullets or numbers, nested indent
//!   • Task lists      [✓] / [ ] prefix
//!   • Tables          space-padded columns, `·` separator, bold header + thin underline
//!   • Horizontal rule short `──────` in muted colour (not full width)
//!   • Soft/hard line breaks

use pulldown_cmark::{Alignment, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

use super::types::{COLOR_ACCENT, COLOR_MUTED, COLOR_REASONING};

// ─── Public entry point ────────────────────────────────────────────────────────

/// Convert `markdown` into ratatui `Line`s ready for embedding in the transcript.
///
/// * `body_color` – default foreground for prose text.
/// * `body_width` – available columns (used for word-wrap & table sizing).
pub fn md_to_lines(markdown: &str, body_color: Color, body_width: usize) -> Vec<Line<'static>> {
    let width = body_width.max(10);
    let mut renderer = MdRenderer::new(body_color, width);
    renderer.render(markdown);
    renderer.finish()
}

// ─── Inline style ─────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct InlineStyle {
    bold: bool,
    italic: bool,
    strikethrough: bool,
    code: bool,
}

impl InlineStyle {
    fn to_ratatui(&self, base_color: Color) -> Style {
        if self.code {
            // inline code: accent colour, slightly dimmed
            return Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::DIM);
        }
        let mut s = Style::default().fg(base_color);
        if self.bold        { s = s.add_modifier(Modifier::BOLD); }
        if self.italic      { s = s.add_modifier(Modifier::ITALIC); }
        if self.strikethrough { s = s.add_modifier(Modifier::CROSSED_OUT); }
        s
    }
}

// ─── Inline buffer ────────────────────────────────────────────────────────────

#[derive(Default)]
struct InlineBuffer {
    spans: Vec<Span<'static>>,
}

impl InlineBuffer {
    fn push(&mut self, text: String, style: Style) {
        if text.is_empty() { return; }
        if let Some(last) = self.spans.last_mut() {
            if last.style == style {
                let mut s = last.content.to_string();
                s.push_str(&text);
                last.content = s.into();
                return;
            }
        }
        self.spans.push(Span::styled(text, style));
    }

    fn is_empty(&self) -> bool {
        self.spans.iter().all(|s| s.content.trim().is_empty())
    }

    fn take(&mut self) -> Vec<Span<'static>> {
        std::mem::take(&mut self.spans)
    }

    fn plain_text(&self) -> String {
        self.spans.iter().map(|s| s.content.as_ref()).collect()
    }
}

// ─── Table accumulator ────────────────────────────────────────────────────────

#[derive(Default)]
struct TableState {
    alignments: Vec<Alignment>,
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
    current_cell: String,
    in_header: bool,
}

// ─── Renderer ─────────────────────────────────────────────────────────────────

struct MdRenderer {
    body_color: Color,
    width: usize,
    out: Vec<Line<'static>>,
    inline: InlineBuffer,
    style: InlineStyle,
    quote_depth: usize,
    list_depth: usize,
    list_ordered: Vec<bool>,
    list_counter: Vec<u64>,
    in_code_block: bool,
    current_lang: String,
    table: Option<TableState>,
    in_table_head: bool,
}

impl MdRenderer {
    fn new(body_color: Color, width: usize) -> Self {
        Self {
            body_color, width,
            out: Vec::new(),
            inline: InlineBuffer::default(),
            style: InlineStyle::default(),
            quote_depth: 0,
            list_depth: 0,
            list_ordered: Vec::new(),
            list_counter: Vec::new(),
            in_code_block: false,
            current_lang: String::new(),
            table: None,
            in_table_head: false,
        }
    }

    fn render(&mut self, markdown: &str) {
        let opts = Options::all();
        let parser = Parser::new_ext(markdown, opts);

        for event in parser {
            match event {
                Event::Start(tag) => self.handle_start(tag),
                Event::End(tag)   => self.handle_end(tag),

                Event::Text(text) => {
                    if self.in_code_block {
                        for line in text.lines() {
                            self.push_code_line(line.to_string());
                        }
                    } else if let Some(tbl) = self.table.as_mut() {
                        tbl.current_cell.push_str(&text);
                    } else {
                        let style = self.style.to_ratatui(self.body_color);
                        self.inline.push(text.to_string(), style);
                    }
                }

                Event::Code(text) => {
                    if let Some(tbl) = self.table.as_mut() {
                        tbl.current_cell.push_str(&text);
                    } else {
                        let style = InlineStyle { code: true, ..Default::default() }
                            .to_ratatui(self.body_color);
                        self.inline.push(text.to_string(), style);
                    }
                }

                Event::SoftBreak => {
                    if self.table.is_none() {
                        let style = self.style.to_ratatui(self.body_color);
                        self.inline.push(" ".to_string(), style);
                    }
                }
                Event::HardBreak => {
                    if self.table.is_none() { self.flush_inline(); }
                }

                // Minimal horizontal rule: short dash sequence, not full-width
                Event::Rule => {
                    self.flush_inline();
                    self.out.push(Line::from(Span::styled(
                        "──────".to_string(),
                        Style::default().fg(COLOR_MUTED),
                    )));
                }

                Event::TaskListMarker(checked) => {
                    let (mark, color) = if checked {
                        ("✓ ", COLOR_ACCENT)
                    } else {
                        ("○ ", COLOR_MUTED)
                    };
                    self.inline.push(mark.to_string(), Style::default().fg(color));
                }

                _ => {}
            }
        }

        self.flush_inline();
    }

    // ── Tag open ──────────────────────────────────────────────────────────────

    fn handle_start(&mut self, tag: Tag) {
        match tag {
            Tag::Heading { level, .. } => {
                self.flush_inline();
                // Blank line before heading (except at very start)
                if !self.out.is_empty() {
                    self.out.push(Line::from(""));
                }
                // No `#` prefix — just style the text itself
                let color = heading_color(level);
                let style = Style::default().fg(color).add_modifier(Modifier::BOLD);
                // Store style as the pending inline style for the heading text
                self.inline.push(String::new(), style); // prime the buffer style
                // We'll apply color via the inline style override below
                self.style.bold = true;
                // We track which heading color to use via a stored span style.
                // The simplest: push a zero-width span to set the dominant style.
                self.inline.spans.clear();
                // We'll push content in Text events; set body_color temporarily via a flag.
                // Actually, simplest approach: override body_color for this heading.
                // We do this by pushing a placeholder that sets the color.
                let _ = color; // used below in handle_end
                // Store color in a pending span so heading text uses it.
                self.inline.push("".to_string(), Style::default().fg(color).add_modifier(Modifier::BOLD));
            }

            Tag::Paragraph => {
                self.flush_inline();
            }

            Tag::BlockQuote(_) => {
                self.flush_inline();
                self.quote_depth += 1;
            }

            Tag::CodeBlock(kind) => {
                self.flush_inline();
                self.in_code_block = true;
                // Capture language hint for syntax highlighting.
                self.current_lang = match &kind {
                    pulldown_cmark::CodeBlockKind::Fenced(lang) => lang.to_string().to_lowercase(),
                    _ => String::new(),
                };
                if !self.out.is_empty() {
                    self.out.push(Line::from(""));
                }
            }

            Tag::List(start) => {
                self.flush_inline();
                self.list_depth += 1;
                self.list_ordered.push(start.is_some());
                self.list_counter.push(start.unwrap_or(1));
            }

            Tag::Item => {
                self.flush_inline();
                let depth = self.list_depth;
                let ordered = *self.list_ordered.last().unwrap_or(&false);
                let indent = "  ".repeat((depth - 1).min(4));
                let bullet = if ordered {
                    let n = self.list_counter.last_mut().unwrap();
                    let s = format!("{indent}{}. ", n);
                    *n += 1;
                    s
                } else {
                    let ch = match depth { 1 => "• ", 2 => "◦ ", _ => "▪ " };
                    format!("{indent}{ch}")
                };
                self.inline.push(bullet, Style::default().fg(COLOR_MUTED));
            }

            Tag::Strong      => self.style.bold = true,
            Tag::Emphasis    => self.style.italic = true,
            Tag::Strikethrough => self.style.strikethrough = true,

            Tag::Table(alignments) => {
                self.flush_inline();
                self.table = Some(TableState { alignments, ..Default::default() });
                self.in_table_head = false;
            }
            Tag::TableHead => { self.in_table_head = true; }
            Tag::TableRow  => {
                if let Some(tbl) = self.table.as_mut() { tbl.current_row = Vec::new(); }
            }
            Tag::TableCell => {
                if let Some(tbl) = self.table.as_mut() { tbl.current_cell = String::new(); }
            }

            _ => {}
        }
    }

    // ── Tag close ─────────────────────────────────────────────────────────────

    fn handle_end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Heading(_) => {
                // Emit heading text — already bold+coloured via the primed span.
                // Flush with the styled spans collected during the heading.
                let spans = self.inline.take();
                self.style.bold = false;
                self.out.push(Line::from(spans));
                // No underrule — just a blank line after
                self.out.push(Line::from(""));
            }

            TagEnd::Paragraph => {
                self.flush_word_wrapped();
                self.out.push(Line::from(""));
            }

            TagEnd::BlockQuote(_) => {
                self.flush_inline();
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.out.push(Line::from(""));
            }

            TagEnd::CodeBlock => {
                self.in_code_block = false;
                self.out.push(Line::from(""));
            }

            TagEnd::List(_) => {
                self.flush_inline();
                self.list_depth = self.list_depth.saturating_sub(1);
                self.list_ordered.pop();
                self.list_counter.pop();
                if self.list_depth == 0 {
                    self.out.push(Line::from(""));
                }
            }

            TagEnd::Item => { self.flush_word_wrapped(); }

            TagEnd::Strong      => self.style.bold = false,
            TagEnd::Emphasis    => self.style.italic = false,
            TagEnd::Strikethrough => self.style.strikethrough = false,

            TagEnd::TableHead => {
                if let Some(tbl) = self.table.as_mut() {
                    if !tbl.current_row.is_empty() {
                        tbl.rows.push(std::mem::take(&mut tbl.current_row));
                    }
                    tbl.in_header = true;
                }
                self.in_table_head = false;
            }

            TagEnd::TableRow => {
                if let Some(tbl) = self.table.as_mut() {
                    if !tbl.current_row.is_empty() {
                        tbl.rows.push(std::mem::take(&mut tbl.current_row));
                    }
                }
            }

            TagEnd::TableCell => {
                if let Some(tbl) = self.table.as_mut() {
                    let cell = std::mem::take(&mut tbl.current_cell);
                    tbl.current_row.push(cell);
                }
            }

            TagEnd::Table => {
                if let Some(tbl) = self.table.take() {
                    let table_lines = render_table(&tbl, self.width);
                    self.out.extend(table_lines);
                    self.out.push(Line::from(""));
                }
            }

            _ => {}
        }
    }

    // ── Body helpers ──────────────────────────────────────────────────────────

    /// Push a code-block line with syntax highlighting.
    fn push_code_line(&mut self, raw: String) {
        let max_chars = self.width.saturating_sub(2);
        let content: String = raw.chars().take(max_chars).collect();
        let lang = self.current_lang.clone();
        let mut spans = vec![Span::styled("  ".to_string(), Style::default())];
        spans.extend(highlight_code_line(&content, &lang));
        self.out.push(Line::from(spans));
    }

    fn flush_inline(&mut self) {
        if self.inline.is_empty() { self.inline.take(); return; }
        let spans = self.inline.take();
        let line = self.prefix_line(spans);
        self.out.push(line);
    }

    fn flush_word_wrapped(&mut self) {
        if self.inline.is_empty() { self.inline.take(); return; }

        let full_text = self.inline.plain_text();
        let dominant_style = self.inline.spans.first()
            .map(|s| s.style)
            .unwrap_or_else(|| Style::default().fg(self.body_color));
        self.inline.take();

        let indent_width = if self.list_depth > 0 {
            let depth = self.list_depth.min(4);
            let ordered = *self.list_ordered.last().unwrap_or(&false);
            let base = 2 * (depth - 1);
            if ordered { base + 3 } else { base + 2 }
        } else { 0 };

        let available = self.width.saturating_sub(indent_width).max(8);
        let indent_str = " ".repeat(indent_width);

        for (i, chunk) in word_wrap(&full_text, available).into_iter().enumerate() {
            let text = if i == 0 { chunk } else { format!("{indent_str}{chunk}") };
            let line = self.prefix_line(vec![Span::styled(text, dominant_style)]);
            self.out.push(line);
        }
    }

    fn prefix_line(&self, mut spans: Vec<Span<'static>>) -> Line<'static> {
        if self.quote_depth > 0 {
            let bar = "▎ ".repeat(self.quote_depth);
            let mut prefixed = vec![Span::styled(
                bar,
                Style::default().fg(COLOR_REASONING).add_modifier(Modifier::ITALIC),
            )];
            prefixed.append(&mut spans);
            Line::from(prefixed)
        } else {
            Line::from(spans)
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush_inline();
        // Strip trailing blank lines.
        while self.out.last().map(|l: &Line| {
            l.spans.is_empty() || l.spans.iter().all(|s| s.content.trim().is_empty())
        }).unwrap_or(false) {
            self.out.pop();
        }
        self.out
    }
}

// ─── Table renderer ───────────────────────────────────────────────────────────

/// Minimal table: space-padded columns separated by `  ·  ` (muted).
/// Header row is bold white; body rows are dim.
/// A thin `─` underline separates header from body.
fn render_table(tbl: &TableState, max_width: usize) -> Vec<Line<'static>> {
    let rows = &tbl.rows;
    if rows.is_empty() { return vec![]; }

    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 { return vec![]; }

    // Natural column widths from content.
    let mut col_widths: Vec<usize> = vec![1usize; col_count];
    for row in rows {
        for (ci, cell) in row.iter().enumerate() {
            if ci < col_count {
                // Use char count (basic — ignores wide/emoji chars for now)
                col_widths[ci] = col_widths[ci].max(cell.chars().count());
            }
        }
    }

    // Budget: separator "  ·  " = 5 chars between columns
    let sep_width = 5;
    let total_sep = sep_width * (col_count - 1);
    let content_budget = max_width.saturating_sub(total_sep).max(col_count);
    let natural_total: usize = col_widths.iter().sum();
    if natural_total > content_budget {
        // Greedy allocation: satisfy smaller columns at their natural width first,
        // then give the remaining budget to larger ones. This prevents a single
        // very wide column from starving narrow columns (e.g. a file-path column
        // next to a long description column).
        let mut order: Vec<usize> = (0..col_count).collect();
        order.sort_by_key(|&i| col_widths[i]);
        let mut remaining = content_budget;
        let mut new_widths = vec![1usize; col_count];
        for (pass, &ci) in order.iter().enumerate() {
            let cols_left = col_count - pass;
            let fair_share = (remaining / cols_left).max(1);
            let w = col_widths[ci].min(fair_share).max(1);
            new_widths[ci] = w;
            remaining = remaining.saturating_sub(w);
        }
        col_widths = new_widths;
    }

    let sep_style   = Style::default().fg(COLOR_MUTED);
    let header_style = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
    let body_style   = Style::default().fg(Color::Rgb(200, 210, 225));

    let mut lines: Vec<Line<'static>> = Vec::new();

    for (ri, row) in rows.iter().enumerate() {
        let is_header = ri == 0 && tbl.in_header;
        let cell_style = if is_header { header_style } else { body_style };

        // Word-wrap each cell into its column width, producing potentially
        // multiple display lines per row.
        let wrapped: Vec<Vec<String>> = (0..col_count)
            .map(|ci| {
                let cell = row.get(ci).map(String::as_str).unwrap_or("");
                word_wrap(cell, col_widths[ci])
            })
            .collect();

        let row_height = wrapped.iter().map(|w| w.len()).max().unwrap_or(1);

        for display_line in 0..row_height {
            let mut spans: Vec<Span<'static>> = Vec::new();
            for ci in 0..col_count {
                if ci > 0 {
                    spans.push(Span::styled("  ·  ".to_string(), sep_style));
                }
                let cell_line = wrapped[ci].get(display_line).map(String::as_str).unwrap_or("");
                let align = tbl.alignments.get(ci).copied().unwrap_or(Alignment::None);
                let padded = align_cell(cell_line, col_widths[ci], align);
                spans.push(Span::styled(padded, cell_style));
            }
            lines.push(Line::from(spans));
        }

        // Thin underline after header row only.
        if is_header {
            let underline_width: usize = col_widths.iter().sum::<usize>()
                + sep_width * col_count.saturating_sub(1);
            lines.push(Line::from(Span::styled(
                "─".repeat(underline_width),
                Style::default().fg(COLOR_MUTED),
            )));
        }
    }

    lines
}

// ─── Syntax highlighter ───────────────────────────────────────────────────────

// Token colour palette (One Dark-inspired)
const C_KEYWORD  : Color = Color::Rgb(198, 120, 221); // purple
const C_TYPE     : Color = Color::Rgb( 86, 182, 194); // cyan
const C_STRING   : Color = Color::Rgb(152, 195, 121); // green
const C_NUMBER   : Color = Color::Rgb(209, 154, 102); // orange
const C_COMMENT  : Color = Color::Rgb( 92,  99, 112); // gray
const C_OPERATOR : Color = Color::Rgb(171, 178, 191); // light (same as normal)
const C_NORMAL   : Color = Color::Rgb(171, 178, 191); // light gray

/// Tokenise and colour-code a single line of source code.
fn highlight_code_line(line: &str, lang: &str) -> Vec<Span<'static>> {
    let keywords: &[&str] = match lang {
        "python" | "py" => &[
            "False","None","True","and","as","assert","async","await",
            "break","class","continue","def","del","elif","else","except",
            "finally","for","from","global","if","import","in","is",
            "lambda","nonlocal","not","or","pass","raise","return","try",
            "while","with","yield","self","cls",
        ],
        "javascript" | "js" | "typescript" | "ts" | "tsx" | "jsx" => &[
            "break","case","catch","class","const","continue","debugger",
            "default","delete","do","else","export","extends","finally",
            "for","function","if","import","in","instanceof","let","new",
            "return","static","super","switch","this","throw","try","typeof",
            "var","void","while","with","yield","async","await","of",
            "true","false","null","undefined","type","interface","enum",
            "implements","abstract","readonly","override","declare",
        ],
        "rust" | "rs" => &[
            "as","async","await","break","const","continue","crate","dyn",
            "else","enum","extern","false","fn","for","if","impl","in",
            "let","loop","match","mod","move","mut","pub","ref","return",
            "self","Self","static","struct","super","trait","true","type",
            "unsafe","use","where","while","Box","Option","Result","Vec",
            "String","Some","None","Ok","Err","println","format","panic",
        ],
        "go" | "golang" => &[
            "break","case","chan","const","continue","default","defer","else",
            "fallthrough","for","func","go","goto","if","import","interface",
            "map","package","range","return","select","struct","switch","type",
            "var","nil","true","false","make","new","len","cap","append",
            "copy","delete","close","panic","recover","error","string","int",
            "int64","float64","bool","byte","rune",
        ],
        "bash" | "sh" | "shell" | "zsh" => &[
            "if","then","else","elif","fi","for","do","done","while","until",
            "case","esac","function","return","local","export","unset","echo",
            "read","source","shift","exit","break","continue","true","false",
            "in","select","set","unset","alias","cd","pwd","test","let",
        ],
        "sql" => &[
            "SELECT","FROM","WHERE","JOIN","LEFT","RIGHT","INNER","OUTER",
            "ON","GROUP","BY","ORDER","HAVING","INSERT","INTO","VALUES",
            "UPDATE","SET","DELETE","CREATE","TABLE","INDEX","VIEW",
            "DROP","ALTER","ADD","COLUMN","PRIMARY","KEY","FOREIGN",
            "REFERENCES","NULL","NOT","AND","OR","IN","IS","LIKE","AS",
            "DISTINCT","COUNT","SUM","AVG","MAX","MIN","LIMIT","OFFSET",
        ],
        _ => &[],
    };

    let comment_starts: &[&str] = match lang {
        "python" | "py" | "bash" | "sh" | "shell" | "zsh" | "yaml" | "toml" | "ruby" => &["#"],
        "sql" => &["--"],
        "html" => &["<!--"],
        _ => &["//", "/*"],
    };

    // Full-line comment fast path
    let trimmed = line.trim_start();
    for cs in comment_starts {
        if trimmed.starts_with(cs) {
            return vec![Span::styled(line.to_string(), Style::default().fg(C_COMMENT))];
        }
    }

    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut pos = 0;
    let mut spans: Vec<Span<'static>> = Vec::new();

    let push = |spans: &mut Vec<Span<'static>>, text: String, color: Color| {
        if text.is_empty() { return; }
        if let Some(last) = spans.last_mut() {
            if last.style.fg == Some(color) {
                let mut s = last.content.to_string();
                s.push_str(&text);
                last.content = s.into();
                return;
            }
        }
        spans.push(Span::styled(text, Style::default().fg(color)));
    };

    while pos < len {
        // String literal (single, double, backtick)
        if matches!(chars[pos], '"' | '\'' | '`') {
            let q = chars[pos];
            let start = pos;
            pos += 1;
            while pos < len {
                if chars[pos] == '\\' { pos += 2; continue; }
                if chars[pos] == q { pos += 1; break; }
                pos += 1;
            }
            let s: String = chars[start..pos].iter().collect();
            push(&mut spans, s, C_STRING);
            continue;
        }

        // Inline comment (// or --)
        let rest: String = chars[pos..].iter().collect();
        let is_inline_comment = comment_starts.iter().any(|cs| rest.starts_with(cs));
        if is_inline_comment {
            push(&mut spans, rest, C_COMMENT);
            break;
        }

        // Number
        if chars[pos].is_ascii_digit()
            || (chars[pos] == '-' && pos + 1 < len && chars[pos + 1].is_ascii_digit())
        {
            let start = pos;
            if chars[pos] == '-' { pos += 1; }
            while pos < len && (chars[pos].is_ascii_digit() || chars[pos] == '.' || chars[pos] == '_' || chars[pos] == 'x' || chars[pos] == 'b') {
                pos += 1;
            }
            let s: String = chars[start..pos].iter().collect();
            push(&mut spans, s, C_NUMBER);
            continue;
        }

        // Identifier or keyword
        if chars[pos].is_alphabetic() || chars[pos] == '_' {
            let start = pos;
            while pos < len && (chars[pos].is_alphanumeric() || chars[pos] == '_') {
                pos += 1;
            }
            let word: String = chars[start..pos].iter().collect();
            let color = if keywords.contains(&word.as_str()) {
                C_KEYWORD
            } else if word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                C_TYPE  // PascalCase → likely a type
            } else {
                C_NORMAL
            };
            push(&mut spans, word, color);
            continue;
        }

        // Operator / punctuation
        let ch = chars[pos].to_string();
        let color = if "=<>!&|+-*/%^~".contains(chars[pos]) { C_OPERATOR } else { C_NORMAL };
        push(&mut spans, ch, color);
        pos += 1;
    }

    if spans.is_empty() {
        spans.push(Span::styled(line.to_string(), Style::default().fg(C_NORMAL)));
    }
    spans
}



fn align_cell(text: &str, width: usize, align: Alignment) -> String {
    let display: String = text.chars().take(width).collect();
    let len = display.chars().count();
    if len >= width { return display; }
    let padding = width - len;
    match align {
        Alignment::Right  => format!("{}{}", " ".repeat(padding), display),
        Alignment::Center => {
            let left = padding / 2;
            format!("{}{}{}", " ".repeat(left), display, " ".repeat(padding - left))
        }
        _ => format!("{}{}", display, " ".repeat(padding)),
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn heading_color(level: HeadingLevel) -> Color {
    match level {
        HeadingLevel::H1 => Color::Rgb(100, 200, 163), // mint (accent)
        HeadingLevel::H2 => Color::Rgb(140, 200, 255), // sky blue
        HeadingLevel::H3 => Color::Rgb(220, 140, 255), // soft violet
        HeadingLevel::H4 | HeadingLevel::H5 | HeadingLevel::H6 => COLOR_MUTED,
    }
}

fn word_wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 { return vec![text.to_string()]; }
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if current_len == 0 {
            current.push_str(word);
            current_len = word_len;
        } else if current_len + 1 + word_len <= width {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
            current_len = word_len;
        }
    }
    if !current.is_empty() { lines.push(current); }
    if lines.is_empty() { lines.push(String::new()); }
    lines
}
