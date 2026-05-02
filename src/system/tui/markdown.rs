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
    let mut r = MdRenderer::new(body_color, width);
    r.render(markdown);
    r.finish()
}

/// Like `md_to_lines` but also returns a source-line map:
/// `source_map[i]` is the raw-markdown line index (0-based) that generated visual line `i`.
///
/// Used by the copy-to-clipboard path to extract only the selected raw markdown lines
/// rather than the whole entry.
pub fn md_to_lines_with_source_map(
    markdown: &str,
    body_color: Color,
    body_width: usize,
) -> (Vec<Line<'static>>, Vec<usize>) {
    let width = body_width.max(10);
    let mut r = MdRenderer::new(body_color, width);
    let mut source_map = r.render_mapped(markdown);
    let last_src = source_map.last().copied().unwrap_or(0);
    let out = r.finish();
    // finish() may flush remaining inline content or strip trailing blank lines.
    while source_map.len() < out.len() {
        source_map.push(last_src);
    }
    source_map.truncate(out.len());
    (out, source_map)
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
        if self.bold {
            s = s.add_modifier(Modifier::BOLD);
        }
        if self.italic {
            s = s.add_modifier(Modifier::ITALIC);
        }
        if self.strikethrough {
            s = s.add_modifier(Modifier::CROSSED_OUT);
        }
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
        if text.is_empty() {
            return;
        }
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

// ─── Shared event dispatcher ─────────────────────────────────────────────────

/// Dispatch a single pulldown-cmark event into the renderer.
/// Extracted so both the plain `render` loop and the source-mapped `render_mapped` loop
/// can share the same event-handling logic without code duplication.
fn dispatch_md_event(r: &mut MdRenderer, event: Event) {
    match event {
        Event::Start(tag) => r.handle_start(tag),
        Event::End(tag) => r.handle_end(tag),

        Event::Text(text) => {
            if r.in_code_block {
                for line in text.lines() {
                    r.push_code_line(line.to_string());
                }
            } else if let Some(tbl) = r.table.as_mut() {
                tbl.current_cell.push_str(&text);
            } else {
                let style = r.style.to_ratatui(r.body_color);
                r.inline.push(text.to_string(), style);
            }
        }

        Event::Code(text) => {
            if let Some(tbl) = r.table.as_mut() {
                tbl.current_cell.push_str(&text);
            } else {
                let style = InlineStyle {
                    code: true,
                    ..Default::default()
                }
                .to_ratatui(r.body_color);
                r.inline.push(text.to_string(), style);
            }
        }

        Event::SoftBreak => {
            if r.table.is_none() {
                let style = r.style.to_ratatui(r.body_color);
                r.inline.push(" ".to_string(), style);
            }
        }
        Event::HardBreak => {
            if r.table.is_none() {
                r.flush_inline();
            }
        }

        Event::Rule => {
            r.flush_inline();
            r.out.push(Line::from(Span::styled(
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
            r.inline.push(mark.to_string(), Style::default().fg(color));
        }

        _ => {}
    }
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
    in_list_item: bool,
}

impl MdRenderer {
    fn new(body_color: Color, width: usize) -> Self {
        Self {
            body_color,
            width,
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
            in_list_item: false,
        }
    }

    fn render(&mut self, markdown: &str) {
        let opts = Options::all();
        for event in Parser::new_ext(markdown, opts) {
            dispatch_md_event(self, event);
        }
        self.flush_inline();
    }

    /// Like `render` but builds a source-line map in parallel.
    /// Returns a `Vec<usize>` where index `i` is the raw source line that generated `self.out[i]`.
    fn render_mapped(&mut self, markdown: &str) -> Vec<usize> {
        // Build byte-offset → line-number table for O(log n) lookups.
        let line_starts: Vec<usize> = std::iter::once(0)
            .chain(markdown.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        let byte_to_line = |b: usize| line_starts.partition_point(|&s| s <= b).saturating_sub(1);

        let opts = Options::all();
        let mut source_map: Vec<usize> = Vec::new();
        let mut table_start_src = 0usize;

        for (event, range) in Parser::new_ext(markdown, opts).into_offset_iter() {
            let cur_src = byte_to_line(range.start);

            if matches!(event, Event::Start(Tag::Table(_))) {
                table_start_src = cur_src;
            }

            let is_end_table = matches!(event, Event::End(TagEnd::Table));
            let before = self.out.len();
            dispatch_md_event(self, event);
            let added = self.out.len() - before;

            if is_end_table && added > 1 {
                // Table visual lines are all flushed at once. Distribute them linearly
                // across [table_start_src..=cur_src] so selecting any portion of the
                // rendered table maps back to a source line within the table.
                let src_span = cur_src.saturating_sub(table_start_src);
                for i in 0..added {
                    let src = table_start_src + (i * src_span) / (added - 1);
                    source_map.push(src);
                }
            } else {
                for _ in 0..added {
                    source_map.push(cur_src);
                }
            }
        }
        source_map
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
                self.inline.push(
                    "".to_string(),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                );
            }

            Tag::Paragraph => {
                if self.in_list_item {
                    return;
                }
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
                // Blank line before code block to visually separate it from preceding text
                // (including bullet/numbered list items that immediately precede code).
                if !self.out.is_empty() {
                    // Only add if last line isn't already blank
                    let already_blank = self
                        .out
                        .last()
                        .map(|l: &Line| {
                            l.spans.is_empty()
                                || l.spans.iter().all(|s| s.content.trim().is_empty())
                        })
                        .unwrap_or(false);
                    if !already_blank {
                        self.out.push(Line::from(""));
                    }
                }
            }

            Tag::List(start) => {
                self.flush_inline();
                self.list_depth += 1;
                self.list_ordered.push(start.is_some());
                self.list_counter.push(start.unwrap_or(1));
            }

            Tag::Item => {
                self.in_list_item = true;
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
                    let ch = match depth {
                        1 => "• ",
                        2 => "◦ ",
                        _ => "▪ ",
                    };
                    format!("{indent}{ch}")
                };
                self.inline.push(bullet, Style::default().fg(COLOR_MUTED));
            }

            Tag::Strong => self.style.bold = true,
            Tag::Emphasis => self.style.italic = true,
            Tag::Strikethrough => self.style.strikethrough = true,

            Tag::Table(alignments) => {
                self.flush_inline();
                self.table = Some(TableState {
                    alignments,
                    ..Default::default()
                });
                self.in_table_head = false;
            }
            Tag::TableHead => {
                self.in_table_head = true;
            }
            Tag::TableRow => {
                if let Some(tbl) = self.table.as_mut() {
                    tbl.current_row = Vec::new();
                }
            }
            Tag::TableCell => {
                if let Some(tbl) = self.table.as_mut() {
                    tbl.current_cell = String::new();
                }
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

            TagEnd::Item => {
                self.in_list_item = false;
                self.flush_word_wrapped();
            }

            TagEnd::Strong => self.style.bold = false,
            TagEnd::Emphasis => self.style.italic = false,
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
        let list_indent = if self.list_depth > 0 {
            let depth = self.list_depth.min(4);
            let ordered = *self.list_ordered.last().unwrap_or(&false);
            let base = 2 * (depth - 1);
            let indent_width = if ordered { base + 3 } else { base + 2 };
            " ".repeat(indent_width)
        } else {
            String::new()
        };
        let indent_chars = list_indent.chars().count();
        let max_chars = self.width.saturating_sub(2 + indent_chars);
        let content: String = raw.chars().take(max_chars).collect();
        let lang = self.current_lang.clone();
        let mut spans = vec![Span::styled(format!("{list_indent}  "), Style::default())];
        spans.extend(highlight_code_line(&content, &lang));
        self.out.push(Line::from(spans));
    }

    fn flush_inline(&mut self) {
        if self.inline.is_empty() {
            self.inline.take();
            return;
        }
        let spans = self.inline.take();
        let line = self.prefix_line(spans);
        self.out.push(line);
    }

    fn flush_word_wrapped(&mut self) {
        if self.inline.is_empty() {
            self.inline.take();
            return;
        }

        let full_text = self.inline.plain_text();
        let dominant_style = self
            .inline
            .spans
            .first()
            .map(|s| s.style)
            .unwrap_or_else(|| Style::default().fg(self.body_color));
        self.inline.take();

        let indent_width = if self.list_depth > 0 {
            let depth = self.list_depth.min(4);
            let ordered = *self.list_ordered.last().unwrap_or(&false);
            let base = 2 * (depth - 1);
            if ordered {
                base + 3
            } else {
                base + 2
            }
        } else {
            0
        };

        let available = self.width.saturating_sub(indent_width).max(8);
        let indent_str = " ".repeat(indent_width);

        for (i, chunk) in word_wrap(&full_text, available).into_iter().enumerate() {
            let text = if i == 0 {
                chunk
            } else {
                format!("{indent_str}{chunk}")
            };
            let line = self.prefix_line(vec![Span::styled(text, dominant_style)]);
            self.out.push(line);
        }
    }

    fn prefix_line(&self, mut spans: Vec<Span<'static>>) -> Line<'static> {
        if self.quote_depth > 0 {
            let bar = "▎ ".repeat(self.quote_depth);
            let mut prefixed = vec![Span::styled(
                bar,
                Style::default()
                    .fg(COLOR_REASONING)
                    .add_modifier(Modifier::ITALIC),
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
        while self
            .out
            .last()
            .map(|l: &Line| {
                l.spans.is_empty() || l.spans.iter().all(|s| s.content.trim().is_empty())
            })
            .unwrap_or(false)
        {
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
    if rows.is_empty() {
        return vec![];
    }

    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 {
        return vec![];
    }

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

    let sep_style = Style::default().fg(COLOR_MUTED);
    let header_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let body_style = Style::default().fg(Color::Rgb(200, 210, 225));

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
                let cell_line = wrapped[ci]
                    .get(display_line)
                    .map(String::as_str)
                    .unwrap_or("");
                let align = tbl.alignments.get(ci).copied().unwrap_or(Alignment::None);
                let padded = align_cell(cell_line, col_widths[ci], align);
                spans.push(Span::styled(padded, cell_style));
            }
            lines.push(Line::from(spans));
        }

        // Thin underline after header row only.
        if is_header {
            let underline_width: usize =
                col_widths.iter().sum::<usize>() + sep_width * col_count.saturating_sub(1);
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
const C_KEYWORD: Color = Color::Rgb(198, 120, 221); // purple
const C_TYPE: Color = Color::Rgb(86, 182, 194); // cyan
const C_STRING: Color = Color::Rgb(152, 195, 121); // green
const C_NUMBER: Color = Color::Rgb(209, 154, 102); // orange
const C_COMMENT: Color = Color::Rgb(92, 99, 112); // gray
const C_OPERATOR: Color = Color::Rgb(171, 178, 191); // light (same as normal)
const C_NORMAL: Color = Color::Rgb(171, 178, 191); // light gray

/// Tokenise and colour-code a single line of source code.
pub fn highlight_code_line(line: &str, lang: &str) -> Vec<Span<'static>> {
    let keywords: &[&str] = match lang {
        "python" | "py" => &[
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield", "self", "cls",
        ],
        "javascript" | "js" | "typescript" | "ts" | "tsx" | "jsx" => &[
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "else",
            "export",
            "extends",
            "finally",
            "for",
            "function",
            "if",
            "import",
            "in",
            "instanceof",
            "let",
            "new",
            "return",
            "static",
            "super",
            "switch",
            "this",
            "throw",
            "try",
            "typeof",
            "var",
            "void",
            "while",
            "with",
            "yield",
            "async",
            "await",
            "of",
            "true",
            "false",
            "null",
            "undefined",
            "type",
            "interface",
            "enum",
            "implements",
            "abstract",
            "readonly",
            "override",
            "declare",
        ],
        "rust" | "rs" => &[
            "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
            "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
            "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
            "trait", "true", "type", "unsafe", "use", "where", "while", "Box", "Option", "Result",
            "Vec", "String", "Some", "None", "Ok", "Err", "println", "format", "panic",
        ],
        "go" | "golang" => &[
            "break",
            "case",
            "chan",
            "const",
            "continue",
            "default",
            "defer",
            "else",
            "fallthrough",
            "for",
            "func",
            "go",
            "goto",
            "if",
            "import",
            "interface",
            "map",
            "package",
            "range",
            "return",
            "select",
            "struct",
            "switch",
            "type",
            "var",
            "nil",
            "true",
            "false",
            "make",
            "new",
            "len",
            "cap",
            "append",
            "copy",
            "delete",
            "close",
            "panic",
            "recover",
            "error",
            "string",
            "int",
            "int64",
            "float64",
            "bool",
            "byte",
            "rune",
        ],
        "bash" | "sh" | "shell" | "zsh" => &[
            "if", "then", "else", "elif", "fi", "for", "do", "done", "while", "until", "case",
            "esac", "function", "return", "local", "export", "unset", "echo", "read", "source",
            "shift", "exit", "break", "continue", "true", "false", "in", "select", "set", "unset",
            "alias", "cd", "pwd", "test", "let",
        ],
        "sql" => &[
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "ON",
            "GROUP",
            "BY",
            "ORDER",
            "HAVING",
            "INSERT",
            "INTO",
            "VALUES",
            "UPDATE",
            "SET",
            "DELETE",
            "CREATE",
            "TABLE",
            "INDEX",
            "VIEW",
            "DROP",
            "ALTER",
            "ADD",
            "COLUMN",
            "PRIMARY",
            "KEY",
            "FOREIGN",
            "REFERENCES",
            "NULL",
            "NOT",
            "AND",
            "OR",
            "IN",
            "IS",
            "LIKE",
            "AS",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
            "LIMIT",
            "OFFSET",
        ],
        "c" | "cpp" | "h" | "hpp" => &[
            "auto",
            "break",
            "case",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "delete",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "friend",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "mutable",
            "namespace",
            "new",
            "operator",
            "private",
            "protected",
            "public",
            "register",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "template",
            "this",
            "throw",
            "try",
            "typedef",
            "typename",
            "union",
            "unsigned",
            "using",
            "virtual",
            "void",
            "volatile",
            "while",
            "nullptr",
            "constexpr",
            "decltype",
            "noexcept",
            "override",
            "final",
            "static_assert",
            "thread_local",
            "alignas",
            "alignof",
            "true",
            "false",
            "NULL",
            "std",
            "string",
            "vector",
            "map",
            "set",
            "pair",
            "shared_ptr",
            "unique_ptr",
            "make_shared",
            "make_unique",
            "cout",
            "cin",
            "endl",
            "cerr",
        ],
        "java" => &[
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "null",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
            "true",
            "false",
            "String",
            "Integer",
            "Boolean",
            "Long",
            "Double",
            "Float",
            "System",
            "Override",
        ],
        "kotlin" => &[
            "as",
            "break",
            "class",
            "continue",
            "do",
            "else",
            "false",
            "for",
            "fun",
            "if",
            "in",
            "interface",
            "is",
            "null",
            "object",
            "package",
            "return",
            "super",
            "this",
            "throw",
            "true",
            "try",
            "typealias",
            "typeof",
            "val",
            "var",
            "when",
            "while",
            "by",
            "catch",
            "constructor",
            "delegate",
            "dynamic",
            "field",
            "file",
            "finally",
            "get",
            "import",
            "init",
            "inner",
            "internal",
            "it",
            "lateinit",
            "noinline",
            "open",
            "operator",
            "out",
            "override",
            "private",
            "protected",
            "public",
            "reified",
            "sealed",
            "set",
            "suspend",
            "tailrec",
            "value",
            "where",
            "Unit",
            "Boolean",
            "Int",
            "Long",
            "Float",
            "Double",
            "String",
            "List",
            "Map",
            "Set",
        ],
        "swift" => &[
            "as",
            "break",
            "case",
            "catch",
            "class",
            "continue",
            "default",
            "defer",
            "do",
            "else",
            "enum",
            "extension",
            "fallthrough",
            "false",
            "for",
            "func",
            "guard",
            "if",
            "import",
            "in",
            "init",
            "inout",
            "internal",
            "is",
            "let",
            "nil",
            "operator",
            "private",
            "protocol",
            "public",
            "repeat",
            "return",
            "self",
            "Self",
            "static",
            "struct",
            "subscript",
            "super",
            "switch",
            "throw",
            "true",
            "try",
            "typealias",
            "var",
            "where",
            "while",
            "yield",
            "Int",
            "String",
            "Double",
            "Float",
            "Bool",
            "Array",
            "Dictionary",
            "Set",
            "Optional",
            "print",
            "fatalError",
        ],
        "ruby" => &[
            "BEGIN",
            "END",
            "alias",
            "and",
            "begin",
            "break",
            "case",
            "class",
            "def",
            "defined?",
            "do",
            "else",
            "elsif",
            "end",
            "ensure",
            "false",
            "for",
            "if",
            "in",
            "module",
            "next",
            "nil",
            "not",
            "or",
            "redo",
            "rescue",
            "retry",
            "return",
            "self",
            "super",
            "then",
            "true",
            "undef",
            "unless",
            "until",
            "when",
            "while",
            "yield",
            "attr_accessor",
            "attr_reader",
            "attr_writer",
            "require",
            "include",
            "extend",
            "raise",
            "puts",
            "p",
            "pp",
        ],
        "html" => &[
            "html",
            "head",
            "body",
            "div",
            "span",
            "p",
            "a",
            "img",
            "ul",
            "ol",
            "li",
            "table",
            "tr",
            "td",
            "th",
            "form",
            "input",
            "button",
            "select",
            "option",
            "textarea",
            "label",
            "script",
            "style",
            "link",
            "meta",
            "title",
            "header",
            "footer",
            "nav",
            "main",
            "section",
            "article",
            "aside",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "class",
            "id",
            "src",
            "href",
            "type",
            "value",
            "name",
            "placeholder",
            "disabled",
            "readonly",
            "required",
            "hidden",
        ],
        "css" => &[
            "color",
            "background",
            "background-color",
            "border",
            "border-radius",
            "box-shadow",
            "display",
            "flex",
            "grid",
            "font",
            "font-size",
            "font-weight",
            "height",
            "width",
            "margin",
            "padding",
            "position",
            "top",
            "left",
            "right",
            "bottom",
            "z-index",
            "overflow",
            "opacity",
            "transform",
            "transition",
            "animation",
            "cursor",
            "align-items",
            "align-self",
            "justify-content",
            "gap",
            "grid-template-columns",
            "important",
            "hover",
            "focus",
            "active",
            "before",
            "after",
            "first-child",
            "last-child",
            "nth-child",
            "root",
            "var",
            "calc",
            "min",
            "max",
            "clamp",
            "media",
            "keyframes",
            "from",
            "to",
        ],
        "lua" => &[
            "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto",
            "if", "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until",
            "while", "print", "pairs", "ipairs", "tostring", "tonumber", "type", "require",
            "table", "string", "math", "io", "os",
        ],
        "php" => &[
            "abstract",
            "and",
            "array",
            "as",
            "break",
            "callable",
            "case",
            "catch",
            "class",
            "clone",
            "const",
            "continue",
            "declare",
            "default",
            "die",
            "do",
            "echo",
            "else",
            "elseif",
            "empty",
            "enddeclare",
            "endfor",
            "endforeach",
            "endif",
            "endswitch",
            "endwhile",
            "eval",
            "exit",
            "extends",
            "final",
            "finally",
            "fn",
            "for",
            "foreach",
            "function",
            "global",
            "goto",
            "if",
            "implements",
            "include",
            "instanceof",
            "insteadof",
            "interface",
            "isset",
            "list",
            "match",
            "namespace",
            "new",
            "or",
            "print",
            "private",
            "protected",
            "public",
            "readonly",
            "require",
            "return",
            "static",
            "switch",
            "throw",
            "trait",
            "try",
            "unset",
            "use",
            "var",
            "while",
            "xor",
            "yield",
            "true",
            "false",
            "null",
            "self",
            "parent",
        ],
        "dart" => &[
            "abstract",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "covariant",
            "default",
            "deferred",
            "do",
            "dynamic",
            "else",
            "enum",
            "export",
            "extends",
            "extension",
            "external",
            "factory",
            "false",
            "final",
            "finally",
            "for",
            "Function",
            "get",
            "hide",
            "if",
            "implements",
            "import",
            "in",
            "interface",
            "is",
            "late",
            "library",
            "mixin",
            "new",
            "null",
            "on",
            "operator",
            "part",
            "required",
            "rethrow",
            "return",
            "sealed",
            "set",
            "show",
            "static",
            "super",
            "switch",
            "this",
            "throw",
            "true",
            "try",
            "typedef",
            "var",
            "void",
            "while",
            "with",
            "yield",
            "int",
            "double",
            "String",
            "bool",
            "List",
            "Map",
            "Set",
            "Future",
            "Stream",
            "print",
        ],
        "elixir" => &[
            "after",
            "and",
            "catch",
            "do",
            "else",
            "end",
            "false",
            "fn",
            "in",
            "not",
            "or",
            "rescue",
            "true",
            "when",
            "unless",
            "use",
            "import",
            "require",
            "alias",
            "def",
            "defp",
            "defmodule",
            "defstruct",
            "defprotocol",
            "defimpl",
            "defmacro",
            "defmacrop",
            "defguard",
            "defguardp",
            "defdelegate",
            "defexception",
            "defcallback",
            "defmacrocallback",
            "defoverridable",
            "case",
            "cond",
            "for",
            "if",
            "unless",
            "receive",
            "try",
            "with",
            "raise",
            "throw",
            "quote",
            "unquote",
            "super",
            "IO",
            "Enum",
            "Map",
            "List",
            "String",
            "Atom",
            "Kernel",
            "Agent",
            "GenServer",
            "Task",
            "Supervisor",
            "Application",
            "Process",
            "Node",
            "Module",
            "Function",
        ],
        "haskell" => &[
            "case",
            "class",
            "data",
            "default",
            "deriving",
            "do",
            "else",
            "forall",
            "foreign",
            "if",
            "import",
            "in",
            "infix",
            "infixl",
            "infixr",
            "instance",
            "let",
            "module",
            "newtype",
            "of",
            "qualified",
            "then",
            "type",
            "where",
            "do",
            "if",
            "then",
            "else",
            "case",
            "of",
            "let",
            "in",
            "where",
            "True",
            "False",
            "Nothing",
            "Just",
            "Left",
            "Right",
            "IO",
            "Maybe",
            "Either",
            "Int",
            "Integer",
            "Float",
            "Double",
            "String",
            "Char",
            "Bool",
            "putStrLn",
            "print",
            "return",
            "pure",
            "map",
            "filter",
            "foldl",
            "foldr",
        ],
        "protobuf" => &[
            "syntax",
            "package",
            "import",
            "option",
            "message",
            "enum",
            "service",
            "rpc",
            "returns",
            "stream",
            "repeated",
            "optional",
            "required",
            "oneof",
            "map",
            "reserved",
            "extensions",
            "extend",
            "to",
            "true",
            "false",
            "null",
            "int32",
            "int64",
            "uint32",
            "uint64",
            "sint32",
            "sint64",
            "fixed32",
            "fixed64",
            "sfixed32",
            "sfixed64",
            "float",
            "double",
            "bool",
            "string",
            "bytes",
        ],
        "dockerfile" => &[
            "FROM",
            "RUN",
            "CMD",
            "LABEL",
            "EXPOSE",
            "ENV",
            "ADD",
            "COPY",
            "ENTRYPOINT",
            "VOLUME",
            "USER",
            "WORKDIR",
            "ARG",
            "ONBUILD",
            "STOPSIGNAL",
            "HEALTHCHECK",
            "SHELL",
            "AS",
            "MAINTAINER",
        ],
        "makefile" => &[
            "ifeq", "ifneq", "ifdef", "ifndef", "else", "endif", "define", "endef", "include",
            "export", "unexport", "override", "private", "vpath", "all", "clean", "install",
            "test", "build", "run", "check", "PHONY", "SHELL", "MAKE", "MAKEFILE", "CURDIR", "RM",
            "CP", "MV",
        ],
        "vue" | "svelte" => &[
            "script",
            "template",
            "style",
            "setup",
            "lang",
            // Also highlight JS/TS keywords since these are component frameworks
            "import",
            "export",
            "from",
            "default",
            "const",
            "let",
            "var",
            "function",
            "return",
            "if",
            "else",
            "for",
            "while",
            "class",
            "extends",
            "new",
            "this",
            "super",
            "true",
            "false",
            "null",
            "undefined",
            "async",
            "await",
            "try",
            "catch",
            "finally",
            "throw",
            "typeof",
            "instanceof",
            "in",
            "of",
            "props",
            "emit",
            "ref",
            "reactive",
            "computed",
            "watch",
            "onMounted",
        ],
        "zig" => &[
            "const",
            "var",
            "fn",
            "pub",
            "return",
            "if",
            "else",
            "while",
            "for",
            "switch",
            "try",
            "catch",
            "error",
            "defer",
            "errdefer",
            "async",
            "await",
            "suspend",
            "resume",
            "usingnamespace",
            "struct",
            "enum",
            "union",
            "opaque",
            "comptime",
            "true",
            "false",
            "null",
            "undefined",
            "unreachable",
            "u8",
            "u16",
            "u32",
            "u64",
            "i8",
            "i16",
            "i32",
            "i64",
            "f16",
            "f32",
            "f64",
            "bool",
            "void",
            "type",
            "anytype",
            "print",
            "alloc",
            "free",
            "create",
            "destroy",
        ],
        "nim" => &[
            "addr",
            "and",
            "as",
            "asm",
            "bind",
            "block",
            "break",
            "case",
            "cast",
            "concept",
            "const",
            "continue",
            "converter",
            "defer",
            "discard",
            "distinct",
            "do",
            "elif",
            "else",
            "end",
            "enum",
            "except",
            "export",
            "finally",
            "for",
            "from",
            "func",
            "if",
            "import",
            "include",
            "interface",
            "iterator",
            "let",
            "macro",
            "method",
            "mixin",
            "mod",
            "nil",
            "not",
            "object",
            "of",
            "or",
            "out",
            "proc",
            "ptr",
            "raise",
            "ref",
            "return",
            "shl",
            "shr",
            "static",
            "template",
            "try",
            "tuple",
            "type",
            "using",
            "var",
            "when",
            "while",
            "with",
            "without",
            "xor",
            "yield",
            "true",
            "false",
            "echo",
            "new",
            "int",
            "float",
            "string",
            "bool",
            "char",
            "seq",
            "array",
            "set",
            "Table",
            "OrderedTable",
            "Option",
        ],
        "r" => &[
            "if",
            "else",
            "repeat",
            "while",
            "function",
            "for",
            "in",
            "next",
            "break",
            "TRUE",
            "FALSE",
            "NULL",
            "NA",
            "Inf",
            "NaN",
            "library",
            "require",
            "source",
            "return",
            "invisible",
            "print",
            "cat",
            "paste",
            "paste0",
            "nchar",
            "substr",
            "grep",
            "gsub",
            "length",
            "which",
            "min",
            "max",
            "sum",
            "mean",
            "range",
            "var",
            "data.frame",
            "matrix",
            "list",
            "c",
            "cbind",
            "rbind",
            "apply",
            "lapply",
            "sapply",
            "vapply",
            "tapply",
            "mapply",
        ],
        "erlang" => &[
            "after",
            "begin",
            "case",
            "catch",
            "cond",
            "end",
            "fun",
            "if",
            "let",
            "of",
            "receive",
            "when",
            "try",
            "query",
            "module",
            "export",
            "import",
            "include",
            "define",
            "spec",
            "type",
            "record",
            "behaviour",
            "behavior",
            "callback",
            "true",
            "false",
            "undefined",
            "ok",
            "error",
            "ignore",
            "stop",
        ],
        "markdown" => &[], // markdown has no keywords — plain text rendering
        "xml" => &[],      // XML has no keywords — plain text rendering
        "json" => &[],     // JSON has no keywords — plain text rendering
        "toml" => &[],     // TOML has no keywords — plain text rendering
        "yaml" => &[],     // YAML has no keywords — plain text rendering
        _ => &[],
    };

    let comment_starts: &[&str] = match lang {
        "python" | "py" | "bash" | "sh" | "shell" | "zsh" | "yaml" | "toml" | "ruby" | "r"
        | "nim" => &["#"],
        "sql" | "haskell" | "elixir" | "erlang" => &["--"],
        "html" | "xml" | "vue" | "svelte" => &["<!--"],
        "lua" => &["--"],
        "protobuf" => &["//"],
        "dockerfile" | "makefile" => &["#"],
        _ => &["//", "/*"],
    };

    // Full-line comment fast path
    let trimmed = line.trim_start();
    for cs in comment_starts {
        if trimmed.starts_with(cs) {
            return vec![Span::styled(
                line.to_string(),
                Style::default().fg(C_COMMENT),
            )];
        }
    }

    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut pos = 0;
    let mut spans: Vec<Span<'static>> = Vec::new();

    let push = |spans: &mut Vec<Span<'static>>, text: String, color: Color| {
        if text.is_empty() {
            return;
        }
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
                if chars[pos] == '\\' {
                    pos += 2;
                    continue;
                }
                if chars[pos] == q {
                    pos += 1;
                    break;
                }
                pos += 1;
            }
            let s: String = chars[start..pos.min(len)].iter().collect();
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
            if chars[pos] == '-' {
                pos += 1;
            }
            while pos < len
                && (chars[pos].is_ascii_digit()
                    || chars[pos] == '.'
                    || chars[pos] == '_'
                    || chars[pos] == 'x'
                    || chars[pos] == 'b')
            {
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
            } else if word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            {
                C_TYPE // PascalCase → likely a type
            } else {
                C_NORMAL
            };
            push(&mut spans, word, color);
            continue;
        }

        // Operator / punctuation
        let ch = chars[pos].to_string();
        let color = if "=<>!&|+-*/%^~".contains(chars[pos]) {
            C_OPERATOR
        } else {
            C_NORMAL
        };
        push(&mut spans, ch, color);
        pos += 1;
    }

    if spans.is_empty() {
        spans.push(Span::styled(
            line.to_string(),
            Style::default().fg(C_NORMAL),
        ));
    }
    spans
}

fn align_cell(text: &str, width: usize, align: Alignment) -> String {
    let display: String = text.chars().take(width).collect();
    let len = display.chars().count();
    if len >= width {
        return display;
    }
    let padding = width - len;
    match align {
        Alignment::Right => format!("{}{}", " ".repeat(padding), display),
        Alignment::Center => {
            let left = padding / 2;
            format!(
                "{}{}{}",
                " ".repeat(left),
                display,
                " ".repeat(padding - left)
            )
        }
        _ => format!("{}{}", display, " ".repeat(padding)),
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Infer a language tag from a file extension for syntax highlighting.
pub fn lang_for_path(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("");
    match ext {
        "rs" => "rust",
        "go" => "go",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" | "mts" => "typescript",
        "tsx" => "tsx",
        "jsx" => "jsx",
        "py" | "pyw" => "python",
        "sh" | "bash" | "zsh" => "bash",
        "sql" => "sql",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "rb" => "ruby",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => "cpp",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "swift" => "swift",
        "html" | "htm" => "html",
        "css" | "scss" | "sass" | "less" => "css",
        "md" | "mdx" => "markdown",
        "xml" | "svg" | "xsl" | "xslt" => "xml",
        "lua" => "lua",
        "php" => "php",
        "r" => "r",
        "dart" => "dart",
        "ex" | "exs" => "elixir",
        "erl" | "hrl" => "erlang",
        "hs" => "haskell",
        "proto" => "protobuf",
        "dockerfile" | "containerfile" => "dockerfile",
        "makefile" | "mk" | "mak" => "makefile",
        "vue" => "vue",
        "svelte" => "svelte",
        "zig" => "zig",
        "nim" => "nim",
        _ => {
            // Fallback: check the filename itself (e.g. Makefile, Dockerfile)
            let filename = path.rsplit('/').next().unwrap_or(path).to_lowercase();
            match filename.as_str() {
                f if f.starts_with("dockerfile") || f.starts_with("containerfile") => "dockerfile",
                f if f == "makefile" || f == "gnumakefile" => "makefile",
                _ => "",
            }
        }
    }
}

fn heading_color(level: HeadingLevel) -> Color {
    match level {
        HeadingLevel::H1 => Color::Rgb(100, 200, 163), // mint (accent)
        HeadingLevel::H2 => Color::Rgb(140, 200, 255), // sky blue
        HeadingLevel::H3 => Color::Rgb(220, 140, 255), // soft violet
        HeadingLevel::H4 | HeadingLevel::H5 | HeadingLevel::H6 => COLOR_MUTED,
    }
}

fn word_wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
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
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}
