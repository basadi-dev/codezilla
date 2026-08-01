use super::types::EntryKind;

#[derive(Debug, Clone)]
pub(super) struct CachedTranscriptEntry {
    pub(super) kind: EntryKind,
    pub(super) title: String,
    pub(super) timestamp: Option<i64>,
    pub(super) completed_at: Option<i64>,
    pub(super) pending: bool,
    pub(super) is_working_timer: bool,
    pub(super) raw_body: String,
    pub(super) body_lines: Vec<String>,
    pub(super) line_count: usize,
    pub(super) collapsed: bool,
}

#[derive(Debug, Default, Clone)]
pub(super) struct TranscriptRenderCache {
    pub(super) width: u16,
    pub(super) dirty: bool,
    pub(super) entries: Vec<CachedTranscriptEntry>,
    pub(super) line_ends: Vec<usize>,
    pub(super) total_lines: usize,
}
