//! Structured error classification for the Codezilla runtime.
//!
//! All error strings that leave the internal pipeline and are displayed to the
//! user pass through here.  The taxonomy provides:
//!
//! - [`ErrorKind`] — machine-readable category for routing / recovery logic.
//! - [`CodError`] — a thin wrapper that pairs a kind with a human message.
//! - [`classify`] — maps raw `anyhow` error strings → `ErrorKind`.
//! - [`humanize`] — rewrites terse/opaque error strings into plain English.

// ─── Error kinds ─────────────────────────────────────────────────────────────

/// High-level error category.  Determines TUI presentation and recovery strategy.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The LLM provider returned a 4xx/5xx response.
    ApiError,
    /// The conversation exceeded the model's context window.
    ContextOverflow,
    /// A streaming connection failed and the non-streaming fallback also failed.
    StreamFailure,
    /// Authentication / API-key problem.
    AuthError,
    /// The requested tool name is not registered.
    ToolNotFound,
    /// The model sent malformed or missing arguments for a tool call.
    ToolInvalidArguments,
    /// A tool timed out during execution.
    ToolTimeout,
    /// A tool raised an I/O or shell error.
    ToolExecutionError,
    /// Permission / sandbox denied the action.
    PermissionDenied,
    /// Thread or turn not found in the session store.
    SessionError,
    /// Configuration file / value problem.
    ConfigError,
    /// An error that could not be classified more specifically.
    Unknown,
}

impl ErrorKind {
    /// Returns a compact display label used in the TUI error entry title.
    pub fn label(self) -> &'static str {
        match self {
            ErrorKind::ApiError => "API Error",
            ErrorKind::ContextOverflow => "Context Limit",
            ErrorKind::StreamFailure => "Stream Error",
            ErrorKind::AuthError => "Auth Error",
            ErrorKind::ToolNotFound => "Tool Not Found",
            ErrorKind::ToolInvalidArguments => "Bad Tool Args",
            ErrorKind::ToolTimeout => "Tool Timeout",
            ErrorKind::ToolExecutionError => "Tool Error",
            ErrorKind::PermissionDenied => "Permission Denied",
            ErrorKind::SessionError => "Session Error",
            ErrorKind::ConfigError => "Config Error",
            ErrorKind::Unknown => "Error",
        }
    }

    /// True for errors that are transient and may resolve on retry.
    #[allow(dead_code)]
    pub fn is_retryable(self) -> bool {
        matches!(self, ErrorKind::StreamFailure | ErrorKind::ApiError)
    }

    /// True for hard errors that should not auto-retry.
    pub fn is_fatal(self) -> bool {
        matches!(
            self,
            ErrorKind::AuthError | ErrorKind::ContextOverflow | ErrorKind::PermissionDenied
        )
    }
}

// ─── Structured error ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CodError {
    pub kind: ErrorKind,
    /// Human-readable message suitable for display in the TUI.
    pub message: String,
}

impl CodError {
    #[allow(dead_code)]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Format as the single-line string shown in transcript error entries.
    #[allow(dead_code)]
    pub fn display_title(&self) -> String {
        self.kind.label().to_string()
    }

    #[allow(dead_code)]
    pub fn display_body(&self) -> String {
        self.message.clone()
    }
}

impl std::fmt::Display for CodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.kind.label(), self.message)
    }
}

// ─── Classification ───────────────────────────────────────────────────────────

/// Classify a raw error string (typically from `anyhow`) into an [`ErrorKind`].
pub fn classify(raw: &str) -> ErrorKind {
    let lower = raw.to_ascii_lowercase();

    // Prefixes set by Codezilla's own error formatting
    if lower.starts_with("tool_not_found") {
        return ErrorKind::ToolNotFound;
    }
    if lower.starts_with("tool_invalid_arguments") {
        return ErrorKind::ToolInvalidArguments;
    }
    if lower.starts_with("tool_execution_timeout") {
        return ErrorKind::ToolTimeout;
    }
    if lower.starts_with("tool_execution") || lower.starts_with("tool_") {
        return ErrorKind::ToolExecutionError;
    }
    if lower.starts_with("thread_not_found") || lower.starts_with("turn_not_found") {
        return ErrorKind::SessionError;
    }
    if lower.starts_with("permission_denied") || lower.starts_with("sandbox_denied") {
        return ErrorKind::PermissionDenied;
    }
    if lower.starts_with("web_fetch_error") || lower.starts_with("web_fetch_read_error") {
        return ErrorKind::ToolExecutionError;
    }

    // HTTP / API patterns
    if is_context_overflow(raw) {
        return ErrorKind::ContextOverflow;
    }
    if lower.contains("401") || lower.contains("unauthori") || lower.contains("invalid api key") {
        return ErrorKind::AuthError;
    }
    if lower.contains("api error") || lower.contains("http error") {
        return ErrorKind::ApiError;
    }
    if lower.contains("stream") && (lower.contains("fail") || lower.contains("error")) {
        return ErrorKind::StreamFailure;
    }

    ErrorKind::Unknown
}

/// Classify and humanize an error string in one call.
/// Returns a [`CodError`] ready for display.
pub fn from_raw(raw: &str) -> CodError {
    let kind = classify(raw);
    let message = humanize(raw, kind);
    CodError { kind, message }
}

// ─── Humanization ─────────────────────────────────────────────────────────────

/// Rewrite a terse / opaque error string into plain-English user copy.
pub fn humanize(raw: &str, kind: ErrorKind) -> String {
    match kind {
        ErrorKind::ContextOverflow => {
            "The conversation is too long for this model's context window. \
             Use /compact to summarise history, or start a new thread with /new."
                .to_string()
        }

        ErrorKind::AuthError => {
            let extra = extract_api_detail(raw);
            format!("Authentication failed — check your API key or token.{extra}")
        }

        ErrorKind::ToolNotFound => {
            // Extract tool name after "tool_not_found: "
            let tool = raw
                .split_once(':')
                .map(|(_, rest)| rest.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("unknown");
            format!(
                "Tool '{tool}' is not available. \
                 The model may have requested a tool that isn't registered in this session."
            )
        }

        ErrorKind::ToolInvalidArguments => {
            let detail = raw
                .split_once(':')
                .and_then(|(_, rest)| rest.split_once(':').map(|(_, d)| d.trim()))
                .unwrap_or(raw);
            format!("Tool called with invalid arguments: {detail}")
        }

        ErrorKind::ToolTimeout => {
            let secs = raw
                .split("exceeded ")
                .nth(1)
                .and_then(|s| s.split('s').next())
                .unwrap_or("?");
            format!("Tool timed out after {secs}s. The command may be hung or slow.")
        }

        ErrorKind::ToolExecutionError => {
            if raw.starts_with("web_fetch_error") || raw.starts_with("web_fetch_read_error") {
                let detail = raw.split_once(':').map(|(_, r)| r.trim()).unwrap_or(raw);
                format!("Web request failed: {detail}")
            } else {
                strip_codezilla_prefix(raw)
            }
        }

        ErrorKind::PermissionDenied => {
            let detail = raw
                .split_once(':')
                .map(|(_, r)| r.trim())
                .unwrap_or("action was blocked by sandbox policy");
            format!("Permission denied: {detail}")
        }

        ErrorKind::SessionError => {
            if raw.starts_with("thread_not_found") {
                "The conversation thread could not be found. \
                 It may have been deleted or the session state is corrupt."
                    .to_string()
            } else {
                "Internal session error — the current turn or thread is in an unexpected state."
                    .to_string()
            }
        }

        ErrorKind::StreamFailure => {
            let api_detail = extract_api_detail(raw);
            format!("The model stream failed and the fallback also failed.{api_detail}")
        }

        ErrorKind::ApiError => {
            let api_detail = extract_api_detail(raw);
            format!("The API returned an error.{api_detail}")
        }

        ErrorKind::ConfigError => strip_codezilla_prefix(raw),

        ErrorKind::Unknown => strip_codezilla_prefix(raw),
    }
}

// ─── Warning humanization ─────────────────────────────────────────────────────

/// Extract a human-readable message from a `Warning` event payload.
///
/// The payload is a JSON `Value`.  We look for a `"message"` string field;
/// if absent we pretty-print the whole value.
pub fn humanize_warning(payload: &serde_json::Value) -> String {
    if let Some(msg) = payload.get("message").and_then(|v| v.as_str()) {
        return msg.to_string();
    }
    // Flatten a simple string value
    if let Some(s) = payload.as_str() {
        return s.to_string();
    }
    // Pretty-print as last resort but strip outer quotes
    serde_json::to_string(payload).unwrap_or_else(|_| payload.to_string())
}

// ─── Context overflow helper (re-export for model_gateway) ───────────────────

/// Returns `true` if the error message indicates a context-window overflow.
/// Kept in sync with the classifier above.
pub fn is_context_overflow(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("context exceeded")
        || lower.contains("maximum context length")
        || lower.contains("context window")
        || lower.contains("prompt is too long")
        || lower.contains("prompt too long")
        || (lower.contains("api error 400") && lower.contains("context"))
}

// ─── Private helpers ──────────────────────────────────────────────────────────

fn extract_api_detail(raw: &str) -> String {
    // Look for JSON body embedded in the error string
    if let Some(start) = raw.find('{') {
        let candidate = &raw[start..];
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(candidate) {
            // Pull "error.message" or "message"
            let msg = v
                .pointer("/error/message")
                .or_else(|| v.pointer("/message"))
                .and_then(|m| m.as_str());
            if let Some(msg) = msg {
                return format!(" ({msg})");
            }
        }
    }
    // Fall back to the last colon-separated segment
    let last = raw.rsplit(':').next().unwrap_or("").trim();
    if last.is_empty() || last == raw {
        String::new()
    } else {
        format!(" ({last})")
    }
}

fn strip_codezilla_prefix(raw: &str) -> String {
    // Remove known snake_case prefixes like "tool_invalid_arguments: "
    if let Some(rest) = raw.split_once(':') {
        let prefix = rest.0.trim();
        if prefix.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
            let body = rest.1.trim();
            if !body.is_empty() {
                return body.to_string();
            }
        }
    }
    raw.to_string()
}
