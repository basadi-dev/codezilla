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
    if lower.contains("401")
        || lower.contains("unauthori")
        || lower.contains("authoriz")
        || lower.contains("invalid api key")
    {
        return ErrorKind::AuthError;
    }
    if lower.contains("api error")
        || lower.contains("http error")
        || lower.contains("api returned an error")
    {
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

// ─── Humanization ────────���────────────────────────────────────────────────────

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
    if let Some(msg) = payload
        .get(super::domain::KEY_MESSAGE)
        .and_then(|v| v.as_str())
    {
        return msg.to_string();
    }
    // Flatten a simple string value
    if let Some(s) = payload.as_str() {
        return s.to_string();
    }
    // Pretty-print as last resort but strip outer quotes
    serde_json::to_string(payload).unwrap_or_else(|_| payload.to_string())
}

// ─── Context overflow detection ───────────────────────────────────────────────

/// Provider-specific context-overflow patterns.
///
/// Each provider has subtly different error messages.  We check these first
/// (exact match), then fall back to the generic patterns below.
fn provider_overflow_patterns(provider_id: &str, lower: &str) -> bool {
    match provider_id {
        "openai" | "openai-compatible" => {
            lower.contains("maximum context length")
                || lower.contains("context_length_exceeded")
                || lower.contains("reduce the length")
                || (lower.contains("400") && lower.contains("context"))
        }
        "anthropic" => {
            lower.contains("prompt is too long")
                || lower.contains("prompt too long")
                || lower.contains("context window")
                || lower.contains("number of tokens")
        }
        "google" | "gemini" => {
            lower.contains("token count")
                || lower.contains("exceeds the maximum")
                || lower.contains("context window")
                || lower.contains("too many tokens")
        }
        "groq" => {
            lower.contains("context length")
                || lower.contains("too many tokens")
                || lower.contains("reduce your prompt")
        }
        "deepseek" => {
            lower.contains("context length")
                || lower.contains("maximum context")
                || lower.contains("too long")
        }
        "ollama" => {
            lower.contains("context exceeded")
                || lower.contains("max context")
                || lower.contains("too many tokens")
        }
        _ => false, // unknown provider → fall through to generic patterns
    }
}

/// Returns `true` if the error message indicates a context-window overflow.
///
/// Checks provider-specific patterns first (when `provider_id` is known),
/// then falls back to generic substring matching.  Logs a warning when the
/// generic fallback matches so we can add new provider patterns.
pub fn is_context_overflow(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    is_context_overflow_for_provider("", &lower)
}

/// Like [`is_context_overflow`] but checks provider-specific patterns first.
/// Callers that know the provider should use this directly.
pub fn is_context_overflow_for_provider(provider_id: &str, lower: &str) -> bool {
    if !provider_id.is_empty() && provider_overflow_patterns(provider_id, lower) {
        return true;
    }

    // Generic fallback patterns (catch-all).
    let hit = lower.contains("context exceeded")
        || lower.contains("maximum context length")
        || lower.contains("context window")
        || lower.contains("prompt is too long")
        || lower.contains("prompt too long")
        || (lower.contains("api error 400") && lower.contains("context"));

    if hit && !provider_id.is_empty() {
        tracing::warn!(
            provider_id,
            error_snippet = %&lower[..lower.len().min(200)],
            "is_context_overflow: matched via generic fallback — consider adding provider-specific pattern"
        );
    }

    hit
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
                return format!(" ({})", sanitize_api_detail(msg));
            }
        }
    }
    if let Some(req_id) = extract_api_request_id(raw) {
        return format!(" ({req_id})");
    }
    // Fall back to the last colon-separated segment
    let last = sanitize_api_detail(raw.rsplit(':').next().unwrap_or("").trim());
    if last.is_empty() || last == raw {
        String::new()
    } else {
        format!(" ({last})")
    }
}

pub fn extract_api_request_id(raw: &str) -> Option<String> {
    raw.split(|c: char| !(c.is_ascii_hexdigit() || c == '-'))
        .find(|tok| is_uuid_v4ish(tok))
        .map(ToOwned::to_owned)
}

fn sanitize_api_detail(input: &str) -> String {
    input
        .trim()
        .trim_matches('"')
        .trim_end_matches("\\\"})")
        .trim_end_matches("\"})")
        .trim_end_matches("})")
        .trim_end_matches(')')
        .trim_end_matches('"')
        .trim()
        .to_string()
}

fn is_uuid_v4ish(s: &str) -> bool {
    if s.len() != 36 {
        return false;
    }
    for (idx, ch) in s.chars().enumerate() {
        let dash = matches!(idx, 8 | 13 | 18 | 23);
        if dash {
            if ch != '-' {
                return false;
            }
        } else if !ch.is_ascii_hexdigit() {
            return false;
        }
    }
    true
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

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── classify ──────────────────────────────────────────────────────────────

    #[test]
    fn classify_context_overflow() {
        assert_eq!(classify("context exceeded"), ErrorKind::ContextOverflow);
        assert_eq!(
            classify("maximum context length exceeded"),
            ErrorKind::ContextOverflow
        );
        assert_eq!(
            classify("the prompt is too long: 150000 tokens"),
            ErrorKind::ContextOverflow
        );
        assert_eq!(
            classify("prompt too long: 200000 tokens > 128000 maximum"),
            ErrorKind::ContextOverflow
        );
        assert_eq!(
            classify("This model's maximum context window"),
            ErrorKind::ContextOverflow
        );
        assert_eq!(
            classify("API error 400: context window exceeded"),
            ErrorKind::ContextOverflow
        );
    }

    #[test]
    fn classify_auth_error() {
        assert_eq!(classify("401 Unauthorized"), ErrorKind::AuthError);
        assert_eq!(classify("invalid api key provided"), ErrorKind::AuthError);
        assert_eq!(
            classify("Authorization failed for provider"),
            ErrorKind::AuthError
        );
    }

    #[test]
    fn classify_tool_errors() {
        assert_eq!(
            classify("tool_not_found: bash_exec"),
            ErrorKind::ToolNotFound
        );
        assert_eq!(
            classify("tool_invalid_arguments: missing field"),
            ErrorKind::ToolInvalidArguments
        );
        assert_eq!(
            classify("tool_execution_timeout: exceeded 30s"),
            ErrorKind::ToolTimeout
        );
        assert_eq!(
            classify("tool_execution: permission denied"),
            ErrorKind::ToolExecutionError
        );
    }

    #[test]
    fn classify_session_errors() {
        assert_eq!(
            classify("thread_not_found: abc123"),
            ErrorKind::SessionError
        );
        assert_eq!(classify("turn_not_found: xyz"), ErrorKind::SessionError);
    }

    #[test]
    fn classify_permission_errors() {
        assert_eq!(
            classify("permission_denied: write to /etc"),
            ErrorKind::PermissionDenied
        );
        assert_eq!(
            classify("sandbox_denied: network access"),
            ErrorKind::PermissionDenied
        );
    }

    #[test]
    fn classify_stream_and_api_errors() {
        assert_eq!(
            classify("stream failed: connection reset"),
            ErrorKind::StreamFailure
        );
        assert_eq!(classify("stream error: timeout"), ErrorKind::StreamFailure);
        assert_eq!(
            classify("API error 500: internal server error"),
            ErrorKind::ApiError
        );
        assert_eq!(classify("http error 502: bad gateway"), ErrorKind::ApiError);
    }

    #[test]
    fn classify_unknown_fallback() {
        assert_eq!(
            classify("something completely unexpected"),
            ErrorKind::Unknown
        );
    }

    // ── is_context_overflow ───────────────────────────────────────────────────

    #[test]
    fn context_overflow_generic_patterns() {
        assert!(is_context_overflow("context exceeded"));
        assert!(is_context_overflow("maximum context length exceeded"));
        assert!(is_context_overflow(
            "This model's maximum context window is 128K tokens"
        ));
        assert!(is_context_overflow("the prompt is too long: 150000 tokens"));
        assert!(is_context_overflow("prompt too long"));
        assert!(is_context_overflow(
            "API error 400: context window exceeded"
        ));
        // Case insensitive
        assert!(is_context_overflow("CONTEXT EXCEEDED"));
        assert!(is_context_overflow("Maximum Context Length"));
    }

    #[test]
    fn context_overflow_negative() {
        assert!(!is_context_overflow("network timeout"));
        assert!(!is_context_overflow("invalid api key"));
        assert!(!is_context_overflow("tool not found"));
        assert!(!is_context_overflow("API error 400: rate limit exceeded"));
    }

    // ── is_context_overflow_for_provider ──────────────────────────────────────

    #[test]
    fn context_overflow_openai_patterns() {
        assert!(is_context_overflow_for_provider(
            "openai",
            "this model's maximum context length is 4096 tokens"
        ));
        assert!(is_context_overflow_for_provider(
            "openai",
            "context_length_exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "openai",
            "Please reduce the length of the messages"
        ));
        assert!(is_context_overflow_for_provider(
            "openai",
            "400 bad request: context too long"
        ));
        // Should NOT match "400" without "context"
        assert!(!is_context_overflow_for_provider(
            "openai",
            "400 bad request: rate limit"
        ));
    }

    #[test]
    fn context_overflow_anthropic_patterns() {
        assert!(is_context_overflow_for_provider(
            "anthropic",
            "prompt is too long: 200000 tokens > 100000 maximum"
        ));
        assert!(is_context_overflow_for_provider(
            "anthropic",
            "prompt too long"
        ));
        assert!(is_context_overflow_for_provider(
            "anthropic",
            "exceeds the context window"
        ));
        assert!(is_context_overflow_for_provider(
            "anthropic",
            "number of tokens exceeded"
        ));
    }

    #[test]
    fn context_overflow_gemini_patterns() {
        assert!(is_context_overflow_for_provider(
            "google",
            "token count exceeds the maximum"
        ));
        assert!(is_context_overflow_for_provider(
            "gemini",
            "exceeds the maximum number of tokens"
        ));
        assert!(is_context_overflow_for_provider(
            "google",
            "context window exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "gemini",
            "too many tokens in request"
        ));
    }

    #[test]
    fn context_overflow_groq_patterns() {
        assert!(is_context_overflow_for_provider(
            "groq",
            "context length exceeded"
        ));
        assert!(is_context_overflow_for_provider("groq", "too many tokens"));
        assert!(is_context_overflow_for_provider(
            "groq",
            "Please reduce your prompt"
        ));
    }

    #[test]
    fn context_overflow_deepseek_patterns() {
        assert!(is_context_overflow_for_provider(
            "deepseek",
            "context length exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "deepseek",
            "maximum context exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "deepseek",
            "prompt too long"
        ));
    }

    #[test]
    fn context_overflow_ollama_patterns() {
        assert!(is_context_overflow_for_provider(
            "ollama",
            "context exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "ollama",
            "max context exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "ollama",
            "too many tokens"
        ));
    }

    #[test]
    fn context_overflow_unknown_provider_falls_through() {
        // Unknown provider should fall through to generic patterns
        assert!(is_context_overflow_for_provider(
            "unknown_provider",
            "context exceeded"
        ));
        assert!(is_context_overflow_for_provider(
            "unknown_provider",
            "maximum context length"
        ));
        // But should NOT match provider-specific-only patterns
        assert!(!is_context_overflow_for_provider(
            "unknown_provider",
            "reduce the length"
        ));
    }

    // ── humanize ──────────────────────────────────────────────────────────────

    #[test]
    fn humanize_context_overflow() {
        let msg = humanize("context exceeded", ErrorKind::ContextOverflow);
        assert!(msg.contains("/compact") || msg.contains("context window"));
    }

    #[test]
    fn humanize_auth_error() {
        let msg = humanize("401 Unauthorized", ErrorKind::AuthError);
        assert!(msg.contains("Authentication failed"));
    }

    #[test]
    fn humanize_tool_not_found() {
        let msg = humanize("tool_not_found: my_tool", ErrorKind::ToolNotFound);
        assert!(msg.contains("my_tool"));
        assert!(msg.contains("not available"));
    }

    #[test]
    fn humanize_unknown() {
        let msg = humanize("some random error", ErrorKind::Unknown);
        assert_eq!(msg, "some random error");
    }

    // ── from_raw ──────────────────────────────────────────────────────────────

    #[test]
    fn from_raw_classifies_and_humanizes() {
        let err = from_raw("context exceeded");
        assert_eq!(err.kind, ErrorKind::ContextOverflow);
        assert!(!err.message.is_empty());
    }

    // ── ErrorKind ─────────────────────────────────────────────────────────────

    #[test]
    fn error_kind_labels() {
        assert_eq!(ErrorKind::ContextOverflow.label(), "Context Limit");
        assert_eq!(ErrorKind::AuthError.label(), "Auth Error");
        assert_eq!(ErrorKind::StreamFailure.label(), "Stream Error");
        assert_eq!(ErrorKind::Unknown.label(), "Error");
    }

    #[test]
    fn error_kind_is_retryable() {
        assert!(ErrorKind::StreamFailure.is_retryable());
        assert!(ErrorKind::ApiError.is_retryable());
        assert!(!ErrorKind::ContextOverflow.is_retryable());
        assert!(!ErrorKind::AuthError.is_retryable());
    }

    #[test]
    fn error_kind_is_fatal() {
        assert!(ErrorKind::ContextOverflow.is_fatal());
        assert!(ErrorKind::AuthError.is_fatal());
        assert!(ErrorKind::PermissionDenied.is_fatal());
        assert!(!ErrorKind::ApiError.is_fatal());
        assert!(!ErrorKind::StreamFailure.is_fatal());
    }
}
