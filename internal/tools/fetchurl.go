package tools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// FetchURLTool fetches a URL and returns its content as clean markdown using Jina AI Reader.
//
// Jina Reader (r.jina.ai) is completely free – no API key required.
// It strips ads, nav bars, and boilerplate, returning only the meaningful text.
// This lets the LLM "read" any webpage, documentation page, GitHub issue, etc.
type FetchURLTool struct {
	httpClient *http.Client
}

// NewFetchURLTool creates a new FetchURLTool.
func NewFetchURLTool() *FetchURLTool {
	return &FetchURLTool{
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (t *FetchURLTool) Name() string { return "fetchURL" }

func (t *FetchURLTool) Description() string {
	return "Fetches a URL and returns its content as clean, readable markdown. Use this to read documentation pages, GitHub issues, blog posts, Stack Overflow answers, or any web page after finding its URL with webSearch. This tool strips ads and navigation — you get only the relevant text. No API key required."
}

func (t *FetchURLTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"url": {
				Type:        "string",
				Description: "The full URL to fetch (must start with http:// or https://).",
			},
			"max_chars": {
				Type:        "integer",
				Description: "Maximum number of characters to return. Defaults to 8000. Set higher for long pages.",
				Default:     8000,
			},
		},
		Required: []string{"url"},
	}
}

func (t *FetchURLTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	rawURL, _ := params["url"].(string)
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: "url cannot be empty"}
	}
	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
		return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: "url must start with http:// or https://"}
	}

	maxChars := 8000
	if m, ok := params["max_chars"].(float64); ok && m > 0 {
		maxChars = int(m)
	}

	// Jina Reader: prepend r.jina.ai/ to any URL — returns clean markdown
	jinaURL := "https://r.jina.ai/" + rawURL

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, jinaURL, nil)
	if err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: "failed to create request", Err: err}
	}
	req.Header.Set("Accept", "text/plain")
	req.Header.Set("User-Agent", "Codezilla/2.0 (AI coding assistant)")
	// Request markdown output explicitly
	req.Header.Set("X-Return-Format", "markdown")

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: "failed to fetch URL", Err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  fmt.Sprintf("server returned HTTP %d: %s", resp.StatusCode, truncateStr(string(body), 200)),
		}
	}

	// Read up to maxChars * 1.2 bytes to account for multi-byte chars
	limitedReader := io.LimitReader(resp.Body, int64(maxChars)*2)
	raw, err := io.ReadAll(limitedReader)
	if err != nil {
		return nil, &ErrToolExecution{ToolName: t.Name(), Message: "failed to read response", Err: err}
	}

	content := string(raw)
	truncated := false
	if len(content) > maxChars {
		content = content[:maxChars]
		// Trim to last newline so we don't break mid-sentence
		if idx := strings.LastIndex(content, "\n"); idx > maxChars/2 {
			content = content[:idx]
		}
		truncated = true
	}

	return map[string]interface{}{
		"success":   true,
		"url":       rawURL,
		"content":   content,
		"truncated": truncated,
	}, nil
}

func truncateStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
