package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/html"
)

// MetasearchConfig holds configuration for the embedded metasearch engine.
type MetasearchConfig struct {
	EnableBing     bool `json:"enable_bing" yaml:"enable_bing"`
	TimeoutSeconds int  `json:"timeout_seconds" yaml:"timeout_seconds"`
	MaxResults     int  `json:"max_results" yaml:"max_results"`
	JitterMs       int  `json:"jitter_ms" yaml:"jitter_ms"`
}

// DefaultMetasearchConfig returns sensible defaults.
func DefaultMetasearchConfig() MetasearchConfig {
	return MetasearchConfig{
		EnableBing:     false,
		TimeoutSeconds: 8,
		MaxResults:     5,
		JitterMs:       0,
	}
}

// userAgents is a small rotation pool, reducing fingerprinting.
var userAgents = []string{
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
}

func randomUA() string {
	return userAgents[rand.Intn(len(userAgents))]
}

// internalResult is a single result from any backend, with a ranking score.
type internalResult struct {
	Title       string
	URL         string
	Description string
	Source      string
	Score       int
}

// SearchResult is the public result returned to the LLM.
type SearchResult struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	Description string `json:"description"`
	Source      string `json:"source"`
}

// WebSearchTool is an embedded metasearch engine — no API keys required.
// It fans out to DuckDuckGo HTML, Wikipedia JSON API, and optionally Bing HTML concurrently.
type WebSearchTool struct {
	cfg        MetasearchConfig
	httpClient *http.Client
}

// NewWebSearchTool creates a WebSearchTool with the given config.
func NewWebSearchTool(cfg MetasearchConfig) *WebSearchTool {
	perBackend := time.Duration(cfg.TimeoutSeconds) * time.Second
	if perBackend <= 0 {
		perBackend = 8 * time.Second
	}
	return &WebSearchTool{
		cfg: cfg,
		httpClient: &http.Client{
			Timeout: perBackend + 3*time.Second,
		},
	}
}

func (t *WebSearchTool) Name() string { return "webSearch" }

func (t *WebSearchTool) Description() string {
	return "Searches the web for up-to-date information by querying DuckDuckGo, Wikipedia, and optionally Bing concurrently. Returns merged, ranked results. No API key required. Use this for current documentation, news, GitHub issues, Stack Overflow answers, or anything beyond your training data."
}

func (t *WebSearchTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"query": {
				Type:        "string",
				Description: "The search query.",
			},
			"count": {
				Type:        "integer",
				Description: "Number of results to return (1–10). Defaults to 5.",
				Default:     5,
			},
		},
		Required: []string{"query"},
	}
}

func (t *WebSearchTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}

	query, _ := params["query"].(string)
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, &ErrInvalidToolParams{ToolName: t.Name(), Message: "query cannot be empty"}
	}

	count := t.cfg.MaxResults
	if count <= 0 {
		count = 5
	}
	if c, ok := params["count"].(float64); ok && c > 0 {
		count = int(c)
	}
	if count > 10 {
		count = 10
	}

	// Optional jitter to reduce rate-limit risk
	if t.cfg.JitterMs > 0 {
		jitter := time.Duration(rand.Intn(t.cfg.JitterMs)) * time.Millisecond
		select {
		case <-time.After(jitter):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	timeout := time.Duration(t.cfg.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 8 * time.Second
	}

	raw := t.fanOut(ctx, query, timeout)
	merged := mergeAndRank(raw, count)

	public := make([]SearchResult, 0, len(merged))
	for _, r := range merged {
		public = append(public, SearchResult{
			Title:       r.Title,
			URL:         r.URL,
			Description: r.Description,
			Source:      r.Source,
		})
	}

	note := ""
	if len(public) == 0 {
		note = "No results found across all backends. The search engines may be rate-limiting this IP. Try again shortly or rephrase the query."
	}

	return map[string]interface{}{
		"success": len(public) > 0,
		"query":   query,
		"results": public,
		"note":    note,
	}, nil
}

// fanOut runs all enabled backends concurrently and collects results.
func (t *WebSearchTool) fanOut(ctx context.Context, query string, timeout time.Duration) []internalResult {
	type backendFn func(context.Context, string) []internalResult

	backends := []backendFn{
		t.duckduckgo,
		t.wikipedia,
	}
	if t.cfg.EnableBing {
		backends = append(backends, t.bing)
	}

	var (
		mu  sync.Mutex
		all []internalResult
		wg  sync.WaitGroup
	)

	for _, fn := range backends {
		wg.Add(1)
		go func(b backendFn) {
			defer wg.Done()
			bCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()
			res := b(bCtx, query)
			mu.Lock()
			all = append(all, res...)
			mu.Unlock()
		}(fn)
	}

	wg.Wait()
	return all
}

// ── DuckDuckGo HTML backend ───────────────────────────────────────────────────

func (t *WebSearchTool) duckduckgo(ctx context.Context, query string) []internalResult {
	form := url.Values{}
	form.Set("q", query)
	form.Set("kl", "wt-wt") // no region bias

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://html.duckduckgo.com/html/",
		strings.NewReader(form.Encode()))
	if err != nil {
		return nil
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", randomUA())
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")
	// Sending an existing kl cookie bypasses DuckDuckGo's JS-required landing page
	req.Header.Set("Cookie", "kl=wt-wt")

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	return parseDDGHTML(resp.Body)
}

// parseDDGHTML extracts results from DuckDuckGo's lite HTML response.
//
// Stable DDG HTML structure:
//
//	<div class="result__body">
//	  <h2 class="result__title">
//	    <a class="result__a" href="//duckduckgo.com/l/?uddg=...&rut=...">Title text</a>
//	  </h2>
//	  <a class="result__snippet">Snippet text</a>
//	</div>
func parseDDGHTML(r io.Reader) []internalResult {
	doc, err := html.Parse(r)
	if err != nil {
		return nil
	}

	var results []internalResult
	pos := 0

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "div" && hasClass(n, "result__body") {
			pos++
			if res, ok := extractDDGResult(n, pos); ok {
				results = append(results, res)
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	return results
}

func extractDDGResult(n *html.Node, pos int) (internalResult, bool) {
	res := internalResult{
		Source: "duckduckgo",
		Score:  maxInt(100-(pos-1)*10, 10),
	}

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			if n.Data == "a" && hasClass(n, "result__a") {
				res.Title = strings.TrimSpace(textContent(n))
				for _, a := range n.Attr {
					if a.Key == "href" {
						res.URL = cleanDDGURL(a.Val)
					}
				}
			}
			if n.Data == "a" && hasClass(n, "result__snippet") && res.Description == "" {
				res.Description = strings.TrimSpace(textContent(n))
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)

	return res, res.URL != "" && res.Title != ""
}

// cleanDDGURL unwraps DuckDuckGo's redirect URLs to get the real destination URL.
func cleanDDGURL(raw string) string {
	// DDG wraps outbound links as //duckduckgo.com/l/?uddg=<encoded_url>&...
	if strings.Contains(raw, "duckduckgo.com/l/") {
		// Make it a full URL so url.Parse handles it
		if strings.HasPrefix(raw, "//") {
			raw = "https:" + raw
		}
		parsed, err := url.Parse(raw)
		if err == nil {
			if u := parsed.Query().Get("uddg"); u != "" {
				decoded, err := url.QueryUnescape(u)
				if err == nil {
					return decoded
				}
			}
		}
	}
	return raw
}

// ── Wikipedia Search API backend ──────────────────────────────────────────────
//
// Official public API — no key, no rate-limit for reasonable use.
// https://www.mediawiki.org/wiki/API:Search

func (t *WebSearchTool) wikipedia(ctx context.Context, query string) []internalResult {
	endpoint := fmt.Sprintf(
		"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&srlimit=3&format=json&srprop=snippet",
		url.QueryEscape(query),
	)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil
	}
	// Wikipedia requests a descriptive User-Agent per their API policy
	req.Header.Set("User-Agent", "Codezilla/2.0 (AI coding assistant; https://github.com/basaid-dev/codezilla)")
	req.Header.Set("Accept", "application/json")

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	if err != nil {
		return nil
	}

	return parseWikipediaJSON(body)
}

func parseWikipediaJSON(data []byte) []internalResult {
	var apiResp struct {
		Query struct {
			Search []struct {
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := json.Unmarshal(data, &apiResp); err != nil {
		return nil
	}

	results := make([]internalResult, 0, len(apiResp.Query.Search))
	for i, s := range apiResp.Query.Search {
		title := s.Title
		// Wikipedia article URL from title
		articleURL := "https://en.wikipedia.org/wiki/" + url.PathEscape(strings.ReplaceAll(title, " ", "_"))
		// Strip HTML tags from snippet (Wikipedia returns <span class="..."> highlights)
		snippet := stripHTMLTags(s.Snippet)

		results = append(results, internalResult{
			Title:       title,
			URL:         articleURL,
			Description: snippet,
			Source:      "wikipedia",
			Score:       80 - i*5, // Wikipedia results are uniformly high quality
		})
	}
	return results
}

// stripHTMLTags removes HTML tags from a string (for Wiki snippets).
func stripHTMLTags(s string) string {
	doc, err := html.Parse(strings.NewReader(s))
	if err != nil {
		return s
	}
	return strings.TrimSpace(textContent(doc))
}

// ── Bing HTML backend (opt-in) ────────────────────────────────────────────────

func (t *WebSearchTool) bing(ctx context.Context, query string) []internalResult {
	endpoint := fmt.Sprintf("https://www.bing.com/search?q=%s&count=5&setlang=en", url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil
	}
	req.Header.Set("User-Agent", randomUA())
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	return parseBingHTML(resp.Body)
}

// parseBingHTML extracts results from Bing's search HTML response.
//
// Bing result structure:
//
//	<li class="b_algo">
//	  <h2><a href="https://...">Title</a></h2>
//	  <div class="b_caption"><p>Snippet text</p></div>
//	</li>
func parseBingHTML(r io.Reader) []internalResult {
	doc, err := html.Parse(r)
	if err != nil {
		return nil
	}

	var results []internalResult
	pos := 0

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "li" && hasClass(n, "b_algo") {
			pos++
			if res, ok := extractBingResult(n, pos); ok {
				results = append(results, res)
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	return results
}

func extractBingResult(n *html.Node, pos int) (internalResult, bool) {
	res := internalResult{
		Source: "bing",
		Score:  maxInt(70-(pos-1)*10, 10),
	}

	inH2 := false

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "h2":
				inH2 = true
				defer func() { inH2 = false }()
			case "a":
				if inH2 && res.Title == "" {
					res.Title = strings.TrimSpace(textContent(n))
					for _, a := range n.Attr {
						if a.Key == "href" && strings.HasPrefix(a.Val, "http") {
							res.URL = a.Val
						}
					}
				}
			case "p":
				if res.Description == "" {
					candidate := strings.TrimSpace(textContent(n))
					if len(candidate) > 20 {
						res.Description = candidate
					}
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)

	return res, res.URL != "" && res.Title != ""
}

// ── Merge & Rank ──────────────────────────────────────────────────────────────

// mergeAndRank deduplicates by normalised URL, sorts by score, and caps at limit.
func mergeAndRank(all []internalResult, limit int) []internalResult {
	seen := make(map[string]bool)
	deduped := make([]internalResult, 0, len(all))

	for _, r := range all {
		if r.URL == "" {
			continue
		}
		key := normaliseURL(r.URL)
		if seen[key] {
			continue
		}
		seen[key] = true
		deduped = append(deduped, r)
	}

	// Insertion sort (results slice is tiny — typically ≤ 20 items)
	for i := 1; i < len(deduped); i++ {
		for j := i; j > 0 && deduped[j].Score > deduped[j-1].Score; j-- {
			deduped[j], deduped[j-1] = deduped[j-1], deduped[j]
		}
	}

	if len(deduped) > limit {
		deduped = deduped[:limit]
	}
	return deduped
}

// normaliseURL strips www., trailing slashes, and tracking params for dedup.
func normaliseURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return strings.ToLower(raw)
	}
	host := strings.ToLower(strings.TrimPrefix(u.Hostname(), "www."))
	path := strings.TrimRight(u.Path, "/")
	return host + path
}

// ── HTML helpers ──────────────────────────────────────────────────────────────

func hasClass(n *html.Node, class string) bool {
	for _, a := range n.Attr {
		if a.Key == "class" {
			for _, c := range strings.Fields(a.Val) {
				if c == class {
					return true
				}
			}
		}
	}
	return false
}

func textContent(n *html.Node) string {
	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			sb.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return sb.String()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
