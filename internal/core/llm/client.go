package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"codezilla/internal/config"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
	"github.com/mozilla-ai/any-llm-go/providers/gemini"
	"github.com/mozilla-ai/any-llm-go/providers/ollama"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

// ErrContextLengthExceeded is returned when the LLM rejects a prompt because
// it exceeds the model's maximum context window. Callers can use errors.Is()
// to detect this and attempt context trimming before retrying.
var ErrContextLengthExceeded = errors.New("context length exceeded")

// bearerAuthTransport injects an Authorization header into every outgoing
// HTTP request. The official Ollama SDK uses SSH key-based challenge-response
// auth, which does not work with bearer-token-based Ollama Cloud endpoints.
// Wrapping the transport is the cleanest way to add the header without
// forking the SDK.
type bearerAuthTransport struct {
	token string
	base  http.RoundTripper
}

func (t *bearerAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone the request so we don't mutate the caller's headers.
	r := req.Clone(req.Context())
	r.Header.Set("Authorization", "Bearer "+t.token)
	return t.base.RoundTrip(r)
}

// basicAuthTransport injects Basic auth credentials into every request.
type basicAuthTransport struct {
	username string
	password string
	base     http.RoundTripper
}

func (t *basicAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r := req.Clone(req.Context())
	r.SetBasicAuth(t.username, t.password)
	return t.base.RoundTrip(r)
}

// headerTransport injects custom headers into every request.
type headerTransport struct {
	headers map[string]string
	base    http.RoundTripper
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r := req.Clone(req.Context())
	for k, v := range t.headers {
		r.Header.Set(k, v)
	}
	return t.base.RoundTrip(r)
}

type Client struct {
	cfg               *config.Config
	mu                sync.RWMutex
	providers         map[string]anyllm.Provider
	modelContextCache map[string]int // cache: model name → context window size
}

// NewClient creates a new thread-safe LLM client registry
func NewClient(cfg *config.Config) *Client {
	return &Client{
		cfg:               cfg,
		providers:         make(map[string]anyllm.Provider),
		modelContextCache: make(map[string]int),
	}
}

// GetProvider retrieves or lazily instantiates a provider by name
func (c *Client) GetProvider(providerName string) (anyllm.Provider, error) {
	if providerName == "" {
		providerName = "ollama"
	}

	c.mu.RLock()
	p, ok := c.providers[providerName]
	c.mu.RUnlock()
	if ok {
		return p, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check after acquiring write lock
	if p, ok := c.providers[providerName]; ok {
		return p, nil
	}

	var provider anyllm.Provider
	var err error

	switch providerName {
	case "openai":
		if c.cfg.LLM.APIKeys.OpenAI == "" {
			return nil, fmt.Errorf("missing OpenAI API Key")
		}
		opts := []anyllm.Option{anyllm.WithAPIKey(c.cfg.LLM.APIKeys.OpenAI)}
		if c.cfg.LLM.OpenAI.BaseURL != "" {
			opts = append(opts, anyllm.WithBaseURL(c.cfg.LLM.OpenAI.BaseURL))
		}
		provider, err = openai.New(opts...)
	case "anthropic":
		if c.cfg.LLM.APIKeys.Anthropic == "" {
			return nil, fmt.Errorf("missing Anthropic API Key")
		}
		provider, err = anthropic.New(anyllm.WithAPIKey(c.cfg.LLM.APIKeys.Anthropic))
	case "gemini":
		if c.cfg.LLM.APIKeys.Gemini == "" {
			return nil, fmt.Errorf("missing Gemini API Key")
		}
		provider, err = gemini.New(anyllm.WithAPIKey(c.cfg.LLM.APIKeys.Gemini))
	case "ollama":
		provider, err = c.buildOllamaProvider()
	default:
		return nil, fmt.Errorf("unsupported provider: %s", providerName)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to initialize provider %s: %w", providerName, err)
	}

	c.providers[providerName] = provider
	return provider, nil
}

// buildOllamaProvider constructs the Ollama provider with proper auth handling.
// The official Ollama SDK uses SSH key challenge-response auth, which does not
// work with bearer-token or basic-auth cloud endpoints. We inject a custom
// http.Client with a RoundTripper that adds the appropriate auth headers.
func (c *Client) buildOllamaProvider() (anyllm.Provider, error) {
	opts := []anyllm.Option{}
	if c.cfg.LLM.Ollama.BaseURL != "" {
		opts = append(opts, anyllm.WithBaseURL(c.cfg.LLM.Ollama.BaseURL))
	}

	// Build a custom HTTP client that injects auth headers, since the
	// Ollama SDK's own auth mechanism (SSH keys) doesn't support cloud tokens.
	needsCustomTransport := c.cfg.LLM.APIKeys.Ollama != "" ||
		(c.cfg.LLM.Ollama.Username != "" && c.cfg.LLM.Ollama.Password != "") ||
		len(c.cfg.LLM.Ollama.Headers) > 0

	if needsCustomTransport {
		transport := http.RoundTripper(http.DefaultTransport)

		// Layer 1: custom headers
		if len(c.cfg.LLM.Ollama.Headers) > 0 {
			transport = &headerTransport{
				headers: c.cfg.LLM.Ollama.Headers,
				base:    transport,
			}
		}

		// Layer 2: auth (bearer takes priority over basic)
		authType := c.cfg.LLM.Ollama.AuthType
		if authType == "" && c.cfg.LLM.APIKeys.Ollama != "" {
			authType = "bearer"
		}

		switch authType {
		case "bearer":
			if c.cfg.LLM.APIKeys.Ollama != "" {
				transport = &bearerAuthTransport{
					token: c.cfg.LLM.APIKeys.Ollama,
					base:  transport,
				}
			}
		case "basic":
			if c.cfg.LLM.Ollama.Username != "" && c.cfg.LLM.Ollama.Password != "" {
				transport = &basicAuthTransport{
					username: c.cfg.LLM.Ollama.Username,
					password: c.cfg.LLM.Ollama.Password,
					base:     transport,
				}
			}
		}

		httpClient := &http.Client{
			Transport: transport,
			// Upper-bound timeout. For streaming the SDK reads chunks
			// incrementally, so 5 min guards against dead connections
			// without killing long-running streams that are actively
			// producing output.
			Timeout: 5 * time.Minute,
		}
		opts = append(opts, anyllm.WithHTTPClient(httpClient))
	}

	return ollama.New(opts...)
}

// Complete executes a non-streaming text completion.
// Pass a non-empty tools slice to enable native function calling.
func (c *Client) Complete(ctx context.Context, providerName, model string, messages []anyllm.Message, temperature float64, tools []anyllm.Tool) (*anyllm.ChatCompletion, error) {
	p, err := c.GetProvider(providerName)
	if err != nil {
		return nil, err
	}

	params := anyllm.CompletionParams{
		Model:       model,
		Messages:    messages,
		Temperature: &temperature,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	result, err := p.Completion(ctx, params)
	if err != nil {
		return nil, wrapContextLengthError(err)
	}
	return result, nil
}

// Stream executes a streaming text completion.
// Pass a non-empty tools slice to enable native function calling in the stream.
func (c *Client) Stream(ctx context.Context, providerName, model string, messages []anyllm.Message, temperature float64, tools []anyllm.Tool) (<-chan anyllm.ChatCompletionChunk, <-chan error, error) {
	p, err := c.GetProvider(providerName)
	if err != nil {
		return nil, nil, wrapContextLengthError(err)
	}

	params := anyllm.CompletionParams{
		Model:       model,
		Messages:    messages,
		Temperature: &temperature,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	streamCh, errCh := p.CompletionStream(ctx, params)
	return streamCh, errCh, nil
}

// ListModels retrieves the list of models from the provider
func (c *Client) ListModels(ctx context.Context, providerName string) ([]string, error) {
	p, err := c.GetProvider(providerName)
	if err != nil {
		return nil, err
	}

	// any-llm-go supports ModelLister interface for providers that implement it
	if lister, ok := p.(anyllm.ModelLister); ok {
		modelsList, err := lister.ListModels(ctx)
		if err != nil {
			return nil, err
		}

		names := make([]string, 0, len(modelsList.Data))
		for _, m := range modelsList.Data {
			names = append(names, m.ID)
		}
		return names, nil
	}

	// Failsafe for providers without list support
	return []string{c.GetDefaultModel()}, nil
}

// GetDefaultModel safely returns a fallback model string
func (c *Client) GetDefaultModel() string {
	if c.cfg.LLM.Models.Default != "" {
		return c.cfg.LLM.Models.Default
	}
	return "qwen2.5-coder:3b" // Failsafe
}

// GetModelContextLength returns the model's maximum context window in tokens.
// For Ollama, it queries the /api/show endpoint to get the real value.
// For other providers, it uses a well-known lookup table.
// Results are cached per model to avoid redundant API calls.
// Returns 0 if the context length cannot be determined.
func (c *Client) GetModelContextLength(ctx context.Context, providerName, model string) int {
	if providerName == "" {
		providerName = "ollama"
	}

	cacheKey := providerName + ":" + model

	// Check cache first
	c.mu.RLock()
	if cached, ok := c.modelContextCache[cacheKey]; ok {
		c.mu.RUnlock()
		return cached
	}
	c.mu.RUnlock()

	var ctxLen int

	// 1. Check user-configured context_lengths map first (works for any provider)
	if len(c.cfg.LLM.ContextLengths) > 0 {
		ctxLen = c.lookupConfiguredContextLength(model)
	}

	// 2. For Ollama, try dynamic discovery via /api/show
	if ctxLen == 0 && providerName == "ollama" {
		ctxLen = c.queryOllamaContextLength(ctx, model)
	}

	// Cache the result (even 0 = "unknown", to avoid retrying)
	c.mu.Lock()
	c.modelContextCache[cacheKey] = ctxLen
	c.mu.Unlock()

	return ctxLen
}

// lookupConfiguredContextLength checks the config's context_lengths map.
// Tries exact match first, then prefix match (e.g., "gpt-4o" matches "gpt-4o-mini").
func (c *Client) lookupConfiguredContextLength(model string) int {
	// Exact match
	if v, ok := c.cfg.LLM.ContextLengths[model]; ok {
		return v
	}

	// Prefix match (e.g., config has "gpt-4o" and model is "gpt-4o-2024-05-13")
	modelLower := strings.ToLower(model)
	for k, v := range c.cfg.LLM.ContextLengths {
		if strings.HasPrefix(modelLower, strings.ToLower(k)) {
			return v
		}
	}

	return 0
}

// queryOllamaContextLength calls Ollama's /api/show endpoint to get the
// model's actual context window size. Falls back to 0 on any error.
func (c *Client) queryOllamaContextLength(ctx context.Context, model string) int {
	baseURL := c.cfg.LLM.Ollama.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"model":   model,
		"verbose": true,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/show", bytes.NewReader(reqBody))
	if err != nil {
		return 0
	}
	req.Header.Set("Content-Type", "application/json")

	// Apply auth headers if configured
	if c.cfg.LLM.APIKeys.Ollama != "" {
		authType := c.cfg.LLM.Ollama.AuthType
		if authType == "" {
			authType = "bearer"
		}
		if authType == "bearer" {
			req.Header.Set("Authorization", "Bearer "+c.cfg.LLM.APIKeys.Ollama)
		} else if authType == "basic" {
			req.SetBasicAuth(c.cfg.LLM.Ollama.Username, c.cfg.LLM.Ollama.Password)
		}
	}
	for k, v := range c.cfg.LLM.Ollama.Headers {
		req.Header.Set(k, v)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0
	}

	// Parse response to extract context length from model_info
	var showResp struct {
		ModelInfo map[string]interface{} `json:"model_info"`
	}
	if err := json.Unmarshal(body, &showResp); err != nil {
		return 0
	}

	// Look for context length in model_info keys
	// Common keys: "<arch>.context_length", "context_length"
	for k, v := range showResp.ModelInfo {
		if strings.HasSuffix(k, ".context_length") || k == "context_length" {
			switch n := v.(type) {
			case float64:
				return int(n)
			case int:
				return n
			}
		}
	}

	return 0
}

// IsContextLengthError checks if an error message indicates the LLM's context window was exceeded.
func IsContextLengthError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	patterns := []string{
		"context_length_exceeded",
		"context length exceeded",
		"prompt too long",
		"maximum context length",
		"max context length",
		"token limit",
		"exceeds the model's max",
		"input is too long",
	}
	for _, p := range patterns {
		if strings.Contains(msg, p) {
			return true
		}
	}
	return false
}

// wrapContextLengthError wraps a provider error with ErrContextLengthExceeded
// if it matches known context-length error patterns.
func wrapContextLengthError(err error) error {
	if err == nil {
		return nil
	}
	if IsContextLengthError(err) {
		return fmt.Errorf("%w: %v", ErrContextLengthExceeded, err)
	}
	return err
}

// IsTransientError checks if a provider error is likely a transient network or server issue
// that could be resolved by retrying with a short backoff (e.g. 500, 502, 503, 429, timeouts).
func IsTransientError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	patterns := []string{
		"internal server error",
		"bad gateway",
		"service unavailable",
		"too many requests",
		"rate limit",
		"timeout",
		"connection reset",
		"eof",
		"broken pipe",
		"gateway timeout",
	}
	for _, p := range patterns {
		if strings.Contains(msg, p) {
			return true
		}
	}
	return false
}
