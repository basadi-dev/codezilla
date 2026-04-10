package llm

import (
	"context"
	"fmt"
	"net/http"
	"sync"

	"codezilla/internal/config"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
	"github.com/mozilla-ai/any-llm-go/providers/gemini"
	"github.com/mozilla-ai/any-llm-go/providers/ollama"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

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
	cfg       *config.Config
	mu        sync.RWMutex
	providers map[string]anyllm.Provider
}

// NewClient creates a new thread-safe LLM client registry
func NewClient(cfg *config.Config) *Client {
	return &Client{
		cfg:       cfg,
		providers: make(map[string]anyllm.Provider),
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

		httpClient := &http.Client{Transport: transport}
		opts = append(opts, anyllm.WithHTTPClient(httpClient))
	}

	return ollama.New(opts...)
}

// Complete executes a non-streaming text completion
func (c *Client) Complete(ctx context.Context, providerName, model string, messages []anyllm.Message, temperature float64) (*anyllm.ChatCompletion, error) {
	p, err := c.GetProvider(providerName)
	if err != nil {
		return nil, err
	}

	params := anyllm.CompletionParams{
		Model:       model,
		Messages:    messages,
		Temperature: &temperature,
	}

	return p.Completion(ctx, params)
}

// Stream executes a streaming text completion
func (c *Client) Stream(ctx context.Context, providerName, model string, messages []anyllm.Message, temperature float64) (<-chan anyllm.ChatCompletionChunk, <-chan error, error) {
	p, err := c.GetProvider(providerName)
	if err != nil {
		return nil, nil, err
	}

	params := anyllm.CompletionParams{
		Model:       model,
		Messages:    messages,
		Temperature: &temperature,
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
