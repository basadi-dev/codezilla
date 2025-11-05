package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const (
	DefaultTimeout = 900 * time.Second
	DefaultBaseURL = "http://localhost:11434/api"
)

// Client represents an Ollama API client
type Client interface {
	Generate(ctx context.Context, request GenerateRequest) (*GenerateResponse, error)
	Chat(ctx context.Context, request ChatRequest) (*ChatResponse, error)
	StreamGenerate(ctx context.Context, request GenerateRequest) (<-chan StreamResponse, error)
	StreamChat(ctx context.Context, request ChatRequest) (<-chan StreamChatResponse, error)
	ListModels(ctx context.Context) (*ListModelsResponse, error)
}

// ClientOptions contains configuration options for the Ollama client
type ClientOptions struct {
	BaseURL    string
	HTTPClient *http.Client
	// Authentication options
	APIKey   string
	AuthType string // "bearer", "basic", or "custom"
	Username string
	Password string
	Headers  map[string]string
}

// clientImpl implements the Client interface
type clientImpl struct {
	baseURL    string
	httpClient *http.Client
	apiKey     string
	authType   string
	username   string
	password   string
	headers    map[string]string
}

// NewClient creates a new Ollama client with the given options
func NewClient(options ...func(*ClientOptions)) Client {
	opts := ClientOptions{
		BaseURL:    DefaultBaseURL,
		HTTPClient: &http.Client{Timeout: DefaultTimeout},
	}

	for _, option := range options {
		option(&opts)
	}

	return &clientImpl{
		baseURL:    opts.BaseURL,
		httpClient: opts.HTTPClient,
		apiKey:     opts.APIKey,
		authType:   opts.AuthType,
		username:   opts.Username,
		password:   opts.Password,
		headers:    opts.Headers,
	}
}

// WithBaseURL sets the base URL for the Ollama API
func WithBaseURL(url string) func(*ClientOptions) {
	return func(o *ClientOptions) {
		o.BaseURL = url
	}
}

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(client *http.Client) func(*ClientOptions) {
	return func(o *ClientOptions) {
		o.HTTPClient = client
	}
}

// WithAPIKey sets the API key for authentication
func WithAPIKey(apiKey string) func(*ClientOptions) {
	return func(o *ClientOptions) {
		o.APIKey = apiKey
		if o.AuthType == "" {
			o.AuthType = "bearer"
		}
	}
}

// WithBasicAuth sets basic authentication credentials
func WithBasicAuth(username, password string) func(*ClientOptions) {
	return func(o *ClientOptions) {
		o.Username = username
		o.Password = password
		o.AuthType = "basic"
	}
}

// WithHeaders sets custom headers for requests
func WithHeaders(headers map[string]string) func(*ClientOptions) {
	return func(o *ClientOptions) {
		o.Headers = headers
	}
}

// GenerateRequest represents a request to the Ollama generate API
type GenerateRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	System    string                 `json:"system"`
	Template  string                 `json:"template,omitempty"`
	Context   []int                  `json:"context,omitempty"`
	Stream    bool                   `json:"stream"`
	Options   map[string]interface{} `json:"options,omitempty"`
	Format    string                 `json:"format,omitempty"`
	KeepAlive string                 `json:"keep_alive,omitempty"`
}

// GenerateResponse represents a response from the Ollama generate API
type GenerateResponse struct {
	Model              string `json:"model"`
	Response           string `json:"response"`
	Context            []int  `json:"context,omitempty"`
	CreatedAt          string `json:"created_at,omitempty"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents a request to the Ollama chat API
type ChatRequest struct {
	Model     string                 `json:"model"`
	Messages  []Message              `json:"messages"`
	Stream    bool                   `json:"stream"`
	Options   map[string]interface{} `json:"options,omitempty"`
	KeepAlive string                 `json:"keep_alive,omitempty"`
}

// ChatResponse represents a response from the Ollama chat API
type ChatResponse struct {
	Model              string  `json:"model"`
	Message            Message `json:"message"`
	CreatedAt          string  `json:"created_at,omitempty"`
	Done               bool    `json:"done"`
	TotalDuration      int64   `json:"total_duration,omitempty"`
	LoadDuration       int64   `json:"load_duration,omitempty"`
	PromptEvalCount    int     `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64   `json:"prompt_eval_duration,omitempty"`
	EvalCount          int     `json:"eval_count,omitempty"`
	EvalDuration       int64   `json:"eval_duration,omitempty"`
}

// StreamResponse represents a streamed response chunk
type StreamResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Context  []int  `json:"context,omitempty"`
	Error    string `json:"error,omitempty"`
}

// StreamChatResponse represents a streamed chat response chunk
type StreamChatResponse struct {
	Model   string  `json:"model"`
	Message Message `json:"message"`
	Done    bool    `json:"done"`
	Error   error   `json:"-"`
}

// ListModelsResponse represents the response from the Ollama list models API
type ListModelsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ModelInfo contains information about an Ollama model
type ModelInfo struct {
	Name       string       `json:"name"`
	ModifiedAt string       `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    ModelDetails `json:"details"`
}

// ModelDetails contains detailed information about a model
type ModelDetails struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// Generate sends a generate request to the Ollama API
func (c *clientImpl) Generate(ctx context.Context, request GenerateRequest) (*GenerateResponse, error) {
	// Create a copy of the request with stream explicitly set to false
	requestCopy := request
	requestCopy.Stream = false // This will always be included in the JSON now

	reqBody, err := json.Marshal(requestCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// We'll skip detailed JSON logging to keep output minimal

	generateURL := fmt.Sprintf("%s/generate", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", generateURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request to %s: %w", generateURL, err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.applyAuth(req)

	// Skipping output to keep messages minimal

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unsuccessful response: %d %s", resp.StatusCode, string(bodyBytes))
	}

	// Read the entire response body for debugging
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Skipping response body output to keep terminal clean

	var response GenerateResponse
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// Chat sends a chat request to the Ollama API
func (c *clientImpl) Chat(ctx context.Context, request ChatRequest) (*ChatResponse, error) {
	// Create a copy of the request with stream set to false
	requestCopy := request
	requestCopy.Stream = false

	reqBody, err := json.Marshal(requestCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	chatURL := fmt.Sprintf("%s/chat", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", chatURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request to %s: %w", chatURL, err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.applyAuth(req)

	// Skipping debug output to reduce noise

	// Send the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Simplified error handling

		return nil, fmt.Errorf("failed to send request to %s: %w", chatURL, err)
	}
	defer resp.Body.Close()

	// Omitting response status and header output

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		errMsg := string(bodyBytes)
		fmt.Fprintf(os.Stderr, "Error response body: %s\n", errMsg)
		return nil, fmt.Errorf("unsuccessful response from %s: %d %s", chatURL, resp.StatusCode, errMsg)
	}

	// Read the entire response body for debugging
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Skipping response body size output

	// Create a new reader from the bytes for JSON decoding
	responseReader := bytes.NewReader(bodyBytes)

	var response ChatResponse
	if err := json.NewDecoder(responseReader).Decode(&response); err != nil {
		// Skipping detailed JSON error output

		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// StreamGenerate sends a generate request to the Ollama API and returns a channel for streaming responses
func (c *clientImpl) StreamGenerate(ctx context.Context, request GenerateRequest) (<-chan StreamResponse, error) {
	// Create a copy of the request with stream explicitly set to true
	requestCopy := request
	requestCopy.Stream = true // This will always be included in the JSON now

	reqBody, err := json.Marshal(requestCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/generate", c.baseURL), bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.applyAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("unsuccessful response: %d %s", resp.StatusCode, string(bodyBytes))
	}

	// Buffer the channel to prevent goroutine leak if consumer stops reading
	responseChannel := make(chan StreamResponse, 10)

	go func() {
		defer close(responseChannel)
		defer resp.Body.Close()

		decoder := json.NewDecoder(resp.Body)
		for {
			var response StreamResponse
			if err := decoder.Decode(&response); err != nil {
				if err != io.EOF {
					response.Error = fmt.Sprintf("failed to decode response: %v", err)
					responseChannel <- response
				}
				break
			}

			select {
			case <-ctx.Done():
				return
			case responseChannel <- response:
			}

			if response.Done {
				break
			}
		}
	}()

	return responseChannel, nil
}

// StreamChat sends a chat request to the Ollama API and returns a channel for streaming responses
func (c *clientImpl) StreamChat(ctx context.Context, request ChatRequest) (<-chan StreamChatResponse, error) {
	// Create a copy of the request with stream explicitly set to true
	requestCopy := request
	requestCopy.Stream = true

	reqBody, err := json.Marshal(requestCopy)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/chat", c.baseURL), bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.applyAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("unsuccessful response: %d %s", resp.StatusCode, string(bodyBytes))
	}

	// Buffer the channel to prevent goroutine leak if consumer stops reading
	responseChannel := make(chan StreamChatResponse, 10)

	go func() {
		defer close(responseChannel)
		defer resp.Body.Close()

		decoder := json.NewDecoder(resp.Body)
		for {
			var response StreamChatResponse
			if err := decoder.Decode(&response); err != nil {
				if err != io.EOF {
					response.Error = fmt.Errorf("failed to decode response: %w", err)
					responseChannel <- response
				}
				break
			}

			select {
			case <-ctx.Done():
				return
			case responseChannel <- response:
			}

			if response.Done {
				break
			}
		}
	}()

	return responseChannel, nil
}

// ListModels retrieves the list of available models from the Ollama API
func (c *clientImpl) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("%s/tags", c.baseURL), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	c.applyAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unsuccessful response: %d %s", resp.StatusCode, string(bodyBytes))
	}

	var response ListModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// applyAuth adds authentication headers to the request
func (c *clientImpl) applyAuth(req *http.Request) {
	// Apply custom headers first
	for key, value := range c.headers {
		req.Header.Set(key, value)
	}

	// Apply authentication based on type
	switch c.authType {
	case "bearer":
		if c.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}
	case "basic":
		if c.username != "" && c.password != "" {
			req.SetBasicAuth(c.username, c.password)
		}
	case "custom":
		// Custom auth is handled by headers
	}
}
