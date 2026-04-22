package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"codezilla/internal/tools"
	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// ──────────────────────────────────────────────────────────────────────────────
// MockLLMResponse is a single scripted response for the mock LLM.
// The orchestrator calls are sequential, so these are consumed in FIFO order.
// ──────────────────────────────────────────────────────────────────────────────

// MockLLMResponse defines a single scripted LLM response.
type MockLLMResponse struct {
	// Text content returned by the model. Mutually exclusive with Error.
	Content string

	// ToolCalls returned by the model (native format).
	ToolCalls []anyllm.ToolCall

	// If set, the LLM call returns this error instead of a response.
	Error error

	// Simulated token usage (optional).
	Usage *anyllm.Usage

	// Reasoning content (for think-block models).
	Reasoning string
}

// ──────────────────────────────────────────────────────────────────────────────
// mockLLMProvider implements anyllm.Provider for testing.
// ──────────────────────────────────────────────────────────────────────────────

type mockLLMProvider struct {
	mu        sync.Mutex
	responses []MockLLMResponse
	callIdx   int
	calls     []mockLLMCall // records all calls for assertions
}

type mockLLMCall struct {
	Model    string
	Messages []anyllm.Message
	Tools    []anyllm.Tool
	Time     time.Time
}

func (m *mockLLMProvider) next() (MockLLMResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.callIdx >= len(m.responses) {
		return MockLLMResponse{}, fmt.Errorf("mock LLM: no more scripted responses (call #%d)", m.callIdx+1)
	}
	resp := m.responses[m.callIdx]
	m.callIdx++
	return resp, nil
}

func (m *mockLLMProvider) recordCall(model string, msgs []anyllm.Message, tools []anyllm.Tool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = append(m.calls, mockLLMCall{
		Model:    model,
		Messages: msgs,
		Tools:    tools,
		Time:     time.Now(),
	})
}

// Calls returns a snapshot of all LLM calls made.
func (m *mockLLMProvider) Calls() []mockLLMCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]mockLLMCall, len(m.calls))
	copy(cp, m.calls)
	return cp
}

// CallCount returns the number of LLM calls made.
func (m *mockLLMProvider) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.calls)
}

// Completion implements anyllm.Provider
func (m *mockLLMProvider) Completion(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	m.recordCall(params.Model, params.Messages, params.Tools)

	resp, err := m.next()
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}

	msg := anyllm.Message{
		Role:    "assistant",
		Content: resp.Content,
	}

	if resp.Reasoning != "" {
		msg.Reasoning = &anyllm.Reasoning{Content: resp.Reasoning}
	}

	if len(resp.ToolCalls) > 0 {
		msg.ToolCalls = resp.ToolCalls
	}

	return &anyllm.ChatCompletion{
		Choices: []anyllm.Choice{
			{Message: msg},
		},
		Usage: resp.Usage,
	}, nil
}

// CompletionStream implements anyllm.Provider
func (m *mockLLMProvider) CompletionStream(ctx context.Context, params anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	m.recordCall(params.Model, params.Messages, params.Tools)

	chunkCh := make(chan anyllm.ChatCompletionChunk, 100)
	errCh := make(chan error, 1)

	resp, err := m.next()
	if err != nil {
		errCh <- err
		close(chunkCh)
		close(errCh)
		return chunkCh, errCh
	}

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		if resp.Error != nil {
			errCh <- resp.Error
			return
		}

		// Emit reasoning tokens if present
		if resp.Reasoning != "" {
			chunkCh <- anyllm.ChatCompletionChunk{
				Choices: []anyllm.ChunkChoice{
					{
						Delta: anyllm.ChunkDelta{
							Reasoning: &anyllm.Reasoning{Content: resp.Reasoning},
						},
					},
				},
			}
		}

		// Emit content tokens (split into chunks for realism)
		if resp.Content != "" {
			words := splitIntoChunks(resp.Content, 10)
			for _, w := range words {
				chunkCh <- anyllm.ChatCompletionChunk{
					Choices: []anyllm.ChunkChoice{
						{
							Delta: anyllm.ChunkDelta{
								Content: w,
							},
						},
					},
				}
			}
		}

		// Emit tool calls
		if len(resp.ToolCalls) > 0 {
			for _, tc := range resp.ToolCalls {
				chunkCh <- anyllm.ChatCompletionChunk{
					Choices: []anyllm.ChunkChoice{
						{
							Delta: anyllm.ChunkDelta{
								ToolCalls: []anyllm.ToolCall{tc},
							},
						},
					},
				}
			}
		}

		// Emit usage in the final chunk
		if resp.Usage != nil {
			chunkCh <- anyllm.ChatCompletionChunk{
				Usage: resp.Usage,
			}
		}
	}()

	return chunkCh, errCh
}

// ──────────────────────────────────────────────────────────────────────────────
// TestHarness wires up a fully testable agent+orchestrator with mock LLM.
// ──────────────────────────────────────────────────────────────────────────────

// TestHarness provides a fully configured agent for unit testing.
type TestHarness struct {
	Agent     Agent
	Provider  *mockLLMProvider
	Config    *Config
	Callbacks *TestCallbackTracker

	// internals for direct access in tests
	rawAgent *agent
}

// TestCallbackTracker records all callback invocations for assertions.
type TestCallbackTracker struct {
	mu sync.Mutex

	ToolExecutions []ToolExecEvent
	ToolPreparings []string
	LLMCalls       []LLMCallEvent
	LLMErrors      []LLMErrorEvent
	ModelRoutings  []ModelRoutedEvent
	VerifyFailures []VerifyFailEvent
	VerifyPassed   int
	StreamTokens   []string
	UsageUpdates   []UsageEvent
	Summarizations int
}

// ToolExecEvent records a tool execution callback.
type ToolExecEvent struct {
	ToolName string
	Params   map[string]interface{}
}

// LLMCallEvent records an OnLLMCall callback.
type LLMCallEvent struct {
	CallNum    int
	MsgCount   int
	ApproxToks int
}

// LLMErrorEvent records an OnLLMError callback.
type LLMErrorEvent struct {
	Model     string
	Error     error
	WillRetry bool
}

// ModelRoutedEvent records an OnModelRouted callback.
type ModelRoutedEvent struct {
	Model  string
	Reason string
}

// VerifyFailEvent records an OnVerifyFailed callback.
type VerifyFailEvent struct {
	Errors   []string
	RetryNum int
}

// UsageEvent records an OnLLMUsage callback.
type UsageEvent struct {
	Turn       TokenUsage
	Session    TokenUsage
	TurnModels map[string]TokenUsage
}

func (t *TestCallbackTracker) recordToolExec(name string, params map[string]interface{}) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ToolExecutions = append(t.ToolExecutions, ToolExecEvent{name, params})
}

func (t *TestCallbackTracker) recordToolPreparing(name string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ToolPreparings = append(t.ToolPreparings, name)
}

func (t *TestCallbackTracker) recordLLMCall(callNum, msgCount, approxToks int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.LLMCalls = append(t.LLMCalls, LLMCallEvent{callNum, msgCount, approxToks})
}

func (t *TestCallbackTracker) recordLLMError(model string, err error, willRetry bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.LLMErrors = append(t.LLMErrors, LLMErrorEvent{model, err, willRetry})
}

func (t *TestCallbackTracker) recordModelRouted(model, reason string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ModelRoutings = append(t.ModelRoutings, ModelRoutedEvent{model, reason})
}

func (t *TestCallbackTracker) recordVerifyFailed(errors []string, retryNum int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.VerifyFailures = append(t.VerifyFailures, VerifyFailEvent{errors, retryNum})
}

func (t *TestCallbackTracker) recordVerifyPassed() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.VerifyPassed++
}

func (t *TestCallbackTracker) recordUsage(turn TokenUsage, session TokenUsage, turnModels map[string]TokenUsage) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.UsageUpdates = append(t.UsageUpdates, UsageEvent{turn, session, turnModels})
}

// ──────────────────────────────────────────────────────────────────────────────
// NewTestHarness creates a test harness with the given scripted responses.
// ──────────────────────────────────────────────────────────────────────────────

// TestHarnessOption configures a test harness.
type TestHarnessOption func(*TestHarness)

// WithMaxIterations sets the max iteration limit.
func WithMaxIterations(n int) TestHarnessOption {
	return func(h *TestHarness) {
		h.Config.MaxIterations = n
	}
}

// WithAutoRoute enables model routing with the given tier models.
func WithAutoRoute(fast, default_, heavy string) TestHarnessOption {
	return func(h *TestHarness) {
		h.Config.AutoRoute = true
		h.Config.FastModel = fast
		h.Config.HeavyModel = heavy
		h.Config.Model = default_
	}
}

// WithLoopDetection configures loop detection parameters.
func WithLoopDetection(window, maxRepeat int) TestHarnessOption {
	return func(h *TestHarness) {
		h.Config.LoopDetectWindow = window
		h.Config.LoopDetectMaxRepeat = maxRepeat
	}
}

// WithSystemPrompt sets a custom system prompt.
func WithSystemPrompt(prompt string) TestHarnessOption {
	return func(h *TestHarness) {
		h.Config.SystemPrompt = prompt
	}
}

// NewTestHarness creates a fully wired test harness.
func NewTestHarness(responses []MockLLMResponse, opts ...TestHarnessOption) *TestHarness {
	provider := &mockLLMProvider{responses: responses}
	callbacks := &TestCallbackTracker{}

	log, _ := logger.New(logger.Config{Silent: true})

	cfg := &Config{
		Model:                "test-model",
		Provider:             "mock",
		MaxTokens:            128000,
		MaxIterations:        50,
		Temperature:          0.0,
		Logger:               log,
		LoopDetectWindow:     10,
		LoopDetectMaxRepeat:  3,
		OnToolExecution:      callbacks.recordToolExec,
		OnToolPreparing:      callbacks.recordToolPreparing,
		OnLLMCall:            callbacks.recordLLMCall,
		OnLLMError:           callbacks.recordLLMError,
		OnModelRouted:        callbacks.recordModelRouted,
		OnVerifyFailed:       callbacks.recordVerifyFailed,
		OnVerifyPassed:       callbacks.recordVerifyPassed,
		OnLLMUsage:           callbacks.recordUsage,
		OnContextSummarizing: func() { callbacks.mu.Lock(); callbacks.Summarizations++; callbacks.mu.Unlock() },
	}

	h := &TestHarness{
		Provider:  provider,
		Config:    cfg,
		Callbacks: callbacks,
	}

	// Apply options
	for _, opt := range opts {
		opt(h)
	}

	// Build the agent without a real LLM client
	permMgr := tools.NewPermissionManager(func(ctx context.Context, request tools.PermissionRequest) (tools.PermissionResponse, error) {
		return tools.PermissionResponse{Granted: true}, nil
	})

	a := &agent{
		config:        cfg,
		context:       NewContext(cfg.MaxTokens, log),
		llmClient:     nil,
		toolRegistry:  nil,
		logger:        log,
		permissionMgr: permMgr,
	}

	if cfg.AutoRoute && (cfg.FastModel != "" || cfg.HeavyModel != "") {
		a.router = NewModelRouter(true, cfg.FastModel, cfg.Model, cfg.HeavyModel)
	}

	if cfg.SystemPrompt != "" {
		a.AddSystemMessage(cfg.SystemPrompt)
	}

	h.Agent = a
	h.rawAgent = a

	return h
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

// splitIntoChunks splits a string into chunks of approximately chunkSize bytes.
func splitIntoChunks(s string, chunkSize int) []string {
	if chunkSize <= 0 {
		return []string{s}
	}
	var chunks []string
	for len(s) > 0 {
		end := chunkSize
		if end > len(s) {
			end = len(s)
		}
		chunks = append(chunks, s[:end])
		s = s[end:]
	}
	return chunks
}

// MakeToolCall creates a properly formatted anyllm.ToolCall for test fixtures.
func MakeToolCall(id, name string, args map[string]interface{}) anyllm.ToolCall {
	argsJSON, _ := json.Marshal(args)
	return anyllm.ToolCall{
		ID:   id,
		Type: "function",
		Function: providers.FunctionCall{
			Name:      name,
			Arguments: string(argsJSON),
		},
	}
}

// MakeUsage creates a simulated anyllm.Usage for test fixtures.
func MakeUsage(prompt, completion, total int) *anyllm.Usage {
	return &anyllm.Usage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      total,
	}
}
