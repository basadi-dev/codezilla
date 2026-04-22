package agent

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ──────────────────────────────────────────────────────────────────────────────
// Interaction Event System
//
// Captures all orchestrator events in a typed, serializable format for:
//   - Debugging and replay
//   - Test assertions
//   - Session analysis
// ──────────────────────────────────────────────────────────────────────────────

// InteractionType classifies the kind of orchestrator event.
type InteractionType string

const (
	// Core loop events
	IxnUserMessage     InteractionType = "user_message"
	IxnLLMRequest      InteractionType = "llm_request"
	IxnLLMResponse     InteractionType = "llm_response"
	IxnLLMStreamToken  InteractionType = "llm_stream_token"
	IxnStateTransition InteractionType = "state_transition"
	IxnComplete        InteractionType = "complete"

	// Tool events
	IxnToolPreparing InteractionType = "tool_preparing"
	IxnToolExecution InteractionType = "tool_execution"
	IxnToolResult    InteractionType = "tool_result"

	// Error events
	IxnErrorRecovery InteractionType = "error_recovery"
	IxnLLMError      InteractionType = "llm_error"

	// Routing events
	IxnModelRouted InteractionType = "model_routed"

	// Reasoning events
	IxnThinkBlock InteractionType = "think_block"

	// Verification events
	IxnVerifyPass InteractionType = "verify_pass"
	IxnVerifyFail InteractionType = "verify_fail"

	// Safety events
	IxnLoopDetected    InteractionType = "loop_detected"
	IxnMaxIterations   InteractionType = "max_iterations"
	IxnContextTrimmed  InteractionType = "context_trimmed"
	IxnContextSummary  InteractionType = "context_summary"

	// Auto-correction events
	IxnLeakedToolCall  InteractionType = "leaked_tool_call"
	IxnThinkOnlyReprompt InteractionType = "think_only_reprompt"
)

// InteractionEvent represents a single orchestrator event.
type InteractionEvent struct {
	Type      InteractionType        `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// String returns a human-readable summary of the event.
func (e InteractionEvent) String() string {
	var parts []string
	parts = append(parts, fmt.Sprintf("[%s]", e.Type))
	for k, v := range e.Data {
		parts = append(parts, fmt.Sprintf("%s=%v", k, v))
	}
	return strings.Join(parts, " ")
}

// InteractionRecorder captures all events during an orchestrator run.
// Thread-safe for concurrent tool execution.
type InteractionRecorder struct {
	mu     sync.Mutex
	events []InteractionEvent
}

// NewInteractionRecorder creates a new empty recorder.
func NewInteractionRecorder() *InteractionRecorder {
	return &InteractionRecorder{}
}

// Record adds an event to the log.
func (r *InteractionRecorder) Record(eventType InteractionType, data map[string]interface{}) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, InteractionEvent{
		Type:      eventType,
		Timestamp: time.Now(),
		Data:      data,
	})
}

// Events returns a snapshot of all recorded events.
func (r *InteractionRecorder) Events() []InteractionEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	cp := make([]InteractionEvent, len(r.events))
	copy(cp, r.events)
	return cp
}

// EventCount returns the number of recorded events.
func (r *InteractionRecorder) EventCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.events)
}

// EventsOfType returns all events matching the given type.
func (r *InteractionRecorder) EventsOfType(t InteractionType) []InteractionEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var filtered []InteractionEvent
	for _, e := range r.events {
		if e.Type == t {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

// HasEvent returns true if any event of the given type was recorded.
func (r *InteractionRecorder) HasEvent(t InteractionType) bool {
	return len(r.EventsOfType(t)) > 0
}

// EventSequence returns the ordered sequence of event types.
// Useful for asserting the exact state machine transitions.
func (r *InteractionRecorder) EventSequence() []InteractionType {
	r.mu.Lock()
	defer r.mu.Unlock()
	types := make([]InteractionType, len(r.events))
	for i, e := range r.events {
		types[i] = e.Type
	}
	return types
}

// Clear removes all recorded events.
func (r *InteractionRecorder) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Scenario represents a reproducible interaction test case.
// ──────────────────────────────────────────────────────────────────────────────

// Scenario defines a reproducible agent interaction for testing.
type Scenario struct {
	// Name is a human-readable identifier for the scenario.
	Name string

	// Category groups related scenarios (e.g., "core_loop", "error_recovery").
	Category string

	// Description explains what this scenario tests.
	Description string

	// UserMessage is the input that triggers the scenario.
	UserMessage string

	// ExpectedStates is the ordered list of orchestrator states that should be visited.
	ExpectedStates []OrchestratorState

	// ExpectedEvents is the ordered list of interaction event types expected.
	ExpectedEvents []InteractionType

	// Assertions is a set of named conditions that must hold after the scenario completes.
	Assertions map[string]func(recorder *InteractionRecorder) error
}

// ──────────────────────────────────────────────────────────────────────────────
// Built-in Scenario Catalog
// ──────────────────────────────────────────────────────────────────────────────

// AllScenarios returns the complete catalog of agent interaction scenarios.
// Each scenario documents a distinct interaction pattern with its expected
// state transitions and event sequence.
func AllScenarios() []Scenario {
	return []Scenario{
		// ── A. Core Loop ──────────────────────────────────────────────
		{
			Name:        "A1_SimpleTextResponse",
			Category:    "core_loop",
			Description: "User asks a question, LLM responds with plain text, no tools involved",
			UserMessage: "What is Go?",
			ExpectedStates: []OrchestratorState{
				StatePrompting, StateStreaming, StateParsing, StateComplete,
			},
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "A2_SingleToolCall",
			Category:    "core_loop",
			Description: "LLM emits a single native tool call, tool executes, LLM summarizes the result",
			UserMessage: "Read the file /etc/hosts",
			ExpectedStates: []OrchestratorState{
				StateStreaming, StateExecutingTools, StatePrompting, StateComplete,
			},
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnToolPreparing, IxnToolExecution,
				IxnToolResult, IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "A3_MultipleToolCallsParallel",
			Category:    "core_loop",
			Description: "LLM emits 2+ tool calls in one turn, all execute in parallel",
			UserMessage: "Read both /etc/hosts and /etc/resolv.conf",
			ExpectedStates: []OrchestratorState{
				StateStreaming, StateExecutingTools, StatePrompting, StateComplete,
			},
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest,
				IxnToolPreparing, IxnToolExecution, IxnToolResult,
				IxnToolPreparing, IxnToolExecution, IxnToolResult,
				IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "A4_ChainedToolCalls",
			Category:    "core_loop",
			Description: "LLM calls tool, reads result, calls another tool, then responds",
			UserMessage: "Find and fix the bug in main.go",
			ExpectedStates: []OrchestratorState{
				StateStreaming, StateExecutingTools,
				StatePrompting, StateExecutingTools,
				StatePrompting, StateComplete,
			},
		},
		{
			Name:        "A5_LeakedToolCallAutoCorrection",
			Category:    "core_loop",
			Description: "LLM outputs tool call as raw JSON text, parser detects and auto-corrects",
			UserMessage: "List the project files",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnLeakedToolCall,
				IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "A6_ThinkOnlyReprompt",
			Category:    "core_loop",
			Description: "LLM produces only <think> block with no visible output, gets re-prompted",
			UserMessage: "Explain the architecture",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnThinkBlock, IxnThinkOnlyReprompt,
				IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},

		// ── B. Error Recovery ─────────────────────────────────────────
		{
			Name:        "B1_ContextLengthRecoverable",
			Category:    "error_recovery",
			Description: "Context too large, auto-trim and retry successfully",
			UserMessage: "Continue working on the task",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnLLMError,
				IxnContextTrimmed, IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "B2_ContextLengthFatal",
			Category:    "error_recovery",
			Description: "Context overflow exhausts all trim retries",
			UserMessage: "Continue",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnLLMError,
				IxnContextTrimmed, IxnLLMRequest, IxnLLMError,
				IxnContextTrimmed, IxnLLMRequest, IxnLLMError,
			},
		},
		{
			Name:        "B3_ToolMismatchDisableRetry",
			Category:    "error_recovery",
			Description: "Tool-call 400 error disables tools and retries as plain text agent",
			UserMessage: "Search for errors",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnLLMError,
				IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "B4_TransientErrorBackoff",
			Category:    "error_recovery",
			Description: "Transient 503/timeout error triggers exponential backoff retry",
			UserMessage: "Hello",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnLLMError,
				IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "B5_UnrecoverableError",
			Category:    "error_recovery",
			Description: "Non-retriable error surfaces as fatal",
			UserMessage: "Do something",
		},
		{
			Name:        "B6_EmptyResponse",
			Category:    "error_recovery",
			Description: "LLM returns empty content, treated as error",
			UserMessage: "Tell me a joke",
		},

		// ── C. Streaming & UI ────────────────────────────────────────
		{
			Name:        "C1_ThinkBlockStreaming",
			Category:    "streaming",
			Description: "<think>...</think> blocks rendered separately from content",
			UserMessage: "Explain recursion",
		},
		{
			Name:        "C2_StreamInterruptedByTool",
			Category:    "streaming",
			Description: "Text streaming stops when tool call is detected",
			UserMessage: "Read main.go",
		},
		{
			Name:        "C3_ModelRouting",
			Category:    "streaming",
			Description: "Router selects fast/default/heavy model based on input",
			UserMessage: "hi",
		},
		{
			Name:        "C4_SpinnerLifecycle",
			Category:    "streaming",
			Description: "Spinner show/update/hide cycle during tool loop",
			UserMessage: "Fix the bug",
		},

		// ── D. Verification ──────────────────────────────────────────
		{
			Name:        "D1_VerifyPass",
			Category:    "verification",
			Description: "File-modifying tool triggers build/lint check that passes",
			UserMessage: "Fix the typo",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnToolExecution,
				IxnToolResult, IxnVerifyPass,
				IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},
		{
			Name:        "D2_VerifyFailSelfCorrect",
			Category:    "verification",
			Description: "Verification fails, error injected, agent retries and fixes",
			UserMessage: "Refactor the module",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnToolExecution,
				IxnToolResult, IxnVerifyFail,
				IxnLLMRequest, IxnToolExecution, IxnToolResult,
				IxnVerifyPass, IxnLLMRequest, IxnLLMResponse, IxnComplete,
			},
		},

		// ── E. Safety ────────────────────────────────────────────────
		{
			Name:        "E1_LoopDetection",
			Category:    "safety",
			Description: "Same tool+args called 3x consecutively triggers abort",
			UserMessage: "Keep reading",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnLLMRequest, IxnToolExecution, IxnToolResult,
				IxnLLMRequest, IxnToolExecution, IxnToolResult,
				IxnLLMRequest, IxnLoopDetected,
			},
		},
		{
			Name:        "E2_MaxIterations",
			Category:    "safety",
			Description: "Agent exceeds MaxIterations limit, aborts with message",
			UserMessage: "Do a complex task",
			ExpectedEvents: []InteractionType{
				IxnUserMessage, IxnMaxIterations,
			},
		},
	}
}
