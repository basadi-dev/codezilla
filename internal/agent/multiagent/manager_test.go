package multiagent

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"codezilla/internal/agent"
	"codezilla/internal/session"
)

// MockAgent is a minimal stub for testing orchestrator logic.
// Clone returns a new independent MockAgent. ProcessCount uses atomic
// operations so the race detector stays happy when multiple goroutines call Clone.
type MockAgent struct {
	processCount int64
}

func (m *MockAgent) ProcessMessage(ctx context.Context, message string) (string, error) {
	atomic.AddInt64(&m.processCount, 1)
	// Simulate some work
	time.Sleep(50 * time.Millisecond)
	return "Mocked Response", nil
}

func (m *MockAgent) ProcessMessageStream(ctx context.Context, message string, onToken func(string), onStreamEnd func()) (string, error) {
	return "", nil
}

func (m *MockAgent) ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error) {
	return nil, nil
}

func (m *MockAgent) AddSystemMessage(message string)       {}
func (m *MockAgent) ReplaceSystemMessage(message string)    {}
func (m *MockAgent) AddUserMessage(message string)          {}
func (m *MockAgent) AddAssistantMessage(message string)     {}
func (m *MockAgent) AddMessage(msg agent.Message)           {}
func (m *MockAgent) GetMessages() []agent.Message           { return nil }
func (m *MockAgent) ClearContext()                          {}
func (m *MockAgent) ClearLastUserMessage()                  {}
func (m *MockAgent) SetModel(model string)                  {}
func (m *MockAgent) SetFastModel(model string)              {}
func (m *MockAgent) SetHeavyModel(model string)             {}
func (m *MockAgent) GetModelForTier(tier agent.RequestTier) string { return "" }
func (m *MockAgent) SetTemperature(temperature float64)     {}
func (m *MockAgent) SetReasoningEffort(effort string)        {}
func (m *MockAgent) SetMaxTokens(maxTokens int)             {}
func (m *MockAgent) SetSessionRecorder(recorder *session.Recorder) {}
func (m *MockAgent) SetAutoRoute(enabled bool)              {}
func (m *MockAgent) ClearTools()                              {}
func (m *MockAgent) FilterTools(predicate func(string) bool)  {}
func (m *MockAgent) ContextStats() (msgCount int, currentTokens int, maxTokens int) {
	return 0, 0, 0
}
func (m *MockAgent) Clone() agent.Agent                     { return &MockAgent{} }

func TestOrchestrator_ExecuteParallel(t *testing.T) {
	mockAgent := &MockAgent{}
	orchestrator := NewOrchestrator(mockAgent, nil)

	tasks := []Task{
		{ID: "t1", Description: "Test Task 1"},
		{ID: "t2", Description: "Test Task 2"},
		{ID: "t3", Description: "Test Task 3"},
	}

	start := time.Now()
	results, err := orchestrator.ExecuteParallel(context.Background(), tasks)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	// 3 tasks of 50ms each running in parallel should complete well under
	// 3× sequential time (150ms). We use a generous 120ms bound to avoid
	// flakiness on loaded CI machines while still catching serial execution.
	if duration > 120*time.Millisecond {
		t.Errorf("Execution took %v (>120ms), tasks may not be running in parallel", duration)
	}
}

func TestOrchestrator_EmptyTasks(t *testing.T) {
	orchestrator := NewOrchestrator(&MockAgent{}, nil)

	results, err := orchestrator.ExecuteParallel(context.Background(), nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if results != nil {
		t.Errorf("Expected nil results for empty tasks, got %v", results)
	}
}

func TestOrchestrator_ContextCancellation(t *testing.T) {
	orchestrator := NewOrchestrator(&MockAgent{}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	results, err := orchestrator.ExecuteParallel(ctx, []Task{
		{ID: "t1", Description: "Should be cancelled"},
	})

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// With immediate cancellation, we may get 0 or 1 results depending on
	// goroutine scheduling. The key assertion is that it doesn't hang.
	_ = results
}

func TestOrchestrator_DAGExecutionOrder(t *testing.T) {
	mockAgent := &MockAgent{}
	orchestrator := NewOrchestrator(mockAgent, nil)

	// DAG: A -> B, A -> C, B+C -> D
	tasks := []Task{
		{ID: "D", DependsOn: []string{"B", "C"}},
		{ID: "C", DependsOn: []string{"A"}},
		{ID: "A"},
		{ID: "B", DependsOn: []string{"A"}},
	}

	results, err := orchestrator.ExecuteParallel(context.Background(), tasks)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(results) != 4 {
		t.Fatalf("Expected 4 results, got %d", len(results))
	}

	// Because of our mock agent sleeping 50ms per task, we can verify sequence
	// by checking task duration roughly or just knowing D should definitely
	// be last, A should be first. Since results are appended upon completion:
	var completionOrder []string
	for _, res := range results {
		completionOrder = append(completionOrder, res.TaskID)
	}

	if len(completionOrder) > 0 && completionOrder[0] != "A" {
		t.Errorf("Task A must complete first, got %s", completionOrder[0])
	}
	if len(completionOrder) > 0 && completionOrder[len(completionOrder)-1] != "D" {
		t.Errorf("Task D must complete last, got %s", completionOrder[len(completionOrder)-1])
	}
}

func TestOrchestrator_DAGDeadlock(t *testing.T) {
	orchestrator := NewOrchestrator(&MockAgent{}, nil)

	// Circular DAG: A → B, B → A — upfront Kahn's algo catches this
	tasks := []Task{
		{ID: "A", DependsOn: []string{"B"}},
		{ID: "B", DependsOn: []string{"A"}},
	}

	_, err := orchestrator.ExecuteParallel(context.Background(), tasks)
	if err == nil {
		t.Fatal("Expected cycle error, got nil")
	}
}

func TestOrchestrator_UnknownDependency(t *testing.T) {
	orchestrator := NewOrchestrator(&MockAgent{}, nil)

	// Task B depends on "X" which doesn't exist
	tasks := []Task{
		{ID: "A"},
		{ID: "B", DependsOn: []string{"X"}},
	}

	_, err := orchestrator.ExecuteParallel(context.Background(), tasks)
	if err == nil {
		t.Fatal("Expected unknown-dependency error, got nil")
	}
}

func TestBuildTaskPrompt_NoDeps(t *testing.T) {
	task := Task{ID: "task-1", Description: "Analyze the repo structure"}
	prompt := buildTaskPrompt(task)

	if prompt == "" {
		t.Fatal("Expected non-empty prompt")
	}
	if !contains(prompt, "task-1") {
		t.Errorf("Prompt missing task ID")
	}
	if !contains(prompt, "Analyze the repo structure") {
		t.Errorf("Prompt missing task description")
	}
	// Should NOT contain any dependency section
	if contains(prompt, "prerequisite") {
		t.Errorf("Prompt unexpectedly contains dependency section")
	}
}

func TestBuildTaskPrompt_WithDeps(t *testing.T) {
	task := Task{
		ID:          "task-2",
		Description: "Write tests based on the analysis",
		Inputs: map[string]interface{}{
			"dependency_outputs": map[string]string{
				"task-1": "The repo has 3 packages: agent, tools, core.",
			},
		},
	}
	prompt := buildTaskPrompt(task)

	if !contains(prompt, "task-1") {
		t.Errorf("Prompt missing dependency task ID")
	}
	if !contains(prompt, "The repo has 3 packages") {
		t.Errorf("Prompt missing dependency output content")
	}
	if !contains(prompt, "prerequisite") {
		t.Errorf("Prompt missing prerequisite section header")
	}
	// Must NOT contain raw Go map formatting
	if contains(prompt, "map[") {
		t.Errorf("Prompt contains raw Go map output — formatting is broken")
	}
}

func contains(s, sub string) bool {
	return strings.Contains(s, sub)
}
