package multiagent

import (
	"context"
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
func (m *MockAgent) SetMaxTokens(maxTokens int)             {}
func (m *MockAgent) SetSessionRecorder(recorder *session.Recorder) {}
func (m *MockAgent) SetAutoRoute(enabled bool)              {}
func (m *MockAgent) ContextStats() (int, int, int)           { return 0, 0, 0 }
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
