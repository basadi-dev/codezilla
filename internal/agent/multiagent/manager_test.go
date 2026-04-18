package multiagent

import (
	"context"
	"testing"
	"time"

	"codezilla/internal/agent"
	"codezilla/internal/session"
)

// MockAgent is a minimal stub for testing orchestrator logic
type MockAgent struct {
	ProcessCount int
}

func (m *MockAgent) ProcessMessage(ctx context.Context, message string) (string, error) {
	m.ProcessCount++
	// Simulate some work
	time.Sleep(10 * time.Millisecond)
	return "Mocked Response", nil
}

func (m *MockAgent) ProcessMessageStream(ctx context.Context, message string, onToken func(string), onStreamEnd func()) (string, error) {
	return "", nil
}

func (m *MockAgent) ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error) {
	return nil, nil
}

func (m *MockAgent) AddSystemMessage(message string) {}
func (m *MockAgent) ReplaceSystemMessage(message string) {}
func (m *MockAgent) AddUserMessage(message string) {}
func (m *MockAgent) AddAssistantMessage(message string) {}
func (m *MockAgent) AddMessage(msg agent.Message) {}
func (m *MockAgent) GetMessages() []agent.Message { return nil }
func (m *MockAgent) ClearContext() {}
func (m *MockAgent) ClearLastUserMessage() {}
func (m *MockAgent) SetModel(model string) {}
func (m *MockAgent) SetFastModel(model string) {}
func (m *MockAgent) SetHeavyModel(model string) {}
func (m *MockAgent) GetModelForTier(tier agent.RequestTier) string { return "" }
func (m *MockAgent) SetTemperature(temperature float64) {}
func (m *MockAgent) SetMaxTokens(maxTokens int) {}
func (m *MockAgent) SetSessionRecorder(recorder *session.Recorder) {}
func (m *MockAgent) SetAutoRoute(enabled bool) {}
func (m *MockAgent) Clone() agent.Agent { return &MockAgent{} }

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

	// Execution should be mostly parallel (3 tasks of 10ms each should take < 30ms total)
	if duration > 28*time.Millisecond {
		t.Errorf("Execution took too long (%v), possibly not running in parallel", duration)
	}
}
