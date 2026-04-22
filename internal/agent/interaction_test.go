package agent

import (
	"testing"
)

// ──────────────────────────────────────────────────────────────────────────────
// Interaction Event System Tests
// ──────────────────────────────────────────────────────────────────────────────

func TestInteractionRecorder_BasicOperations(t *testing.T) {
	r := NewInteractionRecorder()

	if r.EventCount() != 0 {
		t.Fatalf("new recorder should have 0 events, got %d", r.EventCount())
	}

	// Record some events
	r.Record(IxnUserMessage, map[string]interface{}{"text": "hello"})
	r.Record(IxnLLMRequest, map[string]interface{}{"model": "test", "messages": 3})
	r.Record(IxnLLMResponse, map[string]interface{}{"content": "world"})
	r.Record(IxnComplete, nil)

	if r.EventCount() != 4 {
		t.Fatalf("expected 4 events, got %d", r.EventCount())
	}

	// EventsOfType
	llmReqs := r.EventsOfType(IxnLLMRequest)
	if len(llmReqs) != 1 {
		t.Fatalf("expected 1 LLM request event, got %d", len(llmReqs))
	}
	if llmReqs[0].Data["model"] != "test" {
		t.Errorf("expected model=test, got %v", llmReqs[0].Data["model"])
	}

	// HasEvent
	if !r.HasEvent(IxnComplete) {
		t.Error("expected HasEvent(Complete) to be true")
	}
	if r.HasEvent(IxnLoopDetected) {
		t.Error("expected HasEvent(LoopDetected) to be false")
	}

	// EventSequence
	seq := r.EventSequence()
	expected := []InteractionType{IxnUserMessage, IxnLLMRequest, IxnLLMResponse, IxnComplete}
	if len(seq) != len(expected) {
		t.Fatalf("sequence length mismatch: got %d, want %d", len(seq), len(expected))
	}
	for i, s := range seq {
		if s != expected[i] {
			t.Errorf("sequence[%d] = %s, want %s", i, s, expected[i])
		}
	}
}

func TestInteractionRecorder_Clear(t *testing.T) {
	r := NewInteractionRecorder()
	r.Record(IxnUserMessage, nil)
	r.Record(IxnComplete, nil)
	r.Clear()

	if r.EventCount() != 0 {
		t.Fatalf("expected 0 events after Clear, got %d", r.EventCount())
	}
}

func TestInteractionRecorder_ThreadSafety(t *testing.T) {
	r := NewInteractionRecorder()
	done := make(chan struct{})

	// Write from multiple goroutines
	for i := 0; i < 10; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				r.Record(IxnLLMStreamToken, map[string]interface{}{"worker": id, "token": j})
			}
			done <- struct{}{}
		}(i)
	}

	for i := 0; i < 10; i++ {
		<-done
	}

	if r.EventCount() != 1000 {
		t.Errorf("expected 1000 events, got %d", r.EventCount())
	}
}

func TestInteractionEvent_String(t *testing.T) {
	e := InteractionEvent{
		Type: IxnToolExecution,
		Data: map[string]interface{}{"tool": "fileRead"},
	}
	s := e.String()
	if s == "" {
		t.Error("String() should not be empty")
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Scenario Catalog Tests
// ──────────────────────────────────────────────────────────────────────────────

func TestAllScenarios_CatalogComplete(t *testing.T) {
	scenarios := AllScenarios()

	if len(scenarios) < 18 {
		t.Errorf("expected at least 18 scenarios, got %d", len(scenarios))
	}

	// Verify all categories are represented
	categories := make(map[string]int)
	for _, s := range scenarios {
		categories[s.Category]++
	}

	expectedCategories := []string{"core_loop", "error_recovery", "streaming", "verification", "safety"}
	for _, cat := range expectedCategories {
		if categories[cat] == 0 {
			t.Errorf("missing scenarios for category %q", cat)
		}
	}

	// Verify no duplicate names
	names := make(map[string]bool)
	for _, s := range scenarios {
		if names[s.Name] {
			t.Errorf("duplicate scenario name: %s", s.Name)
		}
		names[s.Name] = true
	}

	// Verify all scenarios have required fields
	for _, s := range scenarios {
		if s.Name == "" {
			t.Error("scenario missing Name")
		}
		if s.Category == "" {
			t.Errorf("scenario %s missing Category", s.Name)
		}
		if s.Description == "" {
			t.Errorf("scenario %s missing Description", s.Name)
		}
		if s.UserMessage == "" {
			t.Errorf("scenario %s missing UserMessage", s.Name)
		}
	}
}

func TestAllScenarios_CategoryBreakdown(t *testing.T) {
	scenarios := AllScenarios()

	categories := make(map[string][]string)
	for _, s := range scenarios {
		categories[s.Category] = append(categories[s.Category], s.Name)
	}

	// Log the breakdown for documentation
	for cat, names := range categories {
		t.Logf("Category %q: %d scenarios", cat, len(names))
		for _, n := range names {
			t.Logf("  - %s", n)
		}
	}
}
