package multiagent

import (
	"context"
	"fmt"
	"time"

	"codezilla/internal/agent"
)

// ConcreteWorker implements the Worker interface wrapping a codezilla Agent
type ConcreteWorker struct {
	id    string
	role  WorkerRole
	inner agent.Agent
	bus   *MemoryBus
}

// NewWorker initializes a new concurrent worker.
// The bus parameter provides access to cross-agent shared state and events.
func NewWorker(id string, role WorkerRole, baseAgent agent.Agent, bus *MemoryBus) *ConcreteWorker {
	clonedAgent := baseAgent.Clone()

	// Strip tools — parallel workers do pure LLM completions.
	// The shared tool registry contains closures over the single-threaded TUI
	// (ui.Print, ui.HideThinking, etc.) which would cause garbled output.
	clonedAgent.ClearTools()

	// Prepend specialized identity to the system prompt based on role
	roleInstruction := map[WorkerRole]string{
		RoleGeneric:    "You are a general-purpose assistant. Fulfill the user's request.",
		RoleResearcher: "You are an expert repository researcher. Avoid modifying files. Focus on reading, searching, and understanding context. Share your findings concisely.",
		RoleDeveloper:  "You are an expert developer. Your task is to modify code, write tests, and ensure structural integrity.",
		RoleReviewer:   "You are an expert code reviewer. Analyze the code for logic errors, race conditions, edge cases, and style issues.",
	}

	if instruction, ok := roleInstruction[role]; ok {
		clonedAgent.AddSystemMessage(instruction)
	}

	return &ConcreteWorker{
		id:    id,
		role:  role,
		inner: clonedAgent,
		bus:   bus,
	}
}

func (w *ConcreteWorker) ID() string {
	return w.id
}

func (w *ConcreteWorker) Role() WorkerRole {
	return w.role
}

// Start executes a single task and writes its result. Each worker receives
// exactly one task via its dedicated channel, eliminating task-stealing races.
func (w *ConcreteWorker) Start(ctx context.Context, tasks <-chan Task, results chan<- Result) {
	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-tasks:
			if !ok {
				return // channel closed
			}

			w.executeTask(ctx, task, results)
		}
	}
}

func (w *ConcreteWorker) executeTask(parentCtx context.Context, task Task, results chan<- Result) {
	start := time.Now()

	// Broadcast that this task has started
	if w.bus != nil {
		w.bus.Publish(Event{
			Type:     EventTaskStarted,
			WorkerID: w.id,
			Payload: map[string]interface{}{
				"task_id": task.ID,
				"role":    string(w.role),
				"label":   task.Description,
			},
		})
	}

	// Create task-specific context optionally bound by a deadline
	ctx := parentCtx
	var cancel context.CancelFunc
	if !task.Deadline.IsZero() {
		ctx, cancel = context.WithDeadline(parentCtx, task.Deadline)
		defer cancel()
	}

	prompt := fmt.Sprintf("[Assigned Task: %s]\n\nTask Description:\n%s\n\nInputs:\n%v", task.ID, task.Description, task.Inputs)

	output, err := w.inner.ProcessMessage(ctx, prompt)

	result := Result{
		TaskID:   task.ID,
		WorkerID: w.id,
		Output:   output,
		Error:    err,
		Duration: time.Since(start),
	}

	// Broadcast completion
	if w.bus != nil {
		w.bus.Publish(Event{
			Type:     EventTaskCompleted,
			WorkerID: w.id,
			Payload: map[string]interface{}{
				"task_id":   task.ID,
				"role":      string(w.role),
				"duration":  result.Duration.String(),
				"has_error": err != nil,
			},
		})
	}

	// Send non-blocking or respect context cancellation
	select {
	case <-parentCtx.Done():
		return
	case results <- result:
	}
}
