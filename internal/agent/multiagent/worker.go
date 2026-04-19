package multiagent

import (
	"context"
	"fmt"
	"strings"
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

	// Clear the parent's conversation history from the cloned agent.
	// Parallel workers are independent — they start fresh with only the
	// system prompt. Keeping the parent's tool call / tool result history
	// causes strict providers (Mistral, ministral, etc.) to reject the
	// request with: "Not the same number of function calls and responses".
	clonedAgent.ClearContext()

	// Only retain tools that are strictly read-only and thread-safe.
	// This prevents concurrent file writes and garbled UI spinner outputs from
	// side-effecting tools like multiReplace, runCommand, etc.
	safeTools := map[string]bool{
		"viewFile":         true,
		"listFiles":        true,
		"grepSearch":       true,
		"readURL":          true,
		"webSearch":        true,
		"repoMapGenerator": true,
	}
	clonedAgent.FilterTools(func(toolName string) bool {
		return safeTools[toolName]
	})

	// Route each role to the most appropriate model tier:
	//   Researcher → fast model (quick lookups, scanning)
	//   Developer  → heavy model (complex reasoning, code generation)
	//   Reviewer   → heavy model (deep analysis)
	//   Generic    → default model
	// Falls back to default if the tier model isn't configured.
	switch role {
	case RoleResearcher:
		if m := clonedAgent.GetModelForTier(agent.TierFast); m != "" {
			clonedAgent.SetModel(m)
		}
	case RoleDeveloper, RoleReviewer:
		if m := clonedAgent.GetModelForTier(agent.TierHeavy); m != "" {
			clonedAgent.SetModel(m)
		}
	}

	// Prepend specialized identity to the system prompt based on role.
	// NOTE: All parallel workers are read-only (no file-write tools), so
	// instructions must reflect that constraint regardless of role.
	roleInstruction := map[WorkerRole]string{
		RoleGeneric:    "You are a general-purpose assistant. Fulfill the user's request.",
		RoleResearcher: "You are an expert repository researcher. You have read-only access to the filesystem. Focus on reading, searching, and understanding context. Share your findings concisely.",
		RoleDeveloper:  "You are an expert developer performing analysis. You have read-only access to the filesystem — do NOT attempt to write or edit files. Analyze the code, reason about changes needed, and produce a detailed written plan or code diff that a downstream agent can apply.",
		RoleReviewer:   "You are an expert code reviewer with read-only access to the filesystem. Analyze the code for logic errors, race conditions, edge cases, and style issues.",
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

	prompt := buildTaskPrompt(task)

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
				"label":     task.Description,
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

// buildTaskPrompt constructs a clean, LLM-readable prompt for a worker task.
// Dependency outputs from prerequisite tasks are formatted as clearly labeled
// sections rather than dumped as raw Go map output.
func buildTaskPrompt(task Task) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("## Assigned Task: %s\n\n", task.ID))
	sb.WriteString(fmt.Sprintf("**Description:** %s\n", task.Description))

	// Inject dependency outputs as clearly labeled sections
	if deps, ok := task.Inputs["dependency_outputs"].(map[string]string); ok && len(deps) > 0 {
		sb.WriteString("\n---\n## Context from prerequisite tasks:\n")
		for depID, output := range deps {
			sb.WriteString(fmt.Sprintf("\n### Output from Task `%s`:\n", depID))
			sb.WriteString(output)
			sb.WriteString("\n")
		}
		sb.WriteString("---\n")
	}

	sb.WriteString("\nComplete the task described above.")
	return sb.String()
}
