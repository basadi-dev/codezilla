package multiagent

import (
	"context"
	"time"
)

// Role defining the agent's persona and system prompt bias
type WorkerRole string

const (
	RoleGeneric    WorkerRole = "Generic"
	RoleResearcher WorkerRole = "Researcher"
	RoleDeveloper  WorkerRole = "Developer"
	RoleReviewer   WorkerRole = "Reviewer"
)

// Task represents a discrete unit of work assigned to a worker
type Task struct {
	ID          string                 // Unique identifier for the task
	Description string                 // What needs to be done
	Inputs      map[string]interface{} // Any initial context or variables
	Deadline    time.Time              // Optional deadline for context cancellation
}

// Result represents the outcome of a worker's task execution
type Result struct {
	TaskID    string        // The ID of the task this result belongs to
	WorkerID  string        // Which worker completed the task
	Output    string        // The LLM response / conclusion
	Error     error         // Any error encountered during setup/execution
	Duration  time.Duration // Time taken to complete the task
	TokenUsed int           // Number of tokens consumed
}

// Worker encapsulates a single active agent executing tasks
type Worker interface {
	// ID returns the worker string identifier
	ID() string
	// Role returns what specialization this agent uses
	Role() WorkerRole
	// Start begins the worker listening on the tasks channel
	Start(ctx context.Context, tasks <-chan Task, results chan<- Result)
}
