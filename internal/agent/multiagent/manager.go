package multiagent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"codezilla/internal/agent"
	"codezilla/pkg/logger"
)

const defaultMaxConcurrency = 3

// workerCounter provides globally unique, collision-free worker IDs
var workerCounter uint64

// Orchestrator manages the dispatch and lifecycle of parallel workers
type Orchestrator struct {
	bus            *MemoryBus
	baseAgent      agent.Agent
	logger         *logger.Logger
	maxConcurrency int
}

func NewOrchestrator(baseAgent agent.Agent, log *logger.Logger) *Orchestrator {
	if log == nil {
		log = logger.DefaultLogger()
	}
	return &Orchestrator{
		bus:            NewMemoryBus(),
		baseAgent:      baseAgent,
		logger:         log,
		maxConcurrency: defaultMaxConcurrency,
	}
}

// SetMaxConcurrency configures how many workers can run LLM calls simultaneously.
// Values <= 0 are ignored.
func (o *Orchestrator) SetMaxConcurrency(n int) {
	if n > 0 {
		o.maxConcurrency = n
	}
}

// Bus returns the orchestrator's shared MemoryBus for external event subscription
func (o *Orchestrator) Bus() *MemoryBus {
	return o.bus
}

// ExecuteParallel maps discrete tasks to internal workers and returns when all are complete.
// At most MaxConcurrency workers will be executing LLM calls at any given time.
func (o *Orchestrator) ExecuteParallel(ctx context.Context, tasks []Task) ([]Result, error) {
	if len(tasks) == 0 {
		return nil, nil
	}

	o.logger.Info("Starting parallel multi-agent execution",
		"task_count", len(tasks),
		"max_concurrency", o.maxConcurrency)

	resultChan := make(chan Result, len(tasks))
	sem := make(chan struct{}, o.maxConcurrency) // semaphore

	var wg sync.WaitGroup

	// Each worker gets its own dedicated single-task channel.
	// This eliminates the task-stealing race where one fast worker
	// could grab multiple tasks from a shared channel.
	for _, t := range tasks {
		// Extract role hint if provided, fallback to Generic
		role := RoleGeneric
		if t.Inputs != nil {
			if r, ok := t.Inputs["role_hint"].(string); ok {
				role = WorkerRole(r)
			}
		}

		seq := atomic.AddUint64(&workerCounter, 1)
		workerID := fmt.Sprintf("worker-%s-%d", role, seq)

		taskChan := make(chan Task, 1)
		taskChan <- t
		close(taskChan)

		worker := NewWorker(workerID, role, o.baseAgent, o.bus)

		wg.Add(1)
		go func(w *ConcreteWorker, ch <-chan Task) {
			defer wg.Done()

			// Acquire semaphore slot — blocks if MaxConcurrency workers are active
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				return
			}

			w.Start(ctx, ch, resultChan)
		}(worker, taskChan)
	}

	// Close the results channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	var results []Result
	for res := range resultChan {
		results = append(results, res)
	}

	o.logger.Info("Completed parallel multi-agent execution", "result_count", len(results))
	return results, nil
}
