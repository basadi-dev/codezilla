package multiagent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"codezilla/internal/agent"
	"codezilla/pkg/logger"
)

// Orchestrator manages the dispatch and lifecycle of parallel workers
type Orchestrator struct {
	bus       *MemoryBus
	baseAgent agent.Agent
	logger    *logger.Logger
}

func NewOrchestrator(baseAgent agent.Agent, log *logger.Logger) *Orchestrator {
	if log == nil {
		log = logger.DefaultLogger()
	}
	return &Orchestrator{
		bus:       NewMemoryBus(),
		baseAgent: baseAgent,
		logger:    log,
	}
}

// ExecuteParallel maps discrete tasks to internal workers and returns when all are complete
func (o *Orchestrator) ExecuteParallel(ctx context.Context, tasks []Task) ([]Result, error) {
	if len(tasks) == 0 {
		return nil, nil
	}

	o.logger.Info("Starting parallel multi-agent execution", "task_count", len(tasks))

	taskChan := make(chan Task, len(tasks))
	resultChan := make(chan Result, len(tasks))

	// Pre-fill the task queue
	for _, t := range tasks {
		taskChan <- t
	}
	close(taskChan)

	var wg sync.WaitGroup
	workers := make([]*ConcreteWorker, len(tasks))

	// Simple 1-to-1 mapping for tasks to workers for now.
	// In the future this could be a bounded worker pool.
	for i, t := range tasks {
		// Try to extract role hint if provided, fallback to Generic
		role := RoleGeneric
		if r, ok := t.Inputs["role_hint"].(string); ok {
			role = WorkerRole(r)
		}

		workerID := fmt.Sprintf("worker-%s-%d", role, time.Now().UnixNano())
		workers[i] = NewWorker(workerID, role, o.baseAgent)
		
		wg.Add(1)
		go func(w *ConcreteWorker) {
			defer wg.Done()
			w.Start(ctx, taskChan, resultChan)
		}(workers[i])
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
