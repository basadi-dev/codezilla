package multiagent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

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

	o.logger.Info("Starting DAG parallel execution", "task_count", len(tasks), "max_concurrency", o.maxConcurrency)

	tasksByID := make(map[string]Task, len(tasks))
	inDegree := make(map[string]int, len(tasks))
	adjList := make(map[string][]string, len(tasks))

	for _, t := range tasks {
		tasksByID[t.ID] = t
		inDegree[t.ID] = len(t.DependsOn)
	}

	// ── Step 1: Validate that all DependsOn IDs actually exist ──
	for _, t := range tasks {
		for _, dep := range t.DependsOn {
			if _, exists := tasksByID[dep]; !exists {
				return nil, fmt.Errorf("task %q depends on unknown task %q", t.ID, dep)
			}
			adjList[dep] = append(adjList[dep], t.ID)
		}
	}

	// ── Step 2: Upfront cycle detection via Kahn's algorithm ──
	// This catches circular deps instantly before any goroutines start.
	{
		degree := make(map[string]int, len(tasks))
		for id, deg := range inDegree {
			degree[id] = deg
		}
		queue := make([]string, 0, len(tasks))
		for id, deg := range degree {
			if deg == 0 {
				queue = append(queue, id)
			}
		}
		visited := 0
		for len(queue) > 0 {
			n := queue[0]
			queue = queue[1:]
			visited++
			for _, child := range adjList[n] {
				degree[child]--
				if degree[child] == 0 {
					queue = append(queue, child)
				}
			}
		}
		if visited != len(tasks) {
			return nil, fmt.Errorf("cycle detected in task dependency graph: only %d of %d tasks are reachable", visited, len(tasks))
		}
	}

	// ── Step 3: Dispatch tasks topologically ──
	pending := len(tasks)
	readyChan := make(chan Task, pending)
	for id, deg := range inDegree {
		if deg == 0 {
			readyChan <- tasksByID[id]
		}
	}

	resultChan := make(chan Result, len(tasks))
	sem := make(chan struct{}, o.maxConcurrency)
	var wg sync.WaitGroup

	// completedOutputs is accessed sequentially within the select loop;
	// all reads and writes happen on the same goroutine so no mutex is needed.
	completedOutputs := make(map[string]string, len(tasks))
	var results []Result

	// runningWorkers tracks goroutines that have been dispatched but haven't
	// sent their result yet. It IS written from goroutines (via mu) and read
	// in the main loop for deadlock detection.
	var runningWorkers int
	var mu sync.Mutex

	// Use a ticker rather than time.After to avoid allocating a new timer
	// on every select iteration for the full lifetime of the DAG.
	deadlockTicker := time.NewTicker(50 * time.Millisecond)
	defer deadlockTicker.Stop()

	for pending > 0 {
		select {
		case <-ctx.Done():
			return results, nil

		case t := <-readyChan:


			// Inject dependency outputs as properly typed map so buildTaskPrompt
			// can format them cleanly for the LLM.
			if t.Inputs == nil {
				t.Inputs = make(map[string]interface{})
			}
			depOutputs := make(map[string]string)
			for _, dep := range t.DependsOn {
				if out, ok := completedOutputs[dep]; ok {
					depOutputs[dep] = out
				}
			}
			if len(depOutputs) > 0 {
				t.Inputs["dependency_outputs"] = depOutputs
			}

			role := RoleGeneric
			if r, ok := t.Inputs["role_hint"].(string); ok {
				role = WorkerRole(r)
			}
			seq := atomic.AddUint64(&workerCounter, 1)
			workerID := fmt.Sprintf("worker-%s-%d", role, seq)

			worker := NewWorker(workerID, role, o.baseAgent, o.bus)
			taskChan := make(chan Task, 1)
			taskChan <- t
			close(taskChan)

			wg.Add(1)
			go func(w *ConcreteWorker, ch <-chan Task) {
				defer wg.Done()
				select {
				case sem <- struct{}{}:
					mu.Lock()
					runningWorkers++
					mu.Unlock()

					defer func() {
						mu.Lock()
						runningWorkers--
						mu.Unlock()
						<-sem
					}()
				case <-ctx.Done():
					return
				}
				w.Start(ctx, ch, resultChan)
			}(worker, taskChan)

		case res := <-resultChan:
			// completedOutputs read/write is safe: only happens here in the
			// main select loop (single goroutine).
			results = append(results, res)
			completedOutputs[res.TaskID] = res.Output
			pending--

			for _, childID := range adjList[res.TaskID] {
				inDegree[childID]--
				if inDegree[childID] == 0 {
					readyChan <- tasksByID[childID]
				}
			}

		case <-deadlockTicker.C:
			// Cycle detection already ran upfront, but this catches the
			// edge case where a dependency references an ID that was validated
			// but somehow never enqueued (shouldn't happen, but defensive).
			mu.Lock()
			active := runningWorkers
			mu.Unlock()

			if active == 0 && pending > 0 && len(readyChan) == 0 {
				o.logger.Error("Unexpected stall in DAG execution", "pending_count", pending)
				return results, fmt.Errorf("DAG execution stalled: %d tasks pending but no workers active and no tasks ready", pending)
			}
		}
	}

	wg.Wait()
	o.logger.Info("Completed DAG parallel execution", "result_count", len(results))
	return results, nil
}
