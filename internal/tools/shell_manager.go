package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Job represents a running background process
type Job struct {
	ID        string
	Command   string
	Cmd       *exec.Cmd
	Stdout    *bytes.Buffer
	Stderr    *bytes.Buffer
	StartTime time.Time
	Ctx       context.Context
	Cancel    context.CancelFunc
	Done      chan struct{}
	Err       error
}

type BackgroundJobManager struct {
	jobs sync.Map
}

var globalJobManager *BackgroundJobManager
var managerOnce sync.Once

// GetBackgroundJobManager returns the singleton job manager
func GetBackgroundJobManager() *BackgroundJobManager {
	managerOnce.Do(func() {
		globalJobManager = &BackgroundJobManager{}
	})
	return globalJobManager
}

// StartJob creates and starts a background job
func (m *BackgroundJobManager) StartJob(command string, args []string, dir string, env []string) (*Job, error) {
	ctx, cancel := context.WithCancel(context.Background())

	var cmd *exec.Cmd
	if len(args) > 0 {
		cmd = exec.CommandContext(ctx, args[0], args[1:]...)
	} else {
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}

	if dir != "" {
		cmd.Dir = dir
	}
	if len(env) > 0 {
		cmd.Env = env
	}

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, err
	}

	job := &Job{
		ID:        uuid.New().String()[:8], // Short ID
		Command:   command,
		Cmd:       cmd,
		Stdout:    stdout,
		Stderr:    stderr,
		StartTime: time.Now(),
		Ctx:       ctx,
		Cancel:    cancel,
		Done:      make(chan struct{}),
	}

	m.jobs.Store(job.ID, job)

	// Wait in background
	go func() {
		job.Err = cmd.Wait()
		close(job.Done)
	}()

	return job, nil
}

// GetJob retrieves a job
func (m *BackgroundJobManager) GetJob(id string) (*Job, bool) {
	val, ok := m.jobs.Load(id)
	if !ok {
		return nil, false
	}
	return val.(*Job), true
}

// KillJob forcefully terminates a job
func (m *BackgroundJobManager) KillJob(id string) error {
	job, ok := m.GetJob(id)
	if !ok {
		return fmt.Errorf("job %s not found", id)
	}
	job.Cancel()
	<-job.Done // Wait for process to clean up
	m.jobs.Delete(id)
	return nil
}

// Cleanup finished jobs
func (m *BackgroundJobManager) Cleanup() {
	m.jobs.Range(func(key, value interface{}) bool {
		job := value.(*Job)
		select {
		case <-job.Done:
			m.jobs.Delete(key)
		default:
		}
		return true
	})
}
