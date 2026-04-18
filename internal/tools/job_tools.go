package tools

import (
	"context"
	"fmt"
)

// JobOutputTool reads the output of a running background job
type JobOutputTool struct{}

func NewJobOutputTool() *JobOutputTool {
	return &JobOutputTool{}
}

func (t *JobOutputTool) Name() string { return "jobOutput" }

func (t *JobOutputTool) Description() string {
	return "Read the standard output and error of a running background job."
}

func (t *JobOutputTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"job_id": {
				Type:        "string",
				Description: "The ID of the background job",
			},
		},
		Required: []string{"job_id"},
	}
}

func (t *JobOutputTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}
	jobID, _ := params["job_id"].(string)

	mgr := GetBackgroundJobManager()
	job, ok := mgr.GetJob(jobID)
	if !ok {
		return nil, fmt.Errorf("job %s not found (it may have finished and cleanup cleared it)", jobID)
	}

	stdoutStr := job.Stdout.String()
	stderrStr := job.Stderr.String()

	status := "running"
	select {
	case <-job.Done:
		status = "completed"
		if job.Err != nil {
			status = fmt.Sprintf("completed with error: %v", job.Err)
		}
	default:
	}

	return map[string]interface{}{
		"job_id": jobID,
		"status": status,
		"stdout": stdoutStr,
		"stderr": stderrStr,
	}, nil
}

// JobKillTool forcefully terminates a running job
type JobKillTool struct{}

func NewJobKillTool() *JobKillTool {
	return &JobKillTool{}
}

func (t *JobKillTool) Name() string { return "jobKill" }

func (t *JobKillTool) Description() string {
	return "Forcefully terminate a running background job."
}

func (t *JobKillTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"job_id": {
				Type:        "string",
				Description: "The ID of the background job to kill",
			},
		},
		Required: []string{"job_id"},
	}
}

func (t *JobKillTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	if err := ValidateToolParams(t, params); err != nil {
		return nil, err
	}
	jobID, _ := params["job_id"].(string)

	mgr := GetBackgroundJobManager()
	err := mgr.KillJob(jobID)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"success": true,
		"message": fmt.Sprintf("Job %s terminated", jobID),
	}, nil
}
