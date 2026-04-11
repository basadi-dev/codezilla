package tools

import (
	"context"
)

// SubAgentLauncher is a callback implemented by the core application to spin up a sub-agent
type SubAgentLauncher func(ctx context.Context, task string) (string, error)

type SubAgentTool struct {
	launcher SubAgentLauncher
}

func NewSubAgentTool(launcher SubAgentLauncher) Tool {
	return &SubAgentTool{
		launcher: launcher,
	}
}

func (t *SubAgentTool) Name() string {
	return "subAgent"
}

func (t *SubAgentTool) Description() string {
	return "Launch a sub-agent to break down and solve a complex sub-task autonomously. Use this when a task is too complex, requires multiple steps, or needs its own thought process."
}

func (t *SubAgentTool) ParameterSchema() JSONSchema {
	return JSONSchema{
		Type: "object",
		Properties: map[string]JSONSchema{
			"task": {
				Type:        "string",
				Description: "A fully detailed description of the task for the sub-agent. Include all necessary context, file paths, and exact criteria for success.",
			},
		},
		Required: []string{"task"},
	}
}

func (t *SubAgentTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	task, ok := params["task"].(string)
	if !ok || task == "" {
		return nil, &ErrInvalidToolParams{
			ToolName: t.Name(),
			Message:  "missing or invalid 'task' parameter",
		}
	}

	if t.launcher == nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  "sub-agent launcher is not configured",
		}
	}

	result, err := t.launcher(ctx, task)
	if err != nil {
		return nil, &ErrToolExecution{
			ToolName: t.Name(),
			Message:  "sub-agent failed",
			Err:      err,
		}
	}

	return result, nil
}
