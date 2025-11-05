package agent

import (
	"context"
	"fmt"
	"time"

	"codezilla/internal/tools"
	"codezilla/pkg/logger"
)

// ToolExecutor handles tool execution with permissions and logging
type ToolExecutor struct {
	registry      tools.ToolRegistry
	permissionMgr tools.ToolPermissionManager
	logger        *logger.Logger
}

// NewToolExecutor creates a new tool executor
func NewToolExecutor(registry tools.ToolRegistry, permissionMgr tools.ToolPermissionManager, logger *logger.Logger) *ToolExecutor {
	return &ToolExecutor{
		registry:      registry,
		permissionMgr: permissionMgr,
		logger:        logger,
	}
}

// Execute executes a single tool call
func (te *ToolExecutor) Execute(ctx context.Context, toolCall ToolCall) (interface{}, error) {
	start := time.Now()

	tool, exists := te.registry.GetTool(toolCall.ToolName)
	if !exists {
		return nil, fmt.Errorf("tool '%s' not found", toolCall.ToolName)
	}

	te.logger.Debug("Executing tool", "name", toolCall.ToolName, "params", toolCall.Params)

	// Validate parameters
	if err := tools.ValidateToolParams(tool, toolCall.Params); err != nil {
		te.logger.Error("Tool parameter validation failed", "tool", toolCall.ToolName, "error", err)
		return nil, err
	}

	// Check permissions
	granted, err := te.permissionMgr.RequestPermission(ctx, toolCall.ToolName, toolCall.Params, tool)
	if err != nil {
		te.logger.Error("Permission check failed", "tool", toolCall.ToolName, "error", err)
		return nil, fmt.Errorf("permission check failed: %w", err)
	}

	if !granted {
		te.logger.Warn("Tool execution denied by user", "tool", toolCall.ToolName)
		return nil, fmt.Errorf("permission denied for tool '%s'", toolCall.ToolName)
	}

	// Execute tool
	result, err := tool.Execute(ctx, toolCall.Params)
	duration := time.Since(start)

	if err != nil {
		te.logger.Error("Tool execution failed", "tool", toolCall.ToolName, "duration", duration, "error", err)
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}

	te.logger.Info("Tool executed successfully", "tool", toolCall.ToolName, "duration", duration)
	return result, nil
}

// ExecutionResult represents the result of a tool execution (including tool name)
type ExecutionResult struct {
	ToolName string
	Result   interface{}
	Error    error
}

// ExecuteAll executes multiple tool calls sequentially
func (te *ToolExecutor) ExecuteAll(ctx context.Context, toolCalls []ToolCall) []ExecutionResult {
	results := make([]ExecutionResult, len(toolCalls))

	for i, toolCall := range toolCalls {
		result, err := te.Execute(ctx, toolCall)
		results[i] = ExecutionResult{
			ToolName: toolCall.ToolName,
			Result:   result,
			Error:    err,
		}
	}

	return results
}
