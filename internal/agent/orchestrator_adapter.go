package agent

import (
	"context"

	"codezilla/internal/tools"
	"codezilla/llm"
)

// OrchestratorAdapter adapts the new Orchestrator to the existing Agent interface
type OrchestratorAdapter struct {
	orchestrator *Orchestrator
	toolRegistry tools.ToolRegistry
}

// NewOrchestratorAdapter creates a new adapter
func NewOrchestratorAdapter(
	provider llm.Provider,
	toolRegistry tools.ToolRegistry,
	permissionMgr tools.ToolPermissionManager,
	config *Config,
) Agent {
	orchConfig := &OrchestratorConfig{
		Model:         config.Model,
		Temperature:   config.Temperature,
		MaxTokens:     config.MaxTokens,
		MaxIterations: 10,
		SystemPrompt:  config.SystemPrompt,
	}

	orchestrator := NewOrchestrator(provider, toolRegistry, permissionMgr, orchConfig, config.Logger)

	return &OrchestratorAdapter{
		orchestrator: orchestrator,
		toolRegistry: toolRegistry,
	}
}

// ProcessMessage implements Agent.ProcessMessage
func (a *OrchestratorAdapter) ProcessMessage(ctx context.Context, message string) (string, error) {
	return a.orchestrator.ProcessMessage(ctx, message)
}

// ExecuteTool implements Agent.ExecuteTool
func (a *OrchestratorAdapter) ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error) {
	tool, exists := a.toolRegistry.GetTool(toolName)
	if !exists {
		return nil, ErrToolNotFound
	}

	return tool.Execute(ctx, params)
}

// AddSystemMessage implements Agent.AddSystemMessage
func (a *OrchestratorAdapter) AddSystemMessage(message string) {
	a.orchestrator.AddSystemMessage(message)
}

// AddUserMessage implements Agent.AddUserMessage
func (a *OrchestratorAdapter) AddUserMessage(message string) {
	a.orchestrator.AddUserMessage(message)
}

// AddAssistantMessage implements Agent.AddAssistantMessage
func (a *OrchestratorAdapter) AddAssistantMessage(message string) {
	a.orchestrator.AddAssistantMessage(message)
}

// ClearContext implements Agent.ClearContext
func (a *OrchestratorAdapter) ClearContext() {
	a.orchestrator.ClearContext()
}

// SetModel implements Agent.SetModel
func (a *OrchestratorAdapter) SetModel(model string) {
	a.orchestrator.SetModel(model)
}

// SetTemperature implements Agent.SetTemperature
func (a *OrchestratorAdapter) SetTemperature(temperature float64) {
	a.orchestrator.SetTemperature(temperature)
}

// SetMaxTokens implements Agent.SetMaxTokens
func (a *OrchestratorAdapter) SetMaxTokens(maxTokens int) {
	a.orchestrator.SetMaxTokens(maxTokens)
}
