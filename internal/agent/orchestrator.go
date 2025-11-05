package agent

import (
	"context"
	"fmt"

	"codezilla/internal/tools"
	"codezilla/llm"
	"codezilla/pkg/logger"
)

// Orchestrator coordinates the conversation flow
type Orchestrator struct {
	provider      llm.Provider
	toolExecutor  *ToolExecutor
	contextMgr    *ContextManager
	toolExtractor *ToolExtractor
	config        *OrchestratorConfig
	logger        *logger.Logger
}

// OrchestratorConfig contains configuration for the orchestrator
type OrchestratorConfig struct {
	Model            string
	Temperature      float64
	MaxTokens        int
	MaxIterations    int // Max tool execution iterations
	SystemPrompt     string
}

// NewOrchestrator creates a new orchestrator
func NewOrchestrator(
	provider llm.Provider,
	toolRegistry tools.ToolRegistry,
	permissionMgr tools.ToolPermissionManager,
	config *OrchestratorConfig,
	log *logger.Logger,
) *Orchestrator {
	if log == nil {
		// Create default logger
		log, _ = logger.New(logger.Config{Silent: true})
	}

	if config.MaxIterations == 0 {
		config.MaxIterations = 10 // Default safety limit
	}

	contextMgr := NewContextManager(config.MaxTokens, &HeuristicCounter{}, log)
	toolExecutor := NewToolExecutor(toolRegistry, permissionMgr, log)
	toolExtractor := NewToolExtractor()

	// Add system prompt if provided
	if config.SystemPrompt != "" {
		contextMgr.AddSystemMessage(config.SystemPrompt)
	}

	return &Orchestrator{
		provider:      provider,
		toolExecutor:  toolExecutor,
		contextMgr:    contextMgr,
		toolExtractor: toolExtractor,
		config:        config,
		logger:        log,
	}
}

// ProcessMessage processes a user message and returns the agent's response
func (o *Orchestrator) ProcessMessage(ctx context.Context, message string) (string, error) {
	o.logger.Debug("Processing message", "message", message)

	// Add user message to context
	o.contextMgr.AddUserMessage(message)

	// Generate response with tool execution loop
	response, err := o.generateWithToolExecution(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	// Add final response to context
	o.contextMgr.AddAssistantMessage(response)

	return response, nil
}

// generateWithToolExecution generates a response and handles tool execution loop
func (o *Orchestrator) generateWithToolExecution(ctx context.Context) (string, error) {
	var response string

	for iteration := 0; iteration < o.config.MaxIterations; iteration++ {
		o.logger.Debug("Generation iteration", "iteration", iteration)

		// Generate response from provider
		resp, err := o.provider.Generate(ctx, llm.GenerateRequest{
			Model:       o.config.Model,
			Messages:    o.contextMgr.GetMessages(),
			Temperature: o.config.Temperature,
			MaxTokens:   o.config.MaxTokens,
			Stream:      false,
		})
		if err != nil {
			return "", fmt.Errorf("provider generate failed: %w", err)
		}

		response = resp.Content
		o.logger.Debug("Generated response", "length", len(response))

		// Extract tool calls
		toolCalls := o.toolExtractor.ExtractAll(response)
		if len(toolCalls) == 0 {
			// No tool calls, this is the final response
			o.logger.Debug("No tool calls detected, returning final response")
			return response, nil
		}

		o.logger.Debug("Extracted tool calls", "count", len(toolCalls))

		// Execute tools
		results := o.toolExecutor.ExecuteAll(ctx, toolCalls)

		// Add tool results to context
		for _, result := range results {
			o.contextMgr.AddToolResultMessage(result.ToolName, result.Result, result.Error)
		}

		// Continue loop to generate next response
	}

	// Hit max iterations
	o.logger.Warn("Hit max iterations", "maxIterations", o.config.MaxIterations)
	return response, fmt.Errorf("max iterations (%d) reached", o.config.MaxIterations)
}

// AddUserMessage adds a user message to the context
func (o *Orchestrator) AddUserMessage(message string) {
	o.contextMgr.AddUserMessage(message)
}

// AddAssistantMessage adds an assistant message to the context
func (o *Orchestrator) AddAssistantMessage(message string) {
	o.contextMgr.AddAssistantMessage(message)
}

// AddSystemMessage adds a system message to the context
func (o *Orchestrator) AddSystemMessage(message string) {
	o.contextMgr.AddSystemMessage(message)
}

// ClearContext clears non-system messages from context
func (o *Orchestrator) ClearContext() {
	o.contextMgr.Clear()
}

// SetModel changes the active model
func (o *Orchestrator) SetModel(model string) {
	o.config.Model = model
}

// SetTemperature changes the temperature setting
func (o *Orchestrator) SetTemperature(temperature float64) {
	o.config.Temperature = temperature
}

// SetMaxTokens changes the max tokens setting
func (o *Orchestrator) SetMaxTokens(maxTokens int) {
	o.config.MaxTokens = maxTokens
}

// GetProvider returns the underlying provider
func (o *Orchestrator) GetProvider() llm.Provider {
	return o.provider
}
