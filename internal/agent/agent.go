package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"codezilla/internal/core/llm"
	"codezilla/internal/tools"
	"codezilla/pkg/logger"
	anyllm "github.com/mozilla-ai/any-llm-go"
)

var (
	ErrLLMResponseFormat   = errors.New("invalid LLM response format")
	ErrToolExecutionFailed = errors.New("tool execution failed")
	ErrToolNotFound        = errors.New("tool not found")
)

// Agent interface defines the core functionality of an agent
type Agent interface {
	// ProcessMessage processes a user message and returns the agent's response
	ProcessMessage(ctx context.Context, message string) (string, error)

	// ProcessMessageStream processes a user message and streams tokens via callback.
	// The callback is called for each token chunk. Returns the full final response.
	ProcessMessageStream(ctx context.Context, message string, onToken func(string)) (string, error)

	// ExecuteTool executes a tool with the given parameters
	ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error)

	// AddSystemMessage adds a system message to the context
	AddSystemMessage(message string)

	// AddUserMessage adds a user message to the context
	AddUserMessage(message string)

	// AddAssistantMessage adds an assistant message to the context
	AddAssistantMessage(message string)

	// GetMessages returns a copy of all conversation messages (for save/display)
	GetMessages() []Message

	// ClearContext clears all non-system messages from the conversation context
	ClearContext()

	// ClearLastUserMessage removes the most recently added user message from context.
	// Used to avoid double-adding a message when retrying after an error.
	ClearLastUserMessage()

	// SetModel changes the active model used by the agent
	SetModel(model string)

	// SetTemperature changes the temperature setting
	SetTemperature(temperature float64)

	// SetMaxTokens changes the max tokens setting
	SetMaxTokens(maxTokens int)
}


// Config contains configuration for the agent
type Config struct {
	Model          string
	Provider       string
	MaxTokens      int
	Temperature    float64
	SystemPrompt   string
	LLMClient      *llm.Client
	ToolRegistry   tools.ToolRegistry
	PromptTemplate *PromptTemplate
	Logger         *logger.Logger
	PermissionMgr  tools.ToolPermissionManager
	// AutoPlan controls whether the agent automatically creates todo plans for
	// complex tasks. Defaults to false to avoid unintended planning triggers.
	AutoPlan bool
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		Model:          "qwen2.5-coder:3b",
		Provider:       "ollama",
		MaxTokens:      4000,
		Temperature:    0.7,
		AutoPlan:       false,
		PromptTemplate: DefaultPromptTemplate(),
		Logger:         logger.DefaultLogger(),
	}
}

// agent implements the Agent interface
type agent struct {
	config        *Config
	context       *Context
	llmClient     *llm.Client
	toolRegistry  tools.ToolRegistry
	logger        *logger.Logger
	permissionMgr tools.ToolPermissionManager
}

// NewAgent creates a new agent with the given configuration
func NewAgent(config *Config) Agent {
	if config == nil {
		config = DefaultConfig()
	}

	if config.Logger == nil {
		config.Logger = logger.DefaultLogger()
	}

	// If no permission manager is provided, create one with a default callback that always allows execution
	// This will be replaced by the CLI with a proper interactive callback
	if config.PermissionMgr == nil {
		config.PermissionMgr = tools.NewPermissionManager(func(ctx context.Context, request tools.PermissionRequest) (tools.PermissionResponse, error) {
			// Default behavior: grant permission but don't remember
			return tools.PermissionResponse{Granted: true, RememberMe: false}, nil
		})
	}

	agent := &agent{
		config:        config,
		context:       NewContext(config.MaxTokens, config.Logger),
		llmClient:     config.LLMClient,
		toolRegistry:  config.ToolRegistry,
		logger:        config.Logger,
		permissionMgr: config.PermissionMgr,
	}

	// Add initial system message if provided
	if config.SystemPrompt != "" {
		// Format system prompt with tool information
		var toolSpecs []tools.ToolSpec
		if config.ToolRegistry != nil {
			toolSpecs = config.ToolRegistry.GetToolSpecs()
		}

		formattedPrompt := FormatSystemPrompt(config.SystemPrompt, toolSpecs)
		agent.AddSystemMessage(formattedPrompt)
	}

	return agent
}

// ProcessMessage processes a user message and returns the agent's response
func (a *agent) ProcessMessage(ctx context.Context, message string) (string, error) {
	a.logger.Debug("Processing message", "message", message)
	a.AddUserMessage(message)

	if a.config.AutoPlan && a.shouldCreateTodoPlan(message) {
		a.logger.Debug("Creating automatic todo plan for complex task")
		planResponse, err := a.createAutomaticTodoPlan(ctx, message)
		if err != nil {
			a.logger.Error("Failed to create automatic todo plan", "error", err)
		} else if planResponse != "" {
			a.AddAssistantMessage(planResponse)
		}
	}

	response, err := a.generateResponse(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	finalResponse, err := a.runToolLoop(ctx, response)
	if err != nil {
		return "", err
	}

	a.AddAssistantMessage(finalResponse)
	return finalResponse, nil
}

// ProcessMessageStream streams the initial LLM response token by token via onToken,
// then runs tool calls (non-streamed) and returns the full final response.
func (a *agent) ProcessMessageStream(ctx context.Context, message string, onToken func(string)) (string, error) {
	a.logger.Debug("Processing message (streaming)", "message", message)
	a.AddUserMessage(message)

	if a.config.AutoPlan && a.shouldCreateTodoPlan(message) {
		a.logger.Debug("Creating automatic todo plan for complex task")
		planResponse, err := a.createAutomaticTodoPlan(ctx, message)
		if err != nil {
			a.logger.Error("Failed to create automatic todo plan", "error", err)
		} else if planResponse != "" {
			a.AddAssistantMessage(planResponse)
		}
	}

	// --- Stream the first response ---
	chatMessages := a.buildChatMessages()

	// Ensure system prompt is the first message if needed
	sysPrompt := a.buildSystemPrompt()
	if len(chatMessages) > 0 && chatMessages[0].Role != "system" {
		chatMessages = append([]anyllm.Message{
			{Role: "system", Content: sysPrompt},
		}, chatMessages...)
	}

	streamCh, errCh, err := a.llmClient.Stream(ctx, a.config.Provider, a.config.Model, chatMessages, a.config.Temperature)

	if err != nil {
		// Fall back to non-streaming. The user message is already in context,
		// so we call generateResponse + runToolLoop directly to avoid the
		// double-add that ProcessMessage would cause.
		a.logger.Warn("Streaming failed, falling back to non-streaming", "error", err)
		response, genErr := a.generateResponse(ctx)
		if genErr != nil {
			return "", fmt.Errorf("failed to generate response: %w", genErr)
		}
		finalResponse, toolErr := a.runToolLoop(ctx, response)
		if toolErr != nil {
			return "", toolErr
		}
		a.AddAssistantMessage(finalResponse)
		return finalResponse, nil
	}

	var fullResponse strings.Builder
	var streamErr error

	// Process stream until completion or error
	for {
		select {
		case chunk, ok := <-streamCh:
			if !ok {
				streamCh = nil
			} else {
				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
					fullResponse.WriteString(chunk.Choices[0].Delta.Content)
					if onToken != nil {
						onToken(chunk.Choices[0].Delta.Content)
					}
				}
			}
		case errVal, ok := <-errCh:
			if ok && errVal != nil {
				a.logger.Error("Stream processing error", "error", errVal)
				streamErr = errVal
			}
			errCh = nil
		}

		if streamCh == nil && errCh == nil {
			break
		}
	}

	// If the stream returned an error, propagate it so callers can surface it
	if streamErr != nil {
		return "", streamErr
	}

	firstResponse := strings.TrimSpace(fullResponse.String())
	// Remove leading "Assistant:" if the model echoed it
	firstResponse = strings.TrimPrefix(firstResponse, "Assistant:")
	firstResponse = strings.TrimSpace(firstResponse)

	// If streaming returned no content (e.g. a thinking model that streams
	// internal reasoning with empty content fields), fall back to the
	// non-streaming Complete API which consolidates the full response.
	if firstResponse == "" {
		a.logger.Warn("Streaming returned empty content, falling back to non-streaming Complete")
		response, genErr := a.generateResponse(ctx)
		if genErr != nil {
			return "", fmt.Errorf("non-streaming fallback failed: %w", genErr)
		}
		finalResponse, toolErr := a.runToolLoop(ctx, response)
		if toolErr != nil {
			return "", toolErr
		}
		// Emit the full response via onToken so the caller's display path works
		if onToken != nil && finalResponse != "" {
			onToken(finalResponse)
		}
		a.AddAssistantMessage(finalResponse)
		return finalResponse, nil
	}

	// --- Run tool loop (non-streamed) ---
	finalResponse, err := a.runToolLoop(ctx, firstResponse)
	if err != nil {
		return "", err
	}

	a.AddAssistantMessage(finalResponse)
	return finalResponse, nil
}

// ExecuteTool executes a tool with the given parameters
func (a *agent) ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error) {
	if a.toolRegistry == nil {
		return nil, ErrToolNotFound
	}

	// Get the tool
	tool, found := a.toolRegistry.GetTool(toolName)
	if !found {
		a.logger.Error("Tool not found", "tool", toolName)
		return nil, fmt.Errorf("%w: %s", ErrToolNotFound, toolName)
	}

	// Nil check for tool
	if tool == nil {
		a.logger.Error("Tool is nil", "tool", toolName)
		return nil, fmt.Errorf("tool %s is nil", toolName)
	}

	a.logger.Info("Executing tool", "tool", toolName)
	a.logger.Debug("Tool details", "tool", toolName, "description", tool.Description(), "params", fmt.Sprintf("%v", params))

	// Validate parameters
	err := tools.ValidateToolParams(tool, params)
	if err != nil {
		a.logger.Error("Invalid tool parameters", "tool", toolName, "error", err)
		return nil, err
	}

	// Request execution permission
	if a.permissionMgr != nil {
		a.logger.Debug("Requesting tool execution permission", "tool", toolName)

		granted, err := a.permissionMgr.RequestPermission(ctx, toolName, params, tool)
		if err != nil {
			a.logger.Error("Permission request failed", "tool", toolName, "error", err)
			return nil, fmt.Errorf("failed to request permission: %w", err)
		}

		if !granted {
			a.logger.Info("Permission denied for tool execution", "tool", toolName)
			return nil, tools.ErrPermissionDenied
		}

		a.logger.Debug("Permission granted for tool execution", "tool", toolName)
	}

	// Execute the tool
	startTime := time.Now()
	result, err := tool.Execute(ctx, params)
	duration := time.Since(startTime)

	if err != nil {
		a.logger.Error("Tool execution failed", "tool", toolName, "error", err, "duration", duration.String())
		return nil, fmt.Errorf("%w: %s: %v", ErrToolExecutionFailed, toolName, err)
	}

	a.logger.Info("Tool executed successfully",
		"tool", toolName,
		"duration", duration.String())

	return result, nil
}

// --- Context delegate methods ---

// AddSystemMessage adds a system message to the context
func (a *agent) AddSystemMessage(message string) {
	a.context.AddSystemMessage(message)
}

// AddUserMessage adds a user message to the context
func (a *agent) AddUserMessage(message string) {
	a.context.AddUserMessage(message)
}

// AddAssistantMessage adds an assistant message to the context
func (a *agent) AddAssistantMessage(message string) {
	a.context.AddAssistantMessage(message)
}

// GetMessages returns a copy of all conversation messages
func (a *agent) GetMessages() []Message {
	return a.context.GetMessages()
}

// ClearContext clears all non-system messages from the conversation context
func (a *agent) ClearContext() {
	a.logger.Info("Clearing conversation context (keeping system messages)")
	a.context.ClearContext()
}

// ClearLastUserMessage removes the most recently added user message from context.
func (a *agent) ClearLastUserMessage() {
	a.context.ClearLastUserMessage()
}

// --- Configuration mutators ---

// SetModel changes the active model used by the agent
func (a *agent) SetModel(model string) {
	a.logger.Info("Changing model", "from", a.config.Model, "to", model)
	a.config.Model = model
}

// SetTemperature changes the temperature setting
func (a *agent) SetTemperature(temperature float64) {
	a.logger.Info("Changing temperature", "from", a.config.Temperature, "to", temperature)
	a.config.Temperature = temperature
}

// SetMaxTokens changes the max tokens setting
func (a *agent) SetMaxTokens(maxTokens int) {
	a.logger.Info("Changing max tokens", "from", a.config.MaxTokens, "to", maxTokens)
	a.config.MaxTokens = maxTokens
}
