package agent

import (
	"context"
	"errors"
	"fmt"
	"time"

	"codezilla/internal/core/llm"
	"codezilla/internal/session"
	"codezilla/internal/tools"
	"codezilla/pkg/logger"
)

var (
	ErrLLMResponseFormat   = errors.New("invalid LLM response format")
	ErrToolExecutionFailed = errors.New("tool execution failed")
	ErrToolNotFound        = errors.New("tool not found")
)

type Agent interface {
	ProcessMessage(ctx context.Context, message string) (string, error)
	ProcessMessageStream(ctx context.Context, message string, onToken func(string), onStreamEnd func()) (string, error)
	ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error)
	AddSystemMessage(message string)
	ReplaceSystemMessage(message string)
	AddUserMessage(message string)
	AddAssistantMessage(message string)
	AddMessage(msg Message)
	GetMessages() []Message
	ClearContext()
	ClearLastUserMessage()
	SetModel(model string)
	SetPlannerModel(model string)
	SetSubAgentModel(model string)
	SetSummariserModel(model string)
	SetTemperature(temperature float64)
	SetMaxTokens(maxTokens int)
}

type Config struct {
	Model           string
	PlannerModel    string
	SubAgentModel   string
	Provider        string
	MaxTokens       int
	MaxIterations   int
	Temperature     float64
	SystemPrompt    string
	LLMClient       *llm.Client
	ToolRegistry    tools.ToolRegistry
	PromptTemplate  *PromptTemplate
	Logger          *logger.Logger
	PermissionMgr   tools.ToolPermissionManager
	AutoPlan        bool
	OnToolExecution func(toolName string, params map[string]interface{})
	// OnLLMCall is called before each LLM request with:
	//   callNum    – 1-based call counter
	//   msgCount   – number of messages in the context
	//   approxToks – rough token estimate (chars/4)
	OnLLMCall func(callNum, msgCount, approxToks int)
	// OnToolPreparing is called during streaming when a tool call name is first detected,
	// before the tool actually executes. Use this to update the spinner with the tool name.
	OnToolPreparing func(toolName string)
	// OnLLMStreamEnd is called immediately after a text/tool stream completes natively, before parsing.
	OnLLMStreamEnd func()
	// OnLLMUsage is called after each LLM call with the per-turn and cumulative session token usage.
	// Providers that don't return usage data will not trigger this callback.
	OnLLMUsage func(turn TokenUsage, session TokenUsage)

	// Loop detection: stops the run loop if a tool is called with identical args consecutively.
	// 0 = use defaults (window=10, max_repeat=3).
	LoopDetectWindow    int
	LoopDetectMaxRepeat int

	SummariserModel        string
	ThinkCompressThreshold int
	SlidingWindowSize      int // number of recent non-system messages to keep verbatim (0 = disabled)
	SessionRecorder        *session.Recorder
}

func DefaultConfig() *Config {
	return &Config{
		Model:               "qwen2.5-coder:3b",
		Provider:            "ollama",
		MaxTokens:           4000,
		MaxIterations:       0, // 0 = unlimited
		Temperature:         0.7,
		AutoPlan:            false,
		PromptTemplate:      DefaultPromptTemplate(),
		Logger:              logger.DefaultLogger(),
		LoopDetectWindow:    10,
		LoopDetectMaxRepeat: 3,
	}
}

type agent struct {
	config        *Config
	context       *Context
	llmClient     *llm.Client
	toolRegistry  tools.ToolRegistry
	logger        *logger.Logger
	permissionMgr tools.ToolPermissionManager
}

func NewAgent(config *Config) Agent {
	if config == nil {
		config = DefaultConfig()
	}

	if config.Logger == nil {
		config.Logger = logger.DefaultLogger()
	}

	if config.PermissionMgr == nil {
		config.PermissionMgr = tools.NewPermissionManager(func(ctx context.Context, request tools.PermissionRequest) (tools.PermissionResponse, error) {
			return tools.PermissionResponse{Granted: true, RememberMe: false}, nil
		})
	}

	ctx := NewContext(config.MaxTokens, config.Logger)
	if config.SlidingWindowSize > 0 {
		ctx.SlidingWindowSize = config.SlidingWindowSize
	}

	agent := &agent{
		config:        config,
		context:       ctx,
		llmClient:     config.LLMClient,
		toolRegistry:  config.ToolRegistry,
		logger:        config.Logger,
		permissionMgr: config.PermissionMgr,
	}

	if config.SystemPrompt != "" {
		var toolSpecs []tools.ToolSpec
		if config.ToolRegistry != nil {
			toolSpecs = config.ToolRegistry.GetToolSpecs()
		}
		formattedPrompt := FormatSystemPrompt(config.SystemPrompt, toolSpecs)
		agent.AddSystemMessage(formattedPrompt)
	}

	return agent
}

func (a *agent) ProcessMessage(ctx context.Context, message string) (string, error) {
	orchestrator := NewAgentOrchestrator(a)
	return orchestrator.Run(ctx, message, nil, false)
}

func (a *agent) ProcessMessageStream(ctx context.Context, message string, onToken func(string), onStreamEnd func()) (string, error) {
	orchestrator := NewAgentOrchestrator(a)
	a.config.OnLLMStreamEnd = onStreamEnd
	return orchestrator.Run(ctx, message, onToken, true)
}

func (a *agent) ExecuteTool(ctx context.Context, toolName string, params map[string]interface{}) (interface{}, error) {
	if a.toolRegistry == nil {
		return nil, ErrToolNotFound
	}

	tool, found := a.toolRegistry.GetTool(toolName)
	if !found {
		a.logger.Error("Tool not found", "tool", toolName)
		return nil, fmt.Errorf("%w: %s", ErrToolNotFound, toolName)
	}

	if tool == nil {
		a.logger.Error("Tool is nil", "tool", toolName)
		return nil, fmt.Errorf("tool %s is nil", toolName)
	}

	a.logger.Info("Executing tool", "tool", toolName)
	a.logger.Debug("Tool details", "tool", toolName, "description", tool.Description(), "params", fmt.Sprintf("%v", params))

	err := tools.ValidateToolParams(tool, params)
	if err != nil {
		a.logger.Error("Invalid tool parameters", "tool", toolName, "error", err)
		return nil, err
	}

	if a.permissionMgr != nil {
		granted, err := a.permissionMgr.RequestPermission(ctx, toolName, params, tool)
		if err != nil {
			a.logger.Error("Permission request failed", "tool", toolName, "error", err)
			return nil, fmt.Errorf("failed to request permission: %w", err)
		}

		if !granted {
			a.logger.Info("Permission denied for tool execution", "tool", toolName)
			return nil, tools.ErrPermissionDenied
		}
	}

	if a.config.SessionRecorder != nil {
		a.config.SessionRecorder.Record(session.EventToolStart, map[string]interface{}{
			"tool":   toolName,
			"params": params,
		})
	}

	startTime := time.Now()
	result, err := tool.Execute(ctx, params)
	duration := time.Since(startTime)

	if err != nil {
		if a.config.SessionRecorder != nil {
			a.config.SessionRecorder.Record(session.EventToolResult, map[string]interface{}{
				"tool":     toolName,
				"error":    err.Error(),
				"duration": duration.String(),
			})
		}
		a.logger.Error("Tool execution failed", "tool", toolName, "error", err, "duration", duration.String())
		return nil, fmt.Errorf("%w: %s: %v", ErrToolExecutionFailed, toolName, err)
	}

	if a.config.SessionRecorder != nil {
		a.config.SessionRecorder.Record(session.EventToolResult, map[string]interface{}{
			"tool":     toolName,
			"result":   result,
			"duration": duration.String(),
		})
	}

	a.logger.Info("Tool executed successfully", "tool", toolName, "duration", duration.String())
	return result, nil
}

func (a *agent) AddSystemMessage(message string) {
	a.context.AddSystemMessage(message)
}

func (a *agent) ReplaceSystemMessage(message string) {
	a.context.ReplaceSystemMessage(message)
}

func (a *agent) AddUserMessage(message string) {
	a.context.AddUserMessage(message)
}

func (a *agent) AddAssistantMessage(message string) {
	a.context.AddAssistantMessage(message)
}

func (a *agent) AddMessage(msg Message) {
	a.context.AddMessage(msg)
}

func (a *agent) GetMessages() []Message {
	return a.context.GetMessages()
}

func (a *agent) ClearContext() {
	a.logger.Info("Clearing conversation context (keeping system messages)")
	a.context.ClearContext()
}

func (a *agent) ClearLastUserMessage() {
	a.context.ClearLastUserMessage()
}

func (a *agent) SetModel(model string) {
	a.logger.Info("Changing model", "from", a.config.Model, "to", model)
	a.config.Model = model
}

func (a *agent) SetPlannerModel(model string) {
	a.logger.Info("Changing planner model", "from", a.config.PlannerModel, "to", model)
	a.config.PlannerModel = model
}

func (a *agent) SetSubAgentModel(model string) {
	a.logger.Info("Changing sub-agent model", "from", a.config.SubAgentModel, "to", model)
	a.config.SubAgentModel = model
}

func (a *agent) SetSummariserModel(model string) {
	a.logger.Info("Changing summariser model", "from", a.config.SummariserModel, "to", model)
	a.config.SummariserModel = model
}

func (a *agent) SetTemperature(temperature float64) {
	a.logger.Info("Changing temperature", "from", a.config.Temperature, "to", temperature)
	a.config.Temperature = temperature
}

func (a *agent) SetMaxTokens(maxTokens int) {
	a.logger.Info("Changing max tokens", "from", a.config.MaxTokens, "to", maxTokens)
	a.config.MaxTokens = maxTokens
}
