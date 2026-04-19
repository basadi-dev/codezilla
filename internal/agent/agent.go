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
	SetFastModel(model string)
	SetHeavyModel(model string)
	GetModelForTier(tier RequestTier) string
	SetTemperature(temperature float64)
	SetReasoningEffort(effort string)
	SetMaxTokens(maxTokens int)
	SetSessionRecorder(recorder *session.Recorder)
	SetAutoRoute(enabled bool)
	ContextStats() (msgCount int, currentTokens int, maxTokens int)
	Clone() Agent
	ClearTools() // strips all tools from this agent so it does plain LLM completions
}

type Config struct {
	Model           string
	Provider        string
	MaxTokens       int
	MaxIterations   int
	Temperature     float64
	ReasoningEffort string
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
	// OnContextSummarizing is called when preFlight trimming triggers a fast-model summarization.
	OnContextSummarizing func()
	// OnLLMStreamEnd is called immediately after a text/tool stream completes natively, before parsing.
	OnLLMStreamEnd func()
	// OnLLMUsage is called after each LLM call with the per-turn and cumulative session token usage,
	// plus a per-model breakdown of tokens used in this turn (model name → usage with in/out split).
	// Providers that don't return usage data will not trigger this callback.
	OnLLMUsage func(turn TokenUsage, session TokenUsage, turnModels map[string]TokenUsage)

	// Loop detection: stops the run loop if a tool is called with identical args consecutively.
	// 0 = use defaults (window=10, max_repeat=3).
	LoopDetectWindow    int
	LoopDetectMaxRepeat int

	FastModel              string
	HeavyModel             string
	AutoRoute              bool
	ThinkCompressThreshold int
	SlidingWindowSize      int // number of recent non-system messages to keep verbatim (0 = disabled)
	SessionRecorder        *session.Recorder
	// OnModelRouted is called when the router selects a model for the current request.
	// model is the selected model name, reason is a short human-readable explanation.
	OnModelRouted func(model, reason string)

	// Auto-verification: run build/lint after file-modifying tool calls.
	AutoVerify       bool     // enable post-edit verification
	VerifyCommands   []string // custom verify commands (empty = auto-detect from project type)
	WorkingDirectory string   // project root for running verify commands
	MaxVerifyRetries int      // max retries on verification failure (0 = use default of 2)
	// OnVerifyFailed is called when post-edit verification fails.
	// errors is the list of verification error messages, retryNum is the current retry (0 = first failure).
	OnVerifyFailed func(errors []string, retryNum int)
	// OnVerifyPassed is called when post-edit verification succeeds.
	OnVerifyPassed func()
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
	router        *ModelRouter
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

	// Initialise the model router when auto-routing is enabled
	if config.AutoRoute && (config.FastModel != "" || config.HeavyModel != "") {
		agent.router = NewModelRouter(true, config.FastModel, config.Model, config.HeavyModel)
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

func (a *agent) ContextStats() (int, int, int) {
	msgCount, estimatedTokens := a.context.ContextStats()
	return msgCount, estimatedTokens, a.context.MaxTokens
}

func (a *agent) ClearLastUserMessage() {
	a.context.ClearLastUserMessage()
}

func (a *agent) SetModel(model string) {
	a.logger.Info("Changing model", "from", a.config.Model, "to", model)
	a.config.Model = model
}

func (a *agent) GetModelForTier(tier RequestTier) string {
	if a.router == nil {
		return a.config.Model
	}
	return a.router.ModelForTier(tier)
}

func (a *agent) SetFastModel(model string) {
	a.logger.Info("Changing fast model", "from", a.config.FastModel, "to", model)
	a.config.FastModel = model
	if a.router != nil {
		a.router.FastModel = model
	} else if model != "" && a.config.AutoRoute {
		a.router = NewModelRouter(true, model, a.config.Model, a.config.HeavyModel)
	}
}

func (a *agent) SetHeavyModel(model string) {
	a.logger.Info("Changing heavy model", "from", a.config.HeavyModel, "to", model)
	a.config.HeavyModel = model
	if a.router != nil {
		a.router.HeavyModel = model
	} else if model != "" && a.config.AutoRoute {
		a.router = NewModelRouter(true, a.config.FastModel, a.config.Model, model)
	}
}

func (a *agent) SetAutoRoute(enabled bool) {
	a.logger.Info("Changing auto-route", "from", a.config.AutoRoute, "to", enabled)
	a.config.AutoRoute = enabled
	if enabled && (a.config.FastModel != "" || a.config.HeavyModel != "") {
		if a.router == nil {
			a.router = NewModelRouter(true, a.config.FastModel, a.config.Model, a.config.HeavyModel)
		} else {
			a.router.Enabled = true
		}
	} else if a.router != nil {
		a.router.Enabled = false
	}
}

func (a *agent) SetTemperature(temperature float64) {
	a.logger.Info("Changing temperature", "from", a.config.Temperature, "to", temperature)
	a.config.Temperature = temperature
}

func (a *agent) SetReasoningEffort(effort string) {
	a.logger.Info("Changing reasoning effort", "from", a.config.ReasoningEffort, "to", effort)
	a.config.ReasoningEffort = effort
}

func (a *agent) SetMaxTokens(maxTokens int) {
	a.logger.Info("Changing max tokens", "from", a.config.MaxTokens, "to", maxTokens)
	a.config.MaxTokens = maxTokens
	a.context.SetMaxTokens(maxTokens)
}

func (a *agent) SetSessionRecorder(recorder *session.Recorder) {
	a.logger.Info("Changing session recorder")
	a.config.SessionRecorder = recorder
}

func (a *agent) Clone() Agent {
	// Provide a copy of the config so agents don't mutate the global config
	newConfig := *a.config

	// Null out UI-bound callbacks. These are wired to the single-threaded TUI
	// and will produce garbled output if multiple parallel agents fire them
	// concurrently. The multi-agent orchestrator should provide its own
	// callbacks if it needs to aggregate worker progress.
	newConfig.OnToolExecution = nil
	newConfig.OnToolPreparing = nil
	newConfig.OnLLMCall = nil
	newConfig.OnLLMStreamEnd = nil
	newConfig.OnLLMUsage = nil
	newConfig.OnModelRouted = nil
	newConfig.OnContextSummarizing = nil
	newConfig.SessionRecorder = nil

	newAgent := &agent{
		config:        &newConfig,
		context:       a.context.Clone(),
		llmClient:     a.llmClient,     // Shared thread-safe client
		toolRegistry:  a.toolRegistry,  // Shared thread-safe registry
		logger:        a.logger,
		permissionMgr: a.permissionMgr,
	}

	if a.router != nil {
		newRouter := *a.router
		newAgent.router = &newRouter
	}

	return newAgent
}

func (a *agent) ClearTools() {
	a.toolRegistry = tools.NewToolRegistry()
}
