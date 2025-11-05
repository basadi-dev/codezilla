package core

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"codezilla/internal/agent"
	"codezilla/internal/cli"
	"codezilla/internal/storage"
	"codezilla/internal/tools"
	"codezilla/internal/ui"
	"codezilla/llm"
	"codezilla/llm/ollama"
	"codezilla/pkg/logger"
)

// App represents the core application logic, independent of UI
type App struct {
	config     *cli.Config
	logger     *logger.Logger
	agent      agent.Agent
	provider   llm.Provider
	contextMgr *cli.SimpleContextManager
	tools      tools.ToolRegistry
	ui         ui.UI
}

// NewApp creates a new application instance
func NewApp(config *cli.Config, ui ui.UI) (*App, error) {
	// Initialize logger
	logConfig := logger.Config{
		LogFile:  config.LogFile,
		LogLevel: config.LogLevel,
		Silent:   config.LogSilent,
	}
	log, err := logger.New(logConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize logger: %w", err)
	}

	// Initialize LLM client with authentication
	clientOptions := []func(*ollama.ClientOptions){
		ollama.WithBaseURL(config.OllamaURL),
	}

	// Add authentication if configured
	if config.OllamaAPIKey != "" {
		clientOptions = append(clientOptions, ollama.WithAPIKey(config.OllamaAPIKey))
	} else if config.OllamaUsername != "" && config.OllamaPassword != "" {
		clientOptions = append(clientOptions, ollama.WithBasicAuth(config.OllamaUsername, config.OllamaPassword))
	}

	if len(config.OllamaHeaders) > 0 {
		clientOptions = append(clientOptions, ollama.WithHeaders(config.OllamaHeaders))
	}

	// Create LLM provider (Ollama adapter)
	provider := ollama.NewOllamaAdapter(clientOptions...)

	// Test connection
	ctx := context.Background()
	ui.Print("Checking Ollama connection... ")
	_, err = provider.ListModels(ctx)
	if err != nil {
		ui.Error("Failed")
		return nil, fmt.Errorf("cannot connect to Ollama at %s: %w", config.OllamaURL, err)
	}
	ui.Success("Connected")

	// Initialize tool registry
	toolRegistry := tools.NewToolRegistry()

	// Create permission manager with interactive callback
	permissionMgr := tools.NewPermissionManager(func(ctx context.Context, request tools.PermissionRequest) (tools.PermissionResponse, error) {
		// Hide thinking indicator before showing permission request
		ui.HideThinking()

		// Show permission request to user
		ui.Warning("\n🔧 Tool Permission Request:")
		ui.Print("Tool: %s\n", request.ToolContext.ToolName)
		ui.Print("Description: %s\n", request.Description)
		ui.Print("\n")

		// Ask for permission with a simple prompt
		ui.Print("Allow this action? (y/n/always): ")

		// Read response directly without the usual prompt
		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() {
			return tools.PermissionResponse{Granted: false}, fmt.Errorf("failed to read response")
		}
		response := scanner.Text()

		response = strings.ToLower(strings.TrimSpace(response))

		// Show thinking indicator again after permission
		ui.ShowThinking()

		switch response {
		case "y", "yes":
			return tools.PermissionResponse{Granted: true, RememberMe: false}, nil
		case "always", "a":
			return tools.PermissionResponse{Granted: true, RememberMe: true}, nil
		default:
			return tools.PermissionResponse{Granted: false, RememberMe: false}, nil
		}
	})

	// Apply permission levels from config
	for toolName, permString := range config.ToolPermissions {
		var level tools.PermissionLevel
		switch permString {
		case "never_ask":
			level = tools.NeverAsk
		case "always_ask":
			level = tools.AlwaysAsk
		case "ask_once":
			level = tools.AskOnce
		default:
			level = tools.AlwaysAsk
		}
		permissionMgr.SetDefaultPermissionLevel(toolName, level)
	}

	// Register tools after permission manager is configured
	registerTools(toolRegistry, provider, config, log, permissionMgr)

	// Initialize agent with new orchestrator adapter
	agentConfig := &agent.Config{
		Model:         config.DefaultModel,
		SystemPrompt:  config.SystemPrompt,
		Temperature:   float64(config.Temperature),
		MaxTokens:     config.MaxTokens,
		Logger:        log,
		ToolRegistry:  toolRegistry,
		PermissionMgr: permissionMgr,
	}
	agentInstance := agent.NewOrchestratorAdapter(provider, toolRegistry, permissionMgr, agentConfig)

	// Initialize context manager
	contextMgr := cli.NewSimpleContextManager(10)

	return &App{
		config:     config,
		logger:     log,
		agent:      agentInstance,
		provider:   provider,
		contextMgr: contextMgr,
		tools:      toolRegistry,
		ui:         ui,
	}, nil
}

// Close cleans up application resources
func (app *App) Close() error {
	if app.logger != nil {
		return app.logger.Close()
	}
	return nil
}

// Run starts the main application loop
func (app *App) Run(ctx context.Context) error {
	// Show UI elements
	app.ui.Clear()
	app.ui.ShowBanner()
	app.ui.ShowWelcome(app.config.DefaultModel, app.config.OllamaURL, app.config.RetainContext)

	// Main loop
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
			// Read input (single-line, Enter submits immediately)
			input, err := app.ui.ReadLine()
			if err != nil {
				app.ui.Info("Goodbye!")
				return nil
			}

			input = strings.TrimSpace(input)
			if input == "" {
				continue
			}

			// Handle commands
			if strings.HasPrefix(input, "/") {
				if app.handleCommand(ctx, input) {
					return nil
				}
				continue
			}

			// Process with AI
			if err := app.processInput(ctx, input); err != nil {
				app.ui.Error("Failed to process: %v", err)
			}
		}
	}
}

// processInput processes user input with the AI
func (app *App) processInput(ctx context.Context, input string) error {
	// Show thinking indicator
	app.ui.ShowThinking()
	defer app.ui.HideThinking()

	// Add to context if enabled
	if app.config.RetainContext {
		app.contextMgr.AddMessage("User", input)
		app.agent.AddUserMessage(input)
	} else {
		app.agent.ClearContext()
		app.agent.AddUserMessage(input)
	}

	// Process with agent
	response, err := app.agent.ProcessMessage(ctx, input)
	if err != nil {
		return err
	}

	// Add response to context
	if app.config.RetainContext {
		app.contextMgr.AddMessage("Assistant", response)
		app.agent.AddAssistantMessage(response)
	}

	// Display response
	app.ui.ShowResponse(response)

	return nil
}

// handleCommand processes commands
func (app *App) handleCommand(ctx context.Context, cmd string) bool {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return false
	}

	switch parts[0] {
	case "/help", "/h":
		app.ui.ShowHelp()

	case "/exit", "/quit", "/q":
		app.ui.Success("Goodbye!")
		return true

	case "/clear", "/c":
		app.ui.Clear()
		app.ui.ShowBanner()

	case "/models":
		app.showModels(ctx)

	case "/model":
		if len(parts) > 1 {
			app.changeModel(ctx, strings.Join(parts[1:], " "))
		} else {
			app.ui.Info("Current model: %s", app.config.DefaultModel)
		}

	case "/context":
		app.handleContextCommand(parts)

	case "/tools":
		app.showTools()

	case "/reset":
		app.contextMgr.Clear()
		app.agent.ClearContext()
		app.ui.Success("Conversation reset")

	default:
		app.ui.Warning("Unknown command: %s", parts[0])
		app.ui.Info("Type /help for available commands")
	}

	return false
}

// showModels displays available models
func (app *App) showModels(ctx context.Context) {
	models, err := app.provider.ListModels(ctx)
	if err != nil {
		app.ui.Error("Failed to list models: %v", err)
		return
	}

	var modelNames []string
	for _, model := range models {
		modelNames = append(modelNames, model.Name)
	}

	app.ui.ShowModels(modelNames, app.config.DefaultModel)
}

// changeModel changes the current model
func (app *App) changeModel(ctx context.Context, modelName string) {
	models, err := app.provider.ListModels(ctx)
	if err != nil {
		app.ui.Error("Failed to list models: %v", err)
		return
	}

	found := false
	for _, model := range models {
		if model.Name == modelName {
			found = true
			break
		}
	}

	if !found {
		app.ui.Error("Model '%s' not found", modelName)
		app.ui.Info("Use /models to see available models")
		return
	}

	app.config.DefaultModel = modelName
	app.agent.SetModel(modelName)
	app.ui.Success("Switched to model: %s", modelName)
}

// handleContextCommand handles context-related commands
func (app *App) handleContextCommand(parts []string) {
	if len(parts) > 1 {
		switch parts[1] {
		case "on":
			app.config.RetainContext = true
			app.ui.Success("Context retention enabled")
		case "off":
			app.config.RetainContext = false
			app.contextMgr.Clear()
			app.agent.ClearContext()
			app.ui.Success("Context retention disabled and cleared")
		case "clear":
			app.contextMgr.Clear()
			app.agent.ClearContext()
			app.ui.Success("Context cleared")
		case "show":
			app.ui.ShowContext(app.contextMgr.GetContext())
		default:
			app.ui.Warning("Usage: /context [on|off|clear|show]")
		}
	} else {
		status := "disabled"
		if app.config.RetainContext {
			status = "enabled"
		}
		app.ui.Info("Context retention is %s", status)
	}
}

// showTools displays available tools
func (app *App) showTools() {
	var toolInfos []ui.ToolInfo

	for _, tool := range app.tools.ListTools() {
		toolName := tool.Name()
		perm := app.config.ToolPermissions[toolName]
		if perm == "" {
			perm = "always_ask"
		}

		toolInfos = append(toolInfos, ui.ToolInfo{
			Name:        toolName,
			Description: tool.Description(),
			Permission:  perm,
		})
	}

	app.ui.ShowTools(toolInfos)
}

// registerTools registers all available tools
func registerTools(registry tools.ToolRegistry, provider llm.Provider, config *cli.Config, logger *logger.Logger, permissionMgr tools.ToolPermissionManager) {
	// File operation tools
	registry.RegisterTool(tools.NewFileReadTool())
	registry.RegisterTool(tools.NewFileWriteTool())
	registry.RegisterTool(tools.NewListFilesTool())

	// Create analyzer factory and register analyzer tool
	llmAdapter := NewLLMProviderAdapter(provider)
	analyzerFactory := tools.NewAnalyzerFactory(llmAdapter, logger)

	// Register the analyzer (formerly V2)
	registry.RegisterTool(analyzerFactory.CreateProjectScanAnalyzer())

	// Set default permissions for project scanning tool to never ask (always allow)
	// This tool is safe to run automatically as it only reads files without modifying anything
	permissionMgr.SetDefaultPermissionLevel("projectScanAnalyzer", tools.NeverAsk)

	registry.RegisterTool(tools.NewExecuteTool(30))

	// Todo management tools with dependency injection
	// Determine data directory
	homeDir, _ := os.UserHomeDir()
	dataDir := filepath.Join(homeDir, ".codezilla", "todos")

	// Create repository and manager
	todoRepo := storage.NewFileRepository(dataDir)
	todoManager := tools.NewTodoManagerV2(todoRepo)

	// Register todo tools
	for _, tool := range tools.GetTodoToolsV2(todoManager) {
		registry.RegisterTool(tool)
	}
}
