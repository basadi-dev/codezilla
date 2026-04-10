package core

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"codezilla/internal/agent"
	"codezilla/internal/config"
	"codezilla/internal/core/llm"
	"codezilla/internal/tools"
	"codezilla/internal/ui"
	"codezilla/pkg/logger"
)

// App represents the core application logic, independent of UI
type App struct {
	config    *config.Config
	logger    *logger.Logger
	agent     agent.Agent
	llmClient *llm.Client
	tools     tools.ToolRegistry
	ui        ui.UI
}

// NewApp creates a new application instance
func NewApp(cfg *config.Config, ui ui.UI) (*App, error) {
	// Initialize logger
	logConfig := logger.Config{
		LogFile:  cfg.LogFile,
		LogLevel: cfg.LogLevel,
		Silent:   cfg.LogSilent,
	}
	log, err := logger.New(logConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize logger: %w", err)
	}

	// Initialize LLM factory
	llmClient := llm.NewClient(cfg)

	// Test connection
	ctx := context.Background()
	ui.Info("Connecting to %s…", cfg.LLM.Provider)
	_, err = llmClient.ListModels(ctx, cfg.LLM.Provider)
	if err != nil {
		ui.Error("Cannot connect to provider %s: %v", cfg.LLM.Provider, err)
		return nil, fmt.Errorf("cannot connect to provider %s: %w", cfg.LLM.Provider, err)
	}
	ui.Success("Connected to %s", cfg.LLM.Provider)

	// Initialize tool registry
	toolRegistry := tools.NewToolRegistry()

	// Create permission manager with interactive callback
	permissionMgr := tools.NewPermissionManager(func(ctx context.Context, request tools.PermissionRequest) (tools.PermissionResponse, error) {
		// Hide thinking indicator before showing permission request
		ui.HideThinking()

		// Show permission request to user
		ui.Warning("🔐 Tool Permission Request")
		ui.Print("   Tool:        %s\n", request.ToolContext.ToolName)
		ui.Print("   Description: %s\n", request.Description)
		ui.Print("\n")

		// Ask for permission with a simple prompt
		ui.Print("   Allow? (y/n/always): ")
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
	for toolName, permString := range cfg.ToolPermissions {
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
	registerTools(toolRegistry, llmClient, cfg, log, permissionMgr)

	// Initialize agent
	agentConfig := &agent.Config{
		Model:         cfg.LLM.Models.Default,
		PlannerModel:  cfg.LLM.Models.Planner,
		Provider:      cfg.LLM.Provider,
		SystemPrompt:  cfg.SystemPrompt,
		Temperature:   float64(cfg.Temperature),
		MaxTokens:     cfg.MaxTokens,
		MaxIterations: cfg.MaxIterations,
		Logger:        log,
		LLMClient:     llmClient,
		ToolRegistry:  toolRegistry,
		PermissionMgr: permissionMgr,
		AutoPlan:      false, // disabled by default; users can opt-in via config
		OnToolExecution: func(toolName string, params map[string]interface{}) {
			// Clear spinner before printing so messages don't collide on the same line
			ui.HideThinking()

			detail := ""
			switch toolName {
			case "fileRead":
				if fp, ok := params["file_path"].(string); ok {
					detail = fp
				}
			case "fileWrite":
				if fp, ok := params["file_path"].(string); ok {
					detail = fp
				}
			case "listFiles":
				if dir, ok := params["directory"].(string); ok {
					detail = dir
				} else if dir, ok := params["path"].(string); ok {
					detail = dir
				}
			case "execute":
				if cmd, ok := params["command"].(string); ok {
					if len(cmd) > 60 {
						cmd = cmd[:60] + "..."
					}
					detail = cmd
				}
			}

			if detail != "" {
				ui.Info("🔧 Using tool: %s (%s)", toolName, detail)
			} else {
				ui.Info("🔧 Using tool: %s", toolName)
			}

			// Resume spinner after printing
			ui.ShowThinking()
		},
	}
	agentInstance := agent.NewAgent(agentConfig)

	return &App{
		config:    cfg,
		logger:    log,
		agent:     agentInstance,
		llmClient: llmClient,
		tools:     toolRegistry,
		ui:        ui,
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
	
	connStr := app.config.LLM.Provider
	if app.config.LLM.Provider == "ollama" {
		connStr = app.config.LLM.Ollama.BaseURL
	}
	app.ui.ShowWelcome(app.config.LLM.Models.Default, connStr, app.config.RetainContext)

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

// processInput processes user input with the AI using streaming.
func (app *App) processInput(ctx context.Context, input string) error {
	for {
		// Show thinking indicator while the connection is being established
		app.ui.ShowThinking()

		// If context retention is disabled, clear previous conversation
		if !app.config.RetainContext {
			app.agent.ClearContext()
		}

		// Process with agent in background
		var finalResponse string
		var agentErr error
		done := make(chan struct{})

		go func() {
			defer close(done)

			fr, err := app.agent.ProcessMessageStream(ctx, input, func(token string) {
				// We consume the stream but do not display it since we require the
				// full response to render markdown efficiently with glamour.
			})
			finalResponse = fr
			agentErr = err
		}()

		// Wait for agent to finish
		<-done
		app.ui.HideThinking()

		if agentErr == nil && finalResponse != "" {
			app.ui.ShowResponse(finalResponse)
		}

		if agentErr != nil {
			// Show the error clearly to the user
			app.ui.Error("LLM request failed: %v", agentErr)
			app.ui.Print("\nRetry? (y/n): ")

			retryInput, err := app.ui.ReadLine()
			if err != nil {
				return nil
			}
			retryInput = strings.ToLower(strings.TrimSpace(retryInput))
			if retryInput == "y" || retryInput == "yes" {
				// Remove the failed user message from context so it doesn't double-up on retry
				app.agent.ClearLastUserMessage()
				continue
			}
			return nil
		}

		// Final response is already persisted by the agent's context
		_ = finalResponse
		return nil
	}
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
			app.ui.Info("Current model: %s", app.config.LLM.Models.Default)
		}

	case "/context":
		app.handleContextCommand(parts)

	case "/tools":
		app.showTools()

	case "/reset":
		app.agent.ClearContext()
		app.ui.Success("Conversation reset")

	case "/save":
		if len(parts) < 2 {
			app.ui.Warning("Usage: /save <filename>")
		} else {
			app.saveConversation(parts[1])
		}

	case "/load":
		if len(parts) < 2 {
			app.ui.Warning("Usage: /load <filename>")
		} else {
			app.loadConversation(parts[1])
		}

	default:
		app.ui.Warning("Unknown command: %s", parts[0])
		app.ui.Info("Type /help for available commands")
	}

	return false
}

// saveConversation serialises the conversation context to a JSON file.
func (app *App) saveConversation(filename string) {
	messages := app.agent.GetMessages()
	if len(messages) == 0 {
		app.ui.Warning("No conversation to save.")
		return
	}

	// Build a serialisable structure from agent messages
	type savedMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	saved := make([]savedMessage, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == agent.RoleSystem {
			continue // Don't persist system prompts
		}
		saved = append(saved, savedMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}

	if len(saved) == 0 {
		app.ui.Warning("No conversation to save.")
		return
	}

	raw, err := json.MarshalIndent(saved, "", "  ")
	if err != nil {
		app.ui.Error("Failed to serialise conversation: %v", err)
		return
	}

	if err := os.WriteFile(filename, raw, 0644); err != nil {
		app.ui.Error("Failed to write file: %v", err)
		return
	}

	app.ui.Success("Conversation saved to %s", filename)
}

// loadConversation reads a JSON file and restores the conversation context.
func (app *App) loadConversation(filename string) {
	raw, err := os.ReadFile(filename)
	if err != nil {
		app.ui.Error("Failed to read file: %v", err)
		return
	}

	type savedMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	var messages []savedMessage
	if err := json.Unmarshal(raw, &messages); err != nil {
		// Try legacy format (flat conversation string)
		var legacyData map[string]interface{}
		if legacyErr := json.Unmarshal(raw, &legacyData); legacyErr == nil {
			if conv, ok := legacyData["conversation"].(string); ok && conv != "" {
				app.agent.ClearContext()
				app.agent.AddSystemMessage("The following is a prior conversation that was loaded from disk:\n\n" + conv)
				app.ui.Success("Conversation loaded from %s (legacy format)", filename)
				return
			}
		}
		app.ui.Error("Failed to parse conversation file: %v", err)
		return
	}

	// Reset context and replay messages
	app.agent.ClearContext()
	for _, msg := range messages {
		switch agent.Role(msg.Role) {
		case agent.RoleUser:
			app.agent.AddUserMessage(msg.Content)
		case agent.RoleAssistant:
			app.agent.AddAssistantMessage(msg.Content)
		}
	}

	app.ui.Success("Conversation loaded from %s (%d messages)", filename, len(messages))
}

// showModels displays available models
func (app *App) showModels(ctx context.Context) {
	models, err := app.llmClient.ListModels(ctx, app.config.LLM.Provider)
	if err != nil {
		app.ui.Error("Failed to list models: %v", err)
		return
	}

	app.ui.ShowModels(models, app.config.LLM.Models.Default)
}

// changeModel changes the current model
func (app *App) changeModel(ctx context.Context, modelName string) {
	models, err := app.llmClient.ListModels(ctx, app.config.LLM.Provider)
	if err != nil {
		app.ui.Error("Failed to list models: %v", err)
		return
	}

	found := false
	for _, model := range models {
		if model == modelName {
			found = true
			break
		}
	}

	if !found {
		// Provide a warning but allow overriding since many remote APIs (like OpenAI) don't strictly validate listed vs available
		app.ui.Warning("Model '%s' not explicitly found in model list, but assigning anyway.", modelName)
	}

	app.config.LLM.Models.Default = modelName
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
			app.agent.ClearContext()
			app.ui.Success("Context retention disabled and cleared")
		case "clear":
			app.agent.ClearContext()
			app.ui.Success("Context cleared")
		case "show":
			messages := app.agent.GetMessages()
			if len(messages) == 0 {
				app.ui.ShowContext("")
			} else {
				var parts []string
				for _, msg := range messages {
					if msg.Role == agent.RoleSystem {
						continue
					}
					parts = append(parts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
				}
				app.ui.ShowContext(strings.Join(parts, "\n"))
			}
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
func registerTools(registry tools.ToolRegistry, llmClient *llm.Client, cfg *config.Config, logger *logger.Logger, permissionMgr tools.ToolPermissionManager) {
	// File operation tools
	registry.RegisterTool(tools.NewFileReadTool())
	registry.RegisterTool(tools.NewFileWriteTool())
	registry.RegisterTool(tools.NewListFilesTool())
	registry.RegisterTool(tools.NewFileEditTool())
	registry.RegisterTool(tools.NewGrepSearchTool())
	registry.RegisterTool(tools.NewFileManageTool())

	// Create analyzer factory and register analyzer tool
	analyzerModel := cfg.LLM.Models.Analyzer
	if analyzerModel == "" {
		analyzerModel = cfg.LLM.Models.Default
	}
	analyzerFactory := tools.NewAnalyzerFactory(llmClient, cfg.LLM.Provider, analyzerModel, logger)

	// Register the analyzer (formerly V2)
	registry.RegisterTool(analyzerFactory.CreateProjectScanAnalyzer())

	// Set default permissions for safe read-only tools
	permissionMgr.SetDefaultPermissionLevel("projectScanAnalyzer", tools.NeverAsk)
	permissionMgr.SetDefaultPermissionLevel("grepSearch", tools.NeverAsk)
	
	// Modifying tools default to asking (they'll be managed by user configs usually)
	permissionMgr.SetDefaultPermissionLevel("fileEdit", tools.AlwaysAsk)
	permissionMgr.SetDefaultPermissionLevel("fileManage", tools.AlwaysAsk)

	registry.RegisterTool(tools.NewExecuteTool(30))

	// Todo management tools
	todoMgr := tools.NewTodoManager()
	for _, tool := range tools.GetTodoTools(todoMgr) {
		registry.RegisterTool(tool)
	}
}
