package core

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/term"

	"codezilla/internal/agent"
	"codezilla/internal/agent/multiagent"
	"codezilla/internal/config"
	"codezilla/internal/core/llm"
	"codezilla/internal/db"
	"codezilla/internal/session"
	"codezilla/internal/skills"
	"codezilla/internal/tools"
	"codezilla/internal/ui"
	"codezilla/pkg/logger"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/charmbracelet/lipgloss"
)

// App represents the core application logic, independent of UI
type App struct {
	config        *config.Config
	logger        *logger.Logger
	agent         agent.Agent
	llmClient     *llm.Client
	tools         tools.ToolRegistry
	ui            ui.UI
	skillReg      *skills.Registry
	cachedModels  []string // updated by /models, used for Tab completion
	sessionRecord *session.Recorder
	db            *db.DB // persistent SQLite store

	// Token usage & logic tracking
	lastTurnUsage agent.TokenUsage
	sessionUsage  agent.TokenUsage
	routedModel   string
	turnModels    map[string]agent.TokenUsage // per-model token breakdown for the last turn

	// Task cancellation tracking
	taskCancel context.CancelFunc
	cancelMu   sync.Mutex
}

// NewApp creates a new application instance
func NewApp(cfg *config.Config, ui ui.UI, database *db.DB) (*App, error) {
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

	// Setup Session tracking directory and file
	if cfg.SessionEventsDir != "" {
		if err := os.MkdirAll(cfg.SessionEventsDir, 0755); err != nil {
			log.Warn("Failed to create SessionEventsDir", "error", err)
		}
	}
	sessionPath := ""
	isResuming := false
	if cfg.SessionEventsDir != "" {
		if cfg.ResumeSessionID != "" {
			name := cfg.ResumeSessionID
			if !strings.HasSuffix(name, ".jsonl") {
				name += ".jsonl"
			}
			sessionPath = filepath.Join(cfg.SessionEventsDir, name)
			isResuming = true
		} else {
			b := make([]byte, 4)
			rand.Read(b)
			shortUUID := hex.EncodeToString(b)
			sessionPath = filepath.Join(cfg.SessionEventsDir, fmt.Sprintf("session_%s_%s.jsonl", time.Now().Format("20060102_150405"), shortUUID))
		}
	}

	sessionRecord, err := session.NewRecorder(sessionPath, log)
	if err != nil {
		log.Warn("Failed to initialize session recorder", "error", err)
	}

	// Initialize LLM factory
	llmClient := llm.NewClient(cfg)

	// Test connection
	ctx := context.Background()
	ui.Info("Connecting to %s…", cfg.LLM.Provider)
	models, err := llmClient.ListModels(ctx, cfg.LLM.Provider)
	if err != nil {
		ui.Error("Cannot connect to provider %s: %v", cfg.LLM.Provider, err)
		return nil, fmt.Errorf("cannot connect to provider %s: %w", cfg.LLM.Provider, err)
	}
	ui.Success("Connected to %s", cfg.LLM.Provider)

	// Initialize tool registry
	toolRegistry := tools.NewToolRegistry()

	// Initialize skills registry
	skillRegistry := skills.NewRegistry()
	if err := skillRegistry.LoadFromDir(cfg.Skills.Dir); err != nil {
		// Non-fatal: log and continue
		log.Warn("Skills load warning", "error", err)
	}
	skillRegistry.AutoActivate(cfg.Skills.ActiveSkills)

	// Create permission manager with interactive callback
	var permissionMgr tools.ToolPermissionManager
	permissionMgr = tools.NewPermissionManager(func(ctx context.Context, request tools.PermissionRequest) (tools.PermissionResponse, error) {
		// Hide thinking indicator before showing permission request
		ui.HideThinking()

		// Show permission request to user
		ui.Warning("🔐 Tool Permission Request")
		ui.Print("   Tool:        %s\n", request.ToolContext.ToolName)
		ui.Print("   Description: %s\n", request.Description)
		ui.Print("\n")

		scanner := bufio.NewScanner(os.Stdin)
		for {
			// Ask for permission with a simple prompt
			ui.Print("   Allow? (y/n/always): ")
			if !scanner.Scan() {
				return tools.PermissionResponse{Granted: false}, fmt.Errorf("failed to read response")
			}
			response := strings.ToLower(strings.TrimSpace(scanner.Text()))

			if response == "y" || response == "yes" {
				ui.ShowThinking()
				return tools.PermissionResponse{Granted: true, RememberMe: false}, nil
			} else if response == "n" || response == "no" {
				ui.ShowThinking()
				return tools.PermissionResponse{Granted: false, RememberMe: false}, nil
			} else if response == "a" || strings.HasPrefix(response, "always") {
				ui.ShowThinking()
				if permissionMgr != nil {
					permissionMgr.SetDefaultPermissionLevel(request.ToolContext.ToolName, tools.NeverAsk)
				}
				return tools.PermissionResponse{Granted: true, RememberMe: true}, nil
			}

			ui.Warning("Please answer 'y', 'n', or 'always'")
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
			level = tools.NeverAsk
		}
		permissionMgr.SetDefaultPermissionLevel(toolName, level)
	}

	// Create the todo manager here so onToolExec can close over it
	todoMgr := tools.NewTodoManager()

	// registerTools is called after onToolExec is defined below so that the
	// analyzer's progress reporter can call ui.HideThinking / ui.ShowThinking
	// and avoid interleaving with the spinner.

	// lastToolSummary is read by OnLLMCall to show what the agent just did.
	var lastToolSummary string

	onToolExec := func(toolName string, params map[string]interface{}) {
		// Record tool start in session log
		if sessionRecord != nil {
			sessionRecord.Record(session.EventToolStart, map[string]interface{}{
				"tool":   toolName,
				"params": params,
			})
		}

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
			dir, _ := params["dir"].(string)
			if dir == "" {
				dir, _ = params["directory"].(string)
			}
			if dir == "" {
				dir, _ = params["path"].(string)
			}
			detail = dir
		case "execute":
			if cmd, ok := params["command"].(string); ok {
				if len(cmd) > 60 {
					cmd = cmd[:60] + "..."
				}
				detail = cmd
			}
		case "webSearch":
			if q, ok := params["query"].(string); ok {
				detail = q
			}
		case "fetchURL":
			if u, ok := params["url"].(string); ok {
				detail = u
			}
		case "grepSearch":
			dp, ok := params["path"].(string)
			if !ok {
				dp, _ = params["search_path"].(string)
			}
			if dp != "" {
				if q, ok := params["query"].(string); ok {
					detail = fmt.Sprintf(`"%s" in %s`, q, dp)
				} else {
					detail = dp
				}
			}
		case "subAgent":
			if taskStr, ok := params["task"].(string); ok {
				detail = taskStr
				if len(detail) > 40 {
					detail = detail[:37] + "..."
				}
			}
		case "fileEdit":
			fp, _ := params["file_path"].(string)
			if fp == "" {
				fp, _ = params["target_file"].(string)
			}
			detail = fp
		case "fileManage":
			op, _ := params["action"].(string)
			if op == "" {
				op, _ = params["operation"].(string)
			}
			src, _ := params["path"].(string)
			if src == "" {
				src, _ = params["source_path"].(string)
			}
			dst, _ := params["destination_path"].(string)
			if op != "" {
				if dst != "" {
					detail = fmt.Sprintf("%s: %s -> %s", op, src, dst)
				} else {
					detail = fmt.Sprintf("%s: %s", op, src)
				}
			}
		case "todoCreate":
			name, _ := params["name"].(string)
			desc, _ := params["description"].(string)
			theme := ui.GetTheme()

			// Build plan box content
			var lines []string

			// Title line
			titleStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.AdaptiveColor{Light: "#1E66F5", Dark: "#7AA2F7"})
			plannerModel := cfg.LLM.Models.Heavy
			if plannerModel == "" {
				plannerModel = cfg.LLM.Models.Default
			}
			modelSuffix := lipgloss.NewStyle().Faint(true).Render(" · via " + plannerModel)
			if name != "" {
				lines = append(lines, titleStyle.Render("📋 "+name)+modelSuffix)
			} else {
				lines = append(lines, titleStyle.Render("📋 New Plan")+modelSuffix)
			}
			if desc != "" {
				lines = append(lines, theme.StyleDim.Render(desc))
			}
			lines = append(lines, "")

			// Task items
			if items, ok := params["items"].([]interface{}); ok {
				for _, item := range items {
					content := "Untitled Task"
					if strItem, ok := item.(string); ok {
						content = strItem
					} else if itemMap, ok := item.(map[string]interface{}); ok {
						for _, key := range []string{"content", "name", "title", "item", "task", "description", "text", "label"} {
							if c, ok := itemMap[key].(string); ok && c != "" {
								content = c
								break
							}
						}
						// Last-resort: first non-empty string value
						if content == "Untitled Task" {
							for _, v := range itemMap {
								if s, ok := v.(string); ok && s != "" {
									content = s
									break
								}
							}
						}

						priority, _ := itemMap["priority"].(string)
						priorityBadge := ""
						switch priority {
						case "high":
							priorityBadge = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5F87")).Render("▲") + " "
						case "low":
							priorityBadge = lipgloss.NewStyle().Foreground(lipgloss.Color("#626262")).Render("▽") + " "
						}

						if len(content) > 65 {
							content = content[:62] + "..."
						}

						circleStyle := lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#7C7F93", Dark: "#565F89"})
						lines = append(lines, fmt.Sprintf("  %s  %s%s", circleStyle.Render("○"), priorityBadge, content))
					} else {
						// If it's something else, just cast and truncate
						content = fmt.Sprintf("%v", item)
						if len(content) > 65 {
							content = content[:62] + "..."
						}
						circleStyle := lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#7C7F93", Dark: "#565F89"})
						lines = append(lines, fmt.Sprintf("  %s  %s", circleStyle.Render("○"), content))
					}
				}
			}

			// Plain output — no box
			ui.Print("\n")
			for _, l := range lines {
				ui.Print("%s\n", l)
			}
			ui.Print("\n")
			ui.ShowThinking()
			return
		case "todoManage":
			theme := ui.GetTheme()
			statusIcons := map[string]string{
				"pending":     "○",
				"in_progress": "◐",
				"completed":   "●",
				"cancelled":   "⊘",
			}

			// For todoManage, we only want special interactive rendering for "update" actions
			// other actions (list/analyze) return regular text blocks
			if action, _ := params["action"].(string); action == "update" {
				if plan := todoMgr.CurrentPlan(); plan != nil {
					// Re-render the whole list so the user sees live progress
					ui.Print("\n")
					for _, item := range plan.Items {
						icon := statusIcons[item.Status]
						var iconStyle lipgloss.Style
						switch item.Status {
						case "in_progress":
							iconStyle = theme.StyleYellow
						case "completed":
							iconStyle = theme.StyleGreen
						case "cancelled":
							iconStyle = theme.StyleRed
						default:
							iconStyle = theme.StyleDim
						}
						content := item.Content
						if len(content) > 65 {
							content = content[:62] + "..."
						}
						var lineStyle lipgloss.Style
						if item.Status == "completed" || item.Status == "cancelled" {
							lineStyle = theme.StyleDim
						} else {
							lineStyle = lipgloss.NewStyle()
						}
						ui.Print("  %s  %s\n", iconStyle.Render(icon), lineStyle.Render(content))
					}
				}
				ui.Print("\n")
				ui.ShowThinking()
				return
			}
		case "projectScanAnalyzer":
			dir, _ := params["dir"].(string)
			if dir == "" {
				dir, _ = params["directory"].(string)
			}
			uq, _ := params["userQuery"].(string)
			if uq != "" {
				if dir != "" {
					detail = fmt.Sprintf(`"%s" in %s`, uq, dir)
				} else {
					detail = fmt.Sprintf(`"%s"`, uq)
				}
			} else {
				detail = dir
			}
		}

		// Only show tool description for non-obvious tools (skip todo/file noise)
		showDesc := true
		switch toolName {
		case "todoCreate", "todoManage",
			"fileRead", "fileWrite", "fileManage", "listFiles":
			showDesc = false
		}
		var desc string
		if showDesc {
			if tool, ok := toolRegistry.GetTool(toolName); ok {
				desc = tool.Description()
			}
		}

		displayName := toolName
		switch toolName {
		case "fileRead":
			displayName = "📄 Read File"
		case "fileWrite":
			displayName = "📝 Write File"
		case "listFiles":
			displayName = "📂 List Files"
		case "execute":
			displayName = "💻 Run Command"
		case "webSearch":
			displayName = "🌐 Web Search"
		case "fetchURL":
			displayName = "📥 Fetch URL"
		case "grepSearch":
			displayName = "🔍 Search Code"
		case "subAgent":
			displayName = "🤖 Sub-Agent"
		case "multiReplace":
			displayName = "✏️ Edit File"
		case "fileManage":
			displayName = "🏗️  Manage Files"
		case "todoCreate":
			displayName = "📋 Plan"
		case "todoManage":
			action, _ := params["action"].(string)
			switch action {
			case "update":
				displayName = "" // detail carries full info
			case "list":
				displayName = "📋 Tasks"
			case "analyze":
				displayName = "🧠 Analyze Tasks"
			default:
				displayName = "📋 Manage Plan"
			}
		case "projectScanAnalyzer":
			displayName = "🔍 Analyze Project"
		case "repoMapGenerator":
			displayName = "🗺️ Repo Map"
		}

		if displayName == "" {
			// e.g. todoUpdate: just print detail directly
			if detail != "" {
				ui.Info("   %s", detail)
			}
		} else if detail != "" {
			ui.Info("🛠  %s: %s", displayName, detail)
		} else {
			ui.Info("🛠  %s", displayName)
		}

		if desc != "" {
			ui.Info("   ↳ %s", ui.GetTheme().StyleDim.Render(desc))
		}
		ui.Print("\n") // breathing room between tool lines

		// Track a short summary for the spinner status (include detail for context)
		name := displayName
		if name == "" {
			name = toolName
		}
		if detail != "" {
			short := detail
			if len(short) > 40 {
				short = short[:37] + "..."
			}
			lastToolSummary = name + ": " + short
		} else {
			lastToolSummary = name
		}

		// Update the spinner label so the user knows it's no longer "preparing"
		ui.UpdateThinkingStatus(fmt.Sprintf("executing %s", name))

		// Resume spinner after printing
		ui.ShowThinking()
	}

	// Register tools — must be called after onToolExec is defined below so that the
	// analyzer progress reporter can borrow the UI hide/show spinner hooks.
	registerTools(toolRegistry, llmClient, cfg, log, permissionMgr, todoMgr, ui)

	// Add sub-agent tool
	launcher := func(ctx context.Context, task string) (string, error) {
		subModel := cfg.LLM.Models.Default
		ui.Info("\n🤖 Launching Sub-Agent... %s", lipgloss.NewStyle().Faint(true).Render("· "+subModel))
		ui.Info("   Task: %s\n", task)
		subCfg := *cfg
		subCfg.MaxIterations = 15

		subRegistry := tools.NewToolRegistry()
		for _, t := range toolRegistry.ListTools() {
			if t.Name() != "subAgent" {
				subRegistry.RegisterTool(t)
			}
		}

		subAgentConfig := &agent.Config{
			Model:                  subModel,
			Provider:               subCfg.LLM.Provider,
			SystemPrompt:           "You are a sub-agent. Your goal is to solve the specific task provided by the parent agent. Use tools as necessary. Return a clear and concise summary of your findings and completed actions. Task: " + task,
			Temperature:            float64(subCfg.Temperature),
			MaxTokens:              subCfg.MaxTokens,
			MaxIterations:          subCfg.MaxIterations,
			Logger:                 log,
			LLMClient:              llmClient,
			ToolRegistry:           subRegistry,
			PermissionMgr:          permissionMgr,
			OnToolExecution:        onToolExec,
			LoopDetectWindow:       subCfg.LoopDetectWindow,
			LoopDetectMaxRepeat:    subCfg.LoopDetectMaxRepeat,
			ThinkCompressThreshold: subCfg.ThinkCompressThreshold,
		}

		subAgent := agent.NewAgent(subAgentConfig)
		result, err := subAgent.ProcessMessage(ctx, "Execute this task: "+task)
		if err != nil {
			return "", fmt.Errorf("sub-agent execution failed: %w", err)
		}

		ui.Success("🤖 Sub-Agent completed task\n")
		return result, nil
	}

	toolRegistry.RegisterTool(tools.NewSubAgentTool(launcher))
	permissionMgr.SetDefaultPermissionLevel("subAgent", tools.NeverAsk)

	// Pointer indirection: the OnLLMUsage callback below is created before
	// the App struct, so these local aliases get wired to app fields after
	// the App is constructed.
	var appTurnUsage agent.TokenUsage
	var appSessionUsage agent.TokenUsage

	// Initialize agent
	agentConfig := &agent.Config{
		Model:                  cfg.LLM.Models.Default,
		Provider:               cfg.LLM.Provider,
		SystemPrompt:           cfg.SystemPrompt,
		Temperature:            float64(cfg.Temperature),
		ReasoningEffort:        cfg.ReasoningEffort,
		MaxTokens:              cfg.MaxTokens,
		MaxIterations:          cfg.MaxIterations,
		Logger:                 log,
		LLMClient:              llmClient,
		ToolRegistry:           toolRegistry,
		PermissionMgr:          permissionMgr,
		AutoPlan:               cfg.AutoPlan,
		AutoVerify:             cfg.AutoVerify,
		VerifyProfiles:         cfg.VerifyProfiles,
		WorkingDirectory:       cfg.WorkingDirectory,
		OnToolExecution:        onToolExec,
		LoopDetectWindow:       cfg.LoopDetectWindow,
		LoopDetectMaxRepeat:    cfg.LoopDetectMaxRepeat,
		ThinkCompressThreshold: cfg.ThinkCompressThreshold,
		SlidingWindowSize:      cfg.SlidingWindowSize,
		OnVerifyFailed: func(errors []string, retryNum int) {
			ui.Warning("Verification failed (attempt %d). Agent will self-correct.", retryNum)
			for _, err := range errors {
				ui.Print("  %s\n", err)
			}
			ui.ShowThinking()
			ui.UpdateThinkingStatus("analyzing verification errors...")
		},
		OnVerifyPassed: func() {
			ui.Success("System verification passed ✅")
		},
		OnLLMError: func(model string, err error, willRetry bool) {
			ui.HideThinking()
			if willRetry {
				ui.Warning("⚠️  [%s] %v — retrying...", model, err)
				ui.ShowThinking()
				ui.UpdateThinkingStatus("retrying after error...")
			} else {
				ui.Error("❌ [%s] %v", model, err)
			}
		},
		OnLLMCall: func(callNum, msgCount, approxToks int) {
			// Format token count: "21K" or "800"
			tokLabel := fmt.Sprintf("%d", approxToks)
			if approxToks >= 1000 {
				tokLabel = fmt.Sprintf("%.0fK", float64(approxToks)/1000)
			}

			var label string
			if callNum == 1 {
				label = fmt.Sprintf("LLM call #1 · ~%s tokens", tokLabel)
			} else if lastToolSummary != "" {
				label = fmt.Sprintf("LLM call #%d · ~%s tokens · after %s", callNum, tokLabel, lastToolSummary)
			} else {
				label = fmt.Sprintf("LLM call #%d · ~%s tokens", callNum, tokLabel)
			}
			ui.UpdateThinkingStatus(label)
			ui.RestartThinking()
		},
		OnToolPreparing: func(toolName string) {
			display := toolName
			switch toolName {
			case "fileRead":
				display = "📄 Read File"
			case "fileWrite":
				display = "📝 Write File"
			case "listFiles":
				display = "📂 List Files"
			case "execute":
				display = "💻 Run Command"
			case "webSearch":
				display = "🌐 Web Search"
			case "fetchURL":
				display = "📥 Fetch URL"
			case "grepSearch":
				display = "🔍 Search Code"
			case "subAgent":
				display = "🤖 Sub-Agent"
			case "multiReplace":
				display = "✏️ Edit File"
			case "fileManage":
				display = "🏗️  Manage Files"
			case "todoCreate":
				display = "📋 Plan"
			case "todoManage":
				display = "📋 Manage Plan"
			case "projectScanAnalyzer":
				display = "🔍 Analyze Project"
			case "repoMapGenerator":
				display = "🗺️ Repo Map"
			}
			ui.UpdateThinkingStatus(fmt.Sprintf("preparing %s", display))
			ui.ShowThinking()
		},
		OnContextSummarizing: func() {
			ui.UpdateThinkingStatus("summarizing context buffer")
			ui.RestartThinking()
		},
		OnLLMUsage: func(turn agent.TokenUsage, session agent.TokenUsage, turnModels map[string]agent.TokenUsage) {
			appTurnUsage = turn
			appSessionUsage = session

			if appSessionUsage.TotalTokens > 0 {
				tokenInfo := fmt.Sprintf("session: %s", agent.CompactNumber(appSessionUsage.TotalTokens))
				ui.UpdateTokenUsage(tokenInfo)
			}
		},
		OnModelRouted: func(model, reason string) {
			ui.UpdateThinkingStatus(fmt.Sprintf("routed → %s (%s)", model, reason))
		},
		FastModel:       cfg.LLM.Models.Fast,
		HeavyModel:      cfg.LLM.Models.Heavy,
		AutoRoute:       cfg.AutoRoute,
		SessionRecorder: sessionRecord,
	}
	agentInstance := agent.NewAgent(agentConfig)

	// Restore context from session only when the user explicitly chose to resume
	if isResuming && sessionPath != "" {
		events, err := session.LoadEvents(sessionPath)
		if err == nil {
			var currentAssistantResponse strings.Builder
			for _, evt := range events {
				if evt.Type == session.EventUIInput {
					// Commit any accumulated assistant response before the next user turn
					if currentAssistantResponse.Len() > 0 {
						agentInstance.AddAssistantMessage(currentAssistantResponse.String())
						currentAssistantResponse.Reset()
					}
					if text, ok := evt.Data["text"].(string); ok {
						agentInstance.AddUserMessage(text)
					}
				} else if evt.Type == session.EventToken {
					if token, ok := evt.Data["token"].(string); ok {
						currentAssistantResponse.WriteString(token)
					}
				} else if evt.Type == session.EventToolStart {
					// Commit assistant response before tool execution boundary
					if currentAssistantResponse.Len() > 0 {
						agentInstance.AddAssistantMessage(currentAssistantResponse.String())
						currentAssistantResponse.Reset()
					}
				}
				// EventToolResult skipped — no reliable way to map back to tool call IDs
			}
			// Final commit if the session ended mid-assistant-response
			if currentAssistantResponse.Len() > 0 {
				agentInstance.AddAssistantMessage(currentAssistantResponse.String())
			}
		}
	}

	resultApp := &App{
		config:        cfg,
		logger:        log,
		agent:         agentInstance,
		llmClient:     llmClient,
		tools:         toolRegistry,
		ui:            ui,
		skillReg:      skillRegistry,
		cachedModels:  models,
		sessionRecord: sessionRecord,
		db:            database,
	}

	// Wire the OnLLMUsage callback to write to app struct fields via closure.
	// The callback was created before the App struct existed, so we patch it
	// by overwriting the agent config callback now that we have a reference.
	agentConfig.OnLLMUsage = func(turn agent.TokenUsage, session agent.TokenUsage, turnModels map[string]agent.TokenUsage) {
		resultApp.lastTurnUsage = turn
		resultApp.sessionUsage = session
		resultApp.turnModels = turnModels

		// Update the UI status bar with live token counts
		if session.TotalTokens > 0 {
			tokenInfo := resultApp.buildContextTokenStr(fmt.Sprintf("%s · session: %s", agent.FormatModelBreakdown(turnModels), agent.CompactNumber(session.TotalTokens)))
			ui.UpdateTokenUsage(tokenInfo)
		} else {
			tokenInfo := resultApp.buildContextTokenStr("")
			ui.UpdateTokenUsage(tokenInfo)
		}
	}
	agentConfig.OnModelRouted = func(model, reason string) {
		resultApp.routedModel = model
		ui.UpdateModel(model)
		ui.UpdateThinkingStatus(fmt.Sprintf("routed → %s (%s)", model, reason))
	}
	
	originalOnLLMCall := agentConfig.OnLLMCall
	agentConfig.OnLLMCall = func(callNum, msgCount, approxToks int) {
		if originalOnLLMCall != nil {
			originalOnLLMCall(callNum, msgCount, approxToks)
		}
		
		// Actively refresh the context % string in the UI during mid-stream tool execution jumps.
		// Since preFlightContextTrim runs immediately before this callback, this mathematically 
		// guarantees the status bar accurately reflects the trimmed budget and latest token total 
		// while the spinner spins for the next LLM call.
		var tokenInfo string
		if resultApp.sessionUsage.TotalTokens > 0 {
			baseInfo := fmt.Sprintf("session: %s", agent.CompactNumber(resultApp.sessionUsage.TotalTokens))
			if len(resultApp.turnModels) > 0 {
				baseInfo = fmt.Sprintf("%s · %s", agent.FormatModelBreakdown(resultApp.turnModels), baseInfo)
			}
			tokenInfo = resultApp.buildContextTokenStr(baseInfo)
		} else {
			tokenInfo = resultApp.buildContextTokenStr("")
		}
		ui.UpdateTokenUsage(tokenInfo)
	}

	// Suppress unused variable warnings for the pre-wiring aliases
	_ = appTurnUsage
	_ = appSessionUsage

	// Wire the persistent idle status footer below the input prompt.
	// This shows model | idle | tokens when the user is at the prompt.
	resultApp.ui.SetPromptFooter(func() string {

		consoleWidth := 80
		if w, _, err := term.GetSize(int(os.Stdout.Fd())); err == nil && w > 0 {
			consoleWidth = w
		}
		if consoleWidth > 1 {
			consoleWidth--
		}

		tokenInfo := "0 tokens"
		if resultApp.sessionUsage.TotalTokens > 0 {
			tokenInfo = agent.FormatModelBreakdown(resultApp.turnModels)
			tokenInfo += fmt.Sprintf(" · session: %s", agent.CompactNumber(resultApp.sessionUsage.TotalTokens))
		}

		separator := lipgloss.NewStyle().Faint(true).Render(strings.Repeat("─", consoleWidth))
		statusLine := lipgloss.NewStyle().Faint(true).Render(fmt.Sprintf("📊 %s", tokenInfo))

		return separator + "\r\n" + statusLine
	})

	return resultApp, nil
}

// wireCompleter builds and installs the Tab-completion callback if the UI
// implements the ui.Completer interface.
func (app *App) wireCompleter() {
	c, ok := app.ui.(ui.Completer)
	if !ok {
		return
	}

	type aliasDef struct {
		Primary string
		Aliases []string
		Desc    string
	}

	// Static top-level slash commands
	staticCmds := []aliasDef{
		{Primary: "/help", Aliases: []string{"/h"}, Desc: "Show this help"},
		{Primary: "/exit", Aliases: []string{"/quit", "/q"}, Desc: "Exit the application"},
		{Primary: "/clear", Aliases: []string{"/c"}, Desc: "Clear the screen"},
		{Primary: "/model ", Desc: "Manage or list models [ls|<name>]"},
		{Primary: "/context ", Desc: "Manage context [on|off|clear|show]"},
		{Primary: "/theme ", Desc: "Change UI color theme [tokyonight|dracula|catppuccin]"},
		{Primary: "/tools", Desc: "Show available tools"},
		{Primary: "/skill ", Desc: "Manage skills"},
		{Primary: "/session ", Desc: "Manage sessions [ls|play <id>]"},
		{Primary: "/reset", Desc: "Reset conversation"},
		{Primary: "/save ", Desc: "Save conversation to JSON file"},
		{Primary: "/load ", Desc: "Load conversation from JSON file"},
		{Primary: "/replay ", Desc: "Replay last user message from a loaded JSON file"},
		{Primary: "/tokens", Desc: "Show session token usage"},
		{Primary: "/router ", Desc: "Model routing [on|off|status]"},
		{Primary: "/thinking ", Desc: "Set reasoning effort [low|medium|high|auto|none|off]"},
	}

	c.SetCompleter(func(line string) []ui.Completion {
		if !strings.HasPrefix(line, "/") {
			return nil
		}

		// /context <sub>
		if strings.HasPrefix(line, "/context ") {
			sub := line[len("/context "):]
			opts := []ui.Completion{
				{Text: "on", Description: "Enable context retention"},
				{Text: "off", Description: "Disable and clear context"},
				{Text: "clear", Description: "Clear current context"},
				{Text: "show", Description: "Show current context"},
			}
			return filterPrefix("/context ", opts, sub)
		}

		// /router <sub>
		if strings.HasPrefix(line, "/router ") {
			sub := line[len("/router "):]
			opts := []ui.Completion{
				{Text: "on", Description: "Enable auto model routing"},
				{Text: "off", Description: "Disable routing (use default model)"},
				{Text: "status", Description: "Show current routing config"},
			}
			return filterPrefix("/router ", opts, sub)
		}

		// /thinking <sub>
		if strings.HasPrefix(line, "/thinking ") {
			sub := line[len("/thinking "):]
			opts := []ui.Completion{
				{Text: "low", Description: "Minimal reasoning (fastest)"},
				{Text: "medium", Description: "Balanced reasoning"},
				{Text: "high", Description: "Deep reasoning (slowest)"},
				{Text: "auto", Description: "Let the model decide"},
				{Text: "none", Description: "Disable extended thinking"},
				{Text: "off", Description: "Clear (use provider default)"},
			}
			return filterPrefix("/thinking ", opts, sub)
		}

		// /model arg
		if strings.HasPrefix(line, "/model ") || strings.HasPrefix(line, "/models ") {
			prefix := "/model "
			if strings.HasPrefix(line, "/models ") {
				prefix = "/models "
			}
			sub := line[len(prefix):]

			// Detect if the user is typing a role name
			isRoleCmd := false
			var activeRoleCmd string
			for _, role := range []string{"default ", "fast ", "heavy "} {
				if strings.HasPrefix(sub, role) {
					isRoleCmd = true
					activeRoleCmd = role
					break
				}
			}

			var opts []ui.Completion
			if isRoleCmd {
				subModel := sub[len(activeRoleCmd):]
				opts = make([]ui.Completion, len(app.cachedModels))
				for i, m := range app.cachedModels {
					opts[i] = ui.Completion{Text: m}
				}
				return filterPrefix(prefix+activeRoleCmd, opts, subModel)
			} else {
				opts = make([]ui.Completion, len(app.cachedModels)+5)
				opts[0] = ui.Completion{Text: "ls", Description: "List all available models"}
				opts[1] = ui.Completion{Text: "list", Description: "List all available models"}
				opts[2] = ui.Completion{Text: "default ", Description: "Set default model"}
				opts[3] = ui.Completion{Text: "fast ", Description: "Set fast model (simple Q&A)"}
				opts[4] = ui.Completion{Text: "heavy ", Description: "Set heavy model (complex tasks)"}
				for i, m := range app.cachedModels {
					opts[i+5] = ui.Completion{Text: m}
				}
				return filterPrefix(prefix, opts, sub)
			}
		}

		// /theme arg
		if strings.HasPrefix(line, "/theme ") {
			sub := line[len("/theme "):]
			themes := ui.AvailableThemes()
			opts := make([]ui.Completion, len(themes))
			for i, t := range themes {
				opts[i] = ui.Completion{Text: t}
			}
			return filterPrefix("/theme ", opts, sub)
		}

		// /skill <sub> or /skills <sub>
		skillPrefix := ""
		if strings.HasPrefix(line, "/skill ") {
			skillPrefix = "/skill "
		} else if strings.HasPrefix(line, "/skills ") {
			skillPrefix = "/skills "
		}
		if skillPrefix != "" {
			sub := line[len(skillPrefix):]
			// /skill info <name> or /skill off <name>
			if strings.HasPrefix(sub, "info ") {
				name := sub[len("info "):]
				return filterPrefix(skillPrefix+"info ", app.skillCompletions(false), name)
			}
			if strings.HasPrefix(sub, "off ") {
				name := sub[len("off "):]
				return filterPrefix(skillPrefix+"off ", app.skillCompletions(true), name)
			}
			// Sub-commands + skill names
			opts := []ui.Completion{
				{Text: "ls", Description: "List all available skills"},
				{Text: "list", Description: "List all available skills"},
				{Text: "reset", Description: "Deactivate all skills"},
				{Text: "info ", Description: "Show skill details"},
				{Text: "off ", Description: "Deactivate a specific skill"},
			}
			opts = append(opts, app.skillCompletions(false)...)
			return filterPrefix(skillPrefix, opts, sub)
		}

		// /session <sub> or /sessions <sub>
		sessionPrefix := ""
		if strings.HasPrefix(line, "/session ") {
			sessionPrefix = "/session "
		} else if strings.HasPrefix(line, "/sessions ") {
			sessionPrefix = "/sessions "
		}
		if sessionPrefix != "" {
			sub := line[len(sessionPrefix):]

			// /session play/replay/resume <name>
			if strings.HasPrefix(sub, "play ") {
				name := sub[len("play "):]
				return filterPrefix(sessionPrefix+"play ", app.sessionCompletions(), name)
			}
			if strings.HasPrefix(sub, "replay ") {
				name := sub[len("replay "):]
				return filterPrefix(sessionPrefix+"replay ", app.sessionCompletions(), name)
			}
			if strings.HasPrefix(sub, "resume ") {
				name := sub[len("resume "):]
				return filterPrefix(sessionPrefix+"resume ", app.sessionCompletions(), name)
			}

			opts := []ui.Completion{
				{Text: "ls", Description: "List all available sessions"},
				{Text: "list", Description: "List all available sessions"},
				{Text: "play ", Description: "Replay a session file"},
				{Text: "replay ", Description: "Replay a session file"},
				{Text: "resume ", Description: "Resume a session file"},
			}
			return filterPrefix(sessionPrefix, opts, sub)
		}

		// Top-level — match static commands by prefix
		var matches []ui.Completion
		for _, cmd := range staticCmds {
			matched := strings.HasPrefix(cmd.Primary, line)
			if !matched {
				for _, a := range cmd.Aliases {
					if strings.HasPrefix(a, line) {
						matched = true
						break
					}
				}
			}

			if matched {
				display := cmd.Primary
				if len(cmd.Aliases) > 0 {
					display = cmd.Primary + ", " + strings.Join(cmd.Aliases, ", ")
				}
				matches = append(matches, ui.Completion{
					Text:        cmd.Primary,
					Display:     display,
					Description: cmd.Desc,
				})
			}
		}
		return matches
	})
}

// fuzzyMatch checks if sub is a subsequence of text, ignoring case.
func fuzzyMatch(text, sub string) bool {
	if sub == "" {
		return true
	}
	t := strings.ToLower(text)
	s := strings.ToLower(sub)

	tIdx, sIdx := 0, 0
	for tIdx < len(t) && sIdx < len(s) {
		if t[tIdx] == s[sIdx] {
			sIdx++
		}
		tIdx++
	}
	return sIdx == len(s)
}

// filterPrefix filters opts by sub-prefix (fuzzy) and prepends commandPrefix to each result.
func filterPrefix(commandPrefix string, opts []ui.Completion, sub string) []ui.Completion {
	var out []ui.Completion
	for _, o := range opts {
		if fuzzyMatch(o.Text, sub) {
			out = append(out, ui.Completion{
				Text:        commandPrefix + o.Text,
				Description: o.Description,
			})
		}
	}
	return out
}

// skillCompletions returns the names and descriptions of skills.
func (app *App) skillCompletions(activeOnly bool) []ui.Completion {
	all := app.skillReg.List()
	comps := make([]ui.Completion, 0, len(all))
	for _, s := range all {
		if activeOnly && !app.skillReg.IsActive(s.Name) {
			continue
		}
		desc := s.Description
		if s.Trigger != "" {
			desc += " (trigger: " + s.Trigger + ")"
		}
		comps = append(comps, ui.Completion{
			Text:        s.Name,
			Description: desc,
		})
	}
	return comps
}

// SessionID returns the ID of the current session.
func (app *App) SessionID() string {
	if app.sessionRecord != nil {
		path := app.sessionRecord.Path()
		if path != "" {
			return filepath.Base(path)
		}
	}
	return ""
}

// PrintSessionSummary prints a formatted summary of the current session on exit.
func (app *App) PrintSessionSummary() {
	sid := app.SessionID()
	if sid == "" {
		return
	}

	app.ui.Print("\n%s\n", lipgloss.NewStyle().Bold(true).Render("📊 Session Summary"))
	app.ui.Print("  %s %s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Width(10).Render("ID:"), sid)
	app.ui.Print("  %s %s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Width(10).Render("Model:"), app.config.LLM.Models.Default)
	if app.sessionUsage.TotalTokens > 0 {
		app.ui.Print("  %s %s\n", lipgloss.NewStyle().Foreground(lipgloss.Color("3")).Width(10).Render("Tokens:"), agent.FormatNumber(app.sessionUsage.TotalTokens))
	}
	app.ui.Print("\n  To resume this session, run:\n")
	app.ui.Print("  %s\n\n", lipgloss.NewStyle().Foreground(lipgloss.AdaptiveColor{Light: "#1E66F5", Dark: "#7AA2F7"}).Render(fmt.Sprintf("codezilla resume %s", sid)))
}

// Close cleans up application resources
func (app *App) Close() error {
	if app.sessionRecord != nil {
		_ = app.sessionRecord.Close()
	}
	if app.db != nil {
		_ = app.db.Close()
	}
	if app.logger != nil {
		return app.logger.Close()
	}
	return nil
}

// Run starts the main application loop
func (app *App) Run(ctx context.Context, rootCancel context.CancelFunc) error {
	// Show UI elements
	app.ui.Clear()
	app.ui.ShowBanner()

	// Setup task-aware signal handler for graceful interruption
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Intercept UI-level interrupts (e.g. BubbleTea intercepting ESC/Ctrl+C in raw mode)
	if intUI, ok := app.ui.(interface{ InterruptChan() <-chan struct{} }); ok {
		go func() {
			for range intUI.InterruptChan() {
				app.cancelMu.Lock()
				if app.taskCancel != nil {
					// We are in the middle of a task -> Cancel it!
					app.taskCancel()
					app.taskCancel = nil
					app.cancelMu.Unlock()
					// Stop the live status ticker before showing the warning
					if stopper, ok := app.ui.(ui.LiveStatusStopper); ok {
						stopper.StopLiveStatus()
					}
					app.ui.Warning("\nTask Cancelled via UI")
				} else {
					app.cancelMu.Unlock()
				}
			}
		}()
	}

	go func() {
		for range sigChan {
			app.cancelMu.Lock()
			if app.taskCancel != nil {
				// We are in the middle of a task -> Cancel it!
				app.taskCancel()
				app.taskCancel = nil
				app.cancelMu.Unlock()
				// Stop the live status ticker before showing the warning
				if stopper, ok := app.ui.(ui.LiveStatusStopper); ok {
					stopper.StopLiveStatus()
				}
				app.ui.Warning("\nTask Cancelled")
			} else {
				// Idling -> Shutdown whole app
				app.cancelMu.Unlock()
				app.ui.Info("\nShutting down...")
				app.PrintSessionSummary()
				if rootCancel != nil {
					rootCancel()
				}
			}
		}
	}()

	// Wire Tab auto-complete now that skill registry is populated
	app.wireCompleter()

	connStr := app.config.LLM.Provider
	if app.config.LLM.Provider == "ollama" {
		connStr = app.config.LLM.Ollama.BaseURL
	}
	type modelledWelcome interface {
		ShowWelcomeWithModels(defaultModel, fastModel, heavyModel, ollamaURL string, contextEnabled bool)
	}
	if mw, ok := app.ui.(modelledWelcome); ok {
		mw.ShowWelcomeWithModels(
			app.config.LLM.Models.Default,
			app.config.LLM.Models.Fast,
			app.config.LLM.Models.Heavy,
			connStr,
			app.config.RetainContext,
		)
	} else {
		app.ui.ShowWelcome(app.config.LLM.Models.Default, connStr, app.config.RetainContext)
	}
	app.ui.SetModel(app.config.LLM.Models.Default)

	if sessionID := app.SessionID(); sessionID != "" {
		app.ui.Info("  📝 Session:          %s", app.ui.GetTheme().StyleCyan.Render(sessionID))

		if app.config.ResumeSessionID != "" {
			app.ui.Print("\n%s\n", lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#2E3440")).Background(lipgloss.AdaptiveColor{Light: "#ACB0BE", Dark: "#A3BE8C"}).Render(" Session Restored "))
			msgs := app.agent.GetMessages()
			for _, msg := range msgs {
				if msg.Role == "user" {
					app.ui.Print("\n%s\n", app.ui.GetTheme().StyleBlue.Render("👤 You:"))
					app.ui.Print("%s\n", msg.Content)
				} else if msg.Role == "assistant" {
					content := msg.Content
					if content != "" {
						app.ui.ShowResponse(sanitizeAgentResponse(content))
					}
				}
			}
			if len(msgs) > 0 {
				app.ui.Print("\n")
			}
		}
	}

	// Main loop
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
			// Read input (single-line, Enter submits immediately)
			input, err := app.ui.ReadLine()
			if err != nil {
				app.ui.Info("\n💡 Goodbye!")
				app.PrintSessionSummary()
				return nil
			}

			input = strings.TrimSpace(input)
			if input == "" {
				continue
			}

			// Handle commands
			if strings.HasPrefix(input, "/") {
				// Check if input matches a skill trigger (exact slash command, no extra text)
				fields := strings.Fields(input)
				if len(fields) == 1 {
					if skill := app.skillReg.FindByTrigger(fields[0]); skill != nil {
						app.handleSkillTrigger(skill)
						continue
					}
				}
				if app.handleCommand(ctx, input) {
					return nil
				}
				continue
			}

			// Process with AI
			if err := app.processInput(ctx, input); err != nil {
				if err != context.Canceled {
					app.ui.Error("Failed to process: %v", err)
				}
			}
		}
	}
}

func (app *App) buildContextTokenStr(baseInfo string) string {
	if app.agent == nil {
		return baseInfo
	}
	_, curCtx, maxCtx := app.agent.ContextStats()
	var ctxStr string
	if maxCtx > 0 {
		pct := (curCtx * 100) / maxCtx
		ctxStr = fmt.Sprintf("ctx: %d%% (%s/%s)", pct, agent.CompactNumber(curCtx), agent.CompactNumber(maxCtx))
	} else {
		ctxStr = fmt.Sprintf("ctx: %s", agent.CompactNumber(curCtx))
	}
	
	if baseInfo == "" {
		return ctxStr
	}
	return ctxStr + " · " + baseInfo
}

// processParallelInput decomposes a user prompt into independent sub-tasks,
// dispatches them to parallel worker agents, and feeds the aggregated results
// back into the main conversation context.
func (app *App) processParallelInput(ctx context.Context, prompt string) error {
	ctx, cancelTask := context.WithCancel(ctx)
	app.cancelMu.Lock()
	app.taskCancel = cancelTask
	app.cancelMu.Unlock()
	defer func() {
		app.cancelMu.Lock()
		app.taskCancel = nil
		app.cancelMu.Unlock()
		cancelTask()
	}()

	app.ui.UpdateThinkingStatus("planning parallel execution graph...")
	app.ui.ShowThinking()

	// ── Step 1: Stream the decomposer LLM call ──────────────────────────────
	// Using Stream (not Complete) so we can print live token progress on a
	// dedicated erasable line above the spinner, giving clear visible feedback
	// during what can be a multi-second think on a heavy model.
	//
	// The Heavy model is used because getting the DAG right is the most
	// critical planning step. Falls back through Default → Fast.
	decomposerModel := app.config.LLM.Models.Heavy
	if decomposerModel == "" {
		decomposerModel = app.config.LLM.Models.Default
	}
	if decomposerModel == "" {
		decomposerModel = app.config.LLM.Models.Fast
	}

	decomposePrompt := fmt.Sprintf(
		`You are a task decomposer. Your ONLY job is to output a JSON array — do NOT call any tools, do NOT use tool_calls format.

Break the following user request into independent or sequential sub-tasks.
Return ONLY a raw JSON array of objects. Each object must have these exact fields:
  "id"         - unique short string (e.g. "task-1")
  "description" - what this task does
  "role"        - one of: "Researcher", "Developer", "Reviewer", "Generic"
  "depends_on" - array of task IDs this task depends on (empty array [] if none)

Execution graph rules:
- Tasks with no dependencies run in parallel.
- If task B needs task A's output, set "depends_on": ["A"] on task B.

User request: %s

Output format: start with [ and end with ]. No markdown. No explanation. No tool calls. Raw JSON only.`, prompt)

	decomposeMessages := []anyllm.Message{
		{Role: "user", Content: decomposePrompt},
	}
	temperature := 0.1 // low temperature for deterministic JSON output

	chunkCh, errCh, err := app.llmClient.Stream(ctx, app.config.LLM.Provider, decomposerModel, decomposeMessages, temperature, "", nil)
	if err != nil {
		app.ui.HideThinking()
		return fmt.Errorf("task decomposition failed: %w", err)
	}

	// Create a live 3-line progress block in the viewport for the planner phase
	if plannerUI, ok := app.ui.(ui.PlannerBlockDisplayer); ok {
		plannerUI.CreatePlannerBlock(decomposerModel, "Decomposing tasks...")
	}

	// Accumulate the full response while feeding a live tail into the spinner
	// status label via UpdateThinkingStatus. The spinner goroutine owns all
	// cursor movement — we must NOT emit any raw ANSI ourselves while it runs.
	// UpdateThinkingStatus is mutex-protected and safe to call from any goroutine.
	var decomposeResultBuilder strings.Builder
	var visibleBuf strings.Builder // non-think tokens only, for the status label
	insideThink := false

	for chunk := range chunkCh {
		token := ""
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			token = delta.Content
			// Some reasoning models emit content in Reasoning.Content
			if token == "" && delta.Reasoning != nil {
				token = delta.Reasoning.Content
			}
		}
		if token == "" {
			continue
		}
		decomposeResultBuilder.WriteString(token)

		// Stream tokens to the planner progress block
		if plannerUI, ok := app.ui.(ui.PlannerBlockDisplayer); ok {
			plannerUI.UpdatePlannerBlock(token)
		}

		// Suppress <think>…</think> from the visible label while keeping the
		// full text in the accumulator (needed later for JSON extraction).
		combined := visibleBuf.String() + token
		if !insideThink {
			if idx := strings.Index(combined, "<think>"); idx != -1 {
				insideThink = true
				visibleBuf.Reset()
				visibleBuf.WriteString(combined[:idx])
			} else {
				visibleBuf.Reset()
				visibleBuf.WriteString(combined)
			}
		} else {
			if idx := strings.Index(combined, "</think>"); idx != -1 {
				insideThink = false
				visibleBuf.Reset()
				visibleBuf.WriteString(combined[idx+len("</think>"):])
			}
			// While inside think, don't update visibleBuf
		}

		// Show the last N visible chars in the spinner label.
		// Strip newlines so it stays on one line.
		tail := strings.ReplaceAll(strings.TrimSpace(visibleBuf.String()), "\n", " ")
		tail = strings.ReplaceAll(tail, "\r", "")
		
		// Constrain length to prevent terminal wrapping (which breaks the spinner).
		// Use runes so multi-byte characters aren't corrupted by byte-slicing.
		const labelMax = 40
		runes := []rune(tail)
		if len(runes) > labelMax {
			tail = "…" + string(runes[len(runes)-labelMax+1:])
		}
		if tail != "" {
			app.ui.UpdateThinkingStatus("planning — " + tail)
		}
	}

	// Check streaming error
	if streamErr := <-errCh; streamErr != nil {
		app.ui.HideThinking()
		if plannerUI, ok := app.ui.(ui.PlannerBlockDisplayer); ok {
			plannerUI.CollapsePlannerBlock()
		}
		return fmt.Errorf("task decomposition stream failed: %w", streamErr)
	}

	// Collapse the planner progress block and spinner
	if plannerUI, ok := app.ui.(ui.PlannerBlockDisplayer); ok {
		plannerUI.CollapsePlannerBlock()
	}
	app.ui.HideThinking()
	app.ui.Info("[*] Decomposition complete via %s — building task graph...", decomposerModel)

	decomposeResult := decomposeResultBuilder.String()


	// Parse the JSON array of tasks from the LLM response.
	// Models often wrap output in <think> blocks, markdown fences, or
	// explanatory text. We aggressively extract the JSON array.
	cleaned := decomposeResult

	// 1. Strip <think>...</think> blocks
	for {
		start := strings.Index(cleaned, "<think>")
		end := strings.Index(cleaned, "</think>")
		if start != -1 && end != -1 && end > start {
			cleaned = cleaned[:start] + cleaned[end+len("</think>"):]
		} else {
			break
		}
	}

	// 2. Strip markdown fences
	cleaned = strings.TrimSpace(cleaned)
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	cleaned = strings.TrimSpace(cleaned)

	// 3. Sanitize malformed JSON: models often put literal newlines inside
	//    string values (multi-line bullet lists). Replace bare newlines that
	//    appear inside JSON strings with escaped \\n.
	cleaned = sanitizeJSONNewlines(cleaned)

	// 4. Parse with flexible ID type (models return either string or number)
	var taskDefs []struct {
		ID          json.RawMessage `json:"id"`
		Description string          `json:"description"`
		Role        string          `json:"role"`
		DependsOn   []string        `json:"depends_on"`
	}
	parseJSON := func(data string) error {
		return json.Unmarshal([]byte(data), &taskDefs)
	}

	if err := parseJSON(cleaned); err != nil {
		// Bracket-match extraction: try three sources in order of reliability:
		// 1. cleaned  — post-stripping (handles prose-wrapped JSON)
		// 2. decomposeResult — raw output before stripping (handles models that
		//    emit JSON *inside* <think> blocks, e.g. GLM / Qwen reasoning mode)
		sources := []string{cleaned, decomposeResult}
		extractJSON := func(s string) string {
			if start := strings.Index(s, "["); start != -1 {
				if end := strings.LastIndex(s, "]"); end > start {
					return s[start : end+1]
				}
			}
			return ""
		}
		parsed := false
		for _, src := range sources {
			if extracted := extractJSON(sanitizeJSONNewlines(src)); extracted != "" {
				if parseJSON(extracted) == nil {
					parsed = true
					break
				}
			}
		}
		if !parsed {
			app.ui.Warning("⚠ Model returned non-JSON output for task decomposition — running as single agent.\n  Tip: if this happens often, try a stronger model for the Heavy tier.")
			app.logger.Warn("Parallel decomposition JSON parse failed", "error", err, "raw", decomposeResult)
			return app.processInput(ctx, prompt)
		}
	}

	if len(taskDefs) == 0 {
		app.ui.Warning("⚠ Decomposition returned an empty task list — running as single agent")
		app.logger.Warn("Parallel decomposition: empty task list", "raw", decomposeResult)
		return app.processInput(ctx, prompt)
	}

	// ── Step 2: Build multiagent tasks ──
	tasks := make([]multiagent.Task, len(taskDefs))
	for i, td := range taskDefs {
		// Coerce ID: model may return "task_1" (string) or 1 (number)
		id := strings.Trim(strings.TrimSpace(string(td.ID)), `"`)
		if id == "" || id == "null" {
			id = fmt.Sprintf("task-%d", i+1)
		}
		tasks[i] = multiagent.Task{
			ID:          id,
			Description: td.Description,
			DependsOn:   td.DependsOn,
			Inputs: map[string]interface{}{
				"role_hint": td.Role,
			},
		}
	}

	defaultModel := app.config.LLM.Models.Default
	fastModel := app.config.LLM.Models.Fast
	heavyModel := app.config.LLM.Models.Heavy
	dimStyle := lipgloss.NewStyle().Faint(true)

	// Build role → model mapping (mirrors worker.go logic)
	roleModel := func(role string) string {
		switch role {
		case "Researcher":
			if fastModel != "" {
				return fastModel
			}
		case "Developer", "Reviewer":
			if heavyModel != "" {
				return heavyModel
			}
		}
		return defaultModel
	}

	app.ui.HideThinking()
	app.ui.Info("[>>] Dispatching %d parallel agents...  %s", len(tasks), dimStyle.Render("· decomposed via: "+decomposerModel))
	app.ui.Print("\n")

	// Build taskID → 1-based index for dep references in the display
	taskNumByID := make(map[string]int, len(tasks))
	for i, t := range tasks {
		taskNumByID[t.ID] = i + 1
	}

	indentStyle := lipgloss.NewStyle().Faint(true).Foreground(lipgloss.Color("#6c7086"))
	depStyle    := lipgloss.NewStyle().Faint(true).Foreground(lipgloss.Color("#fab387")) // soft orange for dep refs

	for i, t := range tasks {
		role := "Generic"
		if r, ok := t.Inputs["role_hint"].(string); ok && r != "" {
			role = r
		}
		model := roleModel(role)

		if len(t.DependsOn) == 0 {
			// Independent — can run in parallel with other no-dep tasks
			app.ui.Info("   %d. [+] [%s] %s  %s", i+1, role, t.Description, dimStyle.Render("("+model+")"))
		} else {
			// Build human-readable dep list: "#2, #3"
			depParts := make([]string, 0, len(t.DependsOn))
			for _, depID := range t.DependsOn {
				if n, ok := taskNumByID[depID]; ok {
					depParts = append(depParts, fmt.Sprintf("#%d", n))
				} else {
					depParts = append(depParts, depID)
				}
			}
			depLabel := depStyle.Render("needs " + strings.Join(depParts, ", "))
			prefix   := indentStyle.Render("   ~>")
			app.ui.Info("%s %d. [%s] %s  %s  %s", prefix, i+1, role, t.Description, dimStyle.Render("("+model+")"), depLabel)
		}
	}
	app.ui.Print("\n")

	// ── Step 3: Launch orchestrator with bus→UI bridge ──
	app.ui.ShowThinking()
	app.ui.UpdateThinkingStatus(fmt.Sprintf("running %d agents in parallel", len(tasks)))

	orch := multiagent.NewOrchestrator(app.agent, app.logger)
	if app.config.MaxParallelAgents > 0 {
		orch.SetMaxConcurrency(app.config.MaxParallelAgents)
	}

	// Wire bus events to the BubbleTea UI worker status lines
	if btUI, ok := app.ui.(*ui.BubbleTeaUI); ok {
		// Build taskID → 1-indexed number lookup
		taskNumLookup := make(map[string]int, len(tasks))
		for i, t := range tasks {
			taskNumLookup[t.ID] = i + 1
		}

		eventCh := make(chan multiagent.Event, 50)
		thinkCh := make(chan multiagent.Event, 200) // higher cap — tokens arrive fast
		toolCh := make(chan multiagent.Event, 100)  // tool-call events
		orch.Bus().Subscribe(multiagent.EventTaskStarted, eventCh)
		orch.Bus().Subscribe(multiagent.EventTaskCompleted, eventCh)
		orch.Bus().Subscribe(multiagent.EventWorkerThinking, thinkCh)
		orch.Bus().Subscribe(multiagent.EventWorkerToolCall, toolCh)

		// Forward thinking tokens to the viewport live block.
		go func() {
			for evt := range thinkCh {
				if token, ok := evt.Payload["token"].(string); ok && token != "" {
					btUI.UpdateWorkerBlock(evt.WorkerID, token)
				}
			}
		}()

		// Forward tool-call events to the viewport live block.
		go func() {
			for evt := range toolCh {
				if toolName, ok := evt.Payload["tool_name"].(string); ok && toolName != "" {
					btUI.UpdateWorkerBlockTool(evt.WorkerID, toolName)
				}
			}
		}()

		go func() {
			for evt := range eventCh {
				role := "Generic"
				if r, ok := evt.Payload["role"].(string); ok {
					role = r
				}
				label := ""
				if l, ok := evt.Payload["label"].(string); ok {
					if len(l) > 50 {
						l = l[:47] + "..."
					}
					label = l
				}
				hasError := false
				if he, ok := evt.Payload["has_error"].(bool); ok {
					hasError = he
				}

				taskID, _ := evt.Payload["task_id"].(string)
				num := taskNumLookup[taskID]

				var elapsedStr string
				if durStr, ok := evt.Payload["duration"].(string); ok {
					elapsedStr = durStr
				}
				elapsedDur, _ := time.ParseDuration(elapsedStr)

				if evt.Type == multiagent.EventTaskStarted {
					// Create a live thinking block in the viewport
					btUI.CreateWorkerBlock(ui.WorkerBlockInit{
						WorkerID: evt.WorkerID,
						Number:   num,
						Role:     role,
						Model:    roleModel(role),
						Label:    label,
					})
					// Also update status bar
					btUI.UpdateWorkerStatus(ui.WorkerStatus{
						WorkerID: evt.WorkerID,
						Number:   num,
						Role:     role,
						Model:    roleModel(role),
						Label:    label,
						Started:  time.Now(),
					})
				} else {
					// Collapse the viewport block
					btUI.CollapseWorkerBlock(ui.WorkerStatus{
						WorkerID: evt.WorkerID,
						Number:   num,
						Role:     role,
						Model:    roleModel(role),
						Label:    label,
						Done:     true,
						HasError: hasError,
						Elapsed:  elapsedDur,
					})
					// Also update status bar
					btUI.UpdateWorkerStatus(ui.WorkerStatus{
						WorkerID: evt.WorkerID,
						Number:   num,
						Role:     role,
						Model:    roleModel(role),
						Label:    label,
						Done:     true,
						HasError: hasError,
						Started:  time.Now(),
						Elapsed:  elapsedDur,
					})
				}
			}
		}()
	}

	results, err := orch.ExecuteParallel(ctx, tasks)
	app.ui.HideThinking()

	// Clear worker status lines
	if btUI, ok := app.ui.(*ui.BubbleTeaUI); ok {
		btUI.ClearWorkerStatuses()
	}

	if err != nil {
		return fmt.Errorf("parallel execution failed: %w", err)
	}

	// ── Step 4: Aggregate results and feed back into main context ──
	// Build lookup from taskID → numbered description for consistent display
	taskIndex := make(map[string]string, len(tasks))
	for i, t := range tasks {
		taskIndex[t.ID] = fmt.Sprintf("Task %d", i+1)
	}
	taskDesc := make(map[string]string, len(tasks))
	for _, t := range tasks {
		desc := t.Description
		if len(desc) > 80 {
			desc = desc[:77] + "..."
		}
		taskDesc[t.ID] = desc
	}

	var summary strings.Builder
	summary.WriteString(fmt.Sprintf("## Parallel Execution Results (%d agents)\n\n", len(results)))

	successCount := 0
	for _, r := range results {
		label := taskIndex[r.TaskID]
		if label == "" {
			label = r.TaskID
		}
		desc := taskDesc[r.TaskID]

		if r.Error != nil {
			summary.WriteString(fmt.Sprintf("### ❌ %s — %s (failed after %s)\n", label, desc, r.Duration.Round(time.Millisecond)))
			summary.WriteString(fmt.Sprintf("Error: %v\n\n", r.Error))
		} else {
			successCount++
			summary.WriteString(fmt.Sprintf("### ✅ %s — %s (%s)\n", label, desc, r.Duration.Round(time.Millisecond)))
			output := r.Output
			if len(output) > 2000 {
				output = output[:2000] + "\n\n[... truncated ...]"
			}
			summary.WriteString(output + "\n\n")
		}
	}

	// Calculate total wall time from longest worker
	var maxDuration time.Duration
	for _, r := range results {
		if r.Duration > maxDuration {
			maxDuration = r.Duration
		}
	}

	app.ui.Info("[>>] Parallel execution complete: %d/%d succeeded  %s",
		successCount, len(results),
		dimStyle.Render(fmt.Sprintf("· %s · model: %s", maxDuration.Round(time.Second), defaultModel)))
	app.ui.ShowResponse(summary.String())

	// Inject the aggregated results as an assistant message so the main agent
	// has awareness of what the parallel workers discovered/produced.
	app.agent.AddAssistantMessage("[Parallel Agent Results]\n" + summary.String())

	return nil
}

// processInput processes user input with the AI using streaming.
func (app *App) processInput(ctx context.Context, input string) error {
	ctx, cancelTask := context.WithCancel(ctx)
	app.cancelMu.Lock()
	app.taskCancel = cancelTask
	app.cancelMu.Unlock()
	defer func() {
		app.cancelMu.Lock()
		app.taskCancel = nil
		app.cancelMu.Unlock()
		cancelTask()
	}()

	if app.sessionRecord != nil {
		app.sessionRecord.Record(session.EventUIInput, map[string]interface{}{
			"text": input,
		})
	}

	taskStart := time.Now()

	for {
		// Show thinking indicator while the connection is being established.
		// Clear any previous status label from the last turn.
		app.ui.UpdateThinkingStatus("")
		if app.sessionUsage.TotalTokens > 0 {
			tokenInfo := app.buildContextTokenStr(fmt.Sprintf("session: %s", agent.CompactNumber(app.sessionUsage.TotalTokens)))
			app.ui.UpdateTokenUsage(tokenInfo)
		} else {
			app.ui.UpdateTokenUsage(app.buildContextTokenStr(""))
		}
		app.ui.ShowThinking()

		// If context retention is disabled, clear previous conversation
		if !app.config.RetainContext {
			app.agent.ClearContext()
		}

		// Process with agent in background
		var finalResponse string
		var agentErr error
		done := make(chan struct{})

		// Inject active skill instructions into agent system prompt
		app.updateAgentSystemPrompt()

		var streamingStarted bool
		var thinkBuffer string
		var inThink bool

		var (
			thinkLinesPrinted    int
			thinkCharsPrinted    int
			markdownLinesPrinted int
			markdownCharsPrinted int
			currentLineLen       int
			consoleWidth         int
		)
		consoleWidth, _, _ = term.GetSize(int(os.Stdout.Fd()))
		if consoleWidth <= 0 {
			consoleWidth = 80
		}

		thinkLineStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#565f89")).Italic(true)
		var thinkLineBuffer string // accumulates the current partial think line

		flushThinkLine := func(complete bool) {
			// Emit whatever is in thinkLineBuffer.
			// If complete=true a newline terminates the line.
			// If complete=false we're mid-line; just print the fragment raw (no newline yet).
			if thinkLineBuffer == "" {
				return
			}
			if complete {
				app.ui.Print("  %s\n",
					thinkLineStyle.Render(thinkLineBuffer))
				thinkLinesPrinted++
				thinkLineBuffer = ""
			} else {
				// mid-line: print the fragment raw
				app.ui.Print("  %s",
					thinkLineStyle.Render(thinkLineBuffer))
				thinkLineBuffer = ""
			}
		}

		printThink := func(s string) {
			thinkCharsPrinted += len(s)
			// Walk the string; emit complete lines with prefix, buffer partials.
			for len(s) > 0 {
				nl := strings.IndexByte(s, '\n')
				if nl == -1 {
					// No newline — accumulate
					thinkLineBuffer += s
					break
				}
				// Got a newline: complete the current line
				thinkLineBuffer += s[:nl]
				app.ui.Print("  %s\n",
					thinkLineStyle.Render(thinkLineBuffer))
				thinkLinesPrinted++
				thinkLineBuffer = ""
				s = s[nl+1:]
			}
		}

		printMarkdown := func(s string) {
			app.ui.Print("%s", s)
			markdownCharsPrinted += len(s)
			for _, r := range s {
				if r == '\n' {
					markdownLinesPrinted++
					currentLineLen = 0
				} else {
					currentLineLen++
					if currentLineLen >= consoleWidth {
						markdownLinesPrinted++
						currentLineLen = 0
					}
				}
			}
		}

		go func() {
			defer close(done)

			onStreamEnd := func() {
				// Flush any remaining buffer before wiping
				if thinkBuffer != "" {
					if inThink {
						printThink(thinkBuffer)
						flushThinkLine(false) // flush any partial line
						app.ui.Print("\n\n")
					} else {
						printMarkdown(thinkBuffer)
					}
					thinkBuffer = ""
				}
				// Flush any partial think line that didn't end with \n
				flushThinkLine(false)
				if streamingStarted && markdownLinesPrinted > 0 {
					// In BubbleTea viewport mode, we can't erase previously appended lines.
					// The streamed raw text stays visible; ShowResponse appends the
					// glamour-rendered version below it with a visual separator.
					markdownLinesPrinted = 0
					markdownCharsPrinted = 0
				}
			}

			fr, err := app.agent.ProcessMessageStream(ctx, input, func(token string) {
				if token == "" {
					return
				}
				if app.sessionRecord != nil {
					app.sessionRecord.Record(session.EventToken, map[string]interface{}{
						"token": token,
					})
				}
				if !streamingStarted {
					streamingStarted = true
					// Hide spinner and print the response header before first token.
					app.ui.HideThinking()
					app.ui.Print("\n")
					app.ui.Println("%s", app.ui.GetTheme().StyleGreen.Render("🤖 Assistant:"))
					app.ui.Print("\n")
				} else {
					// Calls #2+: spinner was restarted by OnLLMCall but streamingStarted
					// is still true. Kill it the moment a token arrives.
					app.ui.HideThinking()
				}

				// Small state machine to strip/replace <think> tags dynamically
				thinkBuffer += token

				if !inThink {
					if idx := strings.Index(thinkBuffer, "<think>"); idx != -1 {
						// Print anything before <think> that was buffered
						if idx > 0 {
							printMarkdown(thinkBuffer[:idx])
						}
						// Enable think style and print opening header
						inThink = true
						app.ui.HideThinking()
						thinkLinesPrinted = 2
						app.ui.Print("\n%s\n",
							lipgloss.NewStyle().Foreground(lipgloss.Color("#565f89")).Italic(true).Render("🤔 thinking"))
						thinkBuffer = thinkBuffer[idx+len("<think>"):]
					}
				}

				if inThink {
					if idx := strings.Index(thinkBuffer, "</think>"); idx != -1 {
						// Print anything before </think> with dimmed style
						if idx > 0 {
							printThink(thinkBuffer[:idx])
						}
						flushThinkLine(true) // flush any partial line before closing
						// Close the think block
						inThink = false
						app.ui.Print("\n")

						app.ui.HideThinking()
						thinkBuffer = thinkBuffer[idx+len("</think>"):]
					} else {
						// Safe to print if we leave enough buffer not to split "</think>"
						if len(thinkBuffer) > 9 {
							safeLen := len(thinkBuffer) - 9
							printThink(thinkBuffer[:safeLen])
							thinkBuffer = thinkBuffer[safeLen:]
						}
					}
				} else {
					// Outside think block, safe to flush buffer if we leave enough for "<think>"
					if len(thinkBuffer) > 8 {
						app.ui.HideThinking() // ensure no spinner hides text
						safeLen := len(thinkBuffer) - 8
						printMarkdown(thinkBuffer[:safeLen])
						thinkBuffer = thinkBuffer[safeLen:]
					}
				}
			}, onStreamEnd)

			// ProcessMessageStream uses onStreamEnd to flush any remaining buffer.
			finalResponse = fr
			agentErr = err
		}()

		// Wait for agent to finish
		<-done

		// Ensure spinner is always stopped when the agent finishes its work,
		// regardless of whether tokens were streamed. This prevents hanging
		// spinners on orchestrator errors or rapid loop-detection kills.
		app.ui.HideThinking()

		if agentErr == nil && finalResponse != "" {
			cleanResponse := finalResponse
			// Remove <think> blocks from cleanResponse — they were already streamed
			// to the terminal and should not be duplicated in the glamour render.
			for {
				startIdx := strings.Index(cleanResponse, "<think>")
				endIdx := strings.Index(cleanResponse, "</think>")
				if startIdx != -1 && endIdx != -1 && endIdx > startIdx {
					cleanResponse = cleanResponse[:startIdx] + cleanResponse[endIdx+8:]
				} else {
					break
				}
			}

			// Add a clear separator if we streamed raw text, then render the
			// full final response with Glamour for proper markdown formatting.
			app.ui.ShowResponse(sanitizeAgentResponse(cleanResponse))

			// Calculate duration
			taskDuration := time.Since(taskStart).Round(time.Second)

			// Build a clean summary line with per-model breakdown
			var tokenInfo string
			if app.turnModels != nil && len(app.turnModels) > 0 {
				tokenInfo = agent.FormatModelBreakdown(app.turnModels)
				if app.sessionUsage.TotalTokens > app.lastTurnUsage.TotalTokens {
					tokenInfo += fmt.Sprintf(" · session: %s", agent.CompactNumber(app.sessionUsage.TotalTokens))
				}
				tokenInfo = app.buildContextTokenStr(tokenInfo)
				app.lastTurnUsage = agent.TokenUsage{} // Reset per-turn usage after display
				app.turnModels = nil
			} else {
				tokenInfo = "—"
			}

			statusLine := fmt.Sprintf("📊 %s  |  ⏱️ %s", tokenInfo, taskDuration)
			app.ui.Print("\n%s\n", lipgloss.NewStyle().Faint(true).Render(statusLine))

			// Stop live status ticker now that we've printed the final status line
			if stopper, ok := app.ui.(ui.LiveStatusStopper); ok {
				stopper.StopLiveStatus()
			}
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

// ─── Response sanitizer ──────────────────────────────────────────────────────

var (
	// runTogetherRe fixes sentences that run together without a space:
	// "...task.I've" → "...task. I've"
	runTogetherRe = regexp.MustCompile(`([.!?])([A-Z])`)

	// cotPrefixes: sentence starters that indicate leaked chain-of-thought.
	cotPrefixes = []string{
		"the user asked", "the user wants", "the user said",
		"the user requested", "the user has", "the user provided",
		"the user mentioned", "the user would", "the user need",
		"we need to", "we should ", "we created", "we have created",
		"we will", "we can ", "we must", "we might", "we are ",
		"i think", "i should", "i need to", "i am thinking",
		"perhaps they", "perhaps the user",
		"they didn't", "they want", "they haven't", "they need",
		"now we ", "so we ", "now i ", "so i ",
		"note: the user", "note: user",
	}

	// taskIDRowRe matches a table row that contains a task ID like t1, t2 …
	taskIDRowRe = regexp.MustCompile(`(?i)\bt\d+\b`)
)

// sanitizeAgentResponse removes chain-of-thought leakage and redundant task
// tables from the LLM response before it is shown to the user.
func sanitizeAgentResponse(response string) string {
	response = stripLeadingCoT(response)
	response = stripTaskIDTables(response)
	return strings.TrimSpace(response)
}

// stripLeadingCoT removes leading sentences that look like internal reasoning.
func stripLeadingCoT(s string) string {
	// Normalize run-together sentences so splitting works reliably.
	normalized := runTogetherRe.ReplaceAllString(s, "$1 $2")
	lines := strings.Split(normalized, "\n")

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Empty lines — keep scanning.
		if trimmed == "" {
			continue
		}

		// Markdown structure (heading, fence, quote, list) = real content, stop.
		if strings.HasPrefix(trimmed, "#") ||
			strings.HasPrefix(trimmed, "```") ||
			strings.HasPrefix(trimmed, ">") ||
			strings.HasPrefix(trimmed, "- ") ||
			strings.HasPrefix(trimmed, "* ") {
			return strings.Join(lines[i:], "\n")
		}

		// Split this line into sentences (separated by ". ", "! ", "? ").
		sentRe := regexp.MustCompile(`[.!?]+\s+`)
		parts := sentRe.Split(trimmed, -1)

		var realParts []string
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			lower := strings.ToLower(part)
			isCoT := false
			for _, prefix := range cotPrefixes {
				if strings.HasPrefix(lower, prefix) {
					isCoT = true
					break
				}
			}
			if !isCoT {
				realParts = append(realParts, part)
			}
		}

		if len(realParts) == 0 {
			// Entire line is CoT — skip and keep scanning.
			continue
		}

		// This line has real content. Reconstruct: real parts + remainder of response.
		lines[i] = strings.Join(realParts, " ")
		return strings.Join(lines[i:], "\n")
	}

	// Everything looked like CoT — return original to be safe.
	return s
}

// stripTaskIDTables removes markdown or glamour-style tables whose rows contain
// task IDs (t1, t2 …) — these duplicate the todoCreate checklist already shown.
func stripTaskIDTables(s string) string {
	lines := strings.Split(s, "\n")
	result := make([]string, 0, len(lines))

	i := 0
	for i < len(lines) {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		// Detect start of a table (lines beginning with │ | or separator chars).
		if isTableHeaderLine(trimmed) {
			// Collect all lines that belong to this table.
			tableLines := []string{line}
			j := i + 1
			hasTaskIDs := taskIDRowRe.MatchString(line)
			for j < len(lines) {
				nxt := strings.TrimSpace(lines[j])
				if nxt == "" || isTableBodyLine(nxt) {
					if taskIDRowRe.MatchString(lines[j]) {
						hasTaskIDs = true
					}
					tableLines = append(tableLines, lines[j])
					j++
				} else {
					break
				}
			}
			if hasTaskIDs {
				// Drop the whole table.
				i = j
				continue
			}
			// Not a task-ID table — keep it.
			result = append(result, tableLines...)
			i = j
			continue
		}

		result = append(result, line)
		i++
	}
	return strings.Join(result, "\n")
}

func isTableHeaderLine(s string) bool {
	return strings.HasPrefix(s, "|") ||
		strings.HasPrefix(s, "│") ||
		strings.HasPrefix(s, "╭") ||
		strings.HasPrefix(s, "┌")
}

func isTableBodyLine(s string) bool {
	return strings.HasPrefix(s, "|") ||
		strings.HasPrefix(s, "│") ||
		strings.HasPrefix(s, "├") ||
		strings.HasPrefix(s, "╰") ||
		strings.HasPrefix(s, "└") ||
		strings.HasPrefix(s, "─") ||
		strings.HasPrefix(s, "-") && strings.ContainsAny(s, "|") ||
		strings.HasPrefix(s, " ") && (strings.Contains(s, "│") || strings.Contains(s, "|"))
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
		app.PrintSessionSummary()
		return true

	case "/clear", "/c":
		app.ui.Clear()
		app.ui.ShowBanner()

	case "/model", "/models":
		if len(parts) > 1 {
			arg := strings.ToLower(parts[1])
			if arg == "ls" || arg == "list" {
				app.showModels(ctx)
			} else if arg == "default" || arg == "fast" || arg == "heavy" {
				if len(parts) > 2 {
					app.changeRoleModel(ctx, arg, strings.Join(parts[2:], " "))
				} else {
					app.ui.Warning("Usage: /model %s <model_name>", arg)
				}
			} else {
				app.changeRoleModel(ctx, "default", strings.Join(parts[1:], " "))
			}
		} else {
			if parts[0] == "/models" {
				app.showModels(ctx)
			} else {
				app.ui.Info("Current models:\n  Default: %s\n  Fast:    %s\n  Heavy:   %s",
					app.config.LLM.Models.Default,
					app.config.LLM.Models.Fast,
					app.config.LLM.Models.Heavy)
				app.ui.Info("Type '/model ls' to see available models")
			}
		}

	case "/context":
		app.handleContextCommand(parts)

	case "/theme":
		if len(parts) > 1 {
			themeName := strings.ToLower(parts[1])
			tProvider, ok := ui.ThemeRegistry[themeName]
			if ok {
				app.ui.SetTheme(tProvider())
				app.ui.Clear()
				app.ui.ShowBanner()
				app.ui.Success("Theme changed to: %s", themeName)
			} else {
				app.ui.Warning("Unknown theme '%s'. Available: %s", themeName, strings.Join(ui.AvailableThemes(), ", "))
			}
		} else {
			app.ui.Info("Type '/theme <name>' to switch themes.")
		}

	case "/tools":
		app.showTools()

	case "/skill", "/skills":
		app.handleSkillCommand(parts)

	case "/session", "/sessions":
		app.handleSessionCommand(ctx, parts)

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

	case "/replay":
		if len(parts) < 2 {
			app.ui.Warning("Usage: /replay <filename>")
		} else {
			app.replayConversation(ctx, parts[1])
		}

	case "/tokens":
		if app.sessionUsage.TotalTokens == 0 {
			app.ui.Info("No token usage recorded yet in this session.")
		} else {
			app.ui.Println("")
			app.ui.Println("%s", lipgloss.NewStyle().Bold(true).Render("📊 Session Token Usage"))
			theme := app.ui.GetTheme()
			keyStyle := theme.StyleDim.Width(22)
			app.ui.Print("%s %s\n", keyStyle.Render("  Prompt tokens:"), agent.FormatNumber(app.sessionUsage.PromptTokens))
			app.ui.Print("%s %s\n", keyStyle.Render("  Completion tokens:"), agent.FormatNumber(app.sessionUsage.CompletionTokens))
			if app.sessionUsage.ReasoningTokens > 0 {
				app.ui.Print("%s %s\n", keyStyle.Render("  Reasoning tokens:"), agent.FormatNumber(app.sessionUsage.ReasoningTokens))
			}
			app.ui.Print("%s %s\n", keyStyle.Render("  Total tokens:"), theme.StyleYellow.Render(agent.FormatNumber(app.sessionUsage.TotalTokens)))
			app.ui.Println("")
		}

	case "/router":
		if len(parts) > 1 {
			switch strings.ToLower(parts[1]) {
			case "on":
				app.config.AutoRoute = true
				app.agent.SetAutoRoute(true)
				if app.config.LLM.Models.Fast == "" {
					app.ui.Warning("Auto-routing enabled, but no fast model configured.")
					app.ui.Info("Set one with: /model fast <model_name>")
				} else {
					app.ui.Success("Auto-routing enabled (fast: %s)", app.config.LLM.Models.Fast)
				}
			case "off":
				app.config.AutoRoute = false
				app.agent.SetAutoRoute(false)
				app.ui.Success("Auto-routing disabled — all requests use default model")
			case "status":
				app.showRouterStatus()
			default:
				app.ui.Warning("Usage: /router [on|off|status]")
			}
		} else {
			app.showRouterStatus()
		}

	case "/thinking":
		if len(parts) > 1 {
			effort := strings.ToLower(parts[1])
			switch effort {
			case "low", "medium", "high", "auto", "none":
				app.agent.SetReasoningEffort(effort)
				app.config.ReasoningEffort = effort
				app.ui.Success("Reasoning effort set to: %s", effort)
			case "off", "clear", "reset":
				app.agent.SetReasoningEffort("")
				app.config.ReasoningEffort = ""
				app.ui.Success("Reasoning effort cleared (using provider default)")
			default:
				app.ui.Warning("Unknown effort level '%s'. Valid: low, medium, high, auto, none, off", effort)
			}
		} else {
			current := app.config.ReasoningEffort
			if current == "" {
				current = "(not set — provider default)"
			}
			app.ui.Info("Current reasoning effort: %s", current)
			app.ui.Info("Usage: /thinking [low|medium|high|auto|none|off]")
		}

	case "/parallel":
		if len(parts) < 2 {
			app.ui.Warning("Usage: /parallel <task description>")
			app.ui.Info("Example: /parallel review internal/agent/agent.go, internal/agent/context.go, and internal/agent/orchestrator.go")
		} else {
			prompt := strings.Join(parts[1:], " ")
			if err := app.processParallelInput(ctx, prompt); err != nil {
				if err != context.Canceled {
					app.ui.Error("Parallel execution failed: %v", err)
				}
			}
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

	saved := make([]agent.Message, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == agent.RoleSystem {
			continue // Don't persist system prompts natively so we can inject new ones on load
		}
		saved = append(saved, msg)
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

	var messages []agent.Message
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
		app.agent.AddMessage(msg)
	}

	app.ui.Success("Conversation loaded from %s (%d messages)", filename, len(messages))
}

// replayConversation loads a history file, strips off any trailing agent/tool outputs,
// and resubmits the last user prompt to re-evaluate the conversation dynamically.
func (app *App) replayConversation(ctx context.Context, filename string) {
	app.loadConversation(filename)
	messages := app.agent.GetMessages()

	// Find the last user message
	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleUser {
			lastUserIdx = i
			break
		}
	}

	if lastUserIdx == -1 {
		app.ui.Warning("No user messages found in history to replay.")
		return
	}

	// The prompt to replay is the last user message.
	promptToReplay := messages[lastUserIdx].Content

	// Clear out everything from the last user message onwards
	app.agent.ClearLastUserMessage() // First clear the very last one explicitly to be safe

	// Get current context, and literally strip it back to exactly before the last user prompt
	// By rebuilding it
	app.agent.ClearContext()
	for i := 0; i < lastUserIdx; i++ {
		if messages[i].Role != agent.RoleSystem {
			app.agent.AddMessage(messages[i])
		}
	}

	app.ui.Info("Replaying prompt: %s", promptToReplay)

	// Re-run the processInput block asynchronously as if the user just typed it
	go func() {
		if err := app.processInput(ctx, promptToReplay); err != nil {
			app.ui.Error("Failed to process replay: %v", err)
		}
	}()
}

// showModels displays available models
func (app *App) showModels(ctx context.Context) {
	models, err := app.llmClient.ListModels(ctx, app.config.LLM.Provider)
	if err != nil {
		app.ui.Error("Failed to list models: %v", err)
		return
	}

	// Cache for Tab completion
	app.cachedModels = models

	app.ui.ShowModels(models, app.config.LLM.Models.Default, app.config.LLM.Models.Fast, app.config.LLM.Models.Heavy)
}

// changeRoleModel changes the current model for a specific role
func (app *App) changeRoleModel(ctx context.Context, role, modelName string) {
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

	if role == "fast" {
		app.config.LLM.Models.Fast = modelName
		app.agent.SetFastModel(modelName)
		app.ui.Success("Switched fast model to: %s", modelName)
		if !app.config.AutoRoute {
			app.ui.Info("Tip: enable routing with /router on")
		}
	} else if role == "heavy" {
		app.config.LLM.Models.Heavy = modelName
		app.agent.SetHeavyModel(modelName)
		app.ui.Success("Switched heavy model to: %s", modelName)
		if !app.config.AutoRoute {
			app.ui.Info("Tip: enable routing with /router on")
		}
	} else {
		app.config.LLM.Models.Default = modelName
		app.agent.SetModel(modelName)
		app.ui.SetModel(modelName)
		app.ui.Success("Switched default model to: %s", modelName)
	}
}

// showRouterStatus displays the current model routing configuration.
func (app *App) showRouterStatus() {
	theme := app.ui.GetTheme()
	app.ui.Println("")
	app.ui.Println("%s", lipgloss.NewStyle().Bold(true).Render("🔀 Model Router"))

	status := theme.StyleRed.Render("disabled")
	if app.config.AutoRoute {
		status = theme.StyleGreen.Render("enabled")
	}

	keyStyle := theme.StyleDim.Width(16)
	app.ui.Print("%s %s\n", keyStyle.Render("  Status:"), status)
	app.ui.Print("%s %s\n", keyStyle.Render("  Fast:"), modelOrNotSet(app.config.LLM.Models.Fast, theme))
	app.ui.Print("%s %s\n", keyStyle.Render("  Default:"), app.config.LLM.Models.Default)
	app.ui.Print("%s %s\n", keyStyle.Render("  Heavy:"), modelOrNotSet(app.config.LLM.Models.Heavy, theme))
	app.ui.Println("")
	if !app.config.AutoRoute {
		app.ui.Info("Enable with: /router on")
	}
	if app.config.LLM.Models.Fast == "" && app.config.LLM.Models.Heavy == "" && app.config.AutoRoute {
		app.ui.Info("Set models with: /model fast <name> and /model heavy <name>")
	}
}

// modelOrNotSet returns the model name or a dimmed "(not set)" placeholder.
func modelOrNotSet(model string, theme ui.Theme) string {
	if model != "" {
		return model
	}
	return theme.StyleDim.Render("(not set)")
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

// updateAgentSystemPrompt rebuilds the agent's system prompt by appending
// instructions from all currently active skills. Called before each LLM request
// so mid-session skill activation/deactivation takes effect immediately.
func (app *App) updateAgentSystemPrompt() {
	instructions := app.skillReg.ActiveInstructions()
	basePrompt := app.config.SystemPrompt

	prompt := basePrompt
	if instructions != "" {
		prompt = basePrompt + "\n\n## Active Skills\n\n" + instructions
	}
	app.agent.ReplaceSystemMessage(prompt)
}

// handleSkillTrigger toggles a skill that matched a slash trigger command.
func (app *App) handleSkillTrigger(skill *skills.Skill) {
	if app.skillReg.IsActive(skill.Name) {
		app.skillReg.Deactivate(skill.Name)
		app.ui.Info("Skill deactivated: %s", skill.Name)
	} else {
		_ = app.skillReg.Activate(skill.Name)
		app.ui.Success("Skill activated: %s", skill.Name)
		if skill.Description != "" {
			app.ui.Info("%s", skill.Description)
		}
	}
}

// handleSkillCommand handles the /skill subcommands.
func (app *App) handleSkillCommand(parts []string) {
	if len(parts) < 2 {
		app.ui.Warning("Usage: /skill [ls|list|info <name>|<name>|off <name>|reset]")
		return
	}

	sub := strings.ToLower(parts[1])
	switch sub {
	case "list", "ls":
		allSkills := app.skillReg.List()
		if len(allSkills) == 0 {
			app.ui.Info("No skills loaded. Add .md files to %s", app.config.Skills.Dir)
			return
		}
		app.ui.Println("\n%s", "Available Skills:")
		for _, s := range allSkills {
			status := "inactive"
			if app.skillReg.IsActive(s.Name) {
				status = "active"
			}
			trigger := ""
			if s.Trigger != "" {
				trigger = " (trigger: " + s.Trigger + ")"
			}
			app.ui.Println("  %-20s [%s]%s — %s", s.Name, status, trigger, s.Description)
		}
		app.ui.Println("")

	case "off":
		if len(parts) < 3 {
			app.ui.Warning("Usage: /skill off <name>")
			return
		}
		name := parts[2]
		app.skillReg.Deactivate(name)
		app.ui.Success("Skill deactivated: %s", name)

	case "reset":
		app.skillReg.DeactivateAll()
		app.ui.Success("All skills deactivated")

	case "info":
		if len(parts) < 3 {
			app.ui.Warning("Usage: /skill info <name>")
			return
		}
		name := parts[2]
		s := app.skillReg.Get(name)
		if s == nil {
			app.ui.Error("Skill not found: %s", name)
			return
		}
		app.ui.Println("\n%s — %s", s.Name, s.Description)
		if s.Trigger != "" {
			app.ui.Print("Trigger: %s\n", s.Trigger)
		}
		app.ui.Print("Always-on: %v\n", s.AlwaysOn)
		app.ui.Print("\nInstructions:\n%s\n\n", s.Instructions)

	default:
		// Treat as skill name to activate/toggle
		name := sub
		s := app.skillReg.Get(name)
		if s == nil {
			app.ui.Warning("Unknown skill: %s. Use /skill list to see available skills.", name)
			return
		}
		app.handleSkillTrigger(s)
	}
}

// registerTools is defined in tool_setup.go


// sanitizeJSONNewlines replaces bare newlines inside JSON string values with
// the escaped sequence \n. LLMs frequently produce multi-line descriptions
// with literal linebreaks which are invalid JSON.
func sanitizeJSONNewlines(s string) string {
	var out strings.Builder
	out.Grow(len(s))
	inString := false
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			out.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && inString {
			out.WriteByte(c)
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			out.WriteByte(c)
			continue
		}
		if c == '\n' && inString {
			out.WriteString(`\n`)
			continue
		}
		if c == '\r' && inString {
			continue // drop carriage returns inside strings
		}
		out.WriteByte(c)
	}
	return out.String()
}
