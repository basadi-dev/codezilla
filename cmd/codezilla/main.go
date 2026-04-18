package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"codezilla/internal/config"
	"codezilla/internal/core"
	"codezilla/internal/ui"
)

func main() {
	// Parse command line flags
	var (
		configPath  = flag.String("config", "", "Path to config file")
		noColors    = flag.Bool("no-colors", false, "Disable colored output")
		provider    = flag.String("provider", "", "Override LLM provider (ollama, openai, anthropic, gemini)")
		model       = flag.String("model", "", "Override default model")
		sessionID   = flag.String("session", "", "Session ID to resume")
		resumeFlag  = flag.Bool("resume", false, "Resume the most recent session")
		ollamaURL   = flag.String("ollama-url", "", "Override Ollama API URL")
		temperature = flag.Float64("temperature", -1, "Override temperature (0.0-1.0)")
		maxTokens   = flag.Int("max-tokens", 0, "Override max tokens")
		inline      = flag.Bool("inline", false, "Render on the main terminal buffer instead of the alt-screen (conversation persists in shell scrollback on exit). Also settable via CODEZILLA_INLINE=1.")
		noMouse     = flag.Bool("no-mouse", false, "Disable mouse capture at launch (restores native text selection). Also settable via CODEZILLA_NO_MOUSE=1 or disable_mouse: true in config.")
		mouseMode   = flag.String("mouse", "", "Mouse capture mode: 'on' or 'off'. Overrides --no-mouse when set.")
		version     = flag.Bool("version", false, "Show version")
		help        = flag.Bool("help", false, "Show help")
	)
	flag.Parse()

	// Parse remainder args for resume command or URL scheme
	args := flag.Args()
	if len(args) > 0 {
		if args[0] == "resume" && len(args) > 1 {
			*sessionID = args[1]
		} else if args[0] == "resume" && len(args) == 1 {
			// "codezilla resume" with no ID → pick latest
			*resumeFlag = true
		} else if strings.HasPrefix(args[0], "codezilla://session/") {
			*sessionID = strings.TrimPrefix(args[0], "codezilla://session/")
		}
	}

	// Handle version
	if *version {
		fmt.Println("Codezilla v2.0.0 - Modular Architecture")
		os.Exit(0)
	}

	// Handle help
	if *help {
		printHelp()
		os.Exit(0)
	}

	// Get config path
	if *configPath == "" {
		*configPath = getDefaultConfigPath()
	}

	// Load configuration
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		cfg = config.DefaultConfig()
		fmt.Printf("Note: Using default configuration\n")
	}

	// Apply CLI overrides
	if *provider != "" {
		cfg.LLM.Provider = *provider
	}
	if *model != "" {
		cfg.LLM.Models.Default = *model
	}
	if *sessionID != "" {
		cfg.ResumeSessionID = *sessionID
	} else if *resumeFlag {
		// Find the most recently modified .jsonl session file
		if latest, err := latestSessionFile(cfg.SessionEventsDir); err == nil && latest != "" {
			cfg.ResumeSessionID = latest
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not find latest session: %v\n", err)
		} else {
			fmt.Fprintln(os.Stderr, "No sessions found to resume.")
			os.Exit(1)
		}
	}
	if *ollamaURL != "" {
		cfg.LLM.Ollama.BaseURL = *ollamaURL
	}
	if *temperature >= 0 && *temperature <= 1 {
		cfg.Temperature = float32(*temperature)
	}
	if *maxTokens > 0 {
		cfg.MaxTokens = *maxTokens
	}

	// Apply color settings
	if *noColors {
		cfg.NoColor = true
	}

	// Apply mouse settings. Precedence: explicit --mouse flag, then
	// --no-mouse, then CODEZILLA_NO_MOUSE env, then whatever the config file
	// said (default false = mouse on, matching GitHub Copilot CLI).
	switch strings.ToLower(strings.TrimSpace(*mouseMode)) {
	case "off", "false", "0", "no":
		cfg.DisableMouse = true
	case "on", "true", "1", "yes":
		cfg.DisableMouse = false
	case "":
		if *noMouse {
			cfg.DisableMouse = true
		} else if v := strings.ToLower(strings.TrimSpace(os.Getenv("CODEZILLA_NO_MOUSE"))); v == "1" || v == "true" || v == "yes" {
			cfg.DisableMouse = true
		}
	default:
		fmt.Fprintf(os.Stderr, "Invalid --mouse value %q (expected 'on' or 'off')\n", *mouseMode)
		os.Exit(2)
	}

	// Get history file path
	historyPath, _ := ui.GetDefaultHistoryFilePath()

	// Create unified BubbleTea UI
	appUI, err := ui.NewBubbleTeaUI(historyPath, *inline, cfg.DisableMouse)

	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize UI: %v\n", err)
		os.Exit(1)
	}

	// Disable colors if requested
	if *noColors {
		appUI.DisableColors()
	}

	// Create the core application
	app, err := core.NewApp(cfg, appUI)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize application: %v\n", err)
		os.Exit(1)
	}
	defer app.Close()

	// Define context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// If the UI supports TUIRunner (BubbleTea), let it own the main goroutine.
	// app.Run() will be called from the goroutine launched inside RunTUI.
	if runner, ok := appUI.(ui.TUIRunner); ok {
		if err := runner.RunTUI(ctx, func(ctx context.Context) error {
			return app.Run(ctx, cancel)
		}); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

}

func getDefaultConfigPath() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".config", "codezilla", "config.yaml")
	}
	return "config.yaml"
}

// latestSessionFile returns the basename of the most recently modified .jsonl
// file in dir, or an empty string if none exist.
func latestSessionFile(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("reading sessions dir %s: %w", dir, err)
	}
	type entry struct {
		name    string
		modTime int64
	}
	var files []entry
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		if info, err := e.Info(); err == nil {
			files = append(files, entry{name: e.Name(), modTime: info.ModTime().UnixNano()})
		}
	}
	if len(files) == 0 {
		return "", nil
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime > files[j].modTime
	})
	return files[0].name, nil
}

func printHelp() {
	fmt.Print(`Codezilla - Modular AI-powered coding assistant

Usage:
  codezilla [options]

Options:
  -config string       Path to configuration file
  -provider string     Override LLM provider (ollama, openai, anthropic, gemini)
  -model string        Override default model (e.g., "qwen3:14b")
  -ollama-url string   Override Ollama API URL (e.g., "http://localhost:11434")
  -temperature float   Override temperature (0.0-1.0)
  -max-tokens int      Override max tokens
  -session string      Session ID to resume
  -resume              Resume the most recent session (same as 'resume' with no ID)
  -no-colors           Disable colored output
  -version             Show version information
  -help                Show this help message

Examples:
  # Run with default fancy UI
  codezilla

  # Resume the last session
  codezilla -resume
  codezilla resume

  # Resume a specific session
  codezilla -session session_20260417_190927_dff5cc42.jsonl
  codezilla resume session_20260417_190927_dff5cc42.jsonl

The modular architecture allows easy switching between different UI implementations
while keeping the core functionality unchanged.
`)
}
