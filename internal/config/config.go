package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

// Config holds the application configuration
type Config struct {
	// LLM Multi-Provider configuration
	LLM LLMConfig `json:"llm" yaml:"llm"`

	// Generation parameters (common to all providers)
	Temperature   float32 `json:"temperature" yaml:"temperature"`
	MaxTokens     int     `json:"max_tokens" yaml:"max_tokens"`
	MaxIterations int     `json:"max_iterations" yaml:"max_iterations"`
	SystemPrompt  string  `json:"system_prompt" yaml:"system_prompt"`

	// Loop detection: kill agent if same tool+args repeated consecutively.
	// 0 = use defaults (window=10, max_repeat=3).
	LoopDetectWindow    int `json:"loop_detect_window" yaml:"loop_detect_window"`
	LoopDetectMaxRepeat int `json:"loop_detect_max_repeat" yaml:"loop_detect_max_repeat"`

	// Log configuration
	LogFile   string `json:"log_file" yaml:"log_file"`
	LogLevel  string `json:"log_level" yaml:"log_level"`
	LogSilent bool   `json:"log_silent" yaml:"log_silent"`

	// Context management
	RetainContext   bool   `json:"retain_context" yaml:"retain_context"`
	MaxContextChars int    `json:"max_context_chars" yaml:"max_context_chars"`
	HistoryFile     string `json:"history_file" yaml:"history_file"`

	// Agent behavior
	AutoPlan bool `json:"auto_plan" yaml:"auto_plan"`

	// Embedded metasearch settings (no API keys required)
	Metasearch MetasearchSettings `json:"metasearch" yaml:"metasearch"`

	// Permission settings
	DangerousToolsWarn  bool              `json:"dangerous_tools_warn" yaml:"dangerous_tools_warn"`
	AlwaysAskPermission bool              `json:"always_ask_permission" yaml:"always_ask_permission"`
	ToolPermissions     map[string]string `json:"tool_permissions" yaml:"tool_permissions"`

	// UI settings
	ForceColor bool `json:"force_color" yaml:"force_color"`
	NoColor    bool `json:"no_color" yaml:"no_color"`

	// Working directory
	WorkingDirectory string `json:"working_directory" yaml:"working_directory"`

	// Analyzer settings
	AnalyzerSettings AnalyzerSettings `json:"analyzer_settings" yaml:"analyzer_settings"`

	// Skills system
	Skills SkillsConfig `json:"skills" yaml:"skills"`
}

// MetasearchSettings configures the embedded metasearch engine.
type MetasearchSettings struct {
	EnableBing     bool `json:"enable_bing" yaml:"enable_bing"`             // Enable Bing HTML scraping (opt-in, more fragile)
	TimeoutSeconds int  `json:"timeout_seconds" yaml:"timeout_seconds"`     // Per-backend timeout in seconds
	MaxResults     int  `json:"max_results" yaml:"max_results"`             // Default number of results to return
	JitterMs       int  `json:"jitter_ms" yaml:"jitter_ms"`                 // Random delay per request in ms (0 = off)
}

// SkillsConfig configures the skills system.
type SkillsConfig struct {
	// Dir is the directory to scan for skill markdown files.
	Dir string `json:"dir" yaml:"dir"`
	// ActiveSkills lists skill names to activate automatically at startup.
	ActiveSkills []string `json:"active_skills" yaml:"active_skills"`
}

// AnalyzerSettings contains configuration for the file analyzer
type AnalyzerSettings struct {
	UseLLM             bool    `json:"use_llm" yaml:"use_llm"`                         // Use LLM for file analysis
	Concurrency        int     `json:"concurrency" yaml:"concurrency"`                 // Number of files to analyze concurrently
	RelevanceThreshold float64 `json:"relevance_threshold" yaml:"relevance_threshold"` // Minimum relevance score
	AnalysisTimeout    int     `json:"analysis_timeout" yaml:"analysis_timeout"`       // Timeout per file in seconds
	MaxFileSize        int64   `json:"max_file_size" yaml:"max_file_size"`             // Maximum file size to analyze
}

type LLMModelsConfig struct {
	Default  string `json:"default" yaml:"default"`
	Planner  string `json:"planner,omitempty" yaml:"planner,omitempty"`
	SubAgent string `json:"sub_agent,omitempty" yaml:"sub_agent,omitempty"`
	Analyzer string `json:"analyzer,omitempty" yaml:"analyzer,omitempty"`
}

type LLMAPIKeysConfig struct {
	OpenAI    string `json:"openai" yaml:"openai"`
	Gemini    string `json:"gemini" yaml:"gemini"`
	Anthropic string `json:"anthropic" yaml:"anthropic"`
	Ollama    string `json:"ollama" yaml:"ollama"`
}

type OpenAIConfig struct {
	BaseURL  string `json:"base_url,omitempty" yaml:"base_url,omitempty"`
}

type OllamaConfig struct {
	BaseURL  string            `json:"base_url" yaml:"base_url"`
	AuthType string            `json:"auth_type,omitempty" yaml:"auth_type,omitempty"`
	Username string            `json:"username,omitempty" yaml:"username,omitempty"`
	Password string            `json:"password,omitempty" yaml:"password,omitempty"`
	Headers  map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
}

type LLMConfig struct {
	Provider string           `json:"provider" yaml:"provider"`
	Models   LLMModelsConfig  `json:"models" yaml:"models"`
	APIKeys  LLMAPIKeysConfig `json:"api_keys" yaml:"api_keys"`
	Ollama   OllamaConfig     `json:"ollama" yaml:"ollama"`
	OpenAI   OpenAIConfig     `json:"openai,omitempty" yaml:"openai,omitempty"`
}

// defaultSystemPrompt returns the default system prompt with working directory.
func defaultSystemPrompt(cwd string) string {
	return fmt.Sprintf(`You are Codezilla, a helpful AI coding assistant. You have access to various tools that allow you to interact with the local system, read and write files, execute commands, and more.

Current working directory: %s

IMPORTANT RULES FOR TOOL USAGE:
1. When the user asks to see, show, display, read, or print a file, you MUST use the fileRead tool with the appropriate file_path parameter. DO NOT say you cannot read files — you CAN via the fileRead tool.
2. When the user asks to list files, explore, or scan the project, use the listFiles tool.
3. When the user asks to run a command, you MUST explicitly use the execute tool via XML or JSON formats. NEVER output loose markdown bash blocks expecting them to be executed.
4. When the user asks to write or create a file, use the fileWrite tool.
5. When the user asks about current events, needs up-to-date documentation, or asks you to look something up on the web, use the webSearch tool.
6. For complex multi-step tasks, explain your plan step-by-step before executing.
7. Always show what tool you are using and why.
8. When the user refers to "the project", "this project", or uses relative paths, assume they mean the current working directory.
9. Always reply in markdown format.
10. Be concise, accurate, and helpful.
11. DO NOT make up file contents or command outputs — always use the appropriate tool to get real data.`, cwd)
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	// Get current working directory
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}

	return &Config{
		LLM: LLMConfig{
			Provider: "ollama",
			Models: LLMModelsConfig{
				Default: "qwen3-coder:480b", // Best open-weight coder
			},
			Ollama: OllamaConfig{
				BaseURL: "http://localhost:11434",
			},
		},
		Temperature:         0.7,
		MaxTokens:           1024 * 32,
		MaxIterations:       0, // 0 = unlimited tool iterations
		SystemPrompt:        defaultSystemPrompt(cwd),
		LogFile:             filepath.Join("logs", "codezilla.log"),
		LogLevel:            "info",
		LogSilent:           false,
		RetainContext:       true,
		MaxContextChars:     50000,
		HistoryFile:         filepath.Join(getConfigDir(), "history"),
		AutoPlan:            false,
		DangerousToolsWarn:  true,
		AlwaysAskPermission: false,
		ToolPermissions: map[string]string{
			"fileRead":            "never_ask",
			"listFiles":           "never_ask",
			"projectScanAnalyzer": "never_ask",
			"fileWrite":           "always_ask",
			"execute":             "always_ask",
		},
		ForceColor:       false,
		NoColor:          false,
		WorkingDirectory: cwd,
		Metasearch: MetasearchSettings{
			EnableBing:     false,
			TimeoutSeconds: 8,
			MaxResults:     5,
			JitterMs:       0,
		},
		AnalyzerSettings: AnalyzerSettings{
			UseLLM:             true,
			Concurrency:        5,
			RelevanceThreshold: 0.3,
			AnalysisTimeout:    30,
			MaxFileSize:        1024 * 1024, // 1MB
		},
		Skills: SkillsConfig{
			Dir:          "./skills",
			ActiveSkills: []string{},
		},
	}
}

// LoadConfig loads configuration from a file
func LoadConfig(path string) (*Config, error) {
	config := DefaultConfig()

	// Optionally load a .env file from the current directory, ignoring errors if it doesn't exist.
	// This will populate process-level environment variables seamlessly.
	_ = godotenv.Load()

	// If path doesn't exist, proceed with evaluating env vars
	if _, err := os.Stat(path); err == nil {
		// Read the config file
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}

		// Parse YAML
		if err := yaml.Unmarshal(data, config); err != nil {
			return nil, fmt.Errorf("failed to parse yaml config file: %w", err)
		}
	}

	// Ensure tool permissions map is initialized
	if config.ToolPermissions == nil {
		config.ToolPermissions = make(map[string]string)
	}

	// Always use current working directory
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}
	config.WorkingDirectory = cwd

	// If the config file didn't provide a system prompt (or it's the YAML default
	// placeholder), use the enhanced default. Otherwise, append working directory
	// context to the user-provided prompt so it always knows where it is.
	if config.SystemPrompt == "" {
		config.SystemPrompt = defaultSystemPrompt(cwd)
	} else {
		// Append working directory and tool usage rules if not already present
		if !strings.Contains(config.SystemPrompt, "Current working directory:") {
			config.SystemPrompt += fmt.Sprintf("\n\nCurrent working directory: %s", cwd)
		}
		if !strings.Contains(config.SystemPrompt, "IMPORTANT RULES FOR TOOL USAGE") {
			config.SystemPrompt += `

IMPORTANT RULES FOR TOOL USAGE:
1. When the user asks to see, show, display, read, or print a file, you MUST use the fileRead tool with the appropriate file_path parameter. DO NOT say you cannot read files — you CAN via the fileRead tool.
2. When the user asks to list files, explore, or scan the project, use the listFiles tool.
3. When the user asks to run a command, you MUST explicitly use the execute tool via XML or JSON formats. NEVER output loose markdown bash blocks expecting them to be executed.
4. When the user asks to write or create a file, use the fileWrite tool.
5. For complex multi-step tasks, explain your plan step-by-step before executing.
6. Always show what tool you are using and why.
7. Always reply in markdown format.
8. Be concise, accurate, and helpful.
9. DO NOT make up file contents or command outputs — always use the appropriate tool to get real data.`
		}
	}

	// Check environment variables for authentication (these override config file)
	if apiKey := os.Getenv("OLLAMA_API_KEY"); apiKey != "" {
		config.LLM.APIKeys.Ollama = apiKey
		if config.LLM.Ollama.AuthType == "" {
			config.LLM.Ollama.AuthType = "bearer"
		}
	}
	if username := os.Getenv("OLLAMA_USERNAME"); username != "" {
		config.LLM.Ollama.Username = username
		config.LLM.Ollama.AuthType = "basic"
	}
	if password := os.Getenv("OLLAMA_PASSWORD"); password != "" {
		config.LLM.Ollama.Password = password
	}
	if baseURL := os.Getenv("OLLAMA_BASE_URL"); baseURL != "" {
		config.LLM.Ollama.BaseURL = baseURL
	}
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		config.LLM.OpenAI.BaseURL = baseURL
	}
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		config.LLM.APIKeys.OpenAI = apiKey
	}
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		config.LLM.APIKeys.Anthropic = apiKey
	}
	if apiKey := os.Getenv("GEMINI_API_KEY"); apiKey != "" {
		config.LLM.APIKeys.Gemini = apiKey
	}

	return config, nil
}

// SaveConfig saves configuration to a file
func SaveConfig(config *Config, path string) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	// Marshal to YAML
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal yaml config: %w", err)
	}

	// Write to file with secure permissions
	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// getConfigDir returns the directory for configuration files
func getConfigDir() string {
	// Get user config directory
	configDir, err := os.UserConfigDir()
	if err != nil {
		// Fall back to current directory
		return "./config"
	}

	// Use application-specific subdirectory
	return filepath.Join(configDir, "codezilla")
}
