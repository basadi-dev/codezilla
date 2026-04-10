# Codezilla

A modular AI-powered coding assistant CLI tool that uses Ollama for local LLM inference. Codezilla provides an interactive command-line interface with advanced tool execution capabilities and flexible UI options.

## Features

- **Interactive CLI Interface**: Chat with local LLMs through a clean command-line interface
- **Ollama Integration**: Uses Ollama for local LLM inference with support for multiple models
- **Advanced Tool System**: 
  - File operations (read, write, batch read)
  - Shell command execution
  - Directory listing and file search
  - Project structure scanning
  - Markdown analysis
- **Flexible Tool Call Formats**: Supports XML, JSON, and bash code block formats for tool invocation
- **Multiple UI Modes**: Choose between fancy (with colors and emoji) or minimal UI
- **Context Management**: Maintains conversation history with token management
- **Configurable**: Extensive configuration options via file or command-line flags

## Prerequisites

- Go 1.26 or higher
- Ollama installed and running locally (default: http://localhost:11434)
- A compatible Ollama model installed (default: qwen2.5-coder:3b)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/codezilla.git
cd codezilla
```

2. Build the application:
```bash
make build
```

This creates the binary in `build/codezilla`

3. (Optional) Install to your PATH:
```bash
make install
```

## Usage

### Running Codezilla

```bash
# Run with default fancy UI
./build/codezilla

# Run with minimal UI
./build/codezilla -ui minimal

# Run without colors
./build/codezilla -no-colors

# Use a custom config file
./build/codezilla -config /path/to/config.json

# Show version
./build/codezilla -version

# Show help
./build/codezilla -help
```

### Available Commands

Once inside Codezilla, you can use these slash commands:

- `/help` - Show available commands
- `/exit` or `/quit` - Exit the application
- `/clear` - Clear the screen
- `/model [name]` - Switch to a different model or show current model
- `/models` - List available Ollama models
- `/context` - Show current context information
- `/reset` - Clear conversation context
- `/save <filename>` - Save conversation to file
- `/load <filename>` - Load conversation from file
- `/multiline` - Toggle multiline input mode
- `/version` - Show version information

### Configuration

Codezilla can be configured through:

1. **Command-line flags**:
   - `-config string` - Path to configuration file
   - `-ui string` - UI type: "fancy" (default) or "minimal"
   - `-no-colors` - Disable colored output
   - `-version` - Show version information
   - `-help` - Show help message

2. **Configuration file** (JSON format):
```json
{
  "model": "qwen2.5-coder:3b",
  "ollama_url": "http://localhost:11434/api",
  "max_tokens": 4000,
  "temperature": 0.7,
  "log_file": "codezilla.log",
  "log_level": "info",
  "no_color": false
}
```

Default config location: `~/.config/codezilla/config.json`

## Available Tools

Codezilla comes with a comprehensive set of tools that the AI assistant can use:

1. **File Operations**:
   - `fileRead` - Read contents of a file
   - `fileWrite` - Write content to a file
   - `listFiles` - List files in a directory

2. **Command Execution**:
   - `execute` - Execute shell commands

3. **Project Analysis**:
   - `projectScanAnalyzer` - Deep file-by-file analysis based on user queries
   - `diff` - Show differences between two text inputs

### Tool Call Formats

The AI can invoke tools using three different formats:

1. **XML Format**:
```xml
<tool>
  <name>fileRead</name>
  <params>
    <path>/path/to/file.txt</path>
  </params>
</tool>
```

2. **JSON Format**:
```json
{
  "tool": "fileRead",
  "params": {
    "path": "/path/to/file.txt"
  }
}
```

3. **Bash Code Blocks** (automatically converted to execute tool):
```bash
ls -la /tmp
```

## Development

### Project Structure

```
codezilla/
├── cmd/codezilla/      # Main application entry point
├── internal/
│   ├── agent/          # LLM agent and tool extraction logic
│   ├── cli/            # Command-line interface implementation
│   ├── core/           # Core application logic
│   ├── tools/          # Tool implementations
│   └── ui/             # UI implementations (fancy and minimal)
├── llm/ollama/         # Ollama API client
├── pkg/
│   ├── logger/         # Logging utilities
│   └── style/          # Terminal styling and colors
├── build/              # Build artifacts (created by make)
├── Makefile            # Build and development commands
├── go.mod              # Go module definition
└── README.md           # This file
```

### Make Commands

```bash
# Building
make build       # Build the application to build/codezilla
make install     # Install to $GOPATH/bin
make clean       # Remove build artifacts

# Running
make run         # Run with default fancy UI
make run-minimal # Run with minimal UI

# Development
make test        # Run all tests
make test-coverage # Run tests with coverage report
make fmt         # Format code
make vet         # Run go vet
make lint        # Run golangci-lint (if installed)
make tidy        # Tidy and verify go modules
make check       # Run all checks (tidy, fmt, vet, lint)
make all         # Run checks and build

# Help
make help        # Show all available commands
```

### Building from Source

Requirements:
- Go 1.26 or higher
- Make (optional, but recommended)

Without Make:
```bash
# Build
go build -o build/codezilla ./cmd/codezilla

# Run
./build/codezilla
```

With Make:
```bash
# Build and run
make build
make run
```

MIT License - see LICENSE file for details