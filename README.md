# Codezilla

A modular AI-powered coding assistant CLI tool that uses Ollama, OpenAI, Anthropic, or Gemini for LLM inference. Codezilla provides an interactive command-line interface (TUI) with advanced tool execution capabilities, multi-agent parallel workflows, and rich local context management.

*Note: Codezilla has been fully migrated from Go to Rust for improved performance, memory safety, and concurrency management.*

## Features

- **Interactive Ratatui TUI**: Chat with LLMs through a clean, fast terminal user interface with inline rendering options.
- **Multi-Provider Integration**: Supports Ollama (local), OpenAI, Anthropic (Claude), and Google Gemini.
- **Advanced Tool System**: 
  - File operations (read, write, list, multi-replace)
  - Shell command execution
  - Advanced ripgrep codebase search
  - Web searching via DuckDuckGo
  - Project structure scanning and repo mapping
- **Parallel Multiagent Workflows**: Decompose complex tasks into a DAG and execute them concurrently with dedicated agent workers.
- **Intelligent Context Management**: Automatic sliding-window token management and aggressive context trimming to stay within limits.
- **Persistent Sessions**: JSONL-based session recording and replay hydration.
- **Custom Skills**: Load custom instructions via Markdown files to augment the system prompt.

## Prerequisites

- Rust toolchain (cargo, rustc) 1.75+
- (Optional) Ollama installed and running locally
- API keys for cloud providers (if not using local Ollama)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/codezilla.git
cd codezilla
```

2. Build and install via Cargo:
```bash
cargo install --path .
# or using make
make install
```

## Usage

### Running Codezilla

```bash
# Run with default TUI
codezilla

# Run inline (no alternate screen)
codezilla --inline

# Run with specific config file
codezilla --config /path/to/config.yaml

# Resume latest session
codezilla --resume

# Show help
codezilla --help
```

### Available Slash Commands

Once inside the Codezilla TUI, you can use these slash commands in the input box:

- `/help` - Show available commands
- `/clear` - Clear the screen and reset conversation history
- `/version` - Show version information
- `/session` - Display session management info

### Configuration

Codezilla uses a `config.yaml` file (default location: `~/.config/codezilla/config.yaml`).

```yaml
llm:
  provider: ollama
  models:
    default: qwen2.5-coder:3b
    fast: llama3:8b
    heavy: deepseek-coder-v2

  ollama:
    base_url: "http://localhost:11434/v1"
  openai:
    api_key: ${OPENAI_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

max_tokens: 8192
temperature: 0.7
log_file: logs/codezilla.log
log_level: info
```

## Development

### Project Structure

```
codezilla/
├── Cargo.toml          # Rust dependencies
├── Makefile            # Build commands
├── src/
│   ├── main.rs         # Entry point
│   ├── app/            # App wiring and slash commands
│   ├── agent/          # Agent orchestrator and core state machine
│   ├── config/         # YAML config parser
│   ├── llm/            # Multi-provider LLM clients
│   ├── multiagent/     # Parallel worker DAG executor
│   ├── session/        # JSONL session persistence
│   ├── tools/          # Native tools (file, execute, web_search, etc.)
│   └── ui/             # Ratatui application model
└── logs/               # Application logs
```

### Make Commands

```bash
make build       # Build the application
make install     # Install via Cargo
make clean       # Remove build artifacts
make run         # Run in release mode
make test        # Run unit tests
make lint        # Run clippy
make fmt         # Format code
make check       # Run all checks (fmt, clippy)
make all         # Run checks and build
```