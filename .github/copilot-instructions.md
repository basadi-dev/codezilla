# Codezilla Developer Guide

## Build, Test, and Lint Commands

```bash
# Build the application to bin/codezilla
make build

# Run all tests
make test

# Run tests with coverage report
make test-coverage

# Run a single test
go test -v ./internal/tools/... -run TestToolName

# Format code
make fmt

# Run go vet
make vet

# Run linter (requires golangci-lint)
make lint

# Run all checks (fmt, vet, lint)
make check

# Tidy and verify dependencies
make tidy

# Generate SQL code with sqlc
make db-generate

# Reset database
make db-reset

# Full build pipeline
make all
```

## High-Level Architecture

Codezilla is a modular AI-powered coding assistant CLI built with Go and BubbleTea.

### Core Components

- **UI Layer** (`internal/ui/`): BubbleTea-based TUI with fancy and minimal modes
- **Agent Layer** (`internal/agent/`): LLM agent with tool extraction, orchestration, and loop detection
- **Core Layer** (`internal/core/`): Application orchestration, session management, LLM client
- **Tools Layer** (`internal/tools/`): 20+ tools for file operations, shell execution, project analysis, web search
- **Database** (`internal/db/`): SQLite for persistent session history and config storage
- **Skills** (`internal/skills/`): Extensible command system (e.g., caveman mode)

### LLM Providers

Uses `any-llm-go` for multi-provider support:
- Ollama (local or cloud)
- OpenAI (and compatible APIs)
- Anthropic
- Google Gemini

### Model Routing

The auto-router selects models based on request complexity:
- **fast**: Lightweight model for trivial exchanges (greetings, yes/no)
- **default**: Main workhorse for most tasks
- **heavy**: Powerful model for complex multi-step tasks

Configure in `config.yaml` under `llm.models`.

## Key Conventions

### Configuration

- Default config: `~/.config/codezilla/config.yaml`
- Can override with `-config` flag
- Supports YAML format

### Tool Development

Tools implement the `Tool` interface:
```go
type Tool interface {
    Name() string
    Description() string
    ParameterSchema() JSONSchema
    Execute(ctx context.Context, params map[string]interface{}) (interface{}, error)
}
```

Register tools in `internal/tools/tools.go`.

### Session Handling

- Sessions stored in SQLite at `~/.codezilla/codezilla.db`
- Session events logged to `logs/sessions/*.jsonl`
- Resume with `-session` flag or `codezilla resume`

### Logging

- Default log file: `logs/codezilla.log`
- Configurable log level in `config.yaml`
- Use `pkg/logger` for application logging

### Code Style

- Uses golangci-lint (see `.golangci.yml`)
- Required linters: errcheck, gosimple, govet, ineffassign, staticcheck, typecheck, unused, gosec, gofmt, goimports, misspell, revive
- Go version: 1.26+