.PHONY: build clean run run-debug test lint vet fmt help install check all tidy db-generate db-migrate db-clean db-reset

BINARY_NAME=codezilla
BUILD_DIR=bin
LOG_DIR=logs
GO=go
GOFLAGS=-trimpath
CMD_DIR=cmd/codezilla
VERSION=$(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS=-ldflags "-X main.versionInfo=$(VERSION)"

all: check build

help:
	@echo "Available commands:"
	@echo "  make build      - Build the application"
	@echo "  make install    - Install the application to GOPATH/bin"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make run        - Run the application"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make fmt        - Format code"
	@echo "  make vet        - Run go vet"
	@echo "  make check      - Run fmt, vet and lint"
	@echo "  make all        - Run check and build"
	@echo "  make tidy       - Tidy and verify dependencies"
	@echo ""
	@echo "UI commands:"
	@echo "  make run         - Run with default fancy UI"
	@echo "  make run-minimal - Run with minimal UI"

build:
	@echo "Building $(BINARY_NAME) $(VERSION)..."
	@mkdir -p $(BUILD_DIR)
	$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) ./$(CMD_DIR)

install:
	$(GO) install $(GOFLAGS) $(LDFLAGS) ./$(CMD_DIR)

clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(BINARY_NAME)

run: build
	@echo "Running with default fancy UI..."
	@./$(BUILD_DIR)/$(BINARY_NAME) -model cogito:32b


test:
	$(GO) test -v ./...

test-coverage:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out

lint:
	@if command -v golangci-lint >/dev/null 2>&1; then \
		echo "Running golangci-lint..."; \
		golangci-lint run || echo "Lint check failed, but continuing..."; \
	else \
		echo "Warning: golangci-lint not found in PATH."; \
		echo "To install it, run: curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $$(go env GOPATH)/bin"; \
		echo "Skipping lint check..."; \
	fi

vet:
	$(GO) vet ./...

fmt:
	$(GO) fmt ./...

tidy:
	$(GO) mod tidy
	$(GO) mod verify

check: tidy fmt vet
	@$(MAKE) --no-print-directory lint || true

# UI-specific run targets
.PHONY: run-minimal
run-minimal: build
	@echo "Running with minimal UI..."
	@./$(BUILD_DIR)/$(BINARY_NAME) -ui minimal

# Database commands
.PHONY: db-generate
db-generate:
	@echo "Generating SQL code with sqlc..."
	@if command -v sqlc >/dev/null 2>&1; then \
		sqlc generate; \
	else \
		echo "Warning: sqlc not found in PATH."; \
		echo "To install it, run: go install github.com/sqlc-dev/sqlc/cmd/sqlc@latest"; \
		echo "Skipping code generation..."; \
	fi

.PHONY: db-migrate
db-migrate:
	@echo "Preparing database directory..."
	@mkdir -p ~/.codezilla
	@echo "Database directory ready: ~/.codezilla/codezilla.db (created on first run)"


.PHONY: db-clean
db-clean:
	@echo "Cleaning database files..."
	@rm -rf ~/.codezilla/codezilla.db*
	@echo "Database files removed"

.PHONY: db-reset
db-reset: db-clean db-generate
	@echo "Database reset complete"