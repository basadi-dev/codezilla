.PHONY: build clean run dev dev-fast test lint fmt check all help

BINARY_NAME=codezilla
CARGO=cargo

all: check build

help:
	@echo ""
	@echo "  $(BINARY_NAME) — build targets"
	@echo ""
	@echo "  make dev          Fast debug run (incremental, no optimisation)"
	@echo "  make dev-fast     Fastest possible cold compile (no debug info)"
	@echo "  make run          Release run (fully optimised)"
	@echo "  make build        Release build"
	@echo "  make install      Install via cargo"
	@echo "  make test         Run tests"
	@echo "  make lint         Run clippy"
	@echo "  make fmt          Format code"
	@echo "  make check        fmt + clippy + cargo check"
	@echo "  make all          check + release build"
	@echo ""
	@echo "  Tip: brew install llvm  then uncomment .cargo/config.toml"
	@echo "       to use lld for even faster links."
	@echo ""

# ── Development (fast, debug) ─────────────────────────────────────────────────

# Default dev target: debug profile (incremental, opt=0, minimal debug info).
# First build is still cold; subsequent builds are very fast.
dev:
	$(CARGO) run

# Absolute fastest compile — strips all debug info, only useful to verify
# the code compiles (no useful backtraces).
dev-fast:
	$(CARGO) run --profile dev-fast

# ── Production ────────────────────────────────────────────────────────────────

build:
	@echo "Building $(BINARY_NAME) (release)…"
	$(CARGO) build --release

install:
	@echo "Installing $(BINARY_NAME)…"
	$(CARGO) install --path .

run:
	$(CARGO) run --release

# ── Quality ───────────────────────────────────────────────────────────────────

clean:
	$(CARGO) clean

test:
	$(CARGO) test

lint:
	$(CARGO) clippy -- -D warnings

fmt:
	$(CARGO) fmt

check: fmt
	$(CARGO) clippy -- -D warnings
	$(CARGO) check