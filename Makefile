.PHONY: build clean run dev dev-fast test lint fmt check all help memory-data memory-validate memory-eval

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
	@echo "  make memory-data  Generate synthetic memory-controller training data"
	@echo "  make memory-validate Validate memory-controller JSONL data"
	@echo "  make memory-eval MODEL=name Evaluate memory controller via Ollama"
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

# ── Memory controller training ────────────────────────────────────────────────

memory-data:
	python3 training/memory-controller/scripts/generate_synthetic.py \
		--count 1000 \
		--output training/memory-controller/synthetic.jsonl

memory-validate:
	python3 training/memory-controller/scripts/validate_dataset.py \
		training/memory-controller/examples.jsonl \
		training/memory-controller/synthetic.jsonl

memory-eval:
	@test -n "$(MODEL)" || (echo "MODEL is required, e.g. make memory-eval MODEL=codezilla-memory-controller" && exit 1)
	python3 training/memory-controller/scripts/eval_controller.py --model "$(MODEL)"

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench: build
	@echo "Running benchmark suite…"
	./target/release/$(BINARY_NAME) bench --tasks bench/tasks --output bench/results

bench-filter: build
	@echo "Running filtered benchmarks (FILTER=$(FILTER))…"
	./target/release/$(BINARY_NAME) bench --tasks bench/tasks --output bench/results --filter "$(FILTER)"
