.PHONY: build clean run test lint fmt check all

BINARY_NAME=codezilla
CARGO=cargo

all: check build

help:
	@echo "Available commands:"
	@echo "  make build      - Build the application"
	@echo "  make install    - Install the application via cargo"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make run        - Run the application"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run clippy"
	@echo "  make fmt        - Format code"
	@echo "  make check      - Run fmt, clippy, and build check"
	@echo "  make all        - Run check and build"

build:
	@echo "Building $(BINARY_NAME)..."
	$(CARGO) build --release

install:
	@echo "Installing $(BINARY_NAME)..."
	$(CARGO) install --path .

clean:
	@echo "Cleaning up..."
	$(CARGO) clean

run:
	@echo "Running $(BINARY_NAME)..."
	$(CARGO) run --release

test:
	$(CARGO) test

lint:
	$(CARGO) clippy -- -D warnings

fmt:
	$(CARGO) fmt

check: fmt
	$(CARGO) clippy -- -D warnings
	$(CARGO) check