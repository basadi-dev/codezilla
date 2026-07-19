#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-training/memory-controller/axolotl/qwen3-4b-lora.yml}"

if ! command -v axolotl >/dev/null 2>&1; then
  echo "error: axolotl is not installed" >&2
  echo "install in a GPU-capable Python environment, then rerun:" >&2
  echo "  axolotl train ${CONFIG}" >&2
  exit 127
fi

python3 training/memory-controller/scripts/validate_dataset.py \
  training/memory-controller/examples.jsonl \
  training/memory-controller/synthetic.jsonl

axolotl train "${CONFIG}"
