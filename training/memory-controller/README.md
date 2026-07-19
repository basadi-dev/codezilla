# Memory Controller Training

This directory contains the post-training workflow for Codezilla's memory controller.

## 1. Generate Data

```bash
python3 training/memory-controller/scripts/generate_synthetic.py \
  --count 1000 \
  --output training/memory-controller/synthetic.jsonl
```

## 2. Validate Data

```bash
python3 training/memory-controller/scripts/validate_dataset.py \
  training/memory-controller/examples.jsonl \
  training/memory-controller/synthetic.jsonl
```

## 3. Train LoRA

Install Axolotl in a Python environment with GPU support, then run:

```bash
axolotl train training/memory-controller/axolotl/qwen3-4b-lora.yml
```

The template uses `Qwen/Qwen3-4B`. Change `base_model` if you choose a different base. The Ollama `FROM` model and the adapter base must match.

## 4. Create Ollama Model

After training/exporting the adapter, update `ollama/Modelfile.memory-controller` with:

```text
ADAPTER ../out/qwen3-4b-memory-lora
```

Then:

```bash
cd training/memory-controller/ollama
ollama create codezilla-memory-controller -f Modelfile.memory-controller
```

## 5. Evaluate

```bash
python3 training/memory-controller/scripts/eval_controller.py \
  --model codezilla-memory-controller \
  --base-url http://localhost:11434
```

The model must emit only a `MemoryControllerPlan` JSON object. Codezilla applies that plan through the `run_memory_plan` tool.
