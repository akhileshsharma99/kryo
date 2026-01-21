# Qwen LLM Example

Snapshot Qwen 2.5-0.5B with warm CUDA kernels.

## Setup

```bash
uv sync
```

## Usage

```bash
# Create snapshot (runs setup, freezes at kryo.checkpoint())
kryo snapshot create --name qwen -- uv run python qwen.py

# Restore and run (resumes from checkpoint, runs inference)
kryo run --snapshot qwen
```

## How it works

The `qwen.py` script uses `kryo.checkpoint()` to signal when setup is complete:

```python
import kryo

# Setup (runs once, gets checkpointed)
model = load_model()
model.to("cuda")
kryo.checkpoint()  # Freeze here

# Inference (runs after restore)
result = model(input)
```
