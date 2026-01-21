# Qwen LLM Example

Snapshot Qwen 2.5-0.5B with warm CUDA kernels.

## Setup

```bash
uv sync
```

## Usage

```bash
# Create snapshot
kryo snapshot create --name qwen -- uv run python setup.py

# Run inference (sub-second cold start)
kryo run --snapshot qwen -- uv run python inference.py
```
