# Qwen LLM example

Snapshot [Qwen 2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) with warm CUDA kernels. Tiny on purpose so the example fits a single consumer GPU.

Needs Linux, NVIDIA driver 550+, CRIU, `cuda-checkpoint`, the Kryo CLI, and this directory's venv (`kryo` is pulled from `python/` via uv).

## Setup

```bash
uv sync
```

## Usage

`kryo snapshot create` and `kryo run` must run as root:

```bash
sudo kryo snapshot create --name qwen -- uv run python qwen.py
sudo kryo run --snapshot qwen
```

Recreate the snapshot when the script, model, CUDA toolkit, or NVIDIA driver changes. Restored processes are not portable across machines.

## How it works

`qwen.py` calls `kryo.checkpoint()` after load and warmup. The CLI dumps there; `kryo run` resumes after it and runs one generate.

```python
import kryo

model = load_model()
model.to("cuda")
kryo.checkpoint()

result = model(input)
```
