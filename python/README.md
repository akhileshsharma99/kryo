# Kryo Python SDK

Python helper for [Kryo](https://github.com/akhileshsharma99/kryo) checkpoint signaling.

## Installation

```bash
pip install kryo
```

## Usage

```python
import kryo

# Setup (runs once, gets checkpointed)
model = load_model()
model.to("cuda")

kryo.checkpoint()  # Freeze here

# Inference (runs after restore)
result = model(input)
```

Then use the Kryo CLI:

```bash
# Create snapshot
kryo snapshot create --name mymodel -- python app.py

# Restore and run
kryo run --snapshot mymodel
```

## Requirements

- Linux (CRIU is Linux-only)
- Kryo CLI installed
- NVIDIA driver 550+ (for GPU checkpointing)
