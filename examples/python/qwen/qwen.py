"""Qwen inference example with Kryo checkpointing.

Usage:
    # Create snapshot (runs setup, checkpoints)
    sudo kryo snapshot create --name qwen -- uv run python qwen.py

    # Restore and run (skips setup, runs inference)
    sudo kryo run --snapshot qwen
"""

import kryo
from transformers import AutoModelForCausalLM, AutoTokenizer

# === SETUP PHASE ===
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

print("Warming up CUDA kernels...")
model.generate(**tokenizer("warmup", return_tensors="pt").cuda(), max_new_tokens=1)

print("Setup complete - signaling ready")
kryo.checkpoint()  # Signal ready, wait for restore

# === INFERENCE PHASE (runs after restore) ===
print("Running inference...")
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
