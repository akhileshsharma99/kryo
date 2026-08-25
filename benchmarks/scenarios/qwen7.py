"""Qwen 2.5-7B load, warmup generate, then checkpoint."""

import torch
from _base import maybe_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

_ = torch.zeros(1, device="cuda")
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=1)

maybe_checkpoint()
