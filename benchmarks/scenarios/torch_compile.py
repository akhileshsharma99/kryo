"""torch.compile (OpenAI Triton kernels + CUDA graphs) on Qwen 2.5-7B.

This is the in-process compile warmup a PyTorch serving pod pays on first
start. It is not NVIDIA Triton Inference Server.

Named torch_compile.py so it does not shadow the pip `triton` package
(scenarios/ is on sys.path when these scripts run).
"""

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
model.eval()
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
model = torch.compile(model, mode="reduce-overhead")
with torch.no_grad():
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=1)
        torch.cuda.synchronize()

maybe_checkpoint()
