"""torch.compile (OpenAI Triton kernels + CUDA graphs) on Qwen 2.5-7B.

Named torch_compile.py so it does not shadow the pip `triton` package.
"""

import torch
from _base import checkpoint_or_exit
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


def infer() -> None:
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
        torch.cuda.synchronize()


for _ in range(3):
    infer()
checkpoint_or_exit(resume=infer)
