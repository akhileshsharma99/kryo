"""vLLM engine start + CUDA-graph capture on Qwen 2.5-7B, then checkpoint.

Named vllm_engine.py so it does not shadow the pip `vllm` package.
"""

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from _base import checkpoint_or_exit
from vllm import LLM, SamplingParams

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

_ = torch.zeros(1, device="cuda")

llm = LLM(
    model="Qwen/Qwen2.5-7B",
    dtype="bfloat16",
    max_model_len=512,
    max_num_seqs=4,
    gpu_memory_utilization=0.85,
    disable_log_stats=True,
    enforce_eager=False,
    distributed_executor_backend="uni",
)
params = SamplingParams(max_tokens=1, temperature=0.0)


def infer() -> None:
    _ = llm.generate(["Hello, world!"], params)


infer()
checkpoint_or_exit(resume=infer)
