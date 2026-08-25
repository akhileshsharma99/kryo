"""PyTorch CUDA init + a small GPU matmul."""

import torch
from _base import checkpoint_or_exit

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

_ = torch.zeros(1, device="cuda")
x = torch.randn(1000, 1000, device="cuda")


def infer() -> None:
    y = x @ x.T
    torch.cuda.synchronize()
    del y


infer()
checkpoint_or_exit(resume=infer)
