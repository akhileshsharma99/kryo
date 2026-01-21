"""PyTorch + CUDA scenario."""

from _base import Timing, output_results

with Timing("import"):
    import torch

with Timing("cuda_init"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    _ = torch.zeros(1, device="cuda")

with Timing("first_inference"):
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()

output_results()
