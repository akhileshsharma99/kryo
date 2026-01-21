"""PyTorch CPU only scenario."""

from _base import Timing, output_results

with Timing("import"):
    import torch

with Timing("first_inference"):
    x = torch.randn(1000, 1000)
    y = x @ x.T

output_results()
