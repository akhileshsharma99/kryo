"""NumPy only scenario: numpy import + basic operation."""

from _base import Timing, output_results

with Timing("import"):
    import numpy as np

with Timing("first_inference"):
    x = np.random.randn(1000, 1000)
    y = x @ x.T

output_results()
