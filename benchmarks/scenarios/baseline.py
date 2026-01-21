"""Baseline scenario: stdlib only."""

from _base import Timing, output_results

with Timing("import"):
    import json
    import pathlib

    # Use imports to avoid F401
    _json = json.dumps
    _path = pathlib.Path

output_results()
