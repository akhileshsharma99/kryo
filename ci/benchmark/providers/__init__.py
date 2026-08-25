"""Provider registry. Add new clouds here; the scheduler stays unchanged."""

from __future__ import annotations

from providers.base import Provider
from providers.lambda_cloud import LambdaProvider


def get_provider(name: str) -> Provider:
    """Return a provider implementation by YAML `provider:` name."""
    if name == "lambda":
        return LambdaProvider()
    raise ValueError(f"unknown provider {name!r} (supported: lambda)")
