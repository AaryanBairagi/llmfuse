"""llmfuse: production-grade reliability for LLM calls."""

from llmfuse.client import FuseClient
from llmfuse.provider import Response, Provider
from llmfuse.errors import (
    LLMFuseError,
    ProviderError,
    RateLimitError,
    RetryExhaustedError,
    AllProvidersFailedError,
    CircuitOpenError,
)
from llmfuse.retry import RetryPolicy, is_retryable, retry_call

__version__ = "0.1.0.dev0"

__all__ = [
    "FuseClient",
    "Response",
    "Provider",
    "LLMFuseError",
    "ProviderError",
    "RateLimitError",
    "RetryExhaustedError",
    "AllProvidersFailedError",
    "CircuitOpenError",
    "RetryPolicy",
    "is_retryable",
    "retry_call",
]
