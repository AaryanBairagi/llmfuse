class LLMFuseError(Exception):
    """Base Class for every error llmfuse raises"""


class ProviderError(LLMFuseError):
    """
    A call to an LLM provider failed.
    `retryable says whether trying again could help.`
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool = True,
    ) -> None:

        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class RateLimitError(ProviderError):
    """
    The provider said "slow down" (usualy HTTP 429)
    If the provider tells us how long to wait (the ``Retry-After`` header)
    it goes in ``retry_after`` and retry logic respects it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:

        super().__init__(message, provider=provider, status_code=429, retryable=True)
        self.retry_after = retry_after


class RetryExhaustedError(LLMFuseError):
    """Every retry attempt."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(
            f"Gave up after {attempts} attempt(s) : {last_error!r}"
        )  # !r means print the raw error as it is
        self.attempts = attempts
        self.last_error = last_error


class AllProvidersFailedError(LLMFuseError):
    """Every provider failed to respond."""

    def __init__(self, errors: dict[str, BaseException]):
        summary = "; ".join(f"{name} : {error!r}" for name, error in errors.items())
        super().__init__(f"All Providers Failed : {summary}")
        self.errors = errors


class CircuitOpenError(LLMFuseError):
    """We skipped a provider because its circuit breaker is open."""

    def __init__(self, provider: str):
        super().__init__(
            f"Circuit Breaker for Provider : {provider!r} skipped. Redirecting to next provider."
        )
        self.provider = provider


class ThrottledError(LLMFuseError):
    """We held back from calling a provider because we'd exceed its rate limit."""

    def __init__(self, provider: str, wait: float) -> None:
        super().__init__(
            f"Throttled : Provider {provider!r} would exceed its rate limit. Wait for {wait:.2f} seconds."
        )
        self.provider = provider
        self.wait = wait
