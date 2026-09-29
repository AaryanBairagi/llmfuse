"""Exceptions raised by llmfuse."""

class LLMFuseError(Exception):
    """Base Class for every error llmfuse raises"""


class ProviderError(LLMFuseError):
    """
    A call to an LLM provider failed.
    `retryable says whether trying again could help.`
    """

    def __init__(
            self,
            message : str,
            *,
            provider : str | None = None,
            status_code : int | None = None,
            retryable : bool = True
        ) -> None :

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
            message : str,
            *,
            provider : str | None = None,
            retry_after : float | None = None
        ) -> None :

        super().__init__(message, provider = provider, status_code = 429, retryable = True)
        self.retry_after = retry_after


class RetryExhaustedError(LLMFuseError):
    """Every retry attempt"""
    def __init__(
        self,
        attempts : int,
        last_error : BaseException     
    ) -> None :
        super().__init__(f"Gave up after {attempts} attempt(s) : {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error
