import time
from collections.abc import Callable, Sequence
from functools import partial

from llmfuse._guards import build_guards
from llmfuse.breaker import CircuitBreaker, CircuitState
from llmfuse.errors import (
    AllProvidersFailedError,
    CircuitOpenError,
    LLMFuseError,
    ThrottledError,
)
from llmfuse.limiter import TokenBucket
from llmfuse.provider import Provider, Response
from llmfuse.retry import RetryPolicy, retry_call


class FuseClient:
    
    def __init__(
        self,
        *,
        providers: Sequence[Provider],
        retry: RetryPolicy | None = None,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        requests_per_minute : float | None = None,
        burst: int = 5,    #bucket capacity
        max_wait : float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:

        self.providers = list(providers)
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self.max_wait = max_wait

        self._breakers, self._buckets = build_guards(
            providers,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            requests_per_minute=requests_per_minute,
            burst=burst,
            clock=clock,
        )


    def circuit_state(self, provider_name: str) -> CircuitState:
        return self._breakers[provider_name].state


    def complete(self, prompt: str) -> Response:
        errors: dict[str, BaseException] = {}
        #Loop over providers and instantiate a breaker, token bucket for each provider
        for provider in self.providers:
            breaker = self._breakers[provider.name]
            if not breaker.allow_request():
                errors[provider.name] = CircuitOpenError(provider.name)
                continue

            try:
                text = retry_call(
                    partial(self._call_with_token, provider, prompt),
                    self.retry,
                    sleep=self._sleep,
                )

            except ThrottledError as error:
                errors[provider.name] = error
                continue

            except LLMFuseError as error:
                breaker.record_failure()
                errors[provider.name] = error
                continue

            breaker.record_success()
            return Response(text=text, provider=provider.name)

        raise AllProvidersFailedError(errors)


    def _call_with_token(self, provider: Provider, prompt: str) -> str:
        """One attempt: get a token (waiting up to max_wait), then call the provider."""
        bucket = self._buckets.get(provider.name)
        if bucket is None:
            return provider.complete(prompt) #rate limiting is off

        wait = bucket.time_until_available()
        
        if wait > self.max_wait:
            raise ThrottledError(provider.name, wait)
        
        if wait > 0:
            self._sleep(wait) 
        bucket.consume()
        return provider.complete(prompt)
