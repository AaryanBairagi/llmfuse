import asyncio
import time
from collections.abc import Callable, Sequence

from llmfuse._guards import build_guards
from llmfuse.breaker import CircuitState
from llmfuse.errors import (
    AllProvidersFailedError,
    CircuitOpenError,
    LLMFuseError,
    ThrottledError,
)
from llmfuse.provider import AsyncProvider, Response
from llmfuse.retry import RetryPolicy, retry_call_async


class AsyncFuseClient:
    def __init__(
            self,
            *,
            providers: Sequence[AsyncProvider],
            retry: RetryPolicy | None = None,
            failure_threshold: int = 5,
            reset_timeout: float = 30.0,
            requests_per_minute : float | None = None,
            burst: int = 5,  
            max_wait : float = 1.0,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.monotonic,
    ) -> None:
        
        self.providers = providers
        self.retry = retry or RetryPolicy(),
        self.max_wait = max_wait,

        self._sleep = sleep,
        self._breakers, self._buckets = build_guards(
            providers,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            requests_per_minute=requests_per_minute,
            burst=burst,
            clock=clock,
        )

    def circuit_state(self, provider_name :str) -> CircuitState:
        return self._breakers[provider_name].state

    async def complete(self, prompt:str) -> str:
        errors : dict[str, BaseException] = {}

        for provider in self.providers:
            breaker = self._breakers[provider.name]
            if not breaker.allow_request():
                errors[provider.name] = CircuitOpenError(provider.name)
                continue

            try:
                text = await retry_call_async() 

            except ThrottledError as error:
                errors[provider.name] = error

            except LLMFuseError as error:
                breaker.record_failure()
                errors[provider.name] = error
                continue
            
            breaker.record_success()
            return Response(text=text, provider=provider.name)
        
        raise AllProvidersFailedError(errors)