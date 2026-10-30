import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from typing import Literal

from llmfuse._guards import build_guards
from llmfuse.breaker import CircuitState
from llmfuse.errors import (
    AllProvidersFailedError,
    CircuitOpenError,
    LLMFuseError,
    ProviderError,
    ThrottledError,
)
from llmfuse.provider import AsyncProvider, Response
from llmfuse.retry import RetryPolicy, async_retry_call

Routing = Literal["in_order", "least_busy"]

class AsyncFuseClient:
    """Same reliability rules as FuseClient, but `complete` is a coroutine (use `await`)."""

    def __init__(
        self,
        *,
        providers: Sequence[AsyncProvider],
        retry: RetryPolicy | None = None,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        requests_per_minute: float | None = None,
        burst: int = 5,
        max_wait: float = 1.0,
        routing : Routing = "in_order",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:

        if routing not in ("in_order", "least_busy"):
            raise ValueError(f"routing must be 'in_order' or 'least_busy', not {routing!r}.")    
        
        self.providers = list(providers)
        self.retry = retry or RetryPolicy()
        self.max_wait = max_wait
        self._sleep = sleep
        self._breakers, self._buckets = build_guards(
            providers,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            requests_per_minute=requests_per_minute,
            burst=burst,
            clock=clock,
        )
        self.routing = routing
        self._in_flight = {provider.name: 0 for provider in self.providers} ##every provider initially won't have any running requests


    def circuit_state(self, provider_name: str) -> CircuitState:
        return self._breakers[provider_name].state


    def in_flight(self, provider_name: str) -> int:
        """How many requests this client is running on that provider right now."""
        return self._in_flight[provider_name]


    def _route(self) -> list[AsyncProvider]:
        """The order to try providers in, for this one request."""
        if self.routing == "least_busy":
            return sorted(self.providers, key=lambda provider: self._in_flight[provider.name])
        return self.providers

    
    async def complete(self, prompt: str) -> Response:
        errors: dict[str, BaseException] = {}

        for provider in self._route():
            breaker = self._breakers[provider.name]

            if not breaker.allow_request():
                errors[provider.name] = CircuitOpenError(provider.name)
                continue

            self._in_flight[provider.name] += 1

            try:
                text = await async_retry_call(
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

            finally:
                self._in_flight[provider.name] -= 1

            breaker.record_success()

            return Response(text=text, provider=provider.name)

        raise AllProvidersFailedError(errors)


    async def _call_with_token(self, provider: AsyncProvider, prompt: str) -> str:
        """One attempt: reserve a token (waiting up to max_wait), then call the provider."""

        bucket = self._buckets.get(provider.name)

        if bucket is None:
            return await provider.complete(prompt)

        wait = bucket.time_until_available()
        
        if wait > self.max_wait:
            raise ThrottledError(provider.name, wait)

        bucket.consume()  # reserve the token BEFORE sleeping, so no other request can take it
        
        if wait > 0:
            await self._sleep(wait)
        return await provider.complete(prompt)