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
    DeadlineExceededError,
    LLMFuseError,
    ThrottledError,
)
from llmfuse.provider import AsyncProvider, Response
from llmfuse.retry import RetryPolicy, async_retry_call

Routing = Literal["in_order", "least_busy"]

def _check_timeout(timeout: float | None) -> float | None:
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds (or None for no limit).")
        
    return timeout

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
        timeout: float | None = None,
        hedge_after: float | None = None,
        routing : Routing = "in_order",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:

        if routing not in ("in_order", "least_busy"):
            raise ValueError(f"routing must be 'in_order' or 'least_busy', not {routing!r}.")   

        if hedge_after is not None and hedge_after <= 0:
            raise ValueError("hedge_after must be a positive number of seconds (or None to turn hedging off).") 
        
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
        self.timeout = _check_timeout(timeout)
        self.hedge_after = hedge_after
        self.routing = routing
        self._in_flight = {provider.name: 0 for provider in self.providers} ##every provider initially won't have any running requests, hence start with 0


    def circuit_state(self, provider_name: str) -> CircuitState:
        return self._breakers[provider_name].state


    def in_flight(self, provider_name: str) -> int:
        """How many requests this client is running on that provider right now."""
        return self._in_flight[provider_name]


    def _route(self) -> list[AsyncProvider]:
        """The order to try providers in, for this one request."""
        if self.routing == "least_busy":
            return sorted(self.providers, key=lambda provider: self._in_flight[provider.name])
        
        return list(self.providers) #hedge_after will pop the providers from the list hence return a copy


    async def _race_providers(self, prompt: str, errors: dict[str, BaseException]) -> Response:
        """Hedging: if nobody has answered after `hedge_after` seconds, start the next provider too."""
        waiting = self._route()
        running : set[asyncio.Task[Response | None]] = set()

        def start_next() -> None: #if response not in hedge_after send request to next provider 
            provider = waiting.pop(0) 
            running.add(asyncio.create_task(self._ask(provider, prompt, errors)))

        start_next()

        try:
            while running:
                done, _ = await asyncio.wait(
                    running,
                    timeout = self.hedge_after if waiting else None,
                    return_when = asyncio.FIRST_COMPLETED
                )

                running -= done

                if not done:  #hedge_after passed and nobody answered: start a backup
                    start_next()
                    continue

                for task in done:
                    response = task.result()
                    if response is not None:
                        return response

                for _ in done:  #none of them returned a Response: each one failed, and _ask returned None.
                    if waiting:
                        start_next()

            raise AllProvidersFailedError(errors)
        
        finally:
            for task in running:  # the losers (or everyone, if the deadline hit)
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)


    async def _try_providers(self, prompt: str, errors: dict[str, BaseException]) -> Response:
        """No hedging: try providers one after another until one answers."""
        for provider in self._route():
            response = await self._ask(provider, prompt, errors)
            if response is not None:
                return response

        raise AllProvidersFailedError(errors)
    

    async def _ask(self, provider: AsyncProvider, prompt: str, errors: dict[str, BaseException]) -> Response | None:
        """Try ONE provider, with retries. Returns None if it failed (the reason goes in `errors`)."""
        breaker = self._breakers[provider.name]

        if not breaker.allow_request():
            errors[provider.name] = CircuitOpenError(provider.name)
            return None

        self._in_flight[provider.name] += 1

        try:
            text = await async_retry_call(
                partial(self._call_with_token, provider, prompt),
                self.retry,
                sleep=self._sleep,
            )

        except ThrottledError as error:
            errors[provider.name] = error
            return None

        except LLMFuseError as error:
            breaker.record_failure()
            errors[provider.name] = error
            return None

        finally:
            self._in_flight[provider.name] -= 1

        breaker.record_success()
        return Response(text=text, provider=provider.name)


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


    async def complete(self, prompt:str, *, timeout:float | None = None) -> Response:
        """Ask providers in order. `timeout` caps the WHOLE call: every wait, retry and provider."""
        budget =  _check_timeout(timeout) if timeout is not None else self.timeout
        errors: dict[str, BaseException] = {}

        run = self._race_providers if self.hedge_after is not None else self._try_providers

        if budget is None: #no deadline: same as v0.1
            return await run(prompt, errors) 

        try:
            return await asyncio.wait_for(run(prompt, errors), budget)

        except asyncio.TimeoutError:
            raise DeadlineExceededError(budget, errors) from None
    