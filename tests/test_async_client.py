import asyncio
import itertools
import time

import pytest

from llmfuse import (
    AllProvidersFailedError,
    AsyncFuseClient,
    CircuitOpenError,
    ProviderError,
    RetryPolicy,
    ThrottledError,
)
from llmfuse.breaker import CircuitState
from llmfuse.testing import FakeAsyncProvider, FakeClock

DOWN = ProviderError("503 Service Unavailable" , status_code=500)
FAST = RetryPolicy(max_attempts=3, jitter=False, base_delay=0.01, multiplier=1.0)


async def no_sleep(seconds : float) -> None:
    pass


def make_client(*providers, **settings) -> AsyncFuseClient:
    return AsyncFuseClient(providers=list(providers), retry=FAST, sleep=no_sleep, **settings) 


def test_first_healthy_provider() -> None:
    groq = FakeAsyncProvider("groq", reply="Hi from Groq")
    gemini = FakeAsyncProvider("gemini")
    response = asyncio.run(make_client(groq,gemini).complete("Hello"))
    print(response)
    assert response.provider == "groq"
    assert response.text == "Hi from Groq"
    assert groq.calls == 1
    assert gemini.calls == 0


def test_all_providers_failed() -> None:
    groq = FakeAsyncProvider("groq", reply="Hi from Groq", errors=[DOWN] * 3)
    gemini = FakeAsyncProvider("gemini", reply="Hi from Gemini", errors=[DOWN] * 3)
    with pytest.raises(AllProvidersFailedError) as exc_info:
        asyncio.run(make_client(groq, gemini).complete("hello"))
    assert set(exc_info.value.errors) == {"groq", "gemini"}


def test_bad_api_key_fails_without_retrying() -> None:
    bad_api_key = ProviderError("401 Status Code : Bad API Key" , status_code=401, retryable=False)
    groq = FakeAsyncProvider("groq", reply="Hi from Groq", errors=[bad_api_key])
    gemini = FakeAsyncProvider("gemini", reply="Hi from Gemini")
    client = make_client(groq,gemini)
    response = asyncio.run(client.complete("Hello"))
    assert response.provider == "gemini"
    assert groq.calls == 1
    assert gemini.calls == 1


def test_all_retries_failed_error() -> None:
    groq = FakeAsyncProvider("groq", reply="Hi from Groq", errors=[DOWN] * 3)
    gemini = FakeAsyncProvider("gemini", reply="Hi from Gemini")
    client = make_client(groq,gemini)
    response = asyncio.run(client.complete("Hello"))
    assert response.provider == "gemini"
    assert groq.calls == 3
    assert gemini.calls == 1


def test_bugs_are_not_hidden() -> None:
    groq = FakeAsyncProvider("groq", reply="Hi from Groq", errors=[TypeError("oops")])
    gemini = FakeAsyncProvider("gemini", reply="Hi from Gemini")
    client = make_client(groq,gemini)

    with pytest.raises(TypeError):
        asyncio.run(client.complete("Hello"))

    assert groq.calls == 1
    assert gemini.calls == 0

async def run_client(client: AsyncFuseClient) -> str:
    provider = ""
    for _ in range(3):
        response = await client.complete("Hi")
        provider = response.provider
    return provider


def test_open_circuit_skips_provider() -> None:
    groq = FakeAsyncProvider("groq", reply="Hi from Groq", errors=[DOWN] * 100)
    gemini = FakeAsyncProvider("gemini", reply="Hi from Gemini")
    client = make_client(groq, gemini, failure_threshold=2, clock=FakeClock())

    last_provider = asyncio.run(run_client(client))

    assert last_provider == "gemini"
    assert groq.calls == 6  # 2 requests x 3 attempts, then the circuit opens and Groq is skipped
    assert client.circuit_state("groq") is CircuitState.OPEN


def test_circuit_open_error() -> None:
    groq = FakeAsyncProvider("groq", errors=[DOWN] * 100)
    client = make_client(groq, failure_threshold=1, clock=FakeClock())

    with pytest.raises(AllProvidersFailedError):
        asyncio.run(client.complete("one"))

    with pytest.raises(AllProvidersFailedError) as exc_info:
        asyncio.run(client.complete("two"))

    assert isinstance(exc_info.value.errors["groq"], CircuitOpenError)


def test_throttled_error() -> None:
    groq = FakeAsyncProvider("groq", requests_per_minute=1)
    client = make_client(groq, burst=1, clock=FakeClock())
    asyncio.run(client.complete("one"))

    with pytest.raises(AllProvidersFailedError) as exc_info:
        asyncio.run(client.complete("two"))

    assert isinstance(exc_info.value.errors['groq'] , ThrottledError)


def test_throttled_errors_dont_break_the_circuit() -> None:
    clock = FakeClock()
    groq = FakeAsyncProvider('groq', requests_per_minute = 15)
    gemini = FakeAsyncProvider('gemini')
    client = make_client(groq, gemini, burst=1, failure_threshold=1, clock=clock)

    async def two_requests() -> list[str]:
        first = await client.complete("one")
        second = await client.complete("two")
        return [first.provider, second.provider]

    assert asyncio.run(two_requests()) == ["groq" , "gemini"]
    assert client.circuit_state("groq") is CircuitState.CLOSED


async def run_three_at_once(client):
    await asyncio.gather(*(client.complete(f"q{i}") for i in range(3)))


def test_concurrent_requests_respect_the_rate_limit() -> None:
    # 600 per minute = one token every 0.1 s, bucket holds 1 token.
    call_times: list[float] = []

    class TimedProvider(FakeAsyncProvider):
        async def complete(self, prompt: str) -> str:
            call_times.append(time.monotonic())
            return await super().complete(prompt)

    client = AsyncFuseClient(
        providers=[TimedProvider("groq")], requests_per_minute=600, burst=1, max_wait=1.0
    )

    asyncio.run(run_three_at_once(client))
    gaps = [later - earlier for earlier, later in itertools.pairwise(call_times)]
    assert all(gap >= 0.08 for gap in gaps), gaps  
