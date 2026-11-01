import asyncio
import time

import pytest

from llmfuse import (
    AsyncFuseClient,
    CircuitState,
    DeadlineExceededError,
    ProviderError,
    RetryExhaustedError,
    RetryPolicy,
)
from llmfuse.testing import FakeAsyncProvider

DOWN = ProviderError("503 Service Unavailable", status_code=503)
FAST = RetryPolicy(max_attempts=3, jitter=False, base_delay=0.01, multiplier=1.0)


def test_answer_inside_the_deadline_is_returned() -> None:
    client = AsyncFuseClient(providers=[FakeAsyncProvider("groq", delay=0.01)], timeout=1.0)
    response = asyncio.run(client.complete("hello"))
    assert response.provider == "groq"


def test_slow_provider_hits_the_deadline() -> None:
    client = AsyncFuseClient(providers=[FakeAsyncProvider("groq", delay=5.0)], timeout=0.1)
    start = time.monotonic()
    with pytest.raises(DeadlineExceededError) as caught:
        asyncio.run(client.complete("hello"))
    assert time.monotonic() - start < 1.0  # gave up at ~0.1 s instead of waiting 5 s
    assert caught.value.timeout == 0.1


def test_deadline_covers_retries_and_failover() -> None:
    groq = FakeAsyncProvider("groq", errors=[DOWN] * 3)
    gemini = FakeAsyncProvider("gemini", delay=5.0)
    client = AsyncFuseClient(providers=[groq, gemini], retry=FAST, timeout=0.2)
    with pytest.raises(DeadlineExceededError) as caught:
        asyncio.run(client.complete("hello"))
    assert isinstance(caught.value.errors["groq"], RetryExhaustedError)
    assert gemini.calls == 1


def test_deadline_cuts_a_long_backoff_short() -> None:
    groq = FakeAsyncProvider("groq", errors=[DOWN] * 3)
    slow_retry = RetryPolicy(max_attempts=3, jitter=False, base_delay=10.0)
    client = AsyncFuseClient(providers=[groq], retry=slow_retry, timeout=0.1)
    start = time.monotonic()
    with pytest.raises(DeadlineExceededError):
        asyncio.run(client.complete("hello"))
    assert time.monotonic() - start < 1.0
    assert groq.calls == 1  # cancelled during the 10 s backoff, never tried again


def test_per_call_timeout_overrides_the_client_default() -> None:
    client = AsyncFuseClient(providers=[FakeAsyncProvider("groq", delay=0.2)], timeout=0.05)
    response = asyncio.run(client.complete("hello", timeout=1.0))
    assert response.provider == "groq"
    with pytest.raises(DeadlineExceededError):
        asyncio.run(client.complete("hello"))


def test_deadline_is_not_counted_as_a_provider_failure() -> None:
    client = AsyncFuseClient(
        providers=[FakeAsyncProvider("groq", delay=5.0)], failure_threshold=1, timeout=0.05
    )
    with pytest.raises(DeadlineExceededError):
        asyncio.run(client.complete("hello"))
    assert client.circuit_state("groq") is CircuitState.CLOSED


@pytest.mark.parametrize("bad", [0, -1.0])
def test_timeout_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError):
        AsyncFuseClient(providers=[FakeAsyncProvider("groq")], timeout=bad)
    client = AsyncFuseClient(providers=[FakeAsyncProvider("groq")])
    with pytest.raises(ValueError):
        asyncio.run(client.complete("hello", timeout=bad))


def test_deadline_frees_the_least_busy_counter() -> None:
    client = AsyncFuseClient(
        providers=[FakeAsyncProvider("groq", delay=5.0)], routing="least_busy", timeout=0.05
    )
    with pytest.raises(DeadlineExceededError):
        asyncio.run(client.complete("hello"))
    assert client.in_flight("groq") == 0