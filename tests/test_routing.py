import asyncio

import pytest

from llmfuse import AsyncFuseClient, ProviderError, RetryPolicy
from llmfuse.testing import FakeAsyncProvider

FAST = RetryPolicy(max_attempts=3, jitter=False, multiplier=1.0, base_delay=0.01)
DOWN = ProviderError("503 Service Unavailable", status_code=503)


def run_together(client: AsyncFuseClient, count: int) -> list[str]:
    """Send `count` number of requests; return which provider answered the query"""
    async def main() -> list[str]:
        responses = await asyncio.gather(*(client.complete(f"q{i}") for i in range(count)))
        return[response.provider for response in responses]

    return asyncio.run(main())

def test_in_order_sends_everything_to_the_first_provider() -> None:
    groq = FakeAsyncProvider("groq", delay = 0.05)
    gemini = FakeAsyncProvider("gemini", delay = 0.05)
    client = AsyncFuseClient(providers=[groq , gemini])
    assert run_together(client,4) == ["groq", "groq", "groq", "groq"]


def test_least_busy_spreads_out_requests() -> None:
    groq = FakeAsyncProvider("groq", delay = 0.05)
    gemini = FakeAsyncProvider("gemini", delay = 0.05)
    client = AsyncFuseClient(providers=[groq , gemini], routing="least_busy")
    assert run_together(client,4) == ["groq", "gemini", "groq", "gemini"]


def test_least_busy_still_fails_over() -> None:
    groq = FakeAsyncProvider("groq", errors=[DOWN] * 3)
    gemini = FakeAsyncProvider("gemini")
    client = AsyncFuseClient(providers=[groq, gemini], routing="least_busy", retry=FAST)
    assert asyncio.run(client.complete("hello")).provider == "gemini"
    assert client.in_flight("groq") == 0
    assert client.in_flight("gemini") == 0


def test_least_busy_uses_list_order_when_everyone_is_idle() -> None:
    groq = FakeAsyncProvider("groq", errors=[DOWN] * 3)
    gemini = FakeAsyncProvider("gemini")
    client = AsyncFuseClient(providers=[groq, gemini], routing="least_busy")

    for _ in range(3):
        assert asyncio.run(client.complete("hello")).provider == "groq"

    assert gemini.calls == 0


def test_unknown_requests_are_rejected() -> None:
    groq = FakeAsyncProvider("groq")
    with pytest.raises(ValueError):
        AsyncFuseClient(providers=[groq], routing="random")


def test_in_flight_goes_back_to_zero_even_when_cancelled() -> None:
    client = AsyncFuseClient(providers=[FakeAsyncProvider("groq", delay=5.0)])

    async def main() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.complete("hello"), 0.05)  # give up after 0.05 s

    asyncio.run(main())
    assert client.in_flight("groq") == 0


def test_in_flight_counts_requests_in_progress() -> None:
    client = AsyncFuseClient(providers=[FakeAsyncProvider("groq", delay=0.05)])

    async def main() -> None:
        task = asyncio.create_task(client.complete("hello"))
        await asyncio.sleep(0.01)  # let the task start and pause inside the provider
        assert client.in_flight("groq") == 1
        await task
        assert client.in_flight("groq") == 0

    asyncio.run(main())