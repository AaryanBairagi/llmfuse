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


def test_leats_busy_spreads_out_requests() -> None:
    groq = FakeAsyncProvider("groq", delay = 0.05)
    gemini = FakeAsyncProvider("gemini", delay = 0.05)
    client = AsyncFuseClient(providers=[groq , gemini], routing="least_busy")
    assert run_together(client,4) == ["groq", "gemini", "groq", "gemini"]
