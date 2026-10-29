import asyncio
import time

from llmfuse import AsyncFuseClient, ProviderError, RetryPolicy, as_async
from llmfuse.testing import FakeAsyncProvider, FakeProvider

FAST = RetryPolicy(max_attempts=3, jitter=False)
DOWN = ProviderError("503 - Service Unavailable", status_code=503)


async def no_sleep(seconds : float) -> None:
    pass 


def test_adapter_keeps_name() -> None:
    groqP = as_async(FakeProvider("groq", requests_per_minute=15))
    assert groqP.name == "groq"
    assert groqP.requests_per_minute == 15


def test_adapter_calls_synchronous_provider() -> None:
    groq = FakeProvider("groq", requests_per_minute=15, reply="hi from groq")
    async_groq = as_async(groq)
    response = asyncio.run(async_groq.complete("one"))
    assert response == "hi from groq"
    assert groq.calls == 1


def test_errors_failover_still_give_response() -> None:
    groq = FakeProvider("groq" , errors=[DOWN] * 3)
    claude = FakeProvider("claude" , reply="hi from claude" )
    client = AsyncFuseClient(providers=[as_async(groq), as_async(claude)], retry=FAST, sleep=no_sleep)
    response = asyncio.run(client.complete("q1"))
    assert response.provider == "claude"
    assert groq.calls == 3


def test_sync_and_async_providers_can_be_mixed() -> None:
    groq = FakeProvider("groq" , errors=[DOWN] * 3)
    claude = FakeAsyncProvider("claude" , reply="hi from claude" )
    client = AsyncFuseClient(providers=[as_async(groq), claude], retry=FAST, sleep=no_sleep)
    response = asyncio.run(client.complete("q1"))
    assert response.provider == "claude"


def test_slow_sync_calls_run_at_the_same_time() -> None:
    class SlowProvider(FakeProvider):
        def complete(self, prompt: str) -> str:
            time.sleep(0.2)  # a blocking call, like a real HTTP request
            return super().complete(prompt)

    client = AsyncFuseClient(providers=[as_async(SlowProvider("groq"))])

    async def three_at_once() -> None:
        await asyncio.gather(*(client.complete(f"q{i}") for i in range(3)))

    start = time.monotonic()
    asyncio.run(three_at_once())
    elapsed = time.monotonic() - start
    assert elapsed < 0.45, elapsed 