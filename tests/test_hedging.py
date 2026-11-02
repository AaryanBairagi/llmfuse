import asyncio
import time

import pytest

from llmfuse import (
    AllProvidersFailedError,
    AsyncFuseClient,
    CircuitState,
    DeadlineExceededError,
    ProviderError,
)
from llmfuse.testing import FakeAsyncProvider

BAD_KEY = ProviderError("401 Unauthorized", status_code=401, retryable=False)


def timed(client: AsyncFuseClient) -> tuple[str, float]:
    """Run one request; return (provider that answered, seconds it took)."""
    start = time.monotonic()
    response = asyncio.run(client.complete("hello"))
    return response.provider, time.monotonic() - start


def test_no_backup_when_the_first_provider_is_fast() -> None:
    groq = FakeAsyncProvider("groq", delay=0.01)
    gemini = FakeAsyncProvider("gemini")
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.1)
    assert timed(client)[0] == "groq"
    assert gemini.calls == 0


def test_slow_provider_gets_a_backup_and_the_backup_wins() -> None:
    groq = FakeAsyncProvider("groq", delay=1.0)
    gemini = FakeAsyncProvider("gemini", delay=0.01)
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.05)
    provider, seconds = timed(client)
    assert provider == "gemini"
    assert seconds < 0.5  # ~0.06 s instead of waiting 1 s for groq
    assert (groq.calls, gemini.calls) == (1, 1)


def test_the_original_can_still_win() -> None:
    groq = FakeAsyncProvider("groq", delay=0.15)
    gemini = FakeAsyncProvider("gemini", delay=1.0)
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.05)
    assert timed(client)[0] == "groq"  # the backup started, but groq still finished first


def test_the_loser_is_cancelled_cleanly() -> None:
    groq = FakeAsyncProvider("groq", delay=1.0)
    gemini = FakeAsyncProvider("gemini", delay=0.01)
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.05, failure_threshold=1)

    async def main() -> None:
        await client.complete("hello")
        assert client.in_flight("groq") == 0  # freed before complete() returned
        assert client.circuit_state("groq") is CircuitState.CLOSED  # slow is not broken

    asyncio.run(main())


def test_a_fast_failure_fails_over_without_waiting_for_the_hedge_timer() -> None:
    groq = FakeAsyncProvider("groq", errors=[BAD_KEY])
    gemini = FakeAsyncProvider("gemini")
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=5.0)
    provider, seconds = timed(client)
    assert provider == "gemini"
    assert seconds < 0.5  # didn't sit through the 5 s hedge timer


def test_backups_keep_coming_until_someone_answers() -> None:
    groq = FakeAsyncProvider("groq", delay=1.0)
    gemini = FakeAsyncProvider("gemini", delay=1.0)
    claude = FakeAsyncProvider("claude", delay=0.01)
    client = AsyncFuseClient(providers=[groq, gemini, claude], hedge_after=0.05)
    assert timed(client)[0] == "claude"  # started at ~0.1 s, answered at ~0.11 s
    assert (groq.calls, gemini.calls, claude.calls) == (1, 1, 1)


def test_everyone_failing_still_raises_all_providers_failed() -> None:
    groq = FakeAsyncProvider("groq", errors=[BAD_KEY])
    gemini = FakeAsyncProvider("gemini", errors=[BAD_KEY])
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.05)
    with pytest.raises(AllProvidersFailedError) as caught:
        asyncio.run(client.complete("hello"))
    assert set(caught.value.errors) == {"groq", "gemini"}


def test_the_deadline_still_wins_over_hedging() -> None:
    slow = [FakeAsyncProvider(name, delay=5.0) for name in ("groq", "gemini", "claude")]
    client = AsyncFuseClient(providers=slow, hedge_after=0.05, timeout=0.2)
    start = time.monotonic()
    with pytest.raises(DeadlineExceededError):
        asyncio.run(client.complete("hello"))
    assert time.monotonic() - start < 1.0
    assert [provider.calls for provider in slow] == [1, 1, 1]  # all three were racing
    assert all(client.in_flight(provider.name) == 0 for provider in slow)


def test_bugs_are_not_hidden_when_hedging() -> None:
    groq = FakeAsyncProvider("groq", errors=[TypeError("oops")])
    gemini = FakeAsyncProvider("gemini")
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.05)
    with pytest.raises(TypeError):
        asyncio.run(client.complete("hello"))


@pytest.mark.parametrize("bad", [0, -1.0])
def test_hedge_after_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError):
        AsyncFuseClient(providers=[FakeAsyncProvider("groq")], hedge_after=bad)


def test_the_same_client_works_for_many_requests() -> None:
    groq = FakeAsyncProvider("groq", delay=1.0)
    gemini = FakeAsyncProvider("gemini")
    client = AsyncFuseClient(providers=[groq, gemini], hedge_after=0.05)
    for _ in range(3):
        assert asyncio.run(client.complete("hello")).provider == "gemini"
    assert [provider.name for provider in client.providers] == ["groq", "gemini"]