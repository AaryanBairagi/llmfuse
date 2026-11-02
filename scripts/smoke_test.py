"""Runs against the built package (not the source) to prove it installs and works."""

import asyncio

from llmfuse import AsyncFuseClient, FuseClient, __version__, as_async
from llmfuse.testing import FakeAsyncProvider, FakeProvider

# Sync client: retries, then fails over from "down" to "up".
down = FakeProvider("down", errors=[ConnectionError("down")] * 10)
up = FakeProvider("up", reply="ok")
client = FuseClient(providers=[down, up], sleep=lambda seconds: None)

response = client.complete("ping")
assert response.provider == "up", response
assert response.text == "ok", response

# Async client: "slow" never answers in time, so hedging asks "fast" (a wrapped sync provider).
slow = FakeAsyncProvider("slow", delay=5.0)
fast = as_async(FakeProvider("fast", reply="ok"))
async_client = AsyncFuseClient(providers=[slow, fast], hedge_after=0.05, timeout=2.0)

async_response = asyncio.run(async_client.complete("ping"))
assert async_response.provider == "fast", async_response

print(f"llmfuse {__version__}: smoke test passed")