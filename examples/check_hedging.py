"""Make Groq pretend to be slow, then watch hedging send a backup to Gemini.

    uv run --env-file .env examples/check_hedging.py off    → waits for slow Groq
    uv run --env-file .env examples/check_hedging.py 1.0    → backup to Gemini after 1 s
"""

import asyncio
import os
import sys
import time

from llmfuse import AsyncFuseClient, AsyncProvider, as_async
from llmfuse.providers import GeminiProvider, GroqProvider


class SlowDown:
    """Wraps a provider and adds a delay, to pretend it's having a bad day."""

    def __init__(self, provider: AsyncProvider, delay: float) -> None:
        self.provider = provider
        self.name = provider.name
        self.delay = delay

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(self.delay)
        return await self.provider.complete(prompt)


async def main(hedge_after: float | None) -> None:
    client = AsyncFuseClient(
        providers=[
            SlowDown(as_async(GroqProvider(model=os.environ["GROQ_MODEL"])), delay=3.0),
            as_async(GeminiProvider(model=os.environ["GEMINI_MODEL"])),
        ],
        hedge_after=hedge_after,
    )
    start = time.monotonic()
    response = await client.complete("Reply with just the word: pong")
    print(f"hedge_after={hedge_after} → {response.provider}: {response.text.strip()!r} in {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "1.0"
    asyncio.run(main(None if arg == "off" else float(arg)))