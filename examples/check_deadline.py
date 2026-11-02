import asyncio
import os
import sys
import time

from llmfuse import AsyncFuseClient, DeadlineExceededError, as_async
from llmfuse.providers import GeminiProvider, GroqProvider


async def main(budget: float) -> None:
    client = AsyncFuseClient(
        providers=[
            as_async(GroqProvider(model=os.environ["GROQ_MODEL"])),
            as_async(GeminiProvider(model=os.environ["GEMINI_MODEL"])),
        ],
        timeout= budget
    )

    start = time.monotonic()

    try:
        response = await client.complete("Reply with pong")
        print(f"{response.provider}: {response.text!r} in {time.monotonic() - start:.2f}s")

    except DeadlineExceededError as error:
        print(f"⏱ gave up after {time.monotonic() - start:.2f}s: {error}")


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0))