import asyncio
import os
import sys

from llmfuse import AsyncFuseClient, as_async
from llmfuse.providers import AnthropicProvider, GeminiProvider, GroqProvider


async def main(routing: str) -> None:
    client = AsyncFuseClient(
        providers=[
            as_async(GroqProvider(model=os.environ["GROQ_MODEL"])),
            as_async(GeminiProvider(model=os.environ["GEMINI_MODEL"])),
            as_async(AnthropicProvider(model=os.environ["GEMINI_MODEL"])),
        ],
        routing= routing
    )

    prompts = [f"Reply with just the number {i}" for i in range(1, 5)]
    responses = await asyncio.gather(*(client.complete(p) for p in prompts))

    for prompt, response in zip(prompts, responses, strict=True):
        print(f"{prompt} -> {response.provider} : {response.text.strip()}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "least_busy"))