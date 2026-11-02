"""Ask all four providers AT THE SAME TIME, and compare with asking them one by one."""

import asyncio
import os
import time

from llmfuse import LLMFuseError, as_async
from llmfuse.providers import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    OpenAIProvider,
)

PROVIDERS = [
    (GroqProvider, "GROQ_MODEL"),
    (GeminiProvider, "GEMINI_MODEL"),
    (OpenAIProvider, "OPENAI_MODEL"),
    (AnthropicProvider, "ANTHROPIC_MODEL"),
]


async def ask(provider_class: type, model_env: str) -> str:
    name = provider_class.__name__
    start = time.monotonic()
    try:
        provider = as_async(provider_class(model=os.environ[model_env]))
        reply = await provider.complete("Reply with just the word: pong")
        return f"{name}: {reply!r} in {time.monotonic() - start:.1f}s"
    except (LLMFuseError, KeyError, ValueError) as error:
        return f"{name}: {type(error).__name__}: {error}"


async def main() -> None:
    start = time.monotonic()
    results = await asyncio.gather(*(ask(cls, env) for cls, env in PROVIDERS))
    for line in results:
        print(line)
    print(f"All four together took {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())