"""Call each provider directly (no failover) to see which ones work."""

import os

from llmfuse import LLMFuseError
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

for provider_class, model_env in PROVIDERS:
    name = provider_class.__name__
    try:
        provider = provider_class(model=os.environ[model_env])
        print(f"{name}: {provider.complete('Reply with just the word: pong')}")
    except (LLMFuseError, KeyError, ValueError) as error:
        print(f"{name}: {type(error).__name__}: {error}")