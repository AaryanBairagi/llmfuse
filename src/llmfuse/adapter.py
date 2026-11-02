import asyncio

from llmfuse.provider import Provider


class AsyncAdapter:
    def __init__(self, provider: Provider):
        self.provider = provider
        self.name = provider.name
        self.requests_per_minute : float | None = getattr(provider, "requests_per_minute", None) 

    def complete(self, prompt:str):
        return asyncio.to_thread(self.provider.complete, prompt)


def as_async(provider: Provider):
    """`as_async(GroqProvider(...))` gives you a provider AsyncFuseClient can use."""
    return AsyncAdapter(provider)