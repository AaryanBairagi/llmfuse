from collections.abc import Callable, Sequence
from typing import Protocol

from llmfuse.breaker import CircuitBreaker
from llmfuse.limiter import TokenBucket


class HasName(Protocol):
    name: str

def build_guards(
        providers: Sequence[HasName],
        *,
        failure_threshold: int,
        reset_timeout: float,
        requests_per_minute : float | None,
        burst : int,   
        clock: Callable[[], float],
        ) -> tuple[dict[str, CircuitBreaker], dict[str, TokenBucket]]:
        
        """Check the provider list, then give each provider its own breaker and (optional) bucket."""

        if not providers:
            raise ValueError("FuseClient need atleast one provider.")

        names = [provider.name for provider in providers]
        if len(names) != len(set(names)):
            raise ValueError("Provider names must be unique.")



        breakers = {
            provider.name: CircuitBreaker(failure_threshold, reset_timeout, clock=clock)
            for provider in providers
        }

        buckets : dict[str, TokenBucket] = {}

        for provider in providers:
            own_limit = getattr(provider, "requests_per_minute", None)
            limit = own_limit if own_limit is not None else requests_per_minute
            if limit is not None:
                buckets[provider.name] = TokenBucket(limit / 60, burst, clock=clock)

        return breakers , buckets