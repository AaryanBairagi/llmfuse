"""Retry with exponential backoff and full jitter."""

import random
from collections.abc import Callable
from dataclasses import dataclass , field

from llmfuse.errors import RateLimitError


@dataclass(frozen=True)
class RetryPolicy:
    """Settings that decide how long to wait between retries."""

    max_attempts: int = 4        # total tries, including the first one
    base_delay: float = 1.0      # wait before the first retry (seconds)
    multiplier: float = 2.0      # how fast the wait grows (2.0 = doubling)
    max_delay: float = 30.0      # cap on any single wait
    jitter: bool = True          # randomize waits (strongly recommended)
    rng : Callable[[],float] = field(default=random.random, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max attempts must be atleast 1.")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative.")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1.")


    def backoff(self , retry_number : int) -> float:
        """The maximum wait before the retry `retry_number` (starts at 1)"""
        if(retry_number < 1):
            raise ValueError("retry_number starts at 1")
        return min(self.max_delay , self.base_delay * self.multiplier ** (retry_number - 1)) #eg : 1 ** 2


    def delay_for(self , retry_number : int , error : BaseException | None = None) -> float:
        """The actual seconds to wait before `retry_number`"""
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            return min(self.max_delay, max(0.0, error.retry_after))
        ceiling = self.backoff(retry_number)
        return self.rng() * ceiling if self.jitter else ceiling  #if jitter is ON then wait random amount of time 

