import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from llmfuse.errors import ProviderError, RateLimitError, RetryExhaustedError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Settings that decide how long to wait between retries."""

    max_attempts: int = 4  # total tries, including the first one
    base_delay: float = 1.0  # wait before the first retry (seconds)
    multiplier: float = 2.0  # how fast the wait grows (2.0 = doubling)
    max_delay: float = 30.0  # cap on any single wait
    jitter: bool = True  # randomize waits (strongly recommended)
    rng: Callable[[], float] = field(default=random.random, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max attempts must be atleast 1.")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative.")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1.")

    def backoff(self, retry_number: int) -> float:
        """The maximum wait before the retry `retry_number` (starts at 1)"""
        if retry_number < 1:
            raise ValueError("retry_number starts at 1")
        return min(
            self.max_delay, self.base_delay * self.multiplier ** (retry_number - 1)
        )  # eg : 1 ** 2

    def delay_for(self, retry_number: int, error: BaseException | None = None) -> float:
        """The actual seconds to wait before `retry_number`"""
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            return min(self.max_delay, max(0.0, error.retry_after))
        ceiling = self.backoff(retry_number)
        return self.rng() * ceiling if self.jitter else ceiling
        # if jitter is ON then wait random amount of time


# if its our error then return retryable stored. If python error like connection or timeout then return True
# Anything else like TypeError , KeyError then retryable is False
def is_retryable(error: BaseException) -> bool:
    """Return whether an error is worth retrying."""
    if isinstance(error, ProviderError):
        return error.retryable
    return isinstance(error, (TimeoutError, ConnectionError))


def retry_call[T](
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    should_retry: Callable[[BaseException], bool] = is_retryable,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn()`. If it fails with a retryable error, wait and try again."""
    policy = policy or RetryPolicy()

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as error:
            if not should_retry(error):
                raise
            if attempt == policy.max_attempts:
                raise RetryExhaustedError(attempt, error) from error
            delay = policy.delay_for(attempt, error)
            if on_retry is not None:
                on_retry(attempt, error, delay)
            sleep(delay)

    raise AssertionError("unreachable")
