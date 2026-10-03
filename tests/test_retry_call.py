import pytest

from llmfuse import (
    ProviderError,
    RateLimitError,
    RetryExhaustedError,
    RetryPolicy,
    is_retryable,
    retry_call,
)


class FalseSleep:
    """Pretend to sleep. Just remember how long it was asked to wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds) -> None:
        self.delays.append(seconds)


class Flaky:
    """A fake LLM call. Raises errors for 'fail_times' times, in last returns ok."""

    def __init__(self, fail_times: int, error: Exception):
        self.fail_times = fail_times
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return "ok"


NO_JITTER = RetryPolicy(max_attempts=4, base_delay=1.0, multiplier=2.0, jitter=False)


# ---------------------------------------------------- TESTS ------------------------------------------------------------------


def test_on_first_try_never_sleeps() -> None:
    sleep = FalseSleep()
    assert retry_call(lambda : "hello",  NO_JITTER, sleep = sleep) == "hello"
    assert sleep.delays == []


def test_recovery_after_temporary_failures() -> None:
    sleep = FalseSleep()
    fn = Flaky(fail_times=2 , error = ProviderError(message="503" , status_code=503))
    assert retry_call(fn , NO_JITTER , sleep=sleep) == "ok"
    assert fn.calls == 3
    assert sleep.delays == [1.0 , 2.0]


def test_on_retry_hook_is_called_before_each_wait():
    seen : list[(int , float)] = []
    sleep = FalseSleep()
    fn = Flaky(fail_times= 2 , error= ConnectionError("reset"))
    retry_call(
        fn, NO_JITTER, sleep = sleep, on_retry = lambda attempt, error, delay : seen.append((attempt, delay))
    )
    assert seen == [(1 , 1.0) , (2 , 2.0)]


def test_programming_bugs_are_not_retried() -> None:
    sleep = FalseSleep()
    fn = Flaky(fail_times=1 , error=TypeError("oops"))
    with pytest.raises(TypeError):
        retry_call(fn, NO_JITTER, sleep=sleep)
    assert fn.calls == 1


def test_non_retryable_error_fails_immediately() -> None:
    sleep = FalseSleep()
    fn = Flaky(fail_times=1 , error=ProviderError(message="bad api key" , status_code=401 , retryable=False))
    with pytest.raises(ProviderError):
        retry_call(fn , NO_JITTER, sleep=sleep) 
    assert fn.calls == 1
    assert sleep.delays == []


def test_retry_after_is_respected_in_the_loop():
    sleep = FalseSleep()
    fn = Flaky(fail_times=1, error=RateLimitError("429" , retry_after=7.0))
    assert retry_call(fn, NO_JITTER, sleep=sleep) == "ok"
    assert sleep.delays == [7.0]

def test_retryable_rules() -> None:
    assert is_retryable(TimeoutError())
    assert is_retryable(ConnectionError())
    assert is_retryable(RateLimitError("429"))
    assert not is_retryable(ProviderError(message="404", status_code=404, retryable=False)) 
    assert not is_retryable(KeyError('x'))


def test_gives_up_after_max_attempts() -> None:
    sleep = FalseSleep()
    fn = Flaky(fail_times=100, error=TimeoutError("slow"))
    with pytest.raises(RetryExhaustedError) as exc_info:
        retry_call(fn, NO_JITTER, sleep=sleep)
    assert exc_info.value.attempts == 4
    assert isinstance(exc_info.value.last_error, TimeoutError)
    assert exc_info.value.__cause__ is exc_info.value.last_error  
    assert sleep.delays == [1.0, 2.0, 4.0]  


