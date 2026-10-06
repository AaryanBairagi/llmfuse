import pytest

from llmfuse.errors import RateLimitError
from llmfuse.retry import RetryPolicy


def test_backoff_doubles_then_hits_the_cap() -> None:
    p = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=5.0)
    assert [p.backoff(i) for i in range(1, 6)] == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_no_jitter_returns_the_ceiling_exactly() -> None:
    p = RetryPolicy(jitter=False)
    assert p.delay_for(3) == 4.0


def test_full_jitter_returns_random_value() -> None:
    p = RetryPolicy(rng=lambda: 0.5)  # fake random always 0.5
    assert p.delay_for(3) == 2.0  # 0.5 * 40


def test_retry_after_wins_over_backoff() -> None:
    p = RetryPolicy()
    assert p.delay_for(1, RateLimitError("slow", retry_after=3.0)) == 3.0


def test_retry_after_is_still_capped() -> None:
    p = RetryPolicy(max_delay=10.0)
    assert p.delay_for(1, RateLimitError("slow", retry_after=99.0)) == 10.0


def test_policy_is_immutable():
    p = RetryPolicy()
    with pytest.raises(AttributeError):
        p.max_attempts = 99


@pytest.mark.parametrize(
    "bad_settings",
    [{"max_attempts": 0}, {"max_delay": -1}, {"base_delay": -1}, {"multiplier": 0.5}],
)
def test_bad_settings_are_rejected(bad_settings: dict) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**bad_settings)