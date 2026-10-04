import pytest

from llmfuse.limiter import TokenBucket
from llmfuse.testing import FakeClock


def make_bucket(clock : FakeClock) -> TokenBucket:
    # 3 tokens max, refills 0.5 tokens/second (= 30 requests per minute)
    return TokenBucket(rate=0.5, capacity=3, clock=clock)


def test_initiate_bucket() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    assert bucket.tokens == 3
    assert bucket.time_until_unavailable() == 0.0


def test_allows_burst_up_to_capacity() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    for _ in range(3):
        assert bucket.time_until_unavailable() == 0.0
        bucket.consume()
    assert bucket.time_until_unavailable() == pytest.approx(2.0)      #becuase rate=0.5, tokens=0 => 1-0 / 0.5 = 2.0


def test_refills_over_time() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    for _ in range(3): 
        bucket.consume()
    clock.advance(2.0)
    assert bucket.tokens == 1.0
    assert bucket.time_until_unavailable() == 0.0


def test_partial_token_shorter_wait() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    for _ in range(3):
        bucket.consume()
    clock.advance(1.0)
    assert bucket.tokens == 0.5
    assert bucket.time_until_unavailable() == 1.0


def test_bucket_never_holds_more_than_capacity() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    clock.advance(100.0)
    assert bucket.tokens == 3
    assert bucket.time_until_unavailable() == 0.0


@pytest.mark.parametrize("bad_settings", [{"rate": 0, "capacity": 3}, {"rate": 1, "capacity": 0}])
def test_bad_settings_are_rejected(bad_settings: dict) -> None:
    with pytest.raises(ValueError):
        TokenBucket(**bad_settings)