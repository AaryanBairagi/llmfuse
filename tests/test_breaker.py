import pytest

from llmfuse.breaker import CircuitBreaker, CircuitState
from llmfuse.testing import FakeClock


def make_breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, reset_timeout=30.0, clock=clock)


def test_starts_closed_and_allows_no_request() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()


def test_success_resets_failures_count() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_stays_open_until_timeout() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(29.9)
    assert not breaker.allow_request()
    assert breaker.state is CircuitState.OPEN


def test_transitions_to_half_open_after_timeout() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(30.0)
    assert breaker.allow_request()
    assert breaker.state is CircuitState.HALF_OPEN


def test_trips_open_after_threshold() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()


@pytest.mark.parametrize(
    "bad_settings", [{"failure_threshold": 0}, {"reset_timeout": -1.0}]
)
def test_bad_settings_are_rejected(bad_settings: dict) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(**bad_settings)


def test_half_open_success_closes_the_circuit() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(30.0)
    breaker.allow_request()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_half_open_failure_reopens_and_restarts_the_timer() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(30.0)
    breaker.allow_request()
    breaker.record_failure()  # trial failed: ONE failure is enough to re-open
    assert breaker.state is CircuitState.OPEN
    clock.advance(29.0)
    assert not breaker.allow_request()  # timer restarted at t=30, not t=0
    clock.advance(1.0)
    assert breaker.allow_request()
