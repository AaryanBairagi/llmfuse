import time
from collections.abc import Callable
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed",
    OPEN = "open",
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):

        if failure_threshold < 1:
            raise ValueError("Threshold value must be a positive integer.")

        if reset_timeout <= 0:
            raise ValueError("Reset timeout must be a positive number of seconds.")

        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_request(self) -> bool:
        """Should we call this provider right now?"""
        if self._state is CircuitState.OPEN:
            if self._clock() - self._opened_at >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self):
        if self._state is CircuitState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._failures = 0
