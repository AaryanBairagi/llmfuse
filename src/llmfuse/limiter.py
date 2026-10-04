import time
from collections.abc import Callable


class TokenBucket:

    def __init__(
          self,
          rate : float,
          capacity: float,
          *,
          clock : Callable[[] , float] = time.monotonic,  
    ) -> None:
        
        if rate <= 0:
            raise ValueError("rate must be positive.")

        if capacity < 1:
            raise ValueError("capacity must be atleast 1.")
        
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._tokens = float(capacity) #start full 
        self._last_refill = clock()

    #formula : new_tokens = elpased time x rate  //  distance = speed x time
    def _refill(self) -> None:
        """How long until the next token arrives in the token bucket.
        If token bucket is idle then (no. of tokens = capacity) tokens available. 
        """
        now = self._clock()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now


    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens 


    #formula for waiting time x rate + tokens = 1
    def time_until_unavailable(self) -> float:
        """Seconds until one token is available (0.0 if one is available now)."""
        self._refill()
        if self._tokens >= 1:
            return 0.0
        return (1 - self._tokens) / self.rate


    def consume(self) -> None:
        """Spend one token. Call only after time_until_available() reached 0."""

        self._refill()
        self._tokens -= 1      