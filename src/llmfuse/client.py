import time

from functools import partial
from collections.abc import Sequence , Callable

from llmfuse.provider import Provider , Response
from llmfuse.retry import RetryPolicy , retry_call
from llmfuse.errors import AllProvidersFailedError , LLMFuseError
from llmfuse.breaker import CircuitBreaker, CircuitState                              # NEW

class FuseClient:

    def __init__(
        self,
        *,
        providers : Sequence[Provider],
        retry : RetryPolicy | None = None,
        failure_threshold = 5,
        reset_timeout : float = 30.0,
        sleep : Callable[[float] , None] = time.sleep,
        clock : Callable[[] , float] = time.monotonic,
        ):

        if not providers:
            raise ValueError("FuseClient need atleast one provider.")

        self.providers = list(providers)
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._breakers = {
            provider.name : CircuitBreaker(failure_threshold , reset_timeout , clock=clock) 
            for provider in providers  
        }

    def breaker_state(self, provider_name: str) -> CircuitState:


    def complete(self , prompt : str) -> Response:
        errors : dict[str,BaseException] = {}

        for provider in self.providers:
            ##PHASE 2 - Circuit Breaker , RateLimiter
            try:
                text = retry_call(
                    partial(provider.complete , prompt),
                    self.retry,
                    sleep = self.sleep,
                )

            except LLMFuseError as error:
                errors[provider.name] = error
                continue 
            
            return Response(text=text , provider=provider.name)
        
        raise AllProvidersFailedError(errors)

