import pytest

from llmfuse.breaker import CircuitState
from llmfuse.client import FuseClient
from llmfuse.errors import AllProvidersFailedError, CircuitOpenError, ProviderError, ThrottledError
from llmfuse.retry import RetryPolicy
from llmfuse.testing import FakeClock, FakeProvider

FAST = RetryPolicy(max_attempts=3, jitter=False, base_delay=0.01, multiplier=1.0)
DOWN = ProviderError("503 Service Unavailable", status_code=503)


def no_sleep(seconds: float) -> None:
    pass


def make_client(*providers: FakeProvider) -> FuseClient:
    return FuseClient(providers=list(providers), retry=FAST, sleep=no_sleep)


def make_breaker_client(clock: FakeClock, *providers: FakeProvider) -> FuseClient:
    return FuseClient(
        providers=list(providers),
        retry=FAST,
        reset_timeout=30.0,
        failure_threshold=2,
        sleep=no_sleep,
        clock=clock,
    )


def make_limiter_client(
        clock: FakeClock, *providers: FakeProvider, requests_per_minute: float, 
        burst: int, max_wait: float = 1.0, failure_threshold: int = 5
        ) -> FuseClient:
    
    return FuseClient(
        providers= list(providers),
        retry=FAST,
        requests_per_minute=requests_per_minute,
        burst=burst,
        max_wait=max_wait,
        failure_threshold=failure_threshold,
        sleep=clock.advance,
        clock=clock,
    )


def test_first_healthy_provider_responds() -> None:
    groq = FakeProvider("groq", reply="hi from groq")
    openai = FakeProvider("openai", reply="hi from openai")
    gemini = FakeProvider("gemini", reply="hi from gemini")

    response = make_client(groq, openai, gemini).complete("hello")
    assert response.text == "hi from groq"
    assert response.provider == "groq"
    assert groq.calls == 1
    assert openai.calls == 0
    assert gemini.calls == 0


def test_first_provider_fails_second_healthy_provider_responds() -> None:
    groq = FakeProvider("groq", errors=[DOWN, DOWN, DOWN])
    gemini = FakeProvider("gemini", reply="hi from gemini")
    response = make_client(groq, gemini).complete("hello")
    assert response.text == "hi from gemini"
    assert groq.calls == 3
    assert gemini.calls == 1


def test_temporary_provider_failures_retry_until_success() -> None:
    groq = FakeProvider("groq", errors=[DOWN], reply="hi from groq")
    gemini = FakeProvider("gemini", reply="hi from gemini")
    response = make_client(groq, gemini).complete("hello")
    assert response.text == "hi from groq"
    assert groq.calls == 2
    assert gemini.calls == 0


def test_bad_api_key_is_not_retryable() -> None:
    bad_api_key_error = ProviderError(
        "401 Bad API Key", status_code=401, retryable=False
    )
    groq = FakeProvider("groq", errors=[bad_api_key_error], reply="hello from groq")
    gemini = FakeProvider("gemini", reply="hello from gemini")
    response = make_client(groq, gemini).complete("hello")
    assert response.text == "hello from gemini"
    assert response.provider == "gemini"
    assert groq.calls == 1
    assert gemini.calls == 1


def test_all_providers_fail() -> None:
    groq = FakeProvider("groq", errors=[DOWN, DOWN, DOWN])
    gemini = FakeProvider("gemini", errors=[DOWN, DOWN, DOWN])
    client = make_client(groq, gemini)
    with pytest.raises(AllProvidersFailedError) as exc_info:
        client.complete("hello")
    assert set(exc_info.value.errors) == {"groq", "gemini"}


def test_bugs_are_not_hidden_by_failovers() -> None:
    groq = FakeProvider("groq", errors=[TypeError("unexpected type")])
    gemini = FakeProvider("gemini", errors=[DOWN])
    with pytest.raises(TypeError):
        make_client(groq, gemini).complete("hello")
    assert gemini.calls == 0


def test_needs_atleast_one_provider() -> None:
    with pytest.raises(ValueError):
        FuseClient(providers=[])


def test_open_circuit_skips_providers_instantly() -> None:
    groq = FakeProvider("groq", errors=[DOWN] * 100)
    gemini = FakeProvider("gemini", reply="hi from gemini")
    client = make_breaker_client(FakeClock(), groq, gemini)

    client.complete("hello")
    client.complete("hello")

    assert groq.calls == 6
    assert client.circuit_state("groq") is CircuitState.OPEN


def test_circuit_recovers_after_the_timeout() -> None:
    clock = FakeClock()
    groq = FakeProvider("groq", errors=[DOWN] * 6, reply="hi from groq")
    gemini = FakeProvider("gemini", reply="hi from gemini")
    client = make_breaker_client(clock, groq, gemini)

    client.complete("hello")
    client.complete("hello")  # tripped; groq has now used up its 6 errors
    clock.advance(30.0)

    response = client.complete("hello")  # half-open trial succeeds
    assert response.provider == "groq"
    assert client.circuit_state("groq") is CircuitState.CLOSED


def test_all_circuits_open_reports_circuit_open_errors() -> None:
    groq = FakeProvider("groq", errors=[DOWN] * 100)
    gemini = FakeProvider("gemini", errors=[DOWN] * 100)
    client = make_breaker_client(FakeClock(), groq, gemini)
    for _ in range(2):
        with pytest.raises(AllProvidersFailedError):
            client.complete("hello")

    with pytest.raises(AllProvidersFailedError) as exc_info:
        client.complete("hello")
    assert isinstance(exc_info.value.errors["groq"], CircuitOpenError)
    assert isinstance(exc_info.value.errors["gemini"], CircuitOpenError)
    assert groq.calls == 6 and gemini.calls == 6  # nobody was called the 3rd time


def test_provider_names_must_be_unique() -> None:
    with pytest.raises(ValueError):
        FuseClient(providers=[FakeProvider("groq"), FakeProvider("groq")])


def test_short_wait_but_provider_works() -> None:
    clock = FakeClock()
    groq = FakeProvider("groq", reply="hi from groq.")
    gemini = FakeProvider("gemini", reply="hi from gemini.")
    client = make_limiter_client(clock, groq, gemini, requests_per_minute=60, burst=1)

    assert client.complete("one").provider == "groq"
    assert client.complete("two").provider == "groq"
    assert gemini.calls == 0


def test_every_retry_attempts_to_use_token() -> None:
    clock = FakeClock()
    groq = FakeProvider("groq", errors=[DOWN,DOWN], reply="hi from groq.")
    client = make_limiter_client(clock, groq, requests_per_minute=60, burst=1, failure_threshold=1)
    response = client.complete("hello.")
    
    assert response.provider == "groq"
    assert groq.calls == 3
    assert clock.now == pytest.approx(2.0) #because 3 requests use 3 tokens, bucket empty.


def test_long_wait_fails_without_tripping_breaker() -> None:
    clock = FakeClock()
    groq = FakeProvider("groq", reply="hi from groq.")
    gemini = FakeProvider("gemini", reply="hi from gemini.")
    client = make_limiter_client(clock, groq, gemini, requests_per_minute=15, burst=1, failure_threshold=1) #rate=requests_per_minute/60
    response1 = client.complete("one")
    assert response1.provider == "groq"

    response2 = client.complete("two")
    assert response2.provider == "gemini"
    assert groq.calls == 1
    assert gemini.calls == 1
    assert client.circuit_state("groq") is CircuitState.CLOSED


def test_all_providers_throttled_errors() -> None:
    clock = FakeClock()
    groq = FakeProvider("groq", reply="hi from groq.")
    gemini = FakeProvider("gemini", reply="hi from gemini.")
    client = make_limiter_client(clock, groq, gemini, requests_per_minute=15, burst=1, max_wait=1.0)

    client.complete("one")
    client.complete("two")
    with pytest.raises(AllProvidersFailedError) as apferror:
        client.complete("three")

    assert isinstance(apferror.value.errors["groq"], ThrottledError)
    assert isinstance(apferror.value.errors["gemini"], ThrottledError)