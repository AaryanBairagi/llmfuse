import pytest
from src.llmfuse.client import FuseClient
from src.llmfuse.errors import ProviderError , AllProvidersFailedError
from src.llmfuse.testing import FakeProvider 
from src.llmfuse.retry import RetryPolicy

FAST = RetryPolicy(max_attempts=3 , jitter=False , base_delay=0.01 , multiplier=1.0)
DOWN = ProviderError("503 Service Unavailable" , status_code=503)


def no_sleep(seconds:float) -> None:
    pass


def make_client(*providers : FakeProvider) -> FuseClient:
    return FuseClient(providers=list(providers), retry=FAST, sleep=no_sleep)


def test_first_healthy_provider_responds() -> None:
    groq = FakeProvider("groq" , reply = "hi from groq")
    openai = FakeProvider("openai" , reply = "hi from openai")
    gemini = FakeProvider("gemini" , reply = "hi from gemini")

    response = make_client(groq, openai, gemini).complete("hello")
    assert response.text == "hi from groq"
    assert response.provider == "groq"
    assert groq.calls == 1 
    assert openai.calls == 0
    assert gemini.calls == 0


def test_first_provider_fails_second_healthy_provider_responds() -> None:
    groq = FakeProvider("groq" , errors=[DOWN, DOWN, DOWN])
    gemini = FakeProvider("gemini" , reply = "hi from gemini")
    response = make_client(groq,gemini).complete("hello")
    assert response.text == "hi from gemini"
    assert groq.calls == 3
    assert gemini.calls == 1


def test_temporary_provider_failures_retry_until_success() -> None:
    groq = FakeProvider("groq" , errors=[DOWN] , reply = "hi from groq")
    gemini = FakeProvider("gemini" , reply = "hi from gemini")
    response = make_client(groq,gemini).complete("hello")
    assert response.text == "hi from groq"
    assert groq.calls == 2
    assert gemini.calls == 0


def test_bad_api_key_is_not_retryable() -> None:
    bad_api_key_error = ProviderError("401 Bad API Key" , status_code=401, retryable=False)
    groq = FakeProvider("groq", errors=[bad_api_key_error], reply="hello from groq")
    gemini = FakeProvider("gemini", reply="hello from gemini")
    response = make_client(groq,gemini).complete("hello")
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
        make_client(groq,gemini).complete("hello")
    assert gemini.calls == 0


def test_needs_atleast_one_provider() -> None:
    with pytest.raises(ValueError):
        FuseClient(providers=[])