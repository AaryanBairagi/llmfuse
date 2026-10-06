import json
import os

import pytest

from llmfuse.client import FuseClient
from llmfuse.errors import ProviderError, RateLimitError
from llmfuse.providers import GeminiProvider, GroqProvider
from llmfuse.retry import RetryPolicy
from llmfuse.testing import FakeTransport
from llmfuse.transport import HTTPResponse


def chat_response(text: str) -> HTTPResponse:
    """What a model will respond to a query."""
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    return HTTPResponse(status=200, body=json.dumps(body))


def make_groq(transport: FakeTransport) -> GroqProvider:
    return GroqProvider(
        model="test-model", api_key="test-groq-api-key", transport=transport
    )


def test_returns_answer_text() -> None:
    transport = FakeTransport([chat_response("RAG is retrieval + generation")])
    groq = make_groq(transport)
    answer = groq.complete("hello")
    assert answer == "RAG is retrieval + generation"


def test_sends_an_openai_request() -> None:
    transport = FakeTransport([chat_response("hello")])
    make_groq(transport).complete("yo")

    request = transport.requests[0]

    assert request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert request["json"]["model"] == "test-model"
    assert request["json"]["messages"] == [{"role": "user", "content": "yo"}]
    assert request["headers"]["Authorization"] == "Bearer test-groq-api-key"


def test_429_becomes_rate_limit_error_with_retry_after() -> None:
    transport = FakeTransport([HTTPResponse(429, "slow-down", {"retry-after": "7"})])
    with pytest.raises(RateLimitError) as rlerr:
        make_groq(transport).complete("hello")

    assert rlerr.value.retry_after == 7.0
    assert rlerr.value.provider == "groq"


def test_429_without_retry_after() -> None:
    transport = FakeTransport([HTTPResponse(429, "slow-down")])
    with pytest.raises(RateLimitError) as rlerr:
        make_groq(transport).complete("hello")

    assert rlerr.value.retry_after is None


def test_unparseable_retry_after_falls_backoff() -> None:
    date_header = {"retry-after": "Wed, 21 Oct 2026 07:00:06 GMT"}
    transport = FakeTransport([HTTPResponse(429, "slow-down", date_header)])
    with pytest.raises(RateLimitError) as rlerr:
        make_groq(transport).complete("hello")

    assert rlerr.value.retry_after is None


def test_malfomed_response_shows_not_retryable() -> None:
    transport = FakeTransport([HTTPResponse(200, "<html> This is html code </html>")])
    with pytest.raises(ProviderError) as error:  
        make_groq(transport).complete("hello")
    assert error.value.retryable is False


@pytest.mark.parametrize("network_error", [TimeoutError("slow"), ConnectionError("reset")])
def test_network_errors_are_retryable(network_error: Exception) -> None:
    transport = FakeTransport([network_error])
    with pytest.raises(ProviderError) as perr:
        make_groq(transport).complete("hello")

    assert perr.value.retryable is True


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retryable(status: int):
    transport = FakeTransport([HTTPResponse(status, "your_fault")])
    with pytest.raises(ProviderError) as selferr:
        make_groq(transport).complete("hello")

    assert selferr.value.retryable is False


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retryable(status: int):
    transport = FakeTransport([HTTPResponse(status, "server_fault")])
    with pytest.raises(ProviderError) as selferr:
        make_groq(transport).complete("hello")

    assert selferr.value.status_code == status
    assert selferr.value.retryable is True


def test_api_key_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY" , "test-groq-api") #temporarily change the env variable to a garbage value
    transport = FakeTransport([chat_response("hello")])
    groq = GroqProvider(model="m", transport=transport)
    groq.complete("yo")
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer test-groq-api"


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)        # pretend the env var is NOT set
    with pytest.raises(ValueError, match="GROQ_API_KEY"):    
        GroqProvider(model="m")  



def test_real_adapters_fail_over_through_the_whole_stack() -> None:
    down = HTTPResponse(503, "overloaded")
    groq = GroqProvider(model="m", api_key="k", transport=FakeTransport([down, down, down]))
    gemini = GeminiProvider(
        model="m", api_key="k", transport=FakeTransport([chat_response("hi from gemini")])
    )
    client = FuseClient(
        providers=[groq, gemini],
        retry=RetryPolicy(max_attempts=3, jitter=False),
        sleep=lambda seconds: None,
    )
    response = client.complete("hello")
    assert response.provider == "gemini"
    assert response.text == "hi from gemini"


@pytest.mark.skipif(
    not (os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_MODEL")),
    reason="set GROQ_API_KEY and GROQ_MODEL to run the live test",
)
def test_live_groq_call() -> None:
    provider = GroqProvider(model=os.environ["GROQ_MODEL"], max_tokens=20)
    answer = provider.complete("Reply with just the word: pong")
    assert isinstance(answer, str) and answer.strip()