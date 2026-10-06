import json
import os

import pytest

from llmfuse.errors import ProviderError, RateLimitError
from llmfuse.providers import GroqProvider
from llmfuse.testing import FakeTransport
from llmfuse.transport import HTTPResponse


def chat_response(text : str) -> HTTPResponse:
    """What a model will respond to a query."""
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    return HTTPResponse(status=200, body=json.dumps(body))


def make_groq(transport: FakeTransport) -> GroqProvider:
    return GroqProvider(model="test-model", api_key="test-groq-api-key", transport=transport)


def test_returns_answer_text() -> None:
    transport = FakeTransport([chat_response("RAG is retrieval + generation")])
    groq = make_groq(transport)
    answer = groq.complete("hello")
    assert answer == "RAG is retrieval + generation"


def test_sends_an_openai_request() -> None:
    transport = FakeTransport([chat_response("hello")])
    make_groq(transport).complete("yo")

    request = transport.requests[0]  # was transport["requests"]

    assert request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert request["json"]["model"] == "test-model"
    assert request["json"]["messages"] == [{"role": "user", "content": "yo"}]  # fixed
    assert request["headers"]["Authorization"] == "Bearer test-groq-api-key"


def test_429_becomes_rate_limit_error_with_retry_after() -> None:
    transport = FakeTransport([HTTPResponse(429, "slow-down", {"retry-after" : 7})])
    with pytest.raises(RateLimitError) as rlerr :
        make_groq(transport).complete("hello")

    assert rlerr.value.retry_after == 7.0
    assert rlerr.value.provider == "groq"


def test_429_without_retry_after() -> None:
    transport = FakeTransport([HTTPResponse(429, "slow-down")])
    with pytest.raises(RateLimitError) as rlerr :
        make_groq(transport).complete("hello")

    assert rlerr.value.retry_after is None 


def test_unparseable_retry_after_falls_backoff() -> None:
    date_header = {"retry-after" : "Wed, 21 Oct 2026 07:00:06 GMT"}
    transport = FakeTransport([HTTPResponse(429, "slow-down", date_header)])
    with pytest.raises(RateLimitError) as rlerr :
        make_groq(transport).complete("hello")

    assert rlerr.value.retry_after is None


def test_malformed_response_is_not_retryable() -> None:
    transport = FakeTransport([HTTPResponse(200, "<html> This is html code </html>")])
    with pytest.raises(ProviderError) as error:  # was pytest.raises() with nothing
        make_groq(transport).complete("hello")
    assert error.value.retryable is False
    

@pytest.mark.parametrize("network_error", [TimeoutError("slow"), ConnectionError("reset")])
def test_network_errors_are_retryable(network_error: Exception) -> None:
    transport = FakeTransport([network_error])
    with pytest.raises(ProviderError) as perr:
        make_groq(transport).complete("hello")

    assert perr.value.retryable is True

@pytest.mark.parametrize("status", [400,401,402,403])
def test_client_errors_are_not_retryable(status: int):
    transport = FakeTransport([HTTPResponse(status, "your_fault")])
    with pytest.raises(ProviderError) as selferr:
        make_groq(transport).complete("hello")

    assert selferr.value.retryable is False


@pytest.mark.parametrize("status", [500,501,502,503])
def test_server_errors_are_retryable(status: int):
    transport = FakeTransport([HTTPResponse(status, "server_fault")])
    with pytest.raises(ProviderError) as selferr:
        make_groq(transport).complete("hello")

    assert selferr.value.status_code == status
    assert selferr.value.retryable is True


def test_adapter_remembers_its_requests() -> None:
    groq = GroqProvider(model="model_m", api_key="test_api_key", requests_per_minute=30.0)
    assert groq.requests_per_minute == 30.0
    groq1 = GroqProvider(model="model_m", api_key="test_api_key")
    assert groq1.requests_per_minute is None