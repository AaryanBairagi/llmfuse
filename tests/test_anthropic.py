import json
import os

import pytest

from llmfuse import FuseClient, ProviderError, RateLimitError, RetryPolicy
from llmfuse.providers import AnthropicProvider, GroqProvider
from llmfuse.testing import FakeTransport
from llmfuse.transport import HTTPResponse


def anthropic_response(*blocks: dict[str, str]) -> HTTPResponse:
    """What a successful Anthropic Messages API reply looks like."""
    body = {"type": "message", "role": "assistant", "content": list(blocks)}
    return HTTPResponse(status=200, body=json.dumps(body))


def text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def make_claude(transport: FakeTransport) -> AnthropicProvider:
    return AnthropicProvider(model="test-model", api_key="test-anthropic-key", transport=transport)


def test_sends_an_anthropic_format_request() -> None:
    transport = FakeTransport([anthropic_response(text_block("hi"))])
    make_claude(transport).complete("hello")

    request = transport.requests[0]
    assert request["url"] == "https://api.anthropic.com/v1/messages"
    assert request["headers"]["x-api-key"] == "test-anthropic-key"
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["json"]["model"] == "test-model"  # was: request["headers"]["model"]
    assert "Authorization" not in request["headers"]
    assert request["json"]["max_tokens"] == 1024
    assert request["json"]["messages"] == [{"role": "user", "content": "hello"}]
    

def test_returns_answer_text() -> None:
    transport = FakeTransport([anthropic_response(text_block("RAG is retrieval + generation"))])
    response = make_claude(transport).complete("hello")
    assert response == "RAG is retrieval + generation"


def test_block_joins_and_skips() -> None:
    claude_response = anthropic_response(
        {"type": "thinking", "thinking": "let me think..."},  # not text: skipped
        text_block("Hello, "),
        text_block("world"),
    )

    transport = FakeTransport([claude_response])
    response = make_claude(transport).complete("hello")
    assert response == "Hello, world"


def test_response_without_text_is_not_retryable() -> None:
    claude_response = anthropic_response()   #empty text
    transport = FakeTransport([claude_response])
    with pytest.raises(ProviderError) as error:
        make_claude(transport).complete("hello")
    assert error.value.retryable is False   


def test_429_has_retry_after_working() -> None:
    transport = FakeTransport([HTTPResponse(429, "rate limit error", {"retry-after" : "3"})])
    with pytest.raises(RateLimitError) as error:
        make_claude(transport).complete("hello")
    assert error.value.retry_after == 3.0


def test_529_overloaded_is_retryable() -> None:
    transport = FakeTransport([HTTPResponse(529, "overloaded_error")])
    with pytest.raises(ProviderError) as exc_info:
        make_claude(transport).complete("hello")
    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 529


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_client_errors_are_not_retryable(status: int) -> None:
    transport = FakeTransport([HTTPResponse(status, "client error")])
    with pytest.raises(ProviderError) as error:
        make_claude(transport).complete("hello")

    assert error.value.retryable is False


@pytest.mark.parametrize("status", [500, 501, 502, 503])
def test_client_errors_are_retryable(status: int) -> None:
    transport = FakeTransport([HTTPResponse(status, "server error")])
    with pytest.raises(ProviderError) as error:
        make_claude(transport).complete("hello")

    assert error.value.retryable is True


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model="m")


@pytest.mark.skipif(
    not (os.environ.get('ANTHROPIC_API_KEY') and os.environ.get('ANTHROPIC_MODEL')),
    reason = "set anthropic api key in the environment variables first"
)

def test_live_anthropic_provider() -> None:
    anthropic_model = os.environ['ANTHROPIC_MODEL']
    claude = AnthropicProvider(model=anthropic_model, max_tokens=20)
    answer = claude.complete("Just reply with word: pong")

    assert isinstance(answer,str) and answer.strip()


def test_fails_over_from_chat_format_to_anthropic_format() -> None:
    down = HTTPResponse(503, "overloaded")
    groq = GroqProvider(model="m", api_key="k", transport=FakeTransport([down, down, down]))
    claude = AnthropicProvider(model="m", api_key="k", transport=FakeTransport([anthropic_response(text_block("hi"))]) )
    client = FuseClient(
        providers=[groq, claude],
        retry=RetryPolicy(max_attempts=3, jitter=False),
        sleep=lambda seconds: None,
    )

    response = client.complete("hello")
    assert response.provider == "anthropic"
    assert response.text == "hi"