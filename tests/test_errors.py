import pytest

from llmfuse import LLMFuseError, ProviderError, RateLimitError

def test_provider_error_store_details() -> None:
    err = ProviderError("server down" , status_code = 501 , provider = "groq")
    assert str(err) == "server down"
    assert err.status_code == 501
    assert err.provider == "groq"
    assert err.retryable is True


def test_non_retryable_error() -> None:
    err = ProviderError("bad key" , status_code = 401 , retryable = False)
    assert err.retryable is False


def test_rate_limit_is_always_429_and_retryable() -> None:
    err = RateLimitError("slow down" , retry_after = 7.0)
    assert err.status_code == 429
    assert err.retryable is True
    assert err.retry_after == 7.0


def test_everything_is_catchable_as_llmfuse_error() -> None:
    with pytest.raises(LLMFuseError):
        raise RateLimitError("slow down")


def test_arguments_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        ProviderError("fail", "gemini", 503)  # type: ignore[misc]