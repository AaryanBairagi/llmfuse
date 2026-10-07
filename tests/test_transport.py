import urllib.request
from contextlib import AbstractContextManager, nullcontext

import pytest

from llmfuse.transport import USER_AGENT, urllib_transport


class FakeURLResponse:
    """Stands in for the response object urllib.request.urlopen returns."""

    def __init__(self) -> None:
        self.status = 200
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return b'{"ok": true}'


def test_sends_llmfuse_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, urllib.request.Request] = {}

    def fake_urlopen(
        request: urllib.request.Request, timeout: float
    ) -> AbstractContextManager[FakeURLResponse]:
        sent["request"] = request
        return nullcontext(FakeURLResponse())  # works with `with ... as response:`

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    urllib_transport("https://example.com/v1/chat", {"x-api-key": "k"}, b"{}", 5.0)

    request = sent["request"]
    assert request.get_header("User-agent") == USER_AGENT  # urllib stores it as "User-agent"
    assert request.get_header("X-api-key") == "k"  # our own headers still go through
    assert USER_AGENT.startswith("llmfuse/")