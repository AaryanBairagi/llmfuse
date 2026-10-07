import json
from collections.abc import Sequence
from typing import Any

from llmfuse.transport import HTTPResponse


class FakeProvider:
    """A pretend LLM provider: raises `errors` one by one, then returns `reply`."""

    def __init__(
        self, name: str, *, reply: str = "ok", errors: Sequence[Exception] = (), requests_per_minute : float | None = None 
    ) -> None:
        
        self.name = name
        self.reply = reply
        self.errors = list(errors)
        self.calls = 0
        self.requests_per_minute = requests_per_minute  

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.reply


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTransport:
    """A pretend network: returns (or raises) the given items in order, records requests."""

    def __init__(self, response: Sequence[HTTPResponse | Exception]) -> None:
        self._responses = list(response)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url:str, headers:dict[str,str], body:bytes, timeout:float) -> HTTPResponse:
        self.requests.append({"url":url , "headers":headers, "json" : json.loads(body), "timeout": timeout})
    
        item = self._responses.pop(0)
        
        if isinstance(item , Exception):
            raise item
        
        return item
    