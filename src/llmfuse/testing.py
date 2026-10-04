from collections.abc import Sequence


class FakeProvider:
    """A pretend LLM provider: raises `errors` one by one, then returns `reply`."""

    def __init__(
        self, name: str, *, reply: str = "ok", errors: Sequence[Exception] = ()
    ) -> None:
        self.name = name
        self.reply = reply
        self.errors = list(errors)
        self.calls = 0

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
