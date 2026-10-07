"""Runs against the built package (not the source) to prove it installs and works."""

from llmfuse import FuseClient, __version__
from llmfuse.testing import FakeProvider

down = FakeProvider("down", errors=[ConnectionError("down")] * 10)
up = FakeProvider("up", reply="ok")
client = FuseClient(providers=[down, up], sleep=lambda seconds: None)

response = client.complete("ping")
assert response.provider == "up", response
assert response.text == "ok", response
print(f"llmfuse {__version__}: smoke test passed")