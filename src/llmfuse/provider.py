from typing import Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class Response:
    """What FuseClient().complete hands back to the user"""

    text : str
    provider : str

class Provider(Protocol):
    name:  str

    def complete(self, prompt: str) -> str: ...