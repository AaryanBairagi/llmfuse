import json
import os

from llmfuse.errors import ProviderError
from llmfuse.transport import Transport, error_from_response, urllib_transport

ANTHROPIC_VERSION = "2023-06-01"

class  AnthropicProvider:

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        name: str = "anthropic",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        requests_per_minute : float | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        transport: Transport = urllib_transport
        ) -> None:

        self.model = model
        self.name = name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.requests_per_minute = requests_per_minute

        self._transport = transport
        self._base_url = base_url.rstrip("/")

        if api_key is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')

        if not api_key:
            raise ValueError(f"{name}: no API key. Pass api_key=... or set ANTHROPIC_API_KEY")
        self._api_key = api_key


    def complete(self, prompt: str) -> str:
        url = f"{self._base_url}/messages"

        headers = {
            "x-api-key" : self._api_key,
            "anthropic-version" : ANTHROPIC_VERSION,
            "content-type" : "application/json"
        }

        payload = {
            "model" : self.model,
            "max_tokens" : self.max_tokens,
            "messages" : [{"role" : "user" , "content" : prompt}]
        }

        try:
            response = self._transport(url, headers, json.dumps(payload).encode(), self.timeout)

        except (TimeoutError , ConnectionError) as error:
            raise ProviderError(f"{self.name} network error : {error}", provider=self.name, retryable=True) from error

        if response.status != 200:
            raise error_from_response(self.name , response)
        
        text = self._parse(response.body)
        return text

    
    def _parse(self, body: str) -> str:
        try:
            blocks = json.loads(body)["content"]
            text = "".join(
                block["text"] for block in blocks if block.get("type") == "text"
            )
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            raise ProviderError(
                f"{self.name}: unexpected response format: {error}",
                provider=self.name,
                retryable=False,
            ) from error

        if not text:
            raise ProviderError(
                f"{self.name}: response had no text",
                provider=self.name,
                retryable=False,
            )
        return text