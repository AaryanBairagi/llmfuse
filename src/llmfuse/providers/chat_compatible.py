import json
import os

from llmfuse.errors import ProviderError
from llmfuse.transport import Transport, error_from_response, urllib_transport


class ChatCompatibleProvider:

    default_name : str = "openai_compatible"
    default_base_url : str = ""
    api_key_env : str = ""

    def __init__(
            self,
            *,
            model: str,
            api_key: str | None = None,
            base_url: str | None = None,
            name:str | None = None,
            max_tokens: int = 1024,
            timeout: float = 30.0,
            requests_per_minute : float | None = None,
            transport : Transport = urllib_transport
    ) -> None:

        self.model = model
        self.name = name or self.default_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.requests_per_minute = requests_per_minute

        self._transport = transport

        self._base_url = (base_url or self.default_base_url).rstrip('/')
        if not self._base_url:
            raise ValueError(f"{self.name} : base url is required.")

        if api_key is None and self.api_key_env:
            api_key = os.environ.get(self.api_key_env)

        if self.api_key_env and not api_key:    
            raise ValueError(f"{self.name}: no API key. Pass api_key=... or set {self.api_key_env}")
        
        self._api_key = api_key or ""

    def complete(self, prompt: str) -> str:
        url = f"{self._base_url}/chat/completions"
        
        headers = {"Content-Type" : "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model" : self.model,
            "messages" : [{"role" : "user" , "content" : prompt}],
            "max_tokens" : self.max_tokens
        }

        try:
            response = self._transport(url, headers, json.dumps(payload).encode(), self.timeout)

        except (TimeoutError , ConnectionError) as error: 
            raise ProviderError(
                f"{self.name} - network-error : {error} " , provider=self.name , retryable=True
            ) from error

        if response.status != 200:
            raise error_from_response(self.name , response)
        
        return self._parse(response.body)

    # JSON shape for chat requests: send to /chat/completions, put the prompt in messages, read the answer from choices[0].message.content

    def _parse(self, body: str) -> str:
        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]

        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                f"{self.name}: unexpected response format",
                provider=self.name,  
                retryable=False,
            )

        if not isinstance(content, str):
            raise ProviderError(
                f"{self.name}: response had no text",
                provider=self.name,  
                retryable=False,
            )
        return content


class GroqProvider(ChatCompatibleProvider):
    default_name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"


class GeminiProvider(ChatCompatibleProvider):
    default_name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key_env = "GEMINI_API_KEY"


class OpenAIProvider(ChatCompatibleProvider):
    default_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"


