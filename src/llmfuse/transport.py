import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import version

from llmfuse.errors import ProviderError, RateLimitError


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: str
    headers: dict[str,str] = field(default_factory=dict) #Whenever a new HTTPResponse instance is created without explicit headers, call the function dict() to generate a brand-new, empty dictionary just for this instance.


Transport = Callable[[str, dict[str,str], bytes, float] , HTTPResponse]

USER_AGENT = f"llmfuse/{version('llmfuse')} (+https://github.com/AaryanBairagi/llmfuse)"

def urllib_transport(url:str, headers: dict[str,str], body: bytes, timeout: float) -> HTTPResponse :
    """The real transport using the python's standard library."""
    headers = {"User-Agent" : USER_AGENT, **headers} ##CloudFlare is blocking User Agent
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                status=response.status,
                body=response.read().decode("utf-8"),
                headers={k.lower() : v for k , v in response.headers.items()},
            )
        
    except urllib.error.HTTPError as error:
        return HTTPResponse(
            status = error.code,
            body = error.read().decode("utf-8" , errors="replace"),
            headers = {k.lower() : v for k , v in error.headers.items()},
        )

    except urllib.error.URLError as error:
        raise ConnectionError(str(error.reason)) from error


def parse_retry_after(val: str | None) -> float | None:
    """Retry-After is usually a number of seconds. Anything else: we don't know."""
    if val is None:
        return None
    
    try:
        return max(0.0, float(val))
    
    except ValueError:
        return None



def error_from_response(provider: str, response: HTTPResponse) -> ProviderError:
    """Translate an HTTP error response into ONE of our errors (anti-corruption layer)."""
    message = f"{provider}: HTTP {response.status}: {response.body[:200]}"

    if response.status == 429:
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        return RateLimitError(message, provider=provider, retry_after=retry_after)
    
    retryable = response.status >= 500 or response.status == 408
    return ProviderError(message, provider=provider, status_code=response.status, retryable=retryable)