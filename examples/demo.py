import os

from llmfuse import AllProvidersFailedError, FuseClient
from llmfuse.providers import AnthropicProvider, GeminiProvider, GroqProvider


def main() -> None:
    groq = GroqProvider(model=os.environ["GROQ_MODEL"], requests_per_minute=30)
    gemini = GeminiProvider(model=os.environ["GEMINI_MODEL"], requests_per_minute=15)
    claude = AnthropicProvider(model=os.environ["ANTHROPIC_MODEL"])

    client = FuseClient(
        providers=[groq, gemini, claude],
        requests_per_minute=50,  # default for providers without their own limit (here: claude)
    )

    prompt = "Explain RAG in short."
    try:
        response = client.complete(prompt)
    except AllProvidersFailedError as error:
        print("All providers failed:")
        for name, reason in error.errors.items():
            print(f"  {name}: {reason}")
        raise SystemExit(1) from error

    print(f"[{response.provider}] {response.text}")


if __name__ == "__main__":
    main()