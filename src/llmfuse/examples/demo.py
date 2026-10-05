import os
from llmfuse import FuseClient, AllProvidersFailedError
from llmfuse.providers import GeminiProvider, GroqProvider
import dotenv

dotenv.load_dotenv()


def main() -> None:

    # A deliberately broken provider first, to SEE failover happen:
    groq1 = GroqProvider(
        model=os.environ["GROQ_MODEL"], api_key="bad-key", name="groq-broken"
    )
    groq2 = (GroqProvider(model=os.environ["GROQ_MODEL"]),)
    gemini = (GeminiProvider(model=os.environ["GEMINI_MODEL"]),)

    providers = [groq1, groq2, gemini]
    client = FuseClient(
        providers=providers,
        requests_per_minute=30,
    )

    try:
        prompt = "Explain RAG in short."
        response = client.complete(prompt)

    except AllProvidersFailedError as error:
        print("All providers failed : ")
        for name, reason in error.errors.items():
            print(f" {name} : {reason}")
        raise SystemExit(1) from error

    print(f"{[response.provider]} : {response.text}")


if __name__ == "__main__":
    main()
