from dotenv import load_dotenv

from client import LLMClient
from factory import get_llm_provider
from mongodb_logger import MongoDBLogger
from ollama_metrics import generate_response_with_metrics


def main():
    load_dotenv()

    print("1. Gemini (Hosted LLM)")
    print("2. Ollama (Local LLM)")
    print("3. GPT (Hosted LLM)")

    choice = input("\nSelect your LLM provider (1, 2, or 3): ").strip().lower()

    if choice in ("1", "gemini"):
        provider_name = "gemini"
    elif choice in ("2", "ollama"):
        provider_name = "ollama"
    elif choice in ("3", "gpt", "openai"):
        provider_name = "gpt"
    else:
        raise ValueError("Invalid selection. Please enter 1, 2, or 3.")

    prompt = input("\nEnter your question: ")

    if provider_name == "ollama":
        result = generate_response_with_metrics(prompt)
        answer = result["response"]
        metrics = result["metrics"]
    else:
        provider = get_llm_provider(provider_name)
        client = LLMClient(provider)
        answer = client.generate_response(prompt)
        metrics = {}

    print("\n===== DSA AI TUTOR RESPONSE =====\n")
    print(answer)

    if metrics:
        print("\n===== OLLAMA METRICS =====")
        for name, value in metrics.items():
            print(f"{name}: {value}")

    logger = MongoDBLogger()

    try:
        log_id = logger.log_request(
            provider=provider_name,
            prompt=prompt,
            response=answer,
            metrics=metrics,
        )
        print(f"\nRequest saved to MongoDB. ID: {log_id}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()