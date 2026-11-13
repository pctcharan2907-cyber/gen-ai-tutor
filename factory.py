import os
from Gemini_provider import GeminiProvider
from ollama_provider import ollamaprovider
from gpt_provider import GPTProvider


def get_llm_provider(provider_name: str):
    """Factory function to instantiate and return the requested LLM provider."""
    provider_name = provider_name.strip().lower()

    if provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
        return GeminiProvider(api_key)

    elif provider_name == "ollama":
        return ollamaprovider()

    elif provider_name in ("gpt", "openai"):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
        return GPTProvider(api_key)

    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'. Supported options: gemini, ollama, gpt")
