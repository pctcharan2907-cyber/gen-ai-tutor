import ollama
from base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama LLM Provider."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        dsa_prompt = (
            "You are an expert DSA (Data Structures and Algorithms) tutor. "
            f"Explain clearly with intuition, code examples, and complexity: {prompt}"
        )
        response = ollama.generate(
            model=self.model,
            prompt=dsa_prompt,
        )

        if hasattr(response, "response"):
            return response.response
        elif isinstance(response, dict) and "response" in response:
            return response["response"]
        return str(response)


# Backward compatibility alias for the class name used in user code
ollamaprovider = OllamaProvider
