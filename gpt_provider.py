import os
from openai import OpenAI
from base import LLMProvider


class GPTProvider(LLMProvider):
    """OpenAI GPT LLM Provider with automatic fallback if no API key is configured."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")
        self.model = model
        self.client = None

        if (
            self.api_key
            and self.api_key.strip() not in ("", "your_openai_api_key_here")
            and not self.api_key.startswith("your_")
        ):
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate_response(self, prompt: str) -> str:
        # If OpenAI client is configured with a key, use OpenAI
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert DSA (Data Structures and Algorithms) tutor. "
                                "Explain concepts clearly with code examples and time/space complexity analysis."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                # If OpenAI API call fails (e.g. invalid key or network issue), use fallback
                error_msg = str(e)
                fallback_resp = self._fallback_response(prompt)
                if fallback_resp:
                    return fallback_resp
                return f"Error communicating with OpenAI/GPT: {error_msg}"

        # If no OpenAI key is configured in .env, seamlessly use the available DSA tutor engine
        fallback_resp = self._fallback_response(prompt)
        if fallback_resp:
            return fallback_resp

        return (
            "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file "
            "to use official OpenAI GPT models."
        )

    def _fallback_response(self, prompt: str) -> str:
        """Fallback to Gemini or Ollama if OpenAI key is not configured."""
        try:
            from Gemini_provider import GeminiProvider

            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
            if gemini_key:
                gp = GeminiProvider(api_key=gemini_key)
                return gp.generate_response(prompt)
        except Exception:
            pass

        try:
            from ollama_provider import OllamaProvider

            op = OllamaProvider()
            return op.generate_response(prompt)
        except Exception:
            pass

        return ""


# Backward compatibility aliases
OpenAIProvider = GPTProvider
gpt_provider = GPTProvider
