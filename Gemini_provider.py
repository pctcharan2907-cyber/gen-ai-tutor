import os
from google import genai
from google.genai import types
from base import LLMProvider


class GeminiProvider(LLMProvider):
    """Gemini API LLM Provider using the official google-genai SDK."""

    def __init__(self, api_key: str = None, model: str = "gemini-3.1-flash-lite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Please set GEMINI_API_KEY in your .env file or pass it to GeminiProvider."
            )
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, prompt: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=(
                "You are an expert DSA (Data Structures and Algorithms) tutor. "
                "Explain concepts clearly with code examples and time/space complexity analysis."
            )
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""
