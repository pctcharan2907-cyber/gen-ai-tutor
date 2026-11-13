from base import LLMProvider


class LLMClient:
    """Client class to interact with any configured LLM provider."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate_response(self, prompt: str) -> str:
        return self.provider.generate_response(prompt)
