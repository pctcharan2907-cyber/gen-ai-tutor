from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract Base Class for all LLM providers."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response given a prompt.

        Args:
            prompt (str): The user's input/question.

        Returns:
            str: The LLM's response.
        """
        pass


# Backward compatibility alias
LlmProvider = LLMProvider
