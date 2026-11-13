from dotenv import load_dotenv
from client import LLMClient
from factory import get_llm_provider

load_dotenv()

print("1. Gemini (Hosted LLM)")
print("2. ollama (Local LLM)")
print("3. GPT (Hosted LLM)")

choice = input("\nSelect your LLM provider (1, 2, or 3): ").strip()
if choice == "1" or choice.lower() == "gemini":
    provider_name = "gemini"
elif choice == "2" or choice.lower() == "ollama":
    provider_name = "ollama"
elif choice == "3" or choice.lower() in ("gpt", "openai"):
    provider_name = "gpt"
else:
    raise ValueError("Invalid selection. Please enter 1, 2, or 3.")

provider = get_llm_provider(provider_name)

client = LLMClient(provider)

prompt = input("\nEnter your question: ")

print("\n===== DSA AI TUTOR RESPONSE =====\n")
answer = client.generate_response(prompt)
print(answer)
