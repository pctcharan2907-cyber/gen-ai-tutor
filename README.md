# DSA AI Tutor (Multi-Provider LLM Client)

An extensible, modular Data Structures & Algorithms (DSA) AI Tutor built in Python using Object-Oriented Design (Factory and Strategy patterns). It supports **Google Gemini**, **Ollama (Local LLM)**, and **OpenAI GPT**.

---

## 📁 Project Architecture & File Structure

```text
gen-ai-tutor/
│
├── base.py                 # Abstract Base Class (LLMProvider)
├── Gemini_provider.py       # Google Gemini provider (using google-genai)
├── ollama_provider.py       # Local Ollama provider (e.g. llama3.2)
├── gpt_provider.py          # OpenAI GPT provider (e.g. gpt-4o-mini)
├── client.py                # LLMClient consuming any LLMProvider
├── factory.py               # Factory returning provider instance by name
├── main.py                  # CLI entry point with interactive 3-choice menu
├── .env                     # Environment variables (API keys)
├── .gitignore               # Protects .env and virtual environment
└── requirements.txt         # Required Python dependencies
```

---

## 🚀 Quick Start

### 1. Configure `.env`
Ensure your `.env` contains:
```env
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (optional, required only if using choice 3)
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Run the Tutor CLI
From the root directory:
```bash
python main.py
```

You will see:
```text
1. Gemini (Hosted LLM)
2. ollama (Local LLM)
3. GPT (Hosted LLM)

Select your LLM provider (1, 2, or 3): 
```

### 3. Ask your DSA Questions
Choose `1`, `2`, or `3`, then type your question (e.g., `What is the difference between BFS and DFS?`). The tutor will generate an intuitive explanation with code examples and time/space complexity analysis!

---

## 🛠️ Design Patterns Used
- **Strategy Pattern (`base.py`, `client.py`)**: `LLMClient` accepts any implementation of `LLMProvider` and delegates response generation polymorphically.
- **Factory Pattern (`factory.py`)**: `get_llm_provider(provider_name)` abstracts provider instantiation and environment variable resolution.
