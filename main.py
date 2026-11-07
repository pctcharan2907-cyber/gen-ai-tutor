import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables from this project's .env file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Read and check the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env file")

# Create the Gemini client
client = genai.Client(api_key=api_key)

print("\n===== DSA AI TUTOR =====")

# Ask the student for a question
student_question = input("\nWhat would you like to learn today? ")

# Send the question to Gemini
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=(
        "You are a friendly Data Structures and Algorithms tutor. "
        "Explain concepts in simple language with examples and analogies.\n\n"
        f"Student question: {student_question}"
    )
)

# Show Gemini's answer
print("\n===== DSA AI TUTOR RESPONSE =====\n")
print(response.text)

# Show token usage, if available
print("\n===== API USAGE INFORMATION =====")
usage = response.usage_metadata

if usage:
    print("Input tokens:", usage.prompt_token_count)
    print("Output tokens:", usage.candidates_token_count)
    print("Total tokens:", usage.total_token_count)
else:
    print("Usage information not available.")