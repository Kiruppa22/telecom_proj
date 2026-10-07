import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found")

if not model:
    raise ValueError("ANTHROPIC_MODEL not found")


llm = Anthropic(
    model=model,
    api_key=api_key,
    max_tokens=300,
)

print(f"Model: {model}")
print("=" * 80)

response = llm.complete(
    "Reply with exactly: LlamaIndex Claude connection successful"
)

print("RAW RESPONSE:")
print(repr(response))

print("\nTEXT:")
print(response.text)