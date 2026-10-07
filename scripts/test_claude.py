import os

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
model = os.getenv("ANTHROPIC_MODEL")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY was not found in .env")

if not model:
    raise ValueError("ANTHROPIC_MODEL was not found in .env")


client = Anthropic(api_key=api_key)

response = client.messages.create(
    model=model,
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Claude connection successful",
        }
    ],
)

print(f"Model: {model}")
print(f"Response: {response.content[0].text}")