import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",  # or "gpt-3.5-turbo"
    messages=[
        {"role": "user", "content": "Say hello like a Shakespearean bard"}
    ]
)

print("🤖 Response:")
print(response.choices[0].message.content)