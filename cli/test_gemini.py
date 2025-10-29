import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
contents = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

response = client.models.generate_content(model="gemini-2.0-flash-001", contents=contents)
print(response.text)

if response.usage_metadata:
    print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")

