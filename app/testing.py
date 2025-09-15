import os
import cohere
from openai import OpenAI
from dotenv import load_dotenv

# Load from .env (if running locally)
load_dotenv()

# Keys from environment
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print("Cohere Key:", COHERE_API_KEY[:6] + "..." if COHERE_API_KEY else "❌ MISSING")
print("OpenRouter Key:", OPENROUTER_API_KEY[:6] + "..." if OPENROUTER_API_KEY else "❌ MISSING")

# ---- Test Cohere ----
try:
    co = cohere.Client(COHERE_API_KEY)
    resp = co.embed(
        model="embed-english-v3.0",
        texts=["Hello from Cohere!"],
        input_type="search_document"
    )
    print("✅ Cohere test passed. Embedding size:", len(resp.embeddings[0]))
except Exception as e:
    print("❌ Cohere test failed:", e)

# ---- Test OpenRouter ----
# ---- Test OpenRouter ----
try:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello from OpenRouter!"}],
        max_tokens=20
    )
    print("✅ OpenRouter test passed. Response:", resp.choices[0].message.content)
except Exception as e:
    print("❌ OpenRouter test failed:", e)

