from openai import OpenAI
import cohere

OPENROUTER_API_KEY = "tumhara_openrouter_key"
COHERE_API_KEY = "tumhara_cohere_key"

# Test OpenRouter
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from OpenRouter!"}]
)
print("OpenRouter:", resp.choices[0].message.content)

# Test Cohere
co = cohere.Client(COHERE_API_KEY)
resp = co.embed(model="embed-english-v3.0", texts=["Hello"], input_type="search_query")
print("Cohere embedding len:", len(resp.embeddings[0]))
