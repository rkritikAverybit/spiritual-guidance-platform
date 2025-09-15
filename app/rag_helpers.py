import os
import pickle
import faiss
import numpy as np
import cohere
import config  # for COHERE_API_KEY

SAVE_PATH = "data/vector_store/index.pkl"

# Cohere client
co = cohere.ClientV2(api_key=config.COHERE_API_KEY)

# ---------------------------
# Embedding Functions
# ---------------------------
def embed_texts(texts):
    """Generate embeddings for multiple texts (documents)."""
    inputs = [{"content": [{"type": "text", "text": t}]} for t in texts]
    resp = co.embed(
        inputs=inputs,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
    )
    return np.array(resp.embeddings.float, dtype="float32")

def embed_query(query: str):
    """Generate embedding for a query string."""
    inputs = [{"content": [{"type": "text", "text": query}]}]
    resp = co.embed(
        inputs=inputs,
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    return np.array(resp.embeddings.float, dtype="float32")

# ---------------------------
# Search FAISS Index
# ---------------------------
def search_index(query: str, top_k: int = 3):
    """Search FAISS index for relevant chunks."""
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError("No FAISS index found. Please ingest content first.")

    with open(SAVE_PATH, "rb") as f:
        index, chunks = pickle.load(f)

    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results
