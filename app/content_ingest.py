import os
import re
import pickle
import faiss
import numpy as np
import cohere
import pandas as pd
from PyPDF2 import PdfReader

import supabase_helpers as db
from rag_helpers import embed_texts   # ✅ only query-side embeddings import

import config  # for COHERE_API_KEY

SAVE_PATH = "data/vector_store/index.pkl"

# Cohere client
co = cohere.ClientV2(api_key=config.COHERE_API_KEY)

# ---------------------------
# Helpers
# ---------------------------
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def save_faiss_index(embeddings, chunks, save_path=SAVE_PATH):
    """Save embeddings + chunks into FAISS index."""
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            index, stored_chunks = pickle.load(f)
        index.add(embeddings)
        stored_chunks.extend(chunks)
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        stored_chunks = chunks

    with open(save_path, "wb") as f:
        pickle.dump((index, stored_chunks), f)

# ---------------------------
# Ingestion Functions
# ---------------------------
def ingest_pdf(file_path: str, save_path: str = SAVE_PATH):
    reader = PdfReader(file_path)
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    text = clean_text(text)
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    save_faiss_index(embeddings, chunks, save_path)

def ingest_txt(file_path: str, save_path: str = SAVE_PATH):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = clean_text(text)
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    save_faiss_index(embeddings, chunks, save_path)

def ingest_substack_csv(file_path: str, save_path: str = SAVE_PATH):
    df = pd.read_csv(file_path)
    text = " ".join(df.astype(str).fillna("").values.flatten())
    text = clean_text(text)
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    save_faiss_index(embeddings, chunks, save_path)

def ingest_quotes_file(file_path: str) -> int:
    """Ingest quotes from TXT or CSV into Supabase (not FAISS yet)."""
    quotes = []

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    quotes.append({"content": line, "reference": None})

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            text = str(row.get("content", "")).strip()
            ref = str(row.get("reference", "")).strip()
            if text:
                quotes.append({"content": text, "reference": ref})

    if quotes:
        db.supabase.table("quotes").insert(quotes).execute()
    return len(quotes)

def ingest_quotes_to_index(save_path: str = SAVE_PATH) -> int:
    """Fetch all quotes from Supabase and add them to FAISS index."""
    resp = db.supabase.table("quotes").select("id, content, reference").execute()
    quotes = resp.data

    if not quotes:
        return 0

    chunks = []
    for q in quotes:
        text = q["content"]
        ref = q.get("reference") or ""
        chunks.append(f"{text} — {ref}".strip())

    embeddings = embed_texts(chunks)
    save_faiss_index(embeddings, chunks, save_path)

    return len(chunks)
