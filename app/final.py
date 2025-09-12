# streamlit_app.py - Improved Healrr (full file)
import os
import re
import json
import time
import hashlib
import random
import streamlit as st
import faiss
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import cohere
from openai import OpenAI
from datetime import datetime, date
import pandas as pd

import os
import streamlit as st
from dotenv import load_dotenv

def load_keys():
    cohere_key = None
    openrouter_key = None

    # Try Streamlit secrets first
    try:
        cohere_key = st.secrets["COHERE_API_KEY"]
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        # Local fallback
        load_dotenv()
        cohere_key = os.getenv("COHERE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

    # Validation
    if not cohere_key or not openrouter_key:
        st.error("❌ API keys missing! Please check Streamlit Secrets or .env file.")
        st.stop()

    return cohere_key, openrouter_key

# Call once at top
COHERE_API_KEY, OPENROUTER_API_KEY = load_keys()


# ======================
# 🔧 CONFIG & SETUP
# ======================
st.set_page_config(
    page_title="Healrr 🌿 - Spiritual Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Healrr - Your personal spiritual assistant powered by AI"}
)

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not COHERE_API_KEY or not OPENROUTER_API_KEY:
    st.error("❌ Missing API keys in .env file")
    st.stop()

# Base directories
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
BOOKS_DIR = os.path.join(PROCESSED_DIR, "books")
QUOTES_DIR = os.path.join(PROCESSED_DIR, "quotes")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

for d in [RAW_DIR, PROCESSED_DIR, BOOKS_DIR, QUOTES_DIR, VECTOR_DIR, EXPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# API clients (cached resources)
@st.cache_resource
def get_cohere_client():
    return cohere.ClientV2(api_key=COHERE_API_KEY)

@st.cache_resource
def get_openrouter_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

co = get_cohere_client()
client = get_openrouter_client()

# ======================
# 🎨 Styling
# ======================
st.markdown("""
<style>
    .main-header {text-align:center;background:linear-gradient(90deg,#4CAF50,#8BC34A);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:3rem;font-weight:bold;}
    .subtitle {text-align:center;color:#666;font-size:1.1rem;margin-bottom:1.5rem;}
    .user-message {background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:12px 18px;border-radius:18px 18px 4px 18px;margin:8px 0;max-width:80%;margin-left:auto;box-shadow:0 2px 10px rgba(0,0,0,0.1);}
    .bot-message {background:linear-gradient(135deg,#f093fb,#f5576c);color:white;padding:12px 18px;border-radius:18px 18px 18px 4px;margin:8px 0;max-width:80%;box-shadow:0 2px 10px rgba(0,0,0,0.1);}
    .quote-card {background:linear-gradient(135deg,#ffecd2,#fcb69f);padding:16px;border-radius:12px;border-left:4px solid #ff6b6b;margin:10px 0;font-style:italic;box-shadow:0 2px 8px rgba(0,0,0,0.1);}
    .upload-area {border:2px dashed #4CAF50;border-radius:12px;padding:2rem;text-align:center;background:#f8f9fa;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

# ======================
# Helper utilities
# ======================
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def safe_write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ======================
# Text processing & chunking
# ======================
def extract_text_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'"()\-—–]', '', text)
    return text.strip()

def chunk_text(text, size=400, overlap=50):
    if not text:
        return []
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) >= 80:
            chunks.append(chunk)
    return chunks

def extract_quotes_from_text(text, min_len=20, max_len=280):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    quotes = []
    for s in sentences:
        s_clean = s.strip().strip('"“”‘’').strip()
        if not s_clean:
            continue
        if any(word in s_clean.lower() for word in ["translated", "edition", "copyright", "page", "chapter"]):
            continue
        if not (min_len <= len(s_clean) <= max_len):
            continue
        if not s_clean[0].isalpha():
            s_clean = re.sub(r'^[^A-Za-z]+', '', s_clean).strip()
            if not s_clean or not s_clean[0].isalpha():
                continue
        quotes.append(s_clean)
    return quotes[:200]

# ======================
# Embeddings + FAISS (with caching)
# ======================
def _cohere_embeddings_to_array(resp):
    # Try common shapes
    try:
        arr = np.array(resp.embeddings, dtype="float32")
        return arr
    except Exception:
        pass
    try:
        arr = np.array(resp.embeddings.float, dtype="float32")
        return arr
    except Exception:
        pass
    # fallback
    try:
        return np.array(resp, dtype="float32")
    except Exception:
        raise ValueError("Unknown embeddings format from Cohere")

def get_doc_embeddings(texts, batch_size=90):
    """Generate embeddings with safe progress bar and return np.array (n x d)."""
    if not texts:
        return None
    embeddings = []
    total = len(texts)
    # Always use float progress bar
    progress_bar = st.progress(0.0)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = co.embed(model="embed-english-v3.0", texts=batch, input_type="search_document")
            batch_arr = _cohere_embeddings_to_array(resp)
            # Normalize shape
            if batch_arr.ndim == 1:
                batch_arr = batch_arr.reshape(1, -1)
            embeddings.append(batch_arr)
            progress = (i + len(batch)) / total
            progress_bar.progress(min(1.0, float(progress)))
        except Exception as e:
            progress_bar.empty()
            st.error(f"Embedding error: {e}")
            return None
    progress_bar.empty()
    return np.vstack(embeddings).astype("float32") if embeddings else None

def get_query_embedding(query):
    try:
        resp = co.embed(model="embed-english-v3.0", texts=[query], input_type="search_query")
        arr = _cohere_embeddings_to_array(resp)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            return arr[0].astype("float32")
        return arr.astype("float32")
    except Exception as e:
        st.error(f"Query embedding error: {e}")
        return None

def build_faiss(embeds: np.ndarray, path: str):
    if embeds is None or len(embeds) == 0:
        return False
    try:
        dim = int(embeds.shape[1])
        index = faiss.IndexFlatL2(dim)
        index.add(embeds)
        faiss.write_index(index, path)
        return True
    except Exception as e:
        st.error(f"FAISS index error: {e}")
        return False

def search_faiss(path: str, q_embed, k=2):
    if not os.path.exists(path) or q_embed is None:
        return []
    try:
        index = faiss.read_index(path)
        q_vec = np.asarray(q_embed, dtype="float32").reshape(1, -1)
        if q_vec.shape[1] != index.d:
            return []
        D, I = index.search(q_vec, k)
        return I[0].tolist()
    except Exception:
        return []

# ======================
# Book & Quotes management
# ======================
def save_book_chunks(base_name, chunks, reuse_hash=True):
    # Save chunks json and embeddings + index with caching by hash
    chunks_path = os.path.join(BOOKS_DIR, f"{base_name}_chunks.json")
    safe_write_json(chunks_path, chunks)
    # Compute a content hash so we can cache embeddings
    tmp_file = os.path.join(RAW_DIR, f"{base_name}_chunks_for_hash.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))
    content_hash = file_hash(tmp_file)
    os.remove(tmp_file)

    emb_cache_path = os.path.join(VECTOR_DIR, f"{base_name}_{content_hash}.npy")
    index_path = os.path.join(VECTOR_DIR, f"{base_name}_chunks.index")

    if os.path.exists(emb_cache_path):
        try:
            embeds = np.load(emb_cache_path)
            build_faiss(embeds, index_path)
            return True
        except Exception:
            pass

    embeds = get_doc_embeddings(chunks)
    if embeds is not None:
        np.save(emb_cache_path, embeds)
        build_faiss(embeds, index_path)
        return True
    return False

def save_quotes_file(base_name, quotes):
    quotes_path = os.path.join(QUOTES_DIR, f"{base_name}_quotes.json")
    safe_write_json(quotes_path, quotes)
    # embeddings + index (cache by content)
    tmp_file = os.path.join(RAW_DIR, f"{base_name}_quotes_for_hash.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(quotes))
    content_hash = file_hash(tmp_file)
    os.remove(tmp_file)

    emb_cache_path = os.path.join(VECTOR_DIR, f"{base_name}_quotes_{content_hash}.npy")
    index_path = os.path.join(VECTOR_DIR, f"{base_name}_quotes.index")

    if os.path.exists(emb_cache_path):
        try:
            embeds = np.load(emb_cache_path)
            build_faiss(embeds, index_path)
            return True
        except Exception:
            pass
    # small batches for short quotes
    embeds = get_doc_embeddings(quotes, batch_size=64)
    if embeds is not None:
        np.save(emb_cache_path, embeds)
        build_faiss(embeds, index_path)
        return True
    return False

def get_processed_items():
    """Return list of dicts representing processed items (books and quote-files combined by base name)."""
    items = {}
    # books
    for fname in os.listdir(BOOKS_DIR):
        if fname.endswith("_chunks.json"):
            base = fname.replace("_chunks.json", "")
            items.setdefault(base, {})
            items[base]["chunks_path"] = os.path.join(BOOKS_DIR, fname)
    # quotes
    for fname in os.listdir(QUOTES_DIR):
        if fname.endswith("_quotes.json"):
            base = fname.replace("_quotes.json", "")
            items.setdefault(base, {})
            items[base]["quotes_path"] = os.path.join(QUOTES_DIR, fname)

    result = []
    for base, data in items.items():
        chunks = safe_load_json(data.get("chunks_path", "")) if data.get("chunks_path") else []
        quotes = safe_load_json(data.get("quotes_path", "")) if data.get("quotes_path") else []
        # processed_date from whichever exists
        ptime = None
        if data.get("chunks_path") and os.path.exists(data["chunks_path"]):
            ptime = os.path.getctime(data["chunks_path"])
        elif data.get("quotes_path") and os.path.exists(data["quotes_path"]):
            ptime = os.path.getctime(data["quotes_path"])
        else:
            ptime = datetime.now().timestamp()
        result.append({
            "name": base,
            "chunks": len(chunks),
            "quotes": len(quotes),
            "processed_date": datetime.fromtimestamp(ptime).strftime("%Y-%m-%d %H:%M")
        })
    # sort by processed_date newest first
    result.sort(key=lambda x: x["processed_date"], reverse=True)
    return result

def delete_processed(base):
    # delete book chunks, quotes and indexes safely
    try:
        cp = os.path.join(BOOKS_DIR, f"{base}_chunks.json")
        qp = os.path.join(QUOTES_DIR, f"{base}_quotes.json")
        if os.path.exists(cp):
            os.remove(cp)
        if os.path.exists(qp):
            os.remove(qp)
        # remove any index files with this base
        cand = [f for f in os.listdir(VECTOR_DIR) if f.startswith(base)]
        for c in cand:
            try:
                os.remove(os.path.join(VECTOR_DIR, c))
            except Exception:
                pass
        return True
    except Exception as e:
        st.error(f"Error deleting: {e}")
        return False

# ======================
# Quote of the Day rotation
# ======================
QUOTE_HISTORY_PATH = os.path.join(EXPORTS_DIR, "quote_history.json")
if not os.path.exists(QUOTE_HISTORY_PATH):
    safe_write_json(QUOTE_HISTORY_PATH, {"used": [], "last_date": ""})

def pick_quote_of_the_day():
    # collect all quotes
    all_quotes = []
    for fname in os.listdir(QUOTES_DIR):
        if fname.endswith("_quotes.json"):
            qlist = safe_load_json(os.path.join(QUOTES_DIR, fname))
            all_quotes.extend(qlist)
    if not all_quotes:
        return None
    hist = safe_load_json(QUOTE_HISTORY_PATH) or {"used": [], "last_date": ""}
    used = hist.get("used", [])
    # reset if all used
    available = [q for q in all_quotes if q not in used]
    if not available:
        used = []
        available = all_quotes.copy()
    # select a deterministic one for the day but avoid repeats
    today_idx = date.today().toordinal() % len(available)
    qotd = available[today_idx]
    # update history
    used.append(qotd)
    safe_write_json(QUOTE_HISTORY_PATH, {"used": used, "last_date": str(date.today())})
    return qotd



# ======================
# LLM call (OpenRouter) - safe wrapper
# ======================
def ask_llm(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Healrr, a wise and compassionate spiritual guide."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            return extract_text(resp)
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"LLM connection error: {e}")
                return "Sorry, I'm having trouble connecting right now. Please try again later."
            time.sleep(2 ** attempt)

# ======================
# UI: Header & session init
# ======================
st.markdown('<h1 class="main-header">🌿 Healrr Spiritual Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your personal guide for wisdom, reflection & mindfulness</p>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    # try to load an export (optional)
    try:
        last_export = os.path.join(EXPORTS_DIR, "last_chat.json")
        if os.path.exists(last_export):
            st.session_state["chat_history"] = safe_load_json(last_export)
        else:
            st.session_state["chat_history"] = []
    except Exception:
        st.session_state["chat_history"] = []

if "meditation_sessions" not in st.session_state:
    st.session_state["meditation_sessions"] = []

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "💬 Chat"

# ======================
# Sidebar navigation & stats
# ======================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    pages = ["💬 Chat", "📚 Library", "📜 Quotes", "🧘 Meditation", "⚙️ Settings"]
    selected_page = st.radio("Go to:", pages, key="nav_radio")
    st.session_state["selected_page"] = selected_page

    st.markdown("---")
    processed = get_processed_items()
    if processed:
        st.markdown("### 📊 Quick Stats")
        total_chunks = sum(item["chunks"] for item in processed)
        total_quotes = sum(item["quotes"] for item in processed)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Books", len(processed))
            st.metric("Chunks", total_chunks)
        with col2:
            st.metric("Quotes", total_quotes)
            st.metric("Chats", len(st.session_state["chat_history"]))
    st.markdown("---")
    show_debug = st.checkbox("🔎 Debug Mode", value=False)
    auto_quotes = st.checkbox("✨ Auto-show quotes in chat", value=True)

# ======================
# Pages
# ======================

# ----- Chat page -----
if st.session_state["selected_page"] == "💬 Chat":
    st.markdown("### 💬 Chat with Healrr")
    # show history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>🌿 Healrr:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    user_query = st.chat_input("Ask me anything about spirituality, wisdom, or life guidance...")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.spinner("🌿 Healrr is thinking..."):
            q_embed = get_query_embedding(user_query)
            all_contexts = []
            all_quotes = []

            if q_embed is not None:
                processed = get_processed_items()
                for item in processed:
                    base = item["name"]
                    # safe load chunks and quotes
                    chunks = []
                    quotes = []
                    chunks_path = os.path.join(BOOKS_DIR, f"{base}_chunks.json")
                    quotes_path = os.path.join(QUOTES_DIR, f"{base}_quotes.json")
                    if os.path.exists(chunks_path):
                        chunks = safe_load_json(chunks_path)
                    if os.path.exists(quotes_path):
                        quotes = safe_load_json(quotes_path)

                    # search chunks
                    c_index_path = os.path.join(VECTOR_DIR, f"{base}_chunks.index")
                    if os.path.exists(c_index_path) and chunks:
                        I = search_faiss(c_index_path, q_embed, k=4)
                        # add top matched chunks
                        if I:
                            for idx in I:
                                if 0 <= idx < len(chunks):
                                    all_contexts.append(chunks[idx])
                    # search quotes for sprinkle
                    if auto_quotes:
                        q_index_path = os.path.join(VECTOR_DIR, f"{base}_quotes.index")
                        if os.path.exists(q_index_path) and quotes:
                            Iq = search_faiss(q_index_path, q_embed, k=1)
                            if Iq and 0 <= Iq[0] < len(quotes):
                                all_quotes.append(quotes[Iq[0]])

            # debug
            if show_debug:
                with st.expander("🔎 Debug Information"):
                    st.write(f"Contexts found: {len(all_contexts)}")
                    st.write(f"Quotes found: {len(all_quotes)}")
                    if all_contexts:
                        for i, c in enumerate(all_contexts[:3]):
                            st.write(f"{i+1}. {c[:300]}...")

            # choose top contexts (dedupe & limit)
            seen = set()
            chosen_contexts = []
            for c in all_contexts:
                key = c[:200]
                if key in seen: 
                    continue
                seen.add(key)
                chosen_contexts.append(c)
                if len(chosen_contexts) >= 3:
                    break

            context = "\n\n".join(chosen_contexts)
            prompt = f"""Context from spiritual books:
{context}

User question: {user_query}

Please provide a thoughtful, spiritual response as Healrr, a wise and compassionate spiritual guide. Draw from the context when relevant, but also provide your own wisdom. Be encouraging, insightful, and practical."""

            answer = ask_llm(prompt)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            # add quote if found
            if all_quotes and auto_quotes:
                quote_msg = f"📜 *\"{all_quotes[0]}\"*"
                st.session_state["chat_history"].append({"role": "assistant", "content": quote_msg})

        # save last chat export for persistence (optional)
        try:
            safe_write_json(os.path.join(EXPORTS_DIR, "last_chat.json"), st.session_state["chat_history"])
        except Exception:
            pass
        st.rerun()

    # chat actions
    if st.session_state["chat_history"]:
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()
        with col2:
            if st.button("📥 Export Chat"):
                chat_text = "\n\n".join([f"{m['role'].title()}: {m['content']}" for m in st.session_state["chat_history"]])
                st.download_button("💾 Download", data=chat_text, file_name=f"healrr_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")

# ----- Library page -----
elif st.session_state["selected_page"] == "📚 Library":
    st.markdown("### 📚 Spiritual Library")

    # Books upload
    with st.expander("📤 Upload Book / Transcript / Blog"):
        uploaded_book = st.file_uploader("Upload a PDF or TXT file", type=["pdf","txt"], key="upload_book")
        if uploaded_book:
            with st.spinner("Processing book..."):
                file_path = os.path.join(RAW_DIR, uploaded_book.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_book.read())
                raw_text = extract_text_pdf(file_path) if uploaded_book.name.lower().endswith(".pdf") else open(file_path, encoding='utf-8').read()
                cleaned = clean_text(raw_text)
                chunks = chunk_text(cleaned)
                base = os.path.splitext(uploaded_book.name)[0]
                saved = save_book_chunks(base, chunks)
                if saved:
                    st.success(f"✅ {base} uploaded as Book/Text")
                else:
                    st.error("❌ Failed to process book (see errors above).")
                #st.rerun()

    # Quotes upload
    with st.expander("📜 Upload Quotes File"):
        uploaded_quotes = st.file_uploader("Upload a TXT file (one quote per line)", type=["txt"], key="upload_quotes")
        if uploaded_quotes:
            with st.spinner("Processing quotes..."):
                file_path = os.path.join(RAW_DIR, uploaded_quotes.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_quotes.read())
                lines = open(file_path, encoding='utf-8').read().splitlines()
                quotes = [ln.strip().strip('"“”‘’') for ln in lines if len(ln.strip()) > 5]
                base = os.path.splitext(uploaded_quotes.name)[0]
                saved = save_quotes_file(base, quotes)
                if saved:
                    st.success(f"✅ {base} uploaded as Quotes")
                else:
                    st.error("❌ Failed to process quotes.")
                #st.rerun()

    st.markdown("---")
    st.markdown("### 📘 Uploaded Books / Texts")
    book_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith("_chunks.json")]
    if book_files:
        for bf in book_files:
            base = bf.replace("_chunks.json","")
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(f"📖 {base}")
            with col2:
                if st.button("🗑️ Delete", key=f"del_book_{base}"):
                    ok = delete_processed(base)
                    if ok:
                        st.success(f"✅ Deleted {base}")
                        st.rerun()
                    else:
                        st.error("Error deleting.")
    else:
        st.info("No books uploaded yet.")

    st.markdown("### 📜 Uploaded Quotes Files")
    quote_files = [f for f in os.listdir(QUOTES_DIR) if f.endswith("_quotes.json")]
    if quote_files:
        for qf in quote_files:
            base = qf.replace("_quotes.json","")
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(f"💬 {base}")
            with col2:
                if st.button("🗑️ Delete", key=f"del_quote_{base}"):
                    ok = delete_processed(base)
                    if ok:
                        st.success(f"✅ Deleted {base}")
                        st.rerun()
                    else:
                        st.error("Error deleting.")
    else:
        st.info("No quotes uploaded yet.")

# ----- Quotes page -----
elif st.session_state["selected_page"] == "📜 Quotes":
    st.markdown("### 📜 Quote of the Day")
    q = pick_quote_of_the_day()
    if q:
        st.markdown(f'<div class="quote-card"><h4>🌿 Quote of the Day</h4><p>"{q}"</p></div>', unsafe_allow_html=True)
    else:
        st.info("Upload quotes to see Quote of the Day.")

# ----- Meditation -----
elif st.session_state["selected_page"] == "🧘 Meditation":
    st.markdown("### 🧘 Mindful Meditation")
    st.markdown("""
    <div class="meditation-card">
        <h3>🕯️ Take a Moment to Center Yourself</h3>
        <p>Find a quiet space, close your eyes, and focus on your breath. Let your thoughts come and go like clouds in the sky.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        duration_5 = st.button("🧘 5 Minutes", use_container_width=True)
    with col2:
        duration_10 = st.button("🧘‍♀️ 10 Minutes", use_container_width=True)
    with col3:
        duration_15 = st.button("🧘‍♂️ 15 Minutes", use_container_width=True)

    custom_duration = st.slider("Or choose custom duration (minutes):", 1, 60, 5)
    custom_start = st.button("🚀 Start Custom Timer", use_container_width=True)

    selected_duration = None
    if duration_5:
        selected_duration = 5
    elif duration_10:
        selected_duration = 10
    elif duration_15:
        selected_duration = 15
    elif custom_start:
        selected_duration = custom_duration

    if selected_duration:
        st.markdown(f"### 🎯 {selected_duration}-Minute Meditation Session")
        total_seconds = selected_duration * 60
        progress_bar = st.progress(0.0)
        time_display = st.empty()
        status_display = st.empty()
        start_time = time.time()
        for elapsed in range(total_seconds):
            remaining = total_seconds - elapsed
            mins, secs = divmod(remaining, 60)
            time_display.markdown(f'<div class="timer-display">{mins:02d}:{secs:02d}</div>', unsafe_allow_html=True)
            progress_bar.progress(elapsed/total_seconds)
            if elapsed == 0:
                status_display.info("🌱 Begin by taking three deep breaths...")
            elif elapsed == total_seconds // 2:
                status_display.info("🕯️ You're halfway there. Stay present...")
            time.sleep(1)
        time_display.markdown('<div class="timer-display">✅ Complete!</div>', unsafe_allow_html=True)
        progress_bar.progress(1.0)
        status_display.success("🙏 **Meditation complete!**")
        st.session_state["meditation_sessions"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "duration": selected_duration})
        st.balloons()

    if st.session_state["meditation_sessions"]:
        st.markdown("#### 📊 Your Meditation Journey")
        df = pd.DataFrame(st.session_state["meditation_sessions"])
        total_time = df["duration"].sum()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Sessions", len(df))
        with c2:
            st.metric("Total Minutes", total_time)
        with c3:
            st.metric("Average Session", f"{total_time/len(df):.1f} min")
        with st.expander("📅 Recent Sessions"):
            for s in reversed(st.session_state["meditation_sessions"][-10:]):
                st.write(f"🧘 {s['date']} - {s['duration']} minutes")

# ----- Settings -----
elif st.session_state["selected_page"] == "⚙️ Settings":
    st.markdown("### ⚙️ Settings & Preferences")
    col1, col2 = st.columns(2)
    with col1:
        model_sel = st.selectbox("🤖 AI Model", ["gpt-4o-mini", "gpt-4", "gpt-5-mini"], index=0)
        resp_len = st.slider("🎯 Response Length", 100, 2000, 800, 100)
    with col2:
        search_k = st.selectbox("🔍 Search Results", [2,3,4,5], index=1)
        chunk_size = st.slider("📊 Chunk Size", 300, 800, 400, 50)

    st.markdown("#### 💾 Data Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Clear All Chat History"):
            st.session_state["chat_history"] = []
            safe_write_json(os.path.join(EXPORTS_DIR, "last_chat.json"), st.session_state["chat_history"])
            st.success("✅ Chat history cleared!")
    with c2:
        if st.button("🧘 Reset Meditation History"):
            st.session_state["meditation_sessions"] = []
            st.success("✅ Meditation history reset!")
    with c3:
        if st.button("🔄 Reset All Processed Data"):
            # remove processed files
            for f in os.listdir(BOOKS_DIR):
                os.remove(os.path.join(BOOKS_DIR, f))
            for f in os.listdir(QUOTES_DIR):
                os.remove(os.path.join(QUOTES_DIR, f))
            for f in os.listdir(VECTOR_DIR):
                os.remove(os.path.join(VECTOR_DIR, f))
            st.success("✅ All processed data cleared!")
            st.rerun()

    st.markdown("#### 📤 Export Your Data")
    if st.button("📦 Export Everything"):
        export_data = {
            "chat_history": st.session_state["chat_history"],
            "meditation_sessions": st.session_state["meditation_sessions"],
            "processed": get_processed_items(),
            "export_date": datetime.now().isoformat()
        }
        safe_write_json(os.path.join(EXPORTS_DIR, f"healrr_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json"), export_data)
        st.success("✅ Export saved to exports folder")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;padding:20px;'><p>🌿 <strong>Healrr</strong> - Your AI-powered spiritual companion</p><p><em>May your journey be filled with wisdom, peace, and growth</em> 🙏</p></div>", unsafe_allow_html=True)
