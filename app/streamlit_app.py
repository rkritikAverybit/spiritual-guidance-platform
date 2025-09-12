import os
import re
import json
import time
import asyncio
import streamlit as st
import faiss
import numpy as np
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import cohere
from openai import OpenAI
from datetime import datetime, timedelta
import pandas as pd

# ======================
# 🔧 CONFIG & SETUP
# ======================
st.set_page_config(
    page_title="Healrr 🌿 - Spiritual Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Healrr - Your personal spiritual assistant powered by AI"
    }
)

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not COHERE_API_KEY or not OPENROUTER_API_KEY:
    st.error("❌ Missing API keys in .env file")
    st.stop()

# Directory setup
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

for dir_path in [RAW_DIR, PROCESSED_DIR, VECTOR_DIR, EXPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# API clients
co = cohere.ClientV2(api_key=COHERE_API_KEY)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ======================
# 🎨 CUSTOM CSS & STYLING
# ======================
st.markdown("""
<style>
    /* Main theme */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Chat bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .quote-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 4px solid #ff6b6b;
        margin: 10px 0;
        font-style: italic;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
    
    /* Meditation timer */
    .meditation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .timer-display {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 🔧 ENHANCED HELPER FUNCTIONS
# ======================
@st.cache_data
def extract_text_pdf(path):
    """Extract text from PDF with caching"""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def clean_text(text):
    """Enhanced text cleaning"""
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    return text.strip()

def chunk_text(text, size=500, overlap=50):
    """Enhanced chunking with overlap"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.strip()) > 100:  # Only add substantial chunks
            chunks.append(chunk)
    return chunks

def extract_quotes(text):
    """Enhanced quote extraction"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    quotes = []
    
    for s in sentences:
        s = s.strip()
        # Better filtering criteria
        if (40 <= len(s) <= 280 and 
            not any(word in s.lower() for word in ["translated", "edition", "copyright", "page", "chapter"]) and
            s[0].isupper() and
            any(char in s for char in '.!?')):
            quotes.append(s)
    
    return quotes[:200]

@st.cache_data
def get_doc_embeddings(texts, batch_size=90):
    """Cached embedding generation"""
    embeddings = []
    progress_bar = st.progress(0)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = co.embed(
                model="embed-english-v3.0",
                texts=batch,
                input_type="search_document"
            )
            embeddings.extend(resp.embeddings.float)
            progress_bar.progress((i + batch_size) / len(texts))
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            break
    
    progress_bar.empty()
    return np.array(embeddings).astype("float32") if embeddings else None

def get_query_embedding(query):
    """Get query embedding with error handling"""
    try:
        resp = co.embed(model="embed-english-v3.0", texts=[query], input_type="search_query")
        return np.array(resp.embeddings.float[0]).astype("float32")
    except Exception as e:
        st.error(f"Query embedding error: {str(e)}")
        return None

def build_faiss(embeds, path):
    """Enhanced FAISS index building"""
    if embeds is None or len(embeds) == 0:
        return False
    
    try:
        dim = embeds.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeds)
        faiss.write_index(index, path)
        return True
    except Exception as e:
        st.error(f"FAISS index error: {str(e)}")
        return False

def search_faiss(path, q_embed, k=2):
    """Enhanced FAISS search"""
    if not os.path.exists(path) or q_embed is None:
        return []
    
    try:
        index = faiss.read_index(path)
        if q_embed.shape[0] != index.d:
            return []
        D, I = index.search(np.array([q_embed]), k)
        return I[0]
    except Exception:
        return []

def ask_llm(prompt, max_retries=3):
    """Enhanced LLM call with retries"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Sorry, I'm having trouble connecting right now. Please try again. Error: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff

# ======================
# 📚 BOOK MANAGEMENT FUNCTIONS
# ======================
def get_processed_books():
    """Get list of processed books"""
    books = []
    for fname in os.listdir(PROCESSED_DIR):
        if fname.endswith("_chunks.json"):
            base = fname.replace("_chunks.json", "")
            
            chunks_path = os.path.join(PROCESSED_DIR, f"{base}_chunks.json")
            quotes_path = os.path.join(PROCESSED_DIR, f"{base}_quotes.json")
            
            chunks_count = len(json.load(open(chunks_path))) if os.path.exists(chunks_path) else 0
            quotes_count = len(json.load(open(quotes_path))) if os.path.exists(quotes_path) else 0
            
            books.append({
                "name": base,
                "chunks": chunks_count,
                "quotes": quotes_count,
                "processed_date": datetime.fromtimestamp(os.path.getctime(chunks_path)).strftime("%Y-%m-%d %H:%M")
            })
    
    return books

def delete_book(book_name):
    """Delete a processed book"""
    files_to_delete = [
        os.path.join(PROCESSED_DIR, f"{book_name}_chunks.json"),
        os.path.join(PROCESSED_DIR, f"{book_name}_quotes.json"),
        os.path.join(VECTOR_DIR, f"{book_name}_chunks.index"),
        os.path.join(VECTOR_DIR, f"{book_name}_quotes.index")
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)

# ======================
# 🎯 MAIN APP LAYOUT
# ======================
st.markdown('<h1 class="main-header">🌿 Healrr Spiritual Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your personal guide for wisdom, reflection & mindfulness</p>', unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "meditation_sessions" not in st.session_state:
    st.session_state["meditation_sessions"] = []
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "💬 Chat"

# ======================
# 🔄 SIDEBAR NAVIGATION
# ======================
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    pages = ["💬 Chat", "📚 Library", "📜 Quotes", "🧘 Meditation", "⚙️ Settings"]
    selected_page = st.radio("Go to:", pages, key="nav_radio")
    st.session_state["selected_page"] = selected_page
    
    st.markdown("---")
    
    # Quick stats
    books = get_processed_books()
    if books:
        st.markdown("### 📊 Quick Stats")
        total_chunks = sum(book["chunks"] for book in books)
        total_quotes = sum(book["quotes"] for book in books)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Books", len(books))
            st.metric("Chunks", total_chunks)
        with col2:
            st.metric("Quotes", total_quotes)
            st.metric("Chats", len(st.session_state["chat_history"]))
    
    st.markdown("---")
    
    # Theme toggle
    st.markdown("### 🎨 Preferences")
    show_debug = st.checkbox("🔎 Debug Mode", value=False)
    auto_quotes = st.checkbox("✨ Auto-show quotes", value=True)

# ======================
# 📄 PAGE ROUTING
# ======================

# 💬 CHAT PAGE
if st.session_state["selected_page"] == "💬 Chat":
    st.markdown("### 💬 Chat with Healrr")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history with enhanced styling
        for i, msg in enumerate(st.session_state["chat_history"]):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                    <div class="user-message">
                        <strong>You:</strong> {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div class="bot-message">
                        <strong>🌿 Healrr:</strong> {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_query = st.chat_input("Ask me anything about spirituality, wisdom, or life guidance...")
    
    if user_query:
        # Add user message
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        
        with st.spinner("🌿 Healrr is thinking..."):
            # Get embeddings and search
            all_contexts, all_quotes = [], []
            q_embed = get_query_embedding(user_query)
            
            if q_embed is not None:
                for book in books:
                    base = book["name"]
                    
                    # Load data
                    chunks = json.load(open(os.path.join(PROCESSED_DIR, f"{base}_chunks.json")))
                    quotes = json.load(open(os.path.join(PROCESSED_DIR, f"{base}_quotes.json")))
                    
                    # Search chunks
                    c_index_path = os.path.join(VECTOR_DIR, f"{base}_chunks.index")
                    if os.path.exists(c_index_path):
                        I = search_faiss(c_index_path, q_embed, k=3)
                        all_contexts.extend([chunks[i] for i in I if 0 <= i < len(chunks)])
                    
                    # Search quotes
                    if auto_quotes:
                        q_index_path = os.path.join(VECTOR_DIR, f"{base}_quotes.index")
                        if os.path.exists(q_index_path) and quotes:
                            Iq = search_faiss(q_index_path, q_embed, k=1)
                            if Iq is not None and len(Iq) > 0 and 0 <= Iq[0] < len(quotes):
                                all_quotes.append(quotes[Iq[0]])
            
            # Debug info
            if show_debug:
                with st.expander("🔎 Debug Information"):
                    st.write(f"**Contexts found:** {len(all_contexts)}")
                    st.write(f"**Quotes found:** {len(all_quotes)}")
                    if all_contexts:
                        st.write("**Top contexts:**")
                        for i, context in enumerate(all_contexts[:2]):
                            st.write(f"{i+1}. {context[:200]}...")
            
            # Generate response
            context = "\n\n".join(all_contexts[:3])  # Use top 3 contexts
            prompt = f"""Context from spiritual books:
{context}

User question: {user_query}

Please provide a thoughtful, spiritual response as Healrr, a wise and compassionate spiritual guide. Draw from the context when relevant, but also provide your own wisdom. Be encouraging, insightful, and practical."""

            answer = ask_llm(prompt)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            
            # Add quote if found
            if all_quotes and auto_quotes:
                quote_msg = f"📜 *\"{all_quotes[0]}\"*"
                st.session_state["chat_history"].append({"role": "assistant", "content": quote_msg})
        
        st.rerun()
    
    # Chat actions
    if st.session_state["chat_history"]:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state["chat_history"] = []
                st.rerun()
        with col2:
            if st.button("📥 Export Chat"):
                chat_text = "\n\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state["chat_history"]])
                st.download_button(
                    "💾 Download",
                    data=chat_text,
                    file_name=f"healrr_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

# 📚 LIBRARY PAGE
elif st.session_state["selected_page"] == "📚 Library":
    st.markdown("### 📚 Spiritual Library")
    
    # Upload section
    with st.expander("📤 Upload New Book", expanded=not books):
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "📄 Upload a spiritual text (PDF or TXT)", 
            type=["pdf", "txt"],
            help="Upload spiritual books, texts, or documents to enhance Healrr's knowledge"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded:
            with st.spinner("🔄 Processing your book..."):
                # Save file
                file_path = os.path.join(RAW_DIR, uploaded.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded.read())
                
                # Extract and process text
                raw_text = extract_text_pdf(file_path) if uploaded.name.endswith(".pdf") else open(file_path, encoding='utf-8').read()
                text = clean_text(raw_text)
                chunks = chunk_text(text)
                quotes = extract_quotes(text)
                
                if not chunks:
                    st.error("❌ No readable text found in the file.")
                else:
                    # Save processed data
                    base = uploaded.name.split(".")[0]
                    json.dump(chunks, open(os.path.join(PROCESSED_DIR, f"{base}_chunks.json"), "w"))
                    json.dump(quotes, open(os.path.join(PROCESSED_DIR, f"{base}_quotes.json"), "w"))
                    
                    # Generate embeddings
                    st.info("🧠 Generating embeddings...")
                    if chunks:
                        chunk_embeds = get_doc_embeddings(chunks)
                        if chunk_embeds is not None:
                            build_faiss(chunk_embeds, os.path.join(VECTOR_DIR, f"{base}_chunks.index"))
                    
                    if quotes:
                        quote_embeds = get_doc_embeddings(quotes)
                        if quote_embeds is not None:
                            build_faiss(quote_embeds, os.path.join(VECTOR_DIR, f"{base}_quotes.index"))
                    
                    st.success(f"✅ **{base}** processed successfully!")
                    st.info(f"📊 Generated {len(chunks)} chunks and {len(quotes)} quotes")
                    st.rerun()
    
    # Book management
    if books:
        st.markdown("### 📖 Your Spiritual Library")
        
        # Display books in a nice format
        for book in books:
            with st.expander(f"📘 {book['name']}", expanded=False):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**Processed:** {book['processed_date']}")
                with col2:
                    st.metric("Chunks", book['chunks'])
                with col3:
                    st.metric("Quotes", book['quotes'])
                with col4:
                    if st.button(f"🗑️ Delete", key=f"del_{book['name']}"):
                        delete_book(book['name'])
                        st.success(f"✅ {book['name']} deleted!")
                        st.rerun()

# 📜 QUOTES PAGE
elif st.session_state["selected_page"] == "📜 Quotes":
    st.markdown("### 📜 Wisdom Quotes")
    
    if not books:
        st.info("📚 Upload some books first to see quotes here!")
    else:
        # Quote search
        search_query = st.text_input("🔍 Search quotes...", placeholder="Enter keywords to find relevant quotes")
        
        # Display quotes by book
        for book in books:
            quotes_path = os.path.join(PROCESSED_DIR, f"{book['name']}_quotes.json")
            if os.path.exists(quotes_path):
                quotes = json.load(open(quotes_path))
                
                # Filter quotes if search query provided
                if search_query:
                    quotes = [q for q in quotes if search_query.lower() in q.lower()]
                
                if quotes:
                    with st.expander(f"📖 {book['name']} ({len(quotes)} quotes)", expanded=not search_query):
                        for i, quote in enumerate(quotes[:20]):  # Show first 20
                            st.markdown(f'<div class="quote-card">"{quote}"</div>', unsafe_allow_html=True)
                        
                        if len(quotes) > 20:
                            st.info(f"Showing first 20 of {len(quotes)} quotes")
                        
                        # Export quotes
                        if st.button(f"📥 Export {book['name']} Quotes", key=f"export_{book['name']}"):
                            quotes_text = "\n\n".join([f'"{quote}"' for quote in quotes])
                            st.download_button(
                                "💾 Download Quotes",
                                data=quotes_text,
                                file_name=f"{book['name']}_quotes.txt",
                                mime="text/plain",
                                key=f"download_{book['name']}"
                            )

# 🧘 MEDITATION PAGE
elif st.session_state["selected_page"] == "🧘 Meditation":
    st.markdown("### 🧘 Mindful Meditation")
    
    # Meditation card
    st.markdown("""
    <div class="meditation-card">
        <h3>🕯️ Take a Moment to Center Yourself</h3>
        <p>Find a quiet space, close your eyes, and focus on your breath. Let your thoughts come and go like clouds in the sky.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Timer options
    st.markdown("#### ⏱️ Meditation Timer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        duration_5 = st.button("🧘 5 Minutes", use_container_width=True)
    with col2:
        duration_10 = st.button("🧘‍♀️ 10 Minutes", use_container_width=True)
    with col3:
        duration_15 = st.button("🧘‍♂️ 15 Minutes", use_container_width=True)
    
    # Custom duration
    custom_duration = st.slider("Or choose custom duration (minutes):", 1, 60, 5)
    custom_start = st.button("🚀 Start Custom Timer", use_container_width=True)
    
    # Determine selected duration
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
        progress_bar = st.progress(0)
        time_display = st.empty()
        status_display = st.empty()
        
        # Bell intervals (every minute for longer sessions)
        bell_interval = 60 if selected_duration >= 5 else selected_duration * 60
        
        start_time = time.time()
        
        for elapsed in range(total_seconds):
            remaining = total_seconds - elapsed
            mins, secs = divmod(remaining, 60)
            
            # Update displays
            time_display.markdown(f'<div class="timer-display">{mins:02d}:{secs:02d}</div>', unsafe_allow_html=True)
            
            progress = elapsed / total_seconds
            progress_bar.progress(progress)
            
            # Status messages
            if elapsed == 0:
                status_display.info("🌱 Begin by taking three deep breaths...")
            elif elapsed == total_seconds // 4:
                status_display.info("🌿 Notice your breath flowing naturally...")
            elif elapsed == total_seconds // 2:
                status_display.info("🕯️ You're halfway there. Stay present...")
            elif elapsed == (3 * total_seconds) // 4:
                status_display.info("✨ Almost there. Feel the peace within...")
            
            # Bell sound (visual indicator)
            if elapsed > 0 and elapsed % bell_interval == 0:
                status_display.success("🔔 *Bell sound*")
                time.sleep(0.5)
                status_display.empty()
            
            time.sleep(1)
        
        # Completion
        time_display.markdown('<div class="timer-display">✅ Complete!</div>', unsafe_allow_html=True)
        progress_bar.progress(1.0)
        status_display.success("🙏 **Meditation complete!** Take a moment to notice how you feel.")
        
        # Save session
        st.session_state["meditation_sessions"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration": selected_duration
        })
        
        st.balloons()
    
    # Meditation history
    if st.session_state["meditation_sessions"]:
        st.markdown("#### 📊 Your Meditation Journey")
        
        sessions_df = pd.DataFrame(st.session_state["meditation_sessions"])
        total_time = sessions_df["duration"].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", len(sessions_df))
        with col2:
            st.metric("Total Minutes", total_time)
        with col3:
            st.metric("Average Session", f"{total_time / len(sessions_df):.1f} min")
        
        # Recent sessions
        with st.expander("📅 Recent Sessions"):
            for session in reversed(st.session_state["meditation_sessions"][-10:]):
                st.write(f"🧘 {session['date']} - {session['duration']} minutes")

# ⚙️ SETTINGS PAGE
elif st.session_state["selected_page"] == "⚙️ Settings":
    st.markdown("### ⚙️ Settings & Preferences")
    
    # App settings
    st.markdown("#### 🎛️ App Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("🤖 AI Model", ["gpt-5-mini", "gpt-4", "claude-3-sonnet"], index=0)
        st.slider("🎯 Response Length", 100, 2000, 1000, 100)
    
    with col2:
        st.selectbox("🔍 Search Results", [2, 3, 5], index=1)
        st.slider("📊 Chunk Size", 300, 800, 500, 50)
    
    st.markdown("#### 💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear All Chat History"):
            st.session_state["chat_history"] = []
            st.success("✅ Chat history cleared!")
    
    with col2:
        if st.button("🧘 Reset Meditation History"):
            st.session_state["meditation_sessions"] = []
            st.success("✅ Meditation history reset!")
    
    with col3:
        if st.button("🔄 Reset All Data"):
            st.session_state["chat_history"] = []
            st.session_state["meditation_sessions"] = []
            st.success("✅ All data reset!")
    
    # Export all data
    st.markdown("#### 📤 Export Your Data")
    if st.button("📦 Export Everything"):
        export_data = {
            "chat_history": st.session_state["chat_history"],
            "meditation_sessions": st.session_state["meditation_sessions"],
            "books": books,
            "export_date": datetime.now().isoformat()
        }
        
        st.download_button(
            "💾 Download Complete Export",
            data=json.dumps(export_data, indent=2),
            file_name=f"healrr_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# ======================
# 🎯 FOOTER
# ======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌿 <strong>Healrr</strong> - Your AI-powered spiritual companion</p>
    <p><em>May your journey be filled with wisdom, peace, and growth</em> 🙏</p>
</div>
""", unsafe_allow_html=True)
