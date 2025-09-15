import os
import streamlit as st
from content_ingest import ingest_pdf, ingest_txt, ingest_substack_csv, ingest_quotes_file
import supabase_helpers as db

SAVE_PATH = "data/vector_store/index.pkl"

def run_admin():
    st.title("🛠️ Healrr Admin Panel")

    st.sidebar.header("Admin Options")
    action = st.sidebar.radio("Choose Action", ["Upload Content", "Manage Index", "Manage Quotes"])

    # ---------------------------
    # Upload Content
    # ---------------------------
    if action == "Upload Content":
        st.subheader("📂 Upload Content")
        file = st.file_uploader("Upload PDF / TXT / Substack CSV", type=["pdf", "txt", "csv"])
        if file:
            file_path = os.path.join("data", "raw", file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                if st.button("Ingest PDF"):
                    ingest_pdf(file_path, SAVE_PATH)
                    db.supabase.table("content_sources").insert({
                        "filename": file.name,
                        "type": "pdf"
                    }).execute()
                    st.success(f"✅ PDF {file.name} ingested and indexed!")

            elif file.name.endswith(".txt"):
                if st.button("Ingest TXT"):
                    ingest_txt(file_path, SAVE_PATH)
                    db.supabase.table("content_sources").insert({
                        "filename": file.name,
                        "type": "txt"
                    }).execute()
                    st.success(f"✅ TXT {file.name} ingested and indexed!")

            elif file.name.endswith(".csv"):
                if st.button("Ingest Substack CSV"):
                    ingest_substack_csv(file_path, SAVE_PATH)
                    db.supabase.table("content_sources").insert({
                        "filename": file.name,
                        "type": "csv"
                    }).execute()
                    st.success(f"✅ Substack CSV {file.name} ingested and indexed!")

    # ---------------------------
    # Manage Index
    # ---------------------------
    elif action == "Manage Index":
        st.subheader("📊 Vector Index Management")

        if os.path.exists(SAVE_PATH):
            st.success("✅ Index file found at " + SAVE_PATH)
        else:
            st.warning("⚠️ No index found yet. Please upload content first.")

        if st.button("Clear Index"):
            if os.path.exists(SAVE_PATH):
                os.remove(SAVE_PATH)
                st.success("🗑️ Index cleared successfully!")
                db.supabase.table("content_sources").delete().neq("id", 0).execute()
            else:
                st.info("No index to clear.")

    # ---------------------------
    # Manage Quotes
    # ---------------------------
    elif action == "Manage Quotes":
        st.subheader("📜 Upload Quotes File")
    
        file = st.file_uploader("Upload TXT/CSV quotes", type=["txt", "csv"])
        if file:
            file_path = os.path.join("data", "raw", file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
    
            if st.button("Ingest Quotes"):
                count = ingest_quotes_file(file_path)
                st.success(f"✅ {count} quotes ingested into Supabase!")
    
        # ✅ New button for indexing quotes into FAISS
        if st.button("📊 Index Quotes into FAISS"):
            from content_ingest import ingest_quotes_to_index
            count = ingest_quotes_to_index()
            st.success(f"✅ {count} quotes added into FAISS index!")
    
        st.subheader("📋 Existing Quotes")
        quotes = db.supabase.table("quotes").select("*").limit(20).execute().data
        if quotes:
            for q in quotes:
                st.markdown(f"> {q['content']} — {q.get('reference','')}")
