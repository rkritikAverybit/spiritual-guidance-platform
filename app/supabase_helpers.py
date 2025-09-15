import os
from supabase import create_client, Client
import streamlit as st
from datetime import datetime

# ---------------------------
# Setup Supabase Client
# ---------------------------
def get_supabase_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase_client()

# ---------------------------
# Profiles (Onboarding data)
# ---------------------------
def save_profile(user_id: str, name: str, mood: str, reminder: bool):
    data = {
        "id": user_id,
        "name": name,
        "mood": mood,
        "reminder_preference": reminder,
        "updated_at": datetime.utcnow().isoformat()
    }
    return supabase.table("profiles").upsert(data).execute()

def get_profile(user_id: str):
    resp = supabase.table("profiles").select("*").eq("id", user_id).execute()
    return resp.data[0] if resp.data else None

# ---------------------------
# Sessions
# ---------------------------
def create_session(user_id: str, title: str):
    data = {"user_id": user_id, "title": title}
    resp = supabase.table("sessions").insert(data).execute()
    return resp.data[0] if resp.data else None

def get_sessions(user_id: str):
    resp = supabase.table("sessions").select("*").eq("user_id", user_id).order("created_at").execute()
    return resp.data

# ---------------------------
# Messages
# ---------------------------
def add_message(session_id: str, role: str, content: str):
    data = {"session_id": session_id, "role": role, "content": content}
    return supabase.table("messages").insert(data).execute()

def get_messages(session_id: str):
    resp = supabase.table("messages").select("*").eq("session_id", session_id).order("created_at").execute()
    return resp.data

# ---------------------------
# Quotes
# ---------------------------
def add_quote(content: str, reference: str = None):
    data = {"content": content, "reference": reference}
    return supabase.table("quotes").insert(data).execute()

def get_quotes(limit: int = 10):
    resp = supabase.table("quotes").select("*").limit(limit).execute()
    return resp.data

# ---------------------------
# Content Sources
# ---------------------------
def add_content_source(content_type: str, title: str, file_url: str = None):
    data = {"type": content_type, "title": title, "file_url": file_url}
    return supabase.table("content_sources").insert(data).execute()

def get_content_sources(content_type: str = None):
    query = supabase.table("content_sources").select("*")
    if content_type:
        query = query.eq("type", content_type)
    resp = query.execute()
    return resp.data
