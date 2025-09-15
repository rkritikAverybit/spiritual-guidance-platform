import os
import streamlit as st
from supabase import create_client, Client
from supabase_auth.errors import AuthApiError

# ---------------------------
# Hybrid Secrets (works local + cloud)
# ---------------------------
def get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Auth Functions
# ---------------------------
def signup_user(email: str, password: str):
    """Sign up a new user"""
    try:
        return supabase.auth.sign_up({"email": email, "password": password})
    except AuthApiError as e:
        raise e

def login_user(email: str, password: str):
    """Login existing user"""
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except AuthApiError as e:
        raise e


def logout_user():
    """Clear Supabase auth session and Streamlit state"""
    try:
        supabase.auth.sign_out()  # clear supabase session
    except Exception:
        pass
    st.session_state.clear()


# ---------------------------
# Profiles
# ---------------------------
def save_profile(user_id, name=None, mood="🌞 Peaceful", reminder=True):
    """Upsert profile linked to auth.users.id, handle missing name gracefully"""
    if not name or name.strip() == "":
        name = "Unknown"  # fallback if name not given

    data = {
        "id": user_id,   # must match auth.users.id
        "name": name,
        "mood": mood,
        "reminder_preference": reminder,
    }
    return supabase.table("profiles").upsert(data).execute()


def get_profile(user_id):
    resp = supabase.table("profiles").select("*").eq("id", user_id).execute()
    return resp.data[0] if resp.data else None

# ---------------------------
# Mood Logs
# ---------------------------
def save_mood_log(user_id, mood):
    data = {"user_id": user_id, "mood": mood}
    return supabase.table("mood_logs").insert(data).execute()

def get_last_mood(user_id):
    resp = (
        supabase.table("mood_logs")
        .select("mood")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0]["mood"] if resp.data else None

# ---------------------------
# Sessions
# ---------------------------
def get_or_create_session(user_id, title="New Reflection"):
    resp = (
        supabase.table("sessions")
        .select("id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0]["id"]

    data = {"user_id": user_id, "title": title}
    new_session = supabase.table("sessions").insert(data).execute()
    return new_session.data[0]["id"]

# ---------------------------
# Messages
# ---------------------------
def add_message(session_id, role, content):
    data = {"session_id": session_id, "role": role, "content": content}
    return supabase.table("messages").insert(data).execute()

def get_messages(session_id):
    return supabase.table("messages").select("*").eq("session_id", session_id).execute().data

# ---------------------------
# Quotes
# ---------------------------
def save_favorite_quote(user_id, quote_id):
    data = {"user_id": user_id, "quote_id": quote_id}
    return supabase.table("favorites").insert(data).execute()

def get_favorite_quotes(user_id):
    resp = (
        supabase.table("favorites")
        .select("quotes(content, reference)")
        .eq("user_id", user_id)
        .execute()
    )
    return resp.data

# ---------------------------
# OpenRouter Chat Helper
# ---------------------------
def openrouter_chat(messages):
    import openai

    client = openai.OpenAI(
        api_key=get_secret("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",  # or any other model
        messages=messages,
    )

    return resp.choices[0].message


def get_all_quotes():
    """Fetch all quotes from the database"""
    resp = supabase.table("quotes").select("*").execute()
    return resp.data if resp.data else []

def get_current_user():
    """Return the currently logged-in user from Supabase"""
    try:
        return supabase.auth.get_user()
    except Exception:
        return None
