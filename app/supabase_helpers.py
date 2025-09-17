import os
import streamlit as st
from supabase import create_client, Client
from supabase_auth.errors import AuthApiError
from dotenv import load_dotenv

load_dotenv()

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
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.clear()

# ---------------------------
# Profiles
# ---------------------------
def save_profile(user_id, name=None, mood="🌞 Peaceful", reminder=True):
    """Upsert profile linked to auth.users.id, handle missing name gracefully"""
    if not name or name.strip() == "":
        name = "Unknown"
    data = {
        "id": user_id,
        "name": name,
        "mood": mood,
        "reminder_preference": reminder,
    }
    return supabase.table("profiles").upsert(data).execute()

def get_profile(user_id):
    resp = supabase.table("profiles").select("*").eq("id", user_id).execute()
    return resp.data[0] if resp and resp.data else None

# ---------------------------
# Mood Logs
# ---------------------------
def save_mood_log(user_id, mood):
    """Insert a new mood log entry"""
    data = {"user_id": user_id, "mood": mood}
    return supabase.table("mood_logs").insert(data).execute()

def get_last_mood(user_id):
    """Fetch the latest mood for a user"""
    resp = (
        supabase.table("mood_logs")
        .select("mood")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0]["mood"] if resp and resp.data else None

# ---------------------------
# Sessions
# ---------------------------
def get_or_create_session(user_id, title="New Reflection"):
    """Fetch the latest session or create a new one"""
    resp = (
        supabase.table("sessions")
        .select("id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp and resp.data:
        return resp.data[0]["id"]

    data = {"user_id": user_id, "title": title}
    new_session = supabase.table("sessions").insert(data).execute()
    return new_session.data[0]["id"]

# ---------------------------
# Messages
# ---------------------------
def add_message(session_id, role, content):
    """Save a chat message"""
    data = {"session_id": session_id, "role": role, "content": content}
    return supabase.table("messages").insert(data).execute()

def get_messages(session_id):
    """Fetch all messages in a session"""
    resp = supabase.table("messages").select("*").eq("session_id", session_id).execute()
    return resp.data if resp and resp.data else []

# ---------------------------
# Quotes
# ---------------------------
def get_all_quotes():
    """Fetch all quotes"""
    resp = supabase.table("quotes").select("id, content, reference").execute()
    return resp.data if resp and resp.data else []

def save_favorite_quote(user_id, quote_id):
    """Save a favorite quote for a user"""
    data = {"user_id": user_id, "quote_id": quote_id}
    return supabase.table("favorites").insert(data).execute()

def get_favorites(user_id):
    """Fetch a user's favorite quotes with details"""
    resp = (
        supabase.table("favorites")
        .select("id, quote_id, quotes(content, reference)")
        .eq("user_id", user_id)
        .execute()
    )
    return [
        {
            "id": f["id"],
            "quote_id": f["quote_id"],
            "content": f["quotes"]["content"] if f.get("quotes") else None,
            "reference": f["quotes"]["reference"] if f.get("quotes") else None,
        }
        for f in (resp.data or [])
    ]

def remove_favorite(favorite_id):
    """Remove a favorite quote by ID"""
    return supabase.table("favorites").delete().eq("id", favorite_id).execute()

# ---------------------------
# OpenRouter Chat Helper
# ---------------------------
def openrouter_chat(messages):
    """Send chat messages to OpenRouter (GPT model) and return the first message"""
    import openai

    client = openai.OpenAI(
        api_key=get_secret("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
    )
    return resp.choices[0].message

# ---------------------------
# Current User
# ---------------------------
def get_current_user():
    """Return the currently logged-in user from Supabase"""
    try:
        return supabase.auth.get_user()
    except Exception:
        return None


import time
import streamlit as st

def reminder_loop(user_id):
    """
    Runs a 30-min reminder loop while reminder is enabled in session_state.
    Note: This works while app is open; Streamlit doesn't run true background tasks.
    """
    if st.session_state.get("reminder_enabled", False):
        if "last_reminder" not in st.session_state:
            st.session_state.last_reminder = time.time()

        now = time.time()
        if now - st.session_state.last_reminder >= 1800:  # 30 min = 1800 sec
            st.session_state.last_reminder = now
            st.sidebar.info("🌿 Pause. Take 5 conscious breaths and notice your inner Being.")

# ---------------------------
# Meditation Logs
# ---------------------------
def save_meditation_log(user_id, duration):
    """Save meditation session to Supabase"""
    data = {
        "user_id": user_id,
        "duration": duration
    }
    return supabase.table("meditation_logs").insert(data).execute()

def get_meditation_logs(user_id):
    """Fetch meditation history for a user"""
    resp = (
        supabase.table("meditation_logs")
        .select("duration, timestamp")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .execute()
    )
    return resp.data if resp and resp.data else []

# ---------------------------
# Profile Helpers
# ---------------------------
def update_profile(user_id, updates: dict):
    """Update user profile with given dictionary of fields"""
    return supabase.table("profiles").update(updates).eq("id", user_id).execute()



def get_profile(user_id):
    """Fetch profile for a user"""
    resp = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
    return resp.data if resp and resp.data else None


def logout_user():
    """Logout current user from Supabase auth"""
    try:
        supabase.auth.sign_out()
    except Exception as e:
        print("Logout error:", e)
    return True
