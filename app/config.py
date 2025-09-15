import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
COHERE_API_KEY = get_secret("COHERE_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL or SUPABASE_KEY not found in secrets or .env")
if not COHERE_API_KEY:
    raise ValueError("❌ COHERE_API_KEY not found in secrets or .env")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in secrets or .env")