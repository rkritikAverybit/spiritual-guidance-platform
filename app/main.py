

import streamlit as st
import supabase_helpers as db
import onboarding, chat, quotes, admin
from supabase_auth.errors import AuthApiError

st.set_page_config(page_title="Healrr", layout="wide")

# At the top of main.py
if "user_id" not in st.session_state:
    # Try to restore Supabase session
    user = db.get_current_user()
    if user and user.user:
        st.session_state.user_id = user.user.id
    else:
        st.session_state.user_id = None


# ---------------------------
# Session Defaults
# ---------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "profile_complete" not in st.session_state:
    st.session_state.profile_complete = False
if "mood_checked" not in st.session_state:
    st.session_state.mood_checked = False

# ---------------------------
# Sidebar Auth
# ---------------------------
import streamlit as st
import supabase_helpers as db
from supabase_auth.errors import AuthApiError

st.sidebar.title("🔑 Authentication")

if st.session_state.get("user_id"):
    st.sidebar.success("✅ Logged in")
    if st.sidebar.button("Logout"):
        db.logout_user()
        st.rerun()
else:
    auth_choice = st.sidebar.radio("Choose:", ["Login", "Signup", "Forgot Password"])

    email = st.sidebar.text_input("Email")
    password = None
    name = None

    if auth_choice == "Signup":
        name = st.sidebar.text_input("Your Name")  # 🔹 Take name at signup
        password = st.sidebar.text_input("Password", type="password")
    elif auth_choice == "Login":
        password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(auth_choice):
        if not email:
            st.sidebar.error("Please enter your email.")
        else:
            try:
                if auth_choice == "Signup":
                    if not password or not name:
                        st.sidebar.error("Please enter name, email and password.")
                    else:
                        res = db.signup_user(email, password)
                        if res.user:
                            # ✅ Save profile with name immediately
                            db.save_profile(res.user.id, name, "🌞 Peaceful", True)
                            st.sidebar.success("Signup successful! Please confirm your email 📩")

                elif auth_choice == "Login":
                    res = db.login_user(email, password)
                    if res.user:
                        st.session_state.user_id = res.user.id
                        profile = db.get_profile(res.user.id)
                        if profile:
                            st.sidebar.success(f"Welcome back {profile['name']}! 🎉")
                        else:
                            st.sidebar.success("Welcome! Please complete your profile 🧘")
                        st.rerun()

                elif auth_choice == "Forgot Password":
                    db.supabase.auth.reset_password_for_email(
                        email,
                        options={"redirect_to": "http://localhost:8502"}  # replace with prod URL
                    )
                    st.sidebar.info("Password reset link sent to your email 📩")

            except AuthApiError as e:
                err_msg = str(e)
                if "Email not confirmed" in err_msg:
                    st.sidebar.warning("⚠️ Please confirm your email before logging in.")
                elif "User already registered" in err_msg:
                    st.sidebar.info("ℹ️ This email is already registered. Please login.")
                else:
                    st.sidebar.error(f"❌ Auth failed: {err_msg}")


# ---------------------------
# Main Flow (After Login)
# ---------------------------
if st.session_state.user_id:

    # 1️⃣ Ensure profile exists
    profile = db.get_profile(st.session_state.user_id)
    if not profile:
        st.warning("Complete your onboarding 🧘")
        onboarding.run_onboarding(st.session_state.user_id)
    else:
        st.session_state.profile_complete = True
        st.session_state.name = profile.get("name")

    # 2️⃣ Daily Mood Check
    if st.session_state.profile_complete and not st.session_state.mood_checked:
        st.subheader("💭 How are you feeling right now?")
        mood = st.radio("Choose your current state:",
                        ["🌞 Peaceful", "😌 Calm", "😔 Sad", "🔥 Stressed"])

        if st.button("Save Mood"):
            db.save_mood_log(st.session_state.user_id, mood)
            st.session_state.mood_checked = True
            st.success("Mood saved 💖")
            st.rerun()

    # 3️⃣ Main Navigation
    if st.session_state.profile_complete and st.session_state.mood_checked:
        st.sidebar.header("📍 Navigation")
        page = st.sidebar.radio("Go to:", ["Chat", "Quotes", "Admin"])

        if page == "Chat":
            chat.run_chat(st.session_state.user_id)
        elif page == "Quotes":
            quotes.run_quotes(st.session_state.user_id)
        elif page == "Admin":
            admin.run_admin()
