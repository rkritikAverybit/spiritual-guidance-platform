
# # app/main.py
# import streamlit as st
# import time


# # Import local modules
# import supabase_helpers as db
# import onboarding, chat, quotes, admin

# # ---------------------------
# # Page config (compatible with older Streamlit)
# # ---------------------------
# st.set_page_config(page_title="Healrr", layout="wide", initial_sidebar_state="expanded")


# # ---------------------------
# # Helper: render sidebar profile card
# # ---------------------------
# def render_sidebar_profile(profile):
#     st.sidebar.markdown("### 🌸 Healrr")
#     with st.sidebar.container():
#         if profile:
#             name = profile.get("name", "Seeker")
#             mood = profile.get("mood", "🌞 Peaceful")
#             reminder = profile.get("reminder_preference", True)

#             avatar_html = f"""
#             <div style="display:flex;align-items:center;gap:12px">
#                 <div style="background:#3A9188;color:white;border-radius:50%;width:56px;height:56px;
#                             display:flex;align-items:center;justify-content:center;font-weight:700;font-size:22px">
#                     {name[0].upper() if name else "S"}
#                 </div>
#                 <div>
#                     <div style="font-weight:700">{name}</div>
#                     <div style="font-size:13px;color:#666">{mood} • {'Reminders On' if reminder else 'Reminders Off'}</div>
#                 </div>
#             </div>
#             """
#             st.sidebar.markdown(avatar_html, unsafe_allow_html=True)

#         else:
#             # Not logged in / no profile
#             st.sidebar.markdown(
#                 """
#                 <div style="display:flex;align-items:center;gap:12px">
#                   <div style="background:#3A9188;color:white;border-radius:50%;width:56px;height:56px;
#                               display:flex;align-items:center;justify-content:center;font-weight:700;font-size:22px">
#                     ?
#                   </div>
#                   <div>
#                     <div style="font-weight:700">Welcome</div>
#                     <div style="font-size:13px;color:#666">Please login or signup</div>
#                   </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#         st.sidebar.markdown("---")


# # ---------------------------
# # Restore supabase session on app start (so refresh keeps user logged in)
# # ---------------------------
# if "user_id" not in st.session_state:
#     st.session_state.user_id = None
#     try:
#         # Attempt to get current user from supabase client
#         user_resp = None
#         try:
#             user_resp = db.supabase.auth.get_user()
#         except Exception:
#             # some supabase clients use auth.get_session() or auth.get_user()
#             try:
#                 user_resp = db.supabase.auth.get_session()
#             except Exception:
#                 user_resp = None

#         if user_resp:
#             # supabase.auth.get_user() returns { "data": {"user": {...}} } or object with .user
#             # handle multiple shapes
#             uid = None
#             if isinstance(user_resp, dict):
#                 # older SDK shapes
#                 data = user_resp.get("data") or {}
#                 user = data.get("user") or data.get("session", {}).get("user")
#                 if user:
#                     uid = user.get("id")
#             else:
#                 # object with attribute
#                 user_obj = getattr(user_resp, "user", None) or getattr(user_resp, "data", None)
#                 if user_obj and isinstance(user_obj, dict):
#                     uid = user_obj.get("id")
#                 elif user_obj and hasattr(user_obj, "id"):
#                     uid = user_obj.id

#             if uid:
#                 st.session_state.user_id = uid
#     except Exception:
#         # ignore restore errors silently (we'll show login)
#         st.session_state.user_id = None


# # ---------------------------
# # Sidebar: Authentication / Profile
# # ---------------------------
# st.sidebar.title("🔑 Authentication")

# # If logged-in show profile + logout
# if st.session_state.user_id:
#     profile = None
#     try:
#         profile = db.get_profile(st.session_state.user_id)
#     except Exception:
#         profile = None

#     render_sidebar_profile(profile)

#     st.sidebar.markdown("")  # spacing
#     if st.sidebar.button("🚪 Logout"):
#         try:
#             db.logout_user()   # Supabase se bhi sign out
#         except Exception as e:
#             st.sidebar.warning(f"Logout issue: {e}")
#         st.session_state.clear()   # Streamlit state clear
#         st.rerun()    # Refresh app


# else:
#     # Not logged in: show Login / Signup / Forgot Password
#     render_sidebar_profile(None)

#     auth_choice = st.sidebar.radio("Choose:", ["Login", "Signup", "Forgot Password"])
#     email = st.sidebar.text_input("Email", key="auth_email")
#     password = None
#     name = None

#     if auth_choice == "Signup":
#         name = st.sidebar.text_input("Your name", key="signup_name")
#         password = st.sidebar.text_input("Password", type="password", key="signup_pass")
#     elif auth_choice == "Login":
#         password = st.sidebar.text_input("Password", type="password", key="login_pass")
#     else:
#         # forgot password: no password field
#         pass

#     if st.sidebar.button(auth_choice):
#         if not email:
#             st.sidebar.error("Please enter your email.")
#         else:
#             try:
#                 if auth_choice == "Signup":
#                     if not password or not name:
#                         st.sidebar.error("Please enter name and password to sign up.")
#                     else:
#                         res = db.signup_user(email, password)
#                         # res may contain 'user' or resp.data.user depending on SDK
#                         user_id = None
#                         try:
#                             if hasattr(res, "user") and getattr(res, "user"):
#                                 user_id = res.user.id
#                             elif isinstance(res, dict):
#                                 # handle dict responses
#                                 data = res.get("data") or {}
#                                 user = data.get("user") or data.get("user")
#                                 if user:
#                                     user_id = user.get("id")
#                         except Exception:
#                             user_id = None

#                         # If signup returned a user id, create profile immediately
#                         if user_id:
#                             try:
#                                 db.save_profile(user_id, name, "🌞 Peaceful", True)
#                             except Exception as e:
#                                 st.sidebar.warning(f"Signup OK but saving profile failed: {e}")
#                         st.sidebar.success("Signup initiated. Please check your email to confirm (if required).")

#                 elif auth_choice == "Login":
#                     if not password:
#                         st.sidebar.error("Please enter your password.")
#                     else:
#                         res = db.login_user(email, password)
#                         uid = None
#                         if hasattr(res, "user") and getattr(res, "user"):
#                             uid = res.user.id
#                         elif isinstance(res, dict):
#                             data = res.get("data") or {}
#                             user = data.get("user") or data.get("session", {}).get("user")
#                             if user:
#                                 uid = user.get("id")
#                         if uid:
#                             st.session_state.user_id = uid
#                             st.sidebar.success("Logged in successfully.")
#                             # reload main to show onboarding/mood
#                             st.rerun()
#                         else:
#                             st.sidebar.warning("Login returned no user id; check logs.")

#                 elif auth_choice == "Forgot Password":
#                     # supabase helper should expose reset link wrapper; fallback using client
#                     try:
#                         db.supabase.auth.reset_password_for_email(
#                             email,
#                             options={"redirect_to": "http://localhost:8501"}
#                         )
#                         st.sidebar.info("Password reset link sent to your email (check spam).")
#                     except Exception as e:
#                         st.sidebar.error(f"Failed to send reset email: {e}")

#             except Exception as e:
#                 # friendly error messages for common auth issues
#                 msg = str(e)
#                 if "Email not confirmed" in msg:
#                     st.sidebar.warning("Please confirm your email before logging in.")
#                 elif "User already registered" in msg or "already" in msg.lower():
#                     st.sidebar.info("This email is already registered. Please use Login.")
#                 else:
#                     st.sidebar.error(f"Auth error: {e}")


# # ---------------------------
# # Main app for logged-in users
# # ---------------------------
# if st.session_state.user_id:
#     # ensure profile exists
#     profile = None
#     try:
#         profile = db.get_profile(st.session_state.user_id)
#     except Exception:
#         profile = None


#     # ---------------------------
#     # Run onboarding if not complete
#     # ---------------------------

#     profile = db.get_profile(st.session_state.user_id)
#     if not (profile and profile.get("onboarded")):
#         onboarding.run_onboarding(st.session_state.user_id)
#         st.stop()


#     # profile exists from here
#     st.sidebar.markdown("")  # spacing
#     # Daily mood check (ask every session unless already checked in this run)
#     if "mood_checked" not in st.session_state or not st.session_state.get("mood_checked"):
#         st.header("💭 How are you feeling right now?")
#         mood = st.radio("Choose your current state:", ["🌞 Peaceful", "😌 Calm", "😔 Sad", "🔥 Stressed"], key="daily_mood")
#         if st.button("Save Mood"):
#             try:
#                 db.save_mood_log(st.session_state.user_id, mood)
#                 st.session_state.mood_checked = True
#                 st.success("Mood saved 💖")
#                 time.sleep(0.3)
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Failed to save mood: {e}")
#                 st.stop()

#     # After mood is checked show navigation and pages
#     if st.session_state.get("mood_checked", False):
#         # Navigation
#         st.sidebar.markdown("---")

        
#         page = st.sidebar.radio(
#     "Navigate:",
#     ["Chat","Quotes","Meditation", "Journal", "Settings", "Admin"]
# )



#         # show profile summary at top of main area
#         st.markdown(f"### 👋 Welcome back, **{profile.get('name','Seeker')}**")
#         last_mood = db.get_last_mood(st.session_state.user_id) or profile.get("mood", "🌞 Peaceful")
#         st.caption(f"Current mood: {last_mood}")

#         if page == "Chat":
#             chat.run_chat(st.session_state.user_id)
#             from supabase_helpers import reminder_loop
#             reminder_loop(st.session_state.user_id)
#         elif page == "Quotes":
#             quotes.run_quotes(st.session_state.user_id)
#         elif page == "Admin":
#             # optionally restrict admin only to certain emails (you can add checks here)
#             admin.run_admin()
#         elif page == "Meditation":
#             import meditation
#             meditation.run_meditation()
#         elif page == "Journal":
#             import journal
#             journal.run_journal(st.session_state.user_id)

#         elif page == "Settings":
#             # simple settings inline (we'll later move to settings.py)
#             st.header("⚙️ Settings")
#             new_name = st.text_input("Change display name", value=profile.get("name"))
#             reminder_pref = st.checkbox("Enable reminders every 30 minutes", value=profile.get("reminder_preference", True))
#             if st.button("Save settings"):
#                 try:
#                     db.save_profile(st.session_state.user_id, new_name, profile.get("mood", "🌞 Peaceful"), reminder_pref)
#                     st.success("Settings updated.")
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Failed to update settings: {e}")

# else:
#     # not logged in view (short welcome)
#     st.title("🌸 Welcome to Healrr")
#     st.markdown("Healrr guides you inward toward peace. Please login or signup from the sidebar to begin.")



# app/main.py
import streamlit as st
import time

# Import local modules
import supabase_helpers as db
import onboarding, chat, quotes, admin, meditation, journal

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Healrr", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper: render sidebar profile card
# ---------------------------
def render_sidebar_profile(profile):
    st.sidebar.markdown("### 🌸 Healrr")
    with st.sidebar.container():
        if profile:
            name = profile.get("name", "Seeker")
            mood = profile.get("mood", "🌞 Peaceful")
            reminder = profile.get("reminder_preference", True)

            avatar_html = f"""
            <div style="display:flex;align-items:center;gap:12px">
                <div style="background:#3A9188;color:white;border-radius:50%;width:56px;height:56px;
                            display:flex;align-items:center;justify-content:center;font-weight:700;font-size:22px">
                    {name[0].upper() if name else "S"}
                </div>
                <div>
                    <div style="font-weight:700">{name}</div>
                    <div style="font-size:13px;color:#666">{mood} • {'Reminders On' if reminder else 'Reminders Off'}</div>
                </div>
            </div>
            """
            st.sidebar.markdown(avatar_html, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                """
                <div style="display:flex;align-items:center;gap:12px">
                  <div style="background:#3A9188;color:white;border-radius:50%;width:56px;height:56px;
                              display:flex;align-items:center;justify-content:center;font-weight:700;font-size:22px">
                    ?
                  </div>
                  <div>
                    <div style="font-weight:700">Welcome</div>
                    <div style="font-size:13px;color:#666">Please login or signup</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.sidebar.markdown("---")

# ---------------------------
# Restore supabase session on app start (refresh keeps user logged in)
# ---------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    try:
        user_resp = db.supabase.auth.get_user()
    except Exception:
        user_resp = None

    uid = None
    if user_resp:
        if isinstance(user_resp, dict):
            data = user_resp.get("data") or {}
            user = data.get("user") or data.get("session", {}).get("user")
            if user:
                uid = user.get("id")
        else:
            user_obj = getattr(user_resp, "user", None) or getattr(user_resp, "data", None)
            if user_obj:
                uid = user_obj.get("id") if isinstance(user_obj, dict) else getattr(user_obj, "id", None)

    if uid:
        st.session_state.user_id = uid

# ---------------------------
# Sidebar: Authentication / Profile
# ---------------------------
st.sidebar.title("🔑 Authentication")

if st.session_state.user_id:
    # Logged-in view
    try:
        profile = db.get_profile(st.session_state.user_id)
    except Exception:
        profile = None

    render_sidebar_profile(profile)

    if st.sidebar.button("🚪 Logout"):
        try:
            db.logout_user()   # Supabase se sign out
        except Exception as e:
            st.sidebar.warning(f"Logout issue: {e}")
        st.session_state.clear()
        st.rerun()

else:
    # Not logged in: show Login/Signup
    render_sidebar_profile(None)
    auth_choice = st.sidebar.radio("Choose:", ["Login", "Signup", "Forgot Password"])
    email = st.sidebar.text_input("Email", key="auth_email")
    password = None
    name = None

    if auth_choice == "Signup":
        name = st.sidebar.text_input("Your name", key="signup_name")
        password = st.sidebar.text_input("Password", type="password", key="signup_pass")
    elif auth_choice == "Login":
        password = st.sidebar.text_input("Password", type="password", key="login_pass")

    if st.sidebar.button(auth_choice):
        if not email:
            st.sidebar.error("Please enter your email.")
        else:
            try:
                if auth_choice == "Signup":
                    if not password or not name:
                        st.sidebar.error("Please enter name and password to sign up.")
                    else:
                        res = db.signup_user(email, password)
                        user_id = None
                        if hasattr(res, "user") and getattr(res, "user"):
                            user_id = res.user.id
                        elif isinstance(res, dict):
                            data = res.get("data") or {}
                            user = data.get("user")
                            if user:
                                user_id = user.get("id")
                        if user_id:
                            try:
                                db.save_profile(user_id, name, "🌞 Peaceful", True)
                            except Exception as e:
                                st.sidebar.warning(f"Signup OK but saving profile failed: {e}")
                        st.sidebar.success("Signup initiated. Please check your email to confirm.")

                elif auth_choice == "Login":
                    if not password:
                        st.sidebar.error("Please enter your password.")
                    else:
                        res = db.login_user(email, password)
                        uid = None
                        if hasattr(res, "user") and getattr(res, "user"):
                            uid = res.user.id
                        elif isinstance(res, dict):
                            data = res.get("data") or {}
                            user = data.get("user") or data.get("session", {}).get("user")
                            if user:
                                uid = user.get("id")
                        if uid:
                            st.session_state.user_id = uid
                            st.sidebar.success("Logged in successfully.")
                            st.rerun()
                        else:
                            st.sidebar.warning("Login returned no user id; check logs.")

                elif auth_choice == "Forgot Password":
                    try:
                        db.supabase.auth.reset_password_for_email(
                            email, options={"redirect_to": "http://localhost:8501"}
                        )
                        st.sidebar.info("Password reset link sent to your email.")
                    except Exception as e:
                        st.sidebar.error(f"Failed to send reset email: {e}")
            except Exception as e:
                st.sidebar.error(f"Auth error: {e}")

# ---------------------------
# Main app for logged-in users
# ---------------------------
if st.session_state.user_id:
    try:
        profile = db.get_profile(st.session_state.user_id)
    except Exception:
        profile = None

    # Onboarding
    if not (profile and profile.get("onboarded")):
        onboarding.run_onboarding(st.session_state.user_id)
        st.stop()

    # Mood check
    if "mood_checked" not in st.session_state or not st.session_state.get("mood_checked"):
        st.header("💭 How are you feeling right now?")
        mood = st.radio("Choose your current state:", ["🌞 Peaceful", "😌 Calm", "😔 Sad", "🔥 Stressed"], key="daily_mood")
        if st.button("Save Mood"):
            try:
                db.save_mood_log(st.session_state.user_id, mood)
                st.session_state.mood_checked = True
                st.success("Mood saved 💖")
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save mood: {e}")
                st.stop()

    # After mood check
    if st.session_state.get("mood_checked", False):
        st.sidebar.markdown("---")
        page = st.sidebar.radio("Navigate:", ["Chat","Quotes","Meditation","Journal","Settings","Admin"])
        st.markdown(f"### 👋 Welcome back, **{profile.get('name','Seeker')}**")
        last_mood = db.get_last_mood(st.session_state.user_id) or profile.get("mood", "🌞 Peaceful")
        st.caption(f"Current mood: {last_mood}")

        if page == "Chat":
            chat.run_chat(st.session_state.user_id)
            from supabase_helpers import reminder_loop
            reminder_loop(st.session_state.user_id)
        elif page == "Quotes":
            quotes.run_quotes(st.session_state.user_id)
        elif page == "Admin":
            admin.run_admin()
        elif page == "Meditation":
            meditation.run_meditation()
        elif page == "Journal":
            journal.run_journal(st.session_state.user_id)
        elif page == "Settings":
            st.header("⚙️ Settings")
            new_name = st.text_input("Change display name", value=profile.get("name"))
            reminder_pref = st.checkbox("Enable reminders every 30 minutes", value=profile.get("reminder_preference", True))
            if st.button("Save settings"):
                try:
                    db.save_profile(st.session_state.user_id, new_name, profile.get("mood", "🌞 Peaceful"), reminder_pref)
                    st.success("Settings updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update settings: {e}")
else:
    st.title("🌸 Welcome to Healrr")
    st.markdown("Healrr guides you inward toward peace. Please login or signup from the sidebar to begin.")
