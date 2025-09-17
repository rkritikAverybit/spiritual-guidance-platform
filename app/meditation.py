import streamlit as st
import time
import pandas as pd
from datetime import datetime
import supabase_helpers as db
from streamlit_autorefresh import st_autorefresh

# ------------------------------
# Meditation with Supabase History & Stats
# ------------------------------
def run_meditation():
    st.title("🧘 Healrr Meditation")

    st.markdown("Take a moment to connect with your inner Being 🌿")

    # ------------------------------
    # Quick Options
    # ------------------------------
    st.subheader("⏱️ Quick Start")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("5 min"):
            start_meditation(5)
    with col2:
        if st.button("10 min"):
            start_meditation(10)
    with col3:
        if st.button("15 min"):
            start_meditation(15)

    # ------------------------------
    # Custom Duration
    # ------------------------------
    st.subheader("🎚️ Custom Duration")
    minutes = st.slider("Select meditation duration (minutes):", 1, 60, 5)
    if st.button("Start Custom Meditation"):
        start_meditation(minutes)

    # ------------------------------
    # Active Meditation Session
    # ------------------------------
    if st.session_state.get("meditation_active", False):
        st_autorefresh(interval=1000, limit=None, key="meditation_timer")
        remaining = int(st.session_state.meditation_end - time.time())
        if remaining > 0:
            mins, secs = divmod(remaining, 60)
            st.progress((st.session_state.meditation_total - remaining) / st.session_state.meditation_total)
            st.metric("⏳ Time Remaining", f"{mins:02d}:{secs:02d}")

            # Midway gentle reminder
            halfway = st.session_state.meditation_total // 2
            if abs(remaining - halfway) < 2:  # ~2s tolerance
                st.info("🌸 You are halfway there. Stay present in silence.")
        else:
            st.balloons()
            st.success("🌸 Meditation complete. Be still and rest in your true Self.")

            # Save session history to Supabase
            duration = st.session_state.meditation_total // 60
            try:
                db.save_meditation_log(st.session_state.user_id, duration)
            except Exception as e:
                st.error(f"Failed to save meditation log: {e}")
            st.session_state.meditation_active = False

    # ------------------------------
    # Meditation History (from Supabase)
    # ------------------------------
    st.markdown("---")
    # st.subheader("📖 Meditation History")

    # try:
    #     logs = db.get_meditation_logs(st.session_state.user_id)
    # except Exception as e:
    #     st.error(f"Failed to load meditation history: {e}")
    #     logs = []

    # if logs:
    #     df = pd.DataFrame(logs)
    #     st.dataframe(df)

    #     total_sessions = len(df)
    #     total_minutes = df["duration"].sum()
    #     avg_duration = df["duration"].mean()

    #     st.metric("Total Sessions", total_sessions)
    #     st.metric("Total Minutes", total_minutes)
    #     st.metric("Average Duration", f"{avg_duration:.1f} min")
    # else:
    #     st.info("No meditation sessions yet. Start your first one above 🌿")


# ------------------------------
# Helper function to start meditation
# ------------------------------
def start_meditation(minutes: int):
    st.session_state.meditation_active = True
    st.session_state.meditation_total = minutes * 60
    st.session_state.meditation_end = time.time() + st.session_state.meditation_total
    st.success(f"🧘 Meditation started for {minutes} minutes.")
