import streamlit as st
import time
import supabase_helpers as db

def run_meditation():
    st.title("🧘 Healrr Meditation Timer")

    st.markdown("Take a moment to connect with your inner Being 🌿")

    # Meditation duration input
    minutes = st.number_input("Set meditation duration (minutes):", min_value=1, max_value=60, value=5)

    if st.button("Start Meditation"):
        st.session_state.meditation_active = True
        st.session_state.meditation_end = time.time() + minutes * 60
        st.success(f"🧘 Meditation started for {minutes} minutes.")

    # Show countdown if active
    if st.session_state.get("meditation_active", False):
        remaining = int(st.session_state.meditation_end - time.time())
        if remaining > 0:
            mins, secs = divmod(remaining, 60)
            st.metric("⏳ Time Remaining", f"{mins:02d}:{secs:02d}")
        else:
            st.balloons()
            st.success("🌸 Meditation complete. Be still and rest in your true Self.")
            st.session_state.meditation_active = False

    # Conscious reminder (every 30 minutes toggle)
    st.subheader("Conscious Reminder")
    if "reminder_enabled" not in st.session_state:
        st.session_state.reminder_enabled = False

    reminder_toggle = st.checkbox("Enable 30-min Conscious Breath Reminder", value=st.session_state.reminder_enabled)
    st.session_state.reminder_enabled = reminder_toggle

    if reminder_toggle:
        st.info("Healrr will remind you every 30 minutes to take 5 conscious breaths 🌬️")
