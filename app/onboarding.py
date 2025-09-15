import streamlit as st
import supabase_helpers as db

def run_onboarding(user_id):
    st.title("🌸 Welcome to Healrr")

    # Step 1 → Welcome
    if "step" not in st.session_state:
        st.session_state.step = 1

    # Step 1: Welcome
    if st.session_state.step == 1:
        st.markdown("Welcome to **Healrr** 🌿 — your companion for inner peace and spiritual growth.")
        if st.button("Begin Journey"):
            st.session_state.step = 2

    # Step 2: Intro
    elif st.session_state.step == 2:
        st.markdown("Healrr will gently guide you inward, beyond fear and ego, into the silence of your True Self.")
        if st.button("Continue"):
            st.session_state.step = 3

    # Step 3: Ask Name
    elif st.session_state.step == 3:
        name = st.text_input("✨ What may I call you?")
        if st.button("Next") and name:
            st.session_state.name = name
            st.session_state.step = 4

    # Step 4: Ask Mood
    elif st.session_state.step == 4:
        st.markdown("🌈 How are you feeling right now?")
        mood = st.radio("Choose your current state:", ["🌞 Peaceful", "🌧️ Overwhelmed", "🌙 Reflective"])
        if st.button("Next"):
            st.session_state.mood = mood
            st.session_state.step = 5

    # Step 5: Reminder toggle
    elif st.session_state.step == 5:
        st.markdown("🧘 Would you like Healrr to gently remind you every 30 minutes to breathe consciously?")
        reminder = st.checkbox("Yes, remind me every 30 minutes", value=True)
        if st.button("Finish"):
            st.session_state.reminder = reminder

            # ✅ Save onboarding data to Supabase using real user_id
            db.save_profile(user_id, st.session_state.name, st.session_state.mood, st.session_state.reminder)

            st.success("✅ Your journey begins now...")
            st.session_state.step = 6
            st.session_state.profile_complete = True

    # Step 6: Done → Enter Chat
    elif st.session_state.step == 6:
        st.markdown(f"Welcome, **{st.session_state.name}** 🌸")
        st.markdown("Whenever you’re ready, share what’s on your mind. I am here with you.")
        if st.button("Enter Chat"):
            st.switch_page("app/chat.py")
