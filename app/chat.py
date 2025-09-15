import streamlit as st
import supabase_helpers as db

def run_chat(user_id):
    st.title("💬 Healrr Chat")

    # ---------------------------
    # Load profile & mood
    # ---------------------------
    profile = db.get_profile(user_id)
    if not profile:
        st.error("⚠️ No profile found. Please complete onboarding first.")
        return

    name = profile.get("name", "Seeker")
    last_mood = db.get_last_mood(user_id) or profile.get("mood", "🌞 Peaceful")

    st.markdown(f"👋 Welcome back, **{name}** 🌸")
    st.markdown(f"Your current mood: {last_mood}")

    # ---------------------------
    # Get or create session
    # ---------------------------
    session_id = db.get_or_create_session(user_id)

    # ---------------------------
    # Chat Input
    # ---------------------------
    prompt = st.chat_input("Share what's on your heart...")

    if prompt:
        # Save user message
        db.add_message(session_id, "user", prompt)

        # ---------------------------
        # Prepare messages for LLM
        # ---------------------------
        system_prompt = f"""
        You are Healrr 🌿 — a compassionate spiritual guide.

        - Speak gently and warmly, like a healer.
        - Address the user by their name: {name}.
        - Keep in mind their current mood: {last_mood}.
        - Guide them inward, beyond fear and ego, into the silence of their True Self.
        - Avoid being too clinical or robotic. Speak from presence and love.
        """

        history = db.get_messages(session_id)

        messages_for_llm = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages_for_llm.append({"role": msg["role"], "content": msg["content"]})

        # ---------------------------
        # Get response from LLM
        # ---------------------------
        response = db.openrouter_chat(messages_for_llm)

        # Save assistant message
        db.add_message(session_id, "assistant", response.content)

        # Display conversation
        st.chat_message("user").markdown(prompt)
        st.chat_message("assistant").markdown(response.content)

    # ---------------------------
    # Show previous chat history
    # ---------------------------
    history = db.get_messages(session_id)
    for msg in history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])
