import streamlit as st
import supabase_helpers as db
from rag_helpers import search_index

# ---------------------------
# Build Healrr system prompt
# ---------------------------
def _build_system_prompt(name: str, last_mood: str, mood_summary: str = "") -> str:
    return f"""
You are Healrr — a compassionate spiritual guide.
- Address the user by their name: {name}
- Keep in mind their current mood: {last_mood}
- Recent mood summary: {mood_summary or 'No notable trend.'}
- Purpose: Guide the user inward into silence and presence. Help calm the mind and act from peace, not ego.
- Tone: warm, compassionate, and ego-free.
- Always guide user inward toward the silence of the True Self, which mystics called God.
- Use short, reflective sentences.
- Avoid intellectual debate or over-explaining.
- Prefer gentle practices like conscious breathing, observing silence, or offering a quote.
- Speak as if offering direct presence, not advice.
""".strip()

# ---------------------------
# Run Chat Page
# ---------------------------
def run_chat(user_id: str):
    st.title("💬 Healrr Chat")
    st.caption("Your gentle companion for spiritual presence 🌿")

    # Ensure session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Profile & mood
    profile = db.get_profile(user_id)
    name = profile.get("name", "Beloved") if profile else "Beloved"
    last_mood = profile.get("mood", "🌞 Peaceful") if profile else "🌞 Peaceful"

    mood_summary = ""  # placeholder for future mood analysis

    # Inject CSS for fade-in animation
    st.markdown("""
    <style>
    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }
    </style>
    """, unsafe_allow_html=True)

    # Render chat messages
    for m in st.session_state.messages:
        if m["role"] == "user":
            message_html = (
                f"<div style='background:#E0F7FA; padding:12px; border-radius:12px; "
                f"margin:6px 0; text-align:right; color:#004D40;'>{m['content']}</div>"
            )
        else:
            message_html = (
                f"<div style='background:#F3E5F5; padding:12px; border-radius:12px; "
                f"margin:6px 0; text-align:left; color:#4A148C;'>{m['content']}</div>"
            )

        st.markdown(f"<div style='animation: fadeIn 0.6s;'>{message_html}</div>", unsafe_allow_html=True)

    # Input box
    prompt = st.chat_input("Type your reflection or question here...")
    if prompt:
        # Show immediately
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Build system prompt
        system_prompt = _build_system_prompt(name, last_mood, mood_summary)

        # RAG context
        try:
            retrieved_chunks = search_index(prompt, top_k=3)
            context_text = "\n\n".join(retrieved_chunks)
        except FileNotFoundError:
            context_text = "No vector index found yet. Answering without context."
        except Exception:
            context_text = ""

        # LLM messages
        messages_for_llm = [{"role": "system", "content": system_prompt}]
        if context_text:
            messages_for_llm.append({"role": "system", "content": f"Relevant context:\n{context_text}"})

        # Few-shot style examples
        style_examples = [
            {"role": "user", "content": "I feel anxious."},
            {"role": "assistant", "content": "Beloved, pause. Take 5 conscious breaths. Notice the silence that holds even your anxiety."},
            {"role": "user", "content": "How can I connect with God?"},
            {"role": "assistant", "content": "You don’t need to reach anywhere. Sit quietly, breathe, and notice the still presence already within you."},
        ]
        messages_for_llm.extend(style_examples)

        # Recent history
        recent_history = st.session_state.messages[-8:]
        for m in recent_history:
            messages_for_llm.append({"role": m["role"], "content": m["content"]})

        # Call LLM with loader spinner
        try:
            with st.spinner("🌿 Healrr is reflecting..."):
                response = db.openrouter_chat(messages_for_llm)
            answer = response.content
        except Exception as e:
            answer = f"⚠️ Healrr could not respond: {e}"

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Persist to Supabase
        session_id = db.get_or_create_session(user_id)
        db.add_message(session_id, "user", prompt)
        db.add_message(session_id, "assistant", answer)

        # Force rerun so new message appears styled
        st.rerun()

    # ------------------------------
    # Session Actions
    # ------------------------------
    st.markdown("---")
    with st.expander("⚙️ Session actions"):
        session_id = db.get_or_create_session(user_id)

        new_title = st.text_input("Session Title", value="Reflection")
        if st.button("⭐ Save Reflection", help="Save this session with a custom title"):
            try:
                db.supabase.table("sessions").update({
                    "title": new_title,
                    "is_reflection": True
                }).eq("id", session_id).execute()
                st.success(f"✨ Session saved as '{new_title}'")
                st.balloons()
            except Exception as e:
                st.error(f"Unable to save session: {e}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared.")
            st.rerun()
