# import streamlit as st
# import supabase_helpers as db

# def run_onboarding(user_id: str):
#     st.title("🌸 Welcome to Healrr")

#     # Track step in session state
#     if "onboard_step" not in st.session_state:
#         st.session_state.onboard_step = 1

#     step = st.session_state.onboard_step
#     total_steps = 5
#     st.progress((step - 1) / total_steps)

#     # Step 1 → Welcome
#     if step == 1:
#         st.markdown("<h3 style='text-align:center;'>🌸 Welcome to Healrr 🌸</h3>", unsafe_allow_html=True)

#         st.header("Welcome to **Healrr** 🌿")
#         st.write("Your companion for inner peace and spiritual growth.")
#         if st.button("Begin Journey"):
#             st.session_state.onboard_step = 2
#             st.rerun()

#     # Step 2 → Intro
#     elif step == 2:
#         st.subheader("💫 About Healrr")
#         st.write(
#             "Healrr will gently guide you inward, beyond fear and ego, "
#             "into the silence of your True Self. "
#             "Here, you’ll find peace and clarity."
#         )
#         if st.button("Continue"):
#             st.session_state.onboard_step = 3
#             st.rerun()

#     # Step 3 → Ask Name
#     elif step == 3:
#         st.subheader("✨ What may I call you?")
#         name = st.text_input("Enter your name:", key="onboard_name")
#         if st.button("Next"):
#             if not name.strip():
#                 st.warning("Please enter your name to continue.")
#             else:
#                 st.session_state.onboard_name = name.strip()
#                 st.session_state.onboard_step = 4
#                 st.rerun()

#     # Step 4 → Ask Mood
#     elif step == 4:
#         st.subheader("🌈 How are you feeling right now?")
#         mood = st.radio(
#             "Choose your current state:",
#             ["🌞 Peaceful", "😌 Calm", "😔 Sad", "🔥 Stressed"],
#             key="onboard_mood"
#         )
#         if st.button("Next"):
#             st.session_state.onboard_mood = mood
#             st.session_state.onboard_step = 5
#             st.rerun()

#     # Step 5 → Reminder preference
#     elif step == 5:
#         st.subheader("🧘 Conscious Breathing Reminder")
#         reminder = st.checkbox(
#             "Yes, remind me every 30 minutes to take 5 conscious breaths",
#             value=True,
#             key="onboard_reminder"
#         )
#         if st.button("Finish"):
#             try:
#                 db.save_profile(
#                     user_id,
#                     st.session_state.get("onboard_name", "Seeker"),
#                     st.session_state.get("onboard_mood", "🌞 Peaceful"),
#                     reminder,
#                 )
#                 st.success("✅ Your journey begins now...")
#                 st.session_state.onboard_step = 6
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Failed to save your profile: {e}")

#     # Step 6 → Done
#     elif step == 6:
#         st.header(f"Welcome, **{st.session_state.get('onboard_name','Seeker')}** 🌸")
#         st.write("Whenever you’re ready, share what’s on your mind. I am here with you.")
#         if st.button("Enter Chat"):
#             st.session_state.onboard_complete = True
#             # Reset steps for future new users
#             st.session_state.onboard_step = 1
#             st.rerun()


import streamlit as st
import supabase_helpers as db

def run_onboarding(user_id: str):
    st.markdown(
        "<h2 style='text-align:center;'>🌸 Welcome to Healrr 🌸</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Your companion for inner silence and presence</p>",
        unsafe_allow_html=True
    )

    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1

    step = st.session_state.onboarding_step

    # ---------------------------
    # Step 1 — Name
    # ---------------------------
    if step == 1:
        st.markdown("### What should Healrr call you?")
        name = st.text_input("Your Name", placeholder="e.g., Arjun")
        if st.button("Next ➡️", use_container_width=True) and name.strip():
            db.update_profile(user_id, {"name": name.strip()})
            st.session_state.onboarding_step = 2
            st.rerun()

    # ---------------------------
    # Step 2 — Current Mood
    # ---------------------------
    elif step == 2:
        st.markdown("### How are you feeling right now?")
        mood = st.radio(
            "Choose your current mood",
            ["🌞 Peaceful", "🌧️ Anxious", "🔥 Stressed", "🌙 Calm", "💫 Inspired"],
            horizontal=True,
        )
        if st.button("Next ➡️", use_container_width=True):
            db.update_profile(user_id, {"mood": mood})
            st.session_state.onboarding_step = 3
            st.rerun()

    # ---------------------------
    # Step 3 — Intention
    # ---------------------------
    elif step == 3:
        st.markdown("### What brings you here?")
        intention = st.text_area(
            "Write a few words about your intention (optional)",
            placeholder="e.g., I want to find calm in daily life...",
        )
        if st.button("Finish 🌿", use_container_width=True):
            db.update_profile(user_id, {"intention": intention})
            st.session_state.onboarding_step = 4
            st.rerun()

    # ---------------------------
    # Step 4 — Completed
    # ---------------------------
    elif step == 4:
        st.success("✨ Thank you for sharing. Healrr will now walk with you on this journey.")
        st.markdown("<p style='text-align:center;'>You may now begin chatting with Healrr 🌿</p>", unsafe_allow_html=True)
    
        if st.button("Enter Chat 🌿", use_container_width=True):
            # Save onboarding complete in database
            db.update_profile(user_id, {"onboarded": True})
    
            # Also update session_state
            st.session_state["onboarding_complete"] = True
    
            # Rerun to unlock main app
            st.rerun()
