import random
import streamlit as st
import supabase_helpers as db

def run_quotes(user_id: str):
    st.title("📜 Quotes")

    # ---------------------------
    # Quote of the Day
    # ---------------------------
    st.subheader("✨ Quote of the Day")

    quotes = db.get_all_quotes()
    if quotes:
        quote = random.choice(quotes)
        st.markdown(f"> {quote['content']}")
        if "reference" in quote and quote["reference"]:
            st.caption(f"— {quote['reference']}")

        if st.button("❤️ Save to Favorites"):
            db.save_favorite_quote(user_id, quote["id"])
            st.success("Saved to your favorites!")
    else:
        st.warning("No quotes found. Please add some via Admin panel.")

    # ---------------------------
    # Search Quotes
    # ---------------------------
    st.subheader("🔍 Search Quotes")
    keyword = st.text_input("Search by keyword...")
    if keyword:
        results = db.search_quotes(keyword)
        if results:
            for q in results:
                st.markdown(f"> {q['content']}")
                if "reference" in q and q["reference"]:
                    st.caption(f"— {q['reference']}")
        else:
            st.info("No matching quotes found.")

    # ---------------------------
    # Favorites
    # ---------------------------
    st.subheader("❤️ Your Favorite Quotes")
    favs = db.get_favorite_quotes(user_id)
    if favs:
        for q in favs:
            st.markdown(f"> {q['content']}")
            if "reference" in q and q["reference"]:
                st.caption(f"— {q['reference']}")
    else:
        st.info("You don’t have any favorite quotes yet.")
