import streamlit as st
import supabase_helpers as db
import random

def run_quotes(user_id):
    st.title("📜 Healrr Quotes")

    # ------------------------------
    # QUOTE OF THE DAY (by mood)
    # ------------------------------
    st.subheader("✨ Quote of the Day")

    # get user profile (to know mood)
    profile = db.get_profile(user_id)
    user_mood = profile["mood"] if profile else "🌞 Peaceful"

    # fetch all quotes
    quotes = db.get_all_quotes()

    if not quotes:
        st.warning("No quotes found. Please add some via Admin panel.")
    else:
        # only pick once per session
        if "quote_of_day" not in st.session_state:
            # 🎯 Filter quotes by mood keyword (basic matching)
            mood_quotes = [q for q in quotes if user_mood.split()[0].lower() in (q.get("content","").lower())]

            # if no match found, fallback to random from all
            if not mood_quotes:
                mood_quotes = quotes

            st.session_state.quote_of_day = random.choice(mood_quotes)

        quote_of_day = st.session_state.quote_of_day
        st.markdown(f"> {quote_of_day['content']} — *{quote_of_day.get('reference','')}*")

        if st.button("⭐ Save to Favorites", key=f"fav-{quote_of_day['id']}"):
            try:
                db.save_favorite_quote(user_id, quote_of_day["id"])
                st.success("Saved to favorites!")
            except Exception as e:
                st.error(f"Failed to save favorite: {e}")

    # ------------------------------
    # FAVORITES SECTION
    # ------------------------------
    st.markdown("---")
    st.subheader("🌟 Your Favorites")

    favorites = db.get_favorites(user_id)
    if not favorites:
        st.info("No favorites yet.")

    else:
        from PIL import Image, ImageDraw, ImageFont
        import io

        st.markdown(f"> {quote_of_day['content']} — *{quote_of_day.get('reference','')}*")

        # Save as Image button
        if st.button("🖼️ Save as Image"):
            try:
                # Create blank image
                img = Image.new("RGB", (800, 400), color=(240, 230, 250))  # light lavender bg
                draw = ImageDraw.Draw(img)

                # Fonts (fallback if custom font missing)
                try:
                    font_quote = ImageFont.truetype("arial.ttf", 28)
                    font_ref = ImageFont.truetype("arial.ttf", 20)
                except:
                    font_quote = ImageFont.load_default()
                    font_ref = ImageFont.load_default()

                # Add text
                text = f"“{quote_of_day['content']}”"
                ref = f"- {quote_of_day.get('reference','')}"
                draw.multiline_text((40, 120), text, font=font_quote, fill=(50, 20, 90))
                draw.text((40, 300), ref, font=font_ref, fill=(80, 60, 120))

                # Export as PNG
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Quote Image",
                    data=buf.getvalue(),
                    file_name="healrr_quote.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Image export failed: {e}")

