import streamlit as st
import pandas as pd
import supabase_helpers as db
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt

def run_journal(user_id):
    st.title("📖 My Healing Journal")
    st.caption("Reflect on your journey through meditations, quotes, and insights 🌿")

    # Inject CSS for fade-in
    st.markdown("""
    <style>
    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }
    .card {
      animation: fadeIn 0.8s;
      padding: 12px;
      margin: 10px 0;
      border-radius: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------
    # Meditation History
    # ------------------------------
    st.subheader("🧘 Meditation History")

    try:
        logs = db.get_meditation_logs(user_id)
    except Exception as e:
        st.error(f"Failed to load meditation history: {e}")
        logs = []

    if logs:
        df = pd.DataFrame(logs)

        # Convert timestamps to UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Show meditation logs as cards
        for row in logs:
            st.markdown(
                f"<div class='card' style='background:#FFF3E0;'>"
                f"🧘 {row['duration']} min<br>"
                f"<span style='font-size:12px; color:gray;'>{row['timestamp']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Summary stats
        total_sessions = len(df)
        total_minutes = df["duration"].sum()
        avg_duration = df["duration"].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", total_sessions)
        with col2:
            st.metric("Total Minutes", total_minutes)
        with col3:
            st.metric("Average Duration", f"{avg_duration:.1f} min")

        # Export button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Meditation History (CSV)",
            data=csv,
            file_name="meditation_history.csv",
            mime="text/csv"
        )

        # ------------------------------
        # AI Insights
        # ------------------------------
        st.markdown("---")
        st.subheader("🌟 Weekly Insights")

        # Filter last 7 days with UTC-aware datetime
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        weekly_df = df[df["timestamp"] >= one_week_ago]

        if not weekly_df.empty:
            weekly_sessions = len(weekly_df)
            weekly_minutes = weekly_df["duration"].sum()
            avg_weekly_duration = weekly_df["duration"].mean()

            st.write(f"✨ In the last 7 days, you completed **{weekly_sessions} sessions** "
                     f"for a total of **{weekly_minutes} minutes**. "
                     f"Your average duration was **{avg_weekly_duration:.1f} min**.")

            # Plot sessions trend
            daily_stats = weekly_df.groupby(weekly_df["timestamp"].dt.date)["duration"].sum()

            fig, ax = plt.subplots()
            daily_stats.plot(kind="bar", ax=ax)
            ax.set_title("🗓️ Meditation Minutes (Last 7 Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Minutes")
            st.pyplot(fig)

            # Gentle reflection text
            st.info("🌿 Healrr reflection: Notice your rhythm. Consistency matters more than duration. "
                    "Even a few minutes daily builds inner silence.")
        else:
            st.info("No sessions in the last 7 days. 🌸 Start with a short 5-minute meditation today.")

    else:
        st.info("No meditation sessions logged yet. Start your journey in the Meditation tab 🌸")

    # ------------------------------
    # Favorite Quotes
    # ------------------------------
    st.markdown("---")
    st.subheader("💫 Favorite Quotes")

    try:
        favorites = db.get_favorites(user_id)
    except Exception as e:
        st.error(f"Failed to load favorites: {e}")
        favorites = []

    if favorites:
        for fav in favorites:
            st.markdown(
                f"<div class='card' style='background:#E3F2FD;'>"
                f"“{fav['content']}”<br>"
                f"<span style='font-size:12px; color:gray;'>— {fav.get('reference','')}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("You haven’t saved any quotes yet. Explore the Quotes section to add some ✨")
