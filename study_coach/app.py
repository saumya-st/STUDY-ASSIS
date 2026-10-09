"""The Streamlit page. Run via ``streamlit run smart_study.py``."""

from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from .ai import GroqError, generate_study_plan, get_ai_feedback
from .analytics import export_csv, focus_trend, summarize, time_distribution
from .db import SessionLog, load_sessions, save_session
from .ui import inject_css, metric_tile

PAGE_TITLE = "AI Study Coach"
ACCENT = "#e9648b"


def _study_window_hours(start_time: datetime.time, end_time: datetime.time) -> float:
    """Hours between two clock times; a window ending at or before its start wraps to next day."""
    start_dt = datetime.datetime.combine(datetime.date.today(), start_time)
    end_dt = datetime.datetime.combine(datetime.date.today(), end_time)
    if end_dt <= start_dt:
        end_dt += datetime.timedelta(days=1)
    return round((end_dt - start_dt).seconds / 3600, 2)


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )
    inject_css()

    defaults = {"plan": None, "feedback": None, "last_subjects": [], "last_hours": 0.0}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    # Header
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.markdown("# 🧠 AI Study Coach")
        st.markdown(
            "<p style='color:#b07a8a;font-size:0.9rem;margin-top:-0.5rem;'>"
            "Your personalised academic partner</p>",
            unsafe_allow_html=True,
        )
    with col_date:
        st.markdown(
            "<p style='text-align:right;color:#b07a8a;font-size:0.8rem;padding-top:1.2rem;'>"
            f"{datetime.date.today().strftime('%A, %d %b %Y')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Metrics
    df_all = load_sessions()
    stats = summarize(df_all)
    streak = stats["streak"]
    avg_focus = f"{stats['avg_focus']:.1f}" if stats["avg_focus"] is not None else "—"
    total_hours = f"{stats['total_hours']:.1f}h" if not df_all.empty else "—"

    st.markdown(
        '<div class="metric-row">'
        + metric_tile(f"🔥 {streak}", "Day Streak")
        + metric_tile(str(stats["sessions"]), "Sessions")
        + metric_tile(avg_focus, "Avg Focus")
        + metric_tile(total_hours, "Hours Logged")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Input form
    st.markdown(
        '<div class="section-label">📚 Plan Your Study Session</div>', unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        subjects_raw = st.text_input(
            "Subjects",
            placeholder="e.g. Math, Physics, DSA",
            help="Enter subjects separated by commas",
        )
        subjects = [s.strip() for s in subjects_raw.split(",") if s.strip()] if subjects_raw else []
    with col2:
        start_time = st.time_input("Study starts at", value=datetime.time(9, 0))
    with col3:
        end_time = st.time_input("Study ends at", value=datetime.time(12, 0))

    hours = _study_window_hours(start_time, end_time)
    st.markdown(
        f"<p style='color:{ACCENT};font-size:0.85rem;font-weight:600;'>"
        f"⏱ Total study time: <b>{hours} hours</b> "
        f"({start_time.strftime('%I:%M %p')} → {end_time.strftime('%I:%M %p')})</p>",
        unsafe_allow_html=True,
    )

    priorities: dict[str, int] = {}
    if subjects:
        st.markdown("**Set Priority for each subject (1 = low, 5 = high):**")
        cols = st.columns(len(subjects))
        for i, subj in enumerate(subjects):
            with cols[i]:
                priorities[subj] = st.slider(subj, 1, 5, 3, key=f"pri_{subj}")

    generate = st.button("🚀 Generate My Study Plan", use_container_width=True)

    # Plan generation
    if generate:
        if not subjects:
            st.warning("⚠️ Please enter at least one subject above.")
        else:
            try:
                with st.spinner("✨ Crafting your personalised plan…"):
                    plan = generate_study_plan(subjects, hours, priorities, start_time, end_time)
                st.session_state.plan = plan
                st.session_state.last_subjects = subjects
                st.session_state.last_hours = hours
            except GroqError as e:
                st.error(f"⚠️ {e}")

            if st.session_state.plan:
                history = df_all["focus_score"].tail(5).tolist() if not df_all.empty else []
                try:
                    with st.spinner("💬 Getting AI feedback…"):
                        st.session_state.feedback = get_ai_feedback(st.session_state.plan, history)
                except GroqError as e:
                    st.error(f"⚠️ {e}")

    # Show plan
    if st.session_state.plan:
        st.markdown("---")
        st.markdown('<div class="section-label">📅 Your Study Plan</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="plan-card">{st.session_state.plan.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.feedback:
            with st.expander("💬 AI Feedback on your plan", expanded=True):
                st.markdown(st.session_state.feedback)

        if len(st.session_state.last_subjects) > 1:
            st.markdown(
                '<div class="section-label">📊 Time Distribution</div>', unsafe_allow_html=True
            )
            times = time_distribution(
                st.session_state.last_subjects, priorities, st.session_state.last_hours
            )
            st.bar_chart(
                pd.DataFrame.from_dict(times, orient="index", columns=["Hours"]), color=ACCENT
            )

        st.success("✅ Plan ready. Open the first block and start. You've got this! ⚡")

    # Session logger
    st.markdown("---")
    st.markdown('<div class="section-label">📈 Log Today\'s Session</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**How focused were you today?**")
        focus_score = st.slider("Focus Score", 1, 10, 7, label_visibility="collapsed")
        st.markdown(
            "<p style='color:#b07a8a;font-size:0.8rem;margin-top:-0.5rem;'>"
            f"Score: {focus_score}/10</p>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#fff5f7;border:1px solid #f5b8cb;border-radius:14px;"
            "padding:1rem;text-align:center;'>"
            "<div style='font-family:Playfair Display,serif;font-size:2.5rem;font-weight:800;"
            f"color:{ACCENT};line-height:1;'>{focus_score * 10}%</div>"
            "<div style='color:#b07a8a;font-size:0.7rem;letter-spacing:0.1em;margin-top:0.25rem;'>"
            "PRODUCTIVITY</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    save_col, export_col = st.columns(2)
    with save_col:
        if st.button("💾 Save Session", use_container_width=True):
            if not st.session_state.last_subjects:
                st.warning("⚠️ Generate a plan first so we know what you studied.")
            else:
                save_session(
                    SessionLog(
                        date=str(datetime.date.today()),
                        subjects=", ".join(st.session_state.last_subjects),
                        hours=st.session_state.last_hours,
                        focus_score=focus_score,
                        streak=streak + 1,
                    )
                )
                st.success("🔥 Session saved! Keep the streak alive!")
                st.rerun()

    # History
    if not df_all.empty:
        st.markdown("---")
        with st.expander("📜 Session History & Export", expanded=False):
            display_df = df_all.copy()
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

            with export_col:
                st.download_button(
                    "⬇️ Export CSV",
                    data=export_csv(df_all),
                    file_name=f"study_log_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown("**📈 Focus Score Trend**")
            st.line_chart(focus_trend(df_all), color=ACCENT)

    st.markdown("---")
    st.caption("AI Study Coach · Built with Streamlit + Groq · 2025 🌸")
