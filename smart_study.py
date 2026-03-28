"""
AI Study Coach — Pink Theme, Groq API
Run: streamlit run smart_study.py
"""

import streamlit as st
import os
import datetime
import sqlite3
import logging
import requests
from typing import Optional
from dataclasses import dataclass, asdict

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("study_coach")

PAGE_TITLE = "AI Study Coach"
DB_PATH    = "study_coach.db"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
MODEL      = "llama-3.3-70b-versatile"


# ── Data layer ────────────────────────────────────────────────────────────────

@dataclass
class SessionLog:
    date: str
    subjects: str
    hours: float
    focus_score: int
    streak: int


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            subjects    TEXT    NOT NULL,
            hours       REAL    NOT NULL,
            focus_score INTEGER NOT NULL,
            streak      INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def save_session(log: SessionLog) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (date, subjects, hours, focus_score, streak) VALUES (?,?,?,?,?)",
            (log.date, log.subjects, log.hours, log.focus_score, log.streak),
        )
    logger.info("Session saved: %s", asdict(log))


def load_sessions() -> pd.DataFrame:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT date, subjects, hours, focus_score, streak FROM sessions ORDER BY id"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "subjects", "hours", "focus_score", "streak"])
    return pd.DataFrame([dict(r) for r in rows])


def get_current_streak() -> int:
    df = load_sessions()
    if df.empty:
        return 0
    dates = sorted(pd.to_datetime(df["date"].unique()), reverse=True)
    streak, expected = 0, pd.Timestamp(datetime.date.today())
    for d in dates:
        if d.date() == expected.date():
            streak += 1
            expected -= pd.Timedelta(days=1)
        else:
            break
    return streak


# ── AI helpers ────────────────────────────────────────────────────────────────

def _chat(prompt: str, system: str = "") -> Optional[str]:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = ""
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Add it to your .env file or Streamlit secrets.")
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful academic assistant."},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ Groq API error: {e.response.status_code} — {e.response.text}")
        return None
    except Exception as e:
        logger.error("Groq error: %s", e)
        st.error(f"⚠️ Error: {e}")
        return None


def generate_study_plan(subjects: list[str], hours: float, priorities: dict[str, int]) -> Optional[str]:
    priority_str = ", ".join(f"{s} (priority {p}/5)" for s, p in priorities.items())
    prompt = f"""
Create a detailed, time-blocked study schedule.

Subjects with priorities: {priority_str}
Total available time: {hours} hours

Rules:
- Allocate more time to higher-priority subjects
- Include 5-minute breaks every 45 minutes (Pomodoro-style)
- Add one 15-minute review block at the end
- Format as a clean numbered schedule with times
- End with a short motivational sentence
"""
    return _chat(prompt, system="You are a concise, encouraging academic coach.")


def get_ai_feedback(plan: str, focus_history: list[int]) -> Optional[str]:
    avg = sum(focus_history) / len(focus_history) if focus_history else None
    avg_note = f"The student's average recent focus score is {avg:.1f}/10." if avg else ""
    prompt = f"""
Study plan:
{plan}

{avg_note}

Give exactly 3 bullet points of feedback:
- One strength
- One improvement
- One quick tip
Keep it warm, human, under 120 words.
"""
    return _chat(prompt, system="You are a supportive but honest study buddy.")


# ── UI helpers ────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    /* Hide Streamlit default UI */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; letter-spacing: -0.01em; }

    .stApp { background: #fdf0f3; color: #2d1a20; }

    /* Main content top padding */
    .block-container { padding-top: 2rem !important; }

    /* Metric tiles */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-tile {
        flex: 1; min-width: 120px;
        background: #fff5f7;
        border: 1px solid #f5b8cb;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(233,100,139,0.07);
    }
    .metric-tile .val {
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 800;
        color: #e9648b; line-height: 1;
    }
    .metric-tile .lbl {
        font-size: 0.68rem; color: #b07a8a;
        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem;
    }

    /* Input form card */
    .form-card {
        background: #fff5f7;
        border: 1px solid #f5b8cb;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 16px rgba(233,100,139,0.08);
        position: relative;
        overflow: hidden;
    }
    .form-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e9648b, #f48fb1, #ce93d8);
    }

    /* Plan card */
    .plan-card {
        background: #fff5f7;
        border: 1px solid #f5b8cb;
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 16px rgba(233,100,139,0.08);
        position: relative;
        overflow: hidden;
        line-height: 1.9;
        font-size: 0.95rem;
        color: #2d1a20;
    }
    .plan-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e9648b, #f48fb1, #ce93d8);
    }

    /* Buttons */
    .stButton > button {
        background: #e9648b !important;
        color: #fff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 2px 10px rgba(233,100,139,0.25) !important;
        transition: background 0.15s !important;
        width: 100%;
    }
    .stButton > button:hover { background: #d4476d !important; }

    /* Inputs */
    .stTextInput input, .stNumberInput input {
        background: #fff5f7 !important;
        border-color: #f5b8cb !important;
        border-radius: 8px !important;
        color: #2d1a20 !important;
        font-size: 0.95rem !important;
    }
    .stTextInput label, .stNumberInput label, .stSlider label {
        color: #2d1a20 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* Slider */
    [data-testid="stSlider"] { padding: 0.25rem 0; }

    /* Alerts */
    .stSuccess {
        background: #fce4ec !important;
        border-color: #f48fb1 !important;
        color: #880e4f !important;
        border-radius: 10px !important;
    }
    .stWarning { background: #fff8e1 !important; border-radius: 10px !important; }
    .stError { border-radius: 10px !important; }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        color: #2d1a20 !important;
        background: #fff5f7 !important;
        border: 1px solid #f5b8cb !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background: #fff5f7 !important;
        border: 1px solid #f5b8cb !important;
        border-top: none !important;
    }

    hr { border-color: #f5b8cb !important; }

    .stDownloadButton > button {
        background: transparent !important;
        color: #e9648b !important;
        border: 1.5px solid #e9648b !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    .stDownloadButton > button:hover { background: #fce4ec !important; }

    .stCaption { color: #b07a8a !important; font-size: 0.75rem !important; }
    .stDataFrame { border: 1px solid #f5b8cb !important; border-radius: 10px !important; }
    p, li { color: #2d1a20; }

    /* Section labels */
    .section-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d1a20;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_tile(value: str, label: str) -> str:
    return f'<div class="metric-tile"><div class="val">{value}</div><div class="lbl">{label}</div></div>'


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
    )
    inject_css()

    for k, v in {"plan": None, "feedback": None, "last_subjects": [], "last_hours": 0.0}.items():
        st.session_state.setdefault(k, v)

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.markdown("# 🧠 AI Study Coach")
        st.markdown("<p style='color:#b07a8a;font-size:0.9rem;margin-top:-0.5rem;'>Your personalised academic partner</p>", unsafe_allow_html=True)
    with col_date:
        st.markdown(f"<p style='text-align:right;color:#b07a8a;font-size:0.8rem;padding-top:1.2rem;'>{datetime.date.today().strftime('%A, %d %b %Y')}</p>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────────────────────────
    df_all      = load_sessions()
    streak      = get_current_streak()
    avg_focus   = f"{df_all['focus_score'].mean():.1f}" if not df_all.empty else "—"
    total_hours = f"{df_all['hours'].sum():.1f}h"       if not df_all.empty else "—"

    st.markdown(
        '<div class="metric-row">'
        + metric_tile(f"🔥 {streak}", "Day Streak")
        + metric_tile(str(len(df_all)), "Sessions")
        + metric_tile(avg_focus, "Avg Focus")
        + metric_tile(total_hours, "Hours Logged")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Input form — always visible, no sidebar ───────────────────────────────
    st.markdown('<div class="section-label">📚 Plan Your Study Session</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        subjects_raw = st.text_input("Subjects", placeholder="e.g. Math, Physics, DSA",
                                     help="Enter subjects separated by commas")
        subjects = [s.strip() for s in subjects_raw.split(",") if s.strip()] if subjects_raw else []
    with col2:
        hours = st.number_input("Total hours available", min_value=0.5, max_value=16.0, value=3.0, step=0.5)

    priorities: dict[str, int] = {}
    if subjects:
        st.markdown("**Set Priority for each subject (1 = low, 5 = high):**")
        cols = st.columns(len(subjects))
        for i, subj in enumerate(subjects):
            with cols[i]:
                priorities[subj] = st.slider(subj, 1, 5, 3, key=f"pri_{subj}")

    generate = st.button("🚀 Generate My Study Plan", use_container_width=True)

    # ── Plan generation ───────────────────────────────────────────────────────
    if generate:
        if not subjects:
            st.warning("⚠️ Please enter at least one subject above.")
        else:
            with st.spinner("✨ Crafting your personalised plan…"):
                plan = generate_study_plan(subjects, hours, priorities)
                if plan:
                    st.session_state.plan          = plan
                    st.session_state.last_subjects = subjects
                    st.session_state.last_hours    = hours

            if st.session_state.plan:
                with st.spinner("💬 Getting AI feedback…"):
                    history = df_all["focus_score"].tail(5).tolist() if not df_all.empty else []
                    st.session_state.feedback = get_ai_feedback(st.session_state.plan, history)

    # ── Show plan ─────────────────────────────────────────────────────────────
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
            st.markdown('<div class="section-label">📊 Time Distribution</div>', unsafe_allow_html=True)
            subj_list      = st.session_state.last_subjects
            total_priority = sum(priorities.get(s, 3) for s in subj_list)
            times = {s: round(st.session_state.last_hours * priorities.get(s, 3) / total_priority, 2) for s in subj_list}
            st.bar_chart(pd.DataFrame.from_dict(times, orient="index", columns=["Hours"]), color="#e9648b")

        st.success("✅ Plan ready. Open the first block and start. You've got this! ⚡")

    # ── Session logger ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">📈 Log Today\'s Session</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**How focused were you today?**")
        focus_score = st.slider("Focus Score", 1, 10, 7, label_visibility="collapsed")
        st.markdown(f"<p style='color:#b07a8a;font-size:0.8rem;margin-top:-0.5rem;'>Score: {focus_score}/10</p>", unsafe_allow_html=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#fff5f7;border:1px solid #f5b8cb;border-radius:14px;padding:1rem;text-align:center;'>"
            f"<div style='font-family:Playfair Display,serif;font-size:2.5rem;font-weight:800;color:#e9648b;line-height:1;'>{focus_score * 10}%</div>"
            f"<div style='color:#b07a8a;font-size:0.7rem;letter-spacing:0.1em;margin-top:0.25rem;'>PRODUCTIVITY</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    save_col, export_col = st.columns(2)
    with save_col:
        if st.button("💾 Save Session", use_container_width=True):
            if not st.session_state.last_subjects:
                st.warning("⚠️ Generate a plan first so we know what you studied.")
            else:
                save_session(SessionLog(
                    date=str(datetime.date.today()),
                    subjects=", ".join(st.session_state.last_subjects),
                    hours=st.session_state.last_hours,
                    focus_score=focus_score,
                    streak=streak + 1,
                ))
                st.success("🔥 Session saved! Keep the streak alive!")
                st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    if not df_all.empty:
        st.markdown("---")
        with st.expander("📜 Session History & Export", expanded=False):
            display_df = df_all.copy()
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

            with export_col:
                st.download_button(
                    "⬇️ Export CSV",
                    data=df_all.to_csv(index=False).encode(),
                    file_name=f"study_log_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown("**📈 Focus Score Trend**")
            trend = df_all[["date", "focus_score"]].copy()
            trend["date"] = pd.to_datetime(trend["date"])
            trend = trend.groupby("date")["focus_score"].mean().reset_index().set_index("date")
            st.line_chart(trend, color="#e9648b")

    st.markdown("---")
    st.caption("AI Study Coach · Built with Streamlit + Groq · 2025 🌸")


if __name__ == "__main__":
    main()