"""Groq chat-completion helpers.

This module never touches Streamlit widgets. Failures are raised as :class:`GroqError`
so the UI layer decides how to display them.
"""

from __future__ import annotations

import datetime
import logging
import os

import requests

logger = logging.getLogger("study_coach")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT = 30


class GroqError(Exception):
    """Raised when a Groq request cannot be made or fails."""


def _secret(name: str) -> str:
    """Read a value from Streamlit secrets, returning "" when unavailable."""
    try:
        import streamlit as st

        return str(st.secrets[name])
    except Exception:
        return ""


def resolve_api_key() -> str:
    """Environment variable first, then Streamlit secrets."""
    return os.environ.get("GROQ_API_KEY", "") or _secret("GROQ_API_KEY")


def resolve_model() -> str:
    """Environment variable first, then Streamlit secrets, then the default model."""
    return os.environ.get("GROQ_MODEL", "") or _secret("GROQ_MODEL") or DEFAULT_MODEL


def chat(prompt: str, system: str = "") -> str:
    """Send a single-turn chat completion to Groq and return the reply text."""
    api_key = resolve_api_key()
    if not api_key:
        raise GroqError("GROQ_API_KEY not found. Add it to your .env file or Streamlit secrets.")
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": resolve_model(),
                "messages": [
                    {
                        "role": "system",
                        "content": system or "You are a helpful academic assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout as e:
        raise GroqError("Request timed out. Please try again.") from e
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "?")
        body = getattr(e.response, "text", "")
        raise GroqError(f"Groq API error: {status} - {body}") from e
    except Exception as e:
        logger.error("Groq error: %s", e)
        raise GroqError(f"Error: {e}") from e


def generate_study_plan(
    subjects: list[str],
    hours: float,
    priorities: dict[str, int],
    start_time: datetime.time = datetime.time(9, 0),
    end_time: datetime.time = datetime.time(12, 0),
) -> str:
    """Ask the model for a time-blocked schedule; ``subjects`` is kept for call compatibility."""
    priority_str = ", ".join(f"{s} (priority {p}/5)" for s, p in priorities.items())
    start = start_time.strftime("%I:%M %p")
    end = end_time.strftime("%I:%M %p")
    prompt = f"""
Create a detailed, time-blocked study schedule.

Subjects with priorities: {priority_str}
Study window: {start} to {end} ({hours} hours total)

Rules:
- Start the schedule exactly at {start}
- End the schedule exactly at {end}
- Allocate more time to higher-priority subjects
- Include 5-minute breaks every 45 minutes (Pomodoro-style)
- Add one 15-minute review block before end time
- Format as a clean numbered schedule with real clock times (e.g. 9:00 AM - 9:45 AM)
- End with a short motivational sentence
"""
    return chat(prompt, system="You are a concise, encouraging academic coach.")


def get_ai_feedback(plan: str, focus_history: list[int]) -> str:
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
    return chat(prompt, system="You are a supportive but honest study buddy.")
