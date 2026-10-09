"""Pure, framework-free calculations used by the UI and covered by unit tests."""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict

import pandas as pd

SESSION_COLUMNS = ["date", "subjects", "hours", "focus_score", "streak"]


class Summary(TypedDict):
    streak: int
    sessions: int
    avg_focus: float | None
    total_hours: float


def compute_streak(dates: Iterable[datetime.date], today: datetime.date) -> int:
    """Count consecutive days with at least one session, ending on ``today``.

    Duplicate dates count once. If there is no session on ``today`` the streak is 0.
    """
    unique = set(dates)
    streak = 0
    expected = today
    while expected in unique:
        streak += 1
        expected -= datetime.timedelta(days=1)
    return streak


def session_dates(df: pd.DataFrame) -> list[datetime.date]:
    """Extract session dates from a sessions DataFrame (``date`` column as ISO strings)."""
    if df.empty:
        return []
    return [ts.date() for ts in pd.to_datetime(df["date"])]


def summarize(df: pd.DataFrame, today: datetime.date | None = None) -> Summary:
    """Return the four headline metrics shown at the top of the app."""
    today = today or datetime.date.today()
    if df.empty:
        return Summary(streak=0, sessions=0, avg_focus=None, total_hours=0.0)
    return Summary(
        streak=compute_streak(session_dates(df), today),
        sessions=int(len(df)),
        avg_focus=float(df["focus_score"].mean()),
        total_hours=float(df["hours"].sum()),
    )


def time_distribution(
    subjects: Sequence[str],
    priorities: Mapping[str, int],
    hours: float,
    default_priority: int = 3,
) -> dict[str, float]:
    """Split ``hours`` across subjects proportionally to their priority (rounded to 2 dp)."""
    if not subjects:
        return {}
    weights = {s: priorities.get(s, default_priority) for s in subjects}
    total = sum(weights.values())
    if total <= 0:
        return {s: 0.0 for s in subjects}
    return {s: round(hours * w / total, 2) for s, w in weights.items()}


def focus_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Mean focus score per day, indexed by date, for the trend line chart."""
    trend = df[["date", "focus_score"]].copy()
    trend["date"] = pd.to_datetime(trend["date"])
    return trend.groupby("date")["focus_score"].mean().reset_index().set_index("date")


def export_csv(df: pd.DataFrame) -> bytes:
    """Serialize a sessions DataFrame to UTF-8 CSV bytes for the download button."""
    return df.to_csv(index=False).encode()
