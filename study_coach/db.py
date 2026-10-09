"""SQLite persistence for logged study sessions.

The database path comes from the ``STUDY_DB_PATH`` environment variable and defaults
to ``study_coach.db`` in the working directory. The file and table are created on first use.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import asdict, dataclass

import pandas as pd

from .analytics import SESSION_COLUMNS

logger = logging.getLogger("study_coach")

DEFAULT_DB_PATH = "study_coach.db"


@dataclass
class SessionLog:
    date: str
    subjects: str
    hours: float
    focus_score: int
    streak: int


def db_path() -> str:
    """Resolve the SQLite file path at call time so env overrides are honoured."""
    return os.environ.get("STUDY_DB_PATH") or DEFAULT_DB_PATH


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            subjects    TEXT    NOT NULL,
            hours       REAL    NOT NULL,
            focus_score INTEGER NOT NULL,
            streak      INTEGER NOT NULL DEFAULT 0
        )
        """
    )
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
        return pd.DataFrame(columns=SESSION_COLUMNS)
    return pd.DataFrame([dict(r) for r in rows])
