import pandas as pd
import pytest

from study_coach import db
from study_coach.analytics import SESSION_COLUMNS, export_csv


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("STUDY_DB_PATH", str(path))
    return path


def test_db_path_from_env(tmp_db):
    assert db.db_path() == str(tmp_db)


def test_db_path_default(monkeypatch):
    monkeypatch.delenv("STUDY_DB_PATH", raising=False)
    assert db.db_path() == "study_coach.db"


def test_load_empty_returns_columns(tmp_db):
    df = db.load_sessions()
    assert df.empty
    assert list(df.columns) == SESSION_COLUMNS


def test_save_load_round_trip(tmp_db):
    logs = [
        db.SessionLog("2026-10-08", "Math", 2.0, 8, 1),
        db.SessionLog("2026-10-09", "Math, Physics", 1.5, 6, 2),
    ]
    for log in logs:
        db.save_session(log)

    df = db.load_sessions()
    assert tmp_db.exists()
    assert len(df) == 2
    assert list(df.columns) == SESSION_COLUMNS
    assert df["date"].tolist() == ["2026-10-08", "2026-10-09"]
    assert df["subjects"].tolist() == ["Math", "Math, Physics"]
    assert df["hours"].tolist() == [2.0, 1.5]
    assert df["focus_score"].tolist() == [8, 6]
    assert df["streak"].tolist() == [1, 2]


def test_csv_export_equals_loaded_sessions(tmp_db):
    db.save_session(db.SessionLog("2026-10-09", "DSA", 1.0, 9, 1))
    df = db.load_sessions()
    assert export_csv(df) == db.load_sessions().to_csv(index=False).encode()
    parsed = pd.read_csv(pd.io.common.BytesIO(export_csv(df)))
    assert parsed["subjects"].tolist() == ["DSA"]
