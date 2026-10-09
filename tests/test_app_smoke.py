"""Render the full Streamlit page headlessly and make sure nothing raises."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from study_coach import db

ENTRY = Path(__file__).resolve().parents[1] / "smart_study.py"


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_DB_PATH", str(tmp_path / "smoke.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_page_renders_with_empty_db(tmp_db):
    at = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    assert not at.exception
    assert any("AI Study Coach" in m.value for m in at.markdown)
    # Metric tiles fall back to a dash when there is no data.
    assert any(">—<" in m.value for m in at.markdown)


def test_page_renders_with_history(tmp_db):
    db.save_session(db.SessionLog("2026-10-09", "Math", 2.0, 8, 1))
    at = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert any("2.0h" in m.value for m in at.markdown)


def test_generate_without_subjects_warns(tmp_db):
    at = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("at least one subject" in w.value for w in at.warning)


def test_generate_without_api_key_shows_error(tmp_db, monkeypatch):
    from study_coach import ai

    monkeypatch.setattr(ai, "_secret", lambda name: "")
    at = AppTest.from_file(str(ENTRY), default_timeout=30).run()
    at.text_input[0].input("Math, Physics").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("GROQ_API_KEY not found" in e.value for e in at.error)
