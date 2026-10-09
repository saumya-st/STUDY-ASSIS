import datetime

import pandas as pd
import pytest

from study_coach.analytics import (
    SESSION_COLUMNS,
    compute_streak,
    export_csv,
    focus_trend,
    summarize,
    time_distribution,
)

TODAY = datetime.date(2026, 10, 9)


def d(days_ago: int) -> datetime.date:
    return TODAY - datetime.timedelta(days=days_ago)


class TestComputeStreak:
    def test_empty(self):
        assert compute_streak([], TODAY) == 0

    def test_today_only(self):
        assert compute_streak([TODAY], TODAY) == 1

    def test_consecutive_days(self):
        assert compute_streak([d(2), d(1), d(0)], TODAY) == 3

    def test_gap_breaks_streak(self):
        assert compute_streak([d(3), d(1), d(0)], TODAY) == 2

    def test_streak_must_end_today(self):
        assert compute_streak([d(3), d(2), d(1)], TODAY) == 0

    def test_duplicate_dates_count_once(self):
        assert compute_streak([d(0), d(0), d(1)], TODAY) == 2

    def test_order_independent(self):
        assert compute_streak([d(0), d(2), d(1)], TODAY) == 3


class TestSummarize:
    def test_sample(self, sample_df):
        s = summarize(sample_df, today=TODAY)
        assert s["streak"] == 3
        assert s["sessions"] == 3
        assert s["avg_focus"] == pytest.approx(7.0)
        assert s["total_hours"] == pytest.approx(6.5)

    def test_empty(self):
        s = summarize(pd.DataFrame(columns=SESSION_COLUMNS), today=TODAY)
        assert s == {"streak": 0, "sessions": 0, "avg_focus": None, "total_hours": 0.0}


class TestTimeDistribution:
    def test_proportional_split_sums_to_hours(self):
        subjects = ["Math", "Physics", "DSA"]
        priorities = {"Math": 5, "Physics": 3, "DSA": 2}
        times = time_distribution(subjects, priorities, 4.0)
        assert set(times) == set(subjects)
        assert sum(times.values()) == pytest.approx(4.0, abs=0.02)
        assert times["Math"] == pytest.approx(2.0)
        assert times["Physics"] == pytest.approx(1.2)
        assert times["DSA"] == pytest.approx(0.8)

    def test_missing_priority_defaults_to_three(self):
        times = time_distribution(["A", "B"], {"A": 3}, 2.0)
        assert times == {"A": 1.0, "B": 1.0}

    def test_empty_subjects(self):
        assert time_distribution([], {}, 3.0) == {}


def test_focus_trend_averages_per_day():
    df = pd.DataFrame(
        {"date": ["2026-10-08", "2026-10-08", "2026-10-09"], "focus_score": [6, 8, 5]}
    )
    trend = focus_trend(df)
    assert list(trend.index) == [pd.Timestamp("2026-10-08"), pd.Timestamp("2026-10-09")]
    assert trend.loc["2026-10-08"].item() == pytest.approx(7.0)


def test_export_csv_matches_dataframe(sample_df):
    assert export_csv(sample_df) == sample_df.to_csv(index=False).encode()
    assert export_csv(sample_df).startswith(b"date,subjects,hours,focus_score,streak")
