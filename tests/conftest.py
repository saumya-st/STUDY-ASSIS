import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-10-07", "2026-10-08", "2026-10-09"],
            "subjects": ["Math", "Math, Physics", "DSA"],
            "hours": [2.0, 3.0, 1.5],
            "focus_score": [6, 8, 7],
            "streak": [1, 2, 3],
        }
    )
