import pandas as pd
import pytest
from src.analysis import FitnessAnalyzer

@pytest.fixture
def sample_df():
    data = {
        "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        "day_of_week": ["Thursday", "Friday", "Saturday", "Sunday"],
        "steps": [8000, 7500, 10000, 11000],
        "sleep_hours": [7.5, 6.8, 8.2, 7.0],
        "deep_sleep": [1.8, 1.5, 2.1, 1.6],
        "rem_sleep": [2.0, 1.9, 2.4, 2.0],
        "active_minutes": [80, 70, 95, 100],
        "resting_heart_rate": [60, 62, 58, 59],
        "season": ["winter", "winter", "winter", "spring"],
        "is_weekend": [0, 0, 1, 1]
    }
    return pd.DataFrame(data)

def test_basic_stats(sample_df):
    analyzer = FitnessAnalyzer(sample_df)
    stats = analyzer.basic_stats()
    assert "steps" in stats.columns
    assert stats.loc["mean", "sleep_hours"] == pytest.approx(7.375, 0.01)

def test_weekday_vs_weekend(sample_df):
    analyzer = FitnessAnalyzer(sample_df)
    result = analyzer.weekday_vs_weekend_sleep()
    assert result.loc["Будни", "mean"] == pytest.approx(7.15, 0.01)
    assert result.loc["Выходные", "mean"] == pytest.approx(7.6, 0.01)

def test_most_active_day(sample_df):
    analyzer = FitnessAnalyzer(sample_df)
    assert analyzer.most_active_day() == "Sunday"

def test_correlation(sample_df):
    analyzer = FitnessAnalyzer(sample_df)
    corr = analyzer.steps_sleep_correlation()
    assert isinstance(corr, float)

def test_low_sleep_report(sample_df):
    analyzer = FitnessAnalyzer(sample_df)
    low = analyzer.low_sleep_report(threshold=7.0)
    # 7.0 не меньше 7.0, поэтому только 6.8 подходит
    assert len(low) == 1
    assert low.iloc[0]["sleep_hours"] == 6.8