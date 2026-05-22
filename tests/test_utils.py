import pandas as pd
import pytest
from src.utils import chunked_csv_reader, low_sleep_generator, timeit, log_calls
import tempfile
import os
import logging

@pytest.fixture
def sample_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,sleep_hours\n")
        for i in range(15):
            f.write(f"2026-01-{i+1:02d},{5.5 + i*0.1}\n")
        path = f.name
    yield path
    os.unlink(path)

def test_chunked_csv_reader(sample_csv):
    chunks = list(chunked_csv_reader(sample_csv, chunksize=5))
    assert len(chunks) == 3
    assert len(chunks[0]) == 5
    assert len(chunks[2]) == 5

def test_low_sleep_generator():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                       "sleep_hours": [5.0, 7.0],
                       "day_of_week": ["Mon", "Tue"]})
    gen = low_sleep_generator(df, threshold=6.0)
    results = list(gen)
    assert len(results) == 1
    assert results[0]["sleep_hours"] == 5.0

def test_timeit_decorator(caplog):
    @timeit
    def dummy():
        return 42
    with caplog.at_level(logging.INFO):
        res = dummy()
    assert res == 42
    assert "dummy выполнена за" in caplog.text

def test_log_calls_decorator(caplog):
    @log_calls
    def add(a, b):
        return a + b
    with caplog.at_level(logging.INFO):
        add(3, 5)
    assert "с args=(3, 5)" in caplog.text