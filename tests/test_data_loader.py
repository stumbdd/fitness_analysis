import pandas as pd
import numpy as np
import pytest
import tempfile
import os
from src.data_loader import generate_fitness_data, load_data

@pytest.fixture
def sample_csv():
    # Создаёт временный CSV с минимальными данными.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,day_of_week,steps,sleep_hours,deep_sleep,rem_sleep,active_minutes,resting_heart_rate,season,is_weekend\n")
        f.write("2026-01-01,Thursday,8000,7.5,1.8,2.0,80,60,winter,0\n")
        f.write("2026-01-02,Friday,7500,6.8,1.5,1.9,70,62,winter,0\n")
        f.write("2026-01-03,Saturday,10000,8.2,2.1,2.4,95,58,winter,1\n")
        f.write("2026-03-15,Sunday,11000,7.0,1.6,2.0,100,59,spring,1\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)

def test_load_data(sample_csv):
    df = load_data(sample_csv)
    assert len(df) == 4
    assert "date" in df.columns
    assert df["steps"].dtype == np.int64

def test_generate_fitness_data():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        generate_fitness_data(path)
        df = pd.read_csv(path)
        assert len(df) >= 100  # минимум 100 записей по условию
        assert "sleep_hours" in df.columns
        assert df["sleep_hours"].min() >= 4.0
    finally:
        os.unlink(path)