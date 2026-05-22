import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from src.utils import timeit, log_calls


@log_calls
@timeit
def load_data(filepath: str = "data/fitness_data.csv") -> pd.DataFrame:
    # Загружает CSV с данными.
    df = pd.read_csv(filepath, parse_dates=["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    return df


def generate_fitness_data(output_path: str = "data/fitness_data.csv") -> None:
    #Генерирует синтетические данные фитнес-трекера за ~5 месяцев. Сохраняет в CSV.
    
    np.random.seed(42)
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 5, 25)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # Сезон: январь-февраль - зима, март-май - весна
    season = np.where(dates.month.isin([1, 2]), "winter", "spring")
    is_weekend = (dates.dayofweek >= 5).astype(int)  # 5=Saturday, 6=Sunday

    # Базовый сон: зимой 7.5, весной 6.8 + шум
    base_sleep = np.where(season == "winter", 7.5, 6.8)
    sleep_hours = base_sleep + is_weekend * np.random.uniform(0.5, 1.0, n) + np.random.normal(0, 0.5, n)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)

    # Глубокий сон ~20-25% от общего, с небольшим шумом
    deep_sleep = sleep_hours * np.random.uniform(0.20, 0.25, n) + np.random.normal(0, 0.2, n)
    deep_sleep = np.clip(deep_sleep, 0.5, 3.5)
    # REM сон ~20-30% от общего
    rem_sleep = sleep_hours * np.random.uniform(0.20, 0.30, n) + np.random.normal(0, 0.2, n)
    rem_sleep = np.clip(rem_sleep, 0.5, 3.0)

    # Шаги: будни ~6000-10000, выходные +25%, тренд весной +5%
    steps_base = np.random.randint(6000, 10000, size=n)
    weekend_boost = is_weekend * np.random.uniform(1000, 4000, n)
    spring_trend = (season == "spring") * np.random.uniform(500, 2000, n)
    steps = steps_base + weekend_boost + spring_trend
    steps = steps.astype(int)

    # Активные минуты
    active_minutes = (steps / 100).astype(int) + np.random.randint(-5, 10, size=n)
    active_minutes = np.clip(active_minutes, 20, 120)

    # Пульс покоя: зависит от глубокого сна (чем больше, тем ниже пульс)
    resting_hr = 65 - (deep_sleep * 2) + np.random.normal(0, 2, n)
    resting_hr = np.clip(resting_hr.round(), 50, 75).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "day_of_week": dates.day_name(),
        "steps": steps,
        "sleep_hours": sleep_hours.round(2),
        "deep_sleep": deep_sleep.round(2),
        "rem_sleep": rem_sleep.round(2),
        "active_minutes": active_minutes,
        "resting_heart_rate": resting_hr,
        "season": season,
        "is_weekend": is_weekend
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Данные сохранены в {output_path} ({len(df)} записей)")