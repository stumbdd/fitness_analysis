import pandas as pd
import numpy as np
from src.utils import timeit, log_calls


class FitnessAnalyzer:
    # Класс для анализа данных с фитнес-браслета.

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @log_calls
    @timeit
    def basic_stats(self) -> pd.DataFrame:
        # Возвращает описательную статистику по основным метрикам.
        cols = ["steps", "sleep_hours", "deep_sleep", "rem_sleep", "active_minutes", "resting_heart_rate"]
        return self.df[cols].describe()

    @log_calls
    @timeit
    def weekday_vs_weekend_sleep(self) -> pd.DataFrame:
        # Средний сон в будние и выходные дни.
        return self.df.groupby("is_weekend")["sleep_hours"].agg(["mean", "median", "std"]).rename(
            index={0: "Будни", 1: "Выходные"}
        )

    @log_calls
    @timeit
    def steps_sleep_correlation(self) -> float:
        # Корреляция Пирсона между шагами и продолжительностью сна.
        return self.df["steps"].corr(self.df["sleep_hours"])

    @log_calls
    @timeit
    def most_active_day(self) -> str:
        # День недели с максимальным средним количеством шагов.
        avg_steps = self.df.groupby("day_of_week")["steps"].mean()
        return avg_steps.idxmax()

    @log_calls
    @timeit
    def sleep_season_trend(self) -> pd.DataFrame:
        # Средняя продолжительность сна по сезонам.
        return self.df.groupby("season")["sleep_hours"].agg(["mean", "std"])

    @log_calls
    @timeit
    def deep_sleep_vs_heart_rate(self) -> float:
        # Корреляция между глубоким сном и пульсом покоя.
        return self.df["deep_sleep"].corr(self.df["resting_heart_rate"])

    @log_calls
    @timeit
    def low_sleep_report(self, threshold: float = 6.0) -> pd.DataFrame:
        # Дни с продолжительностью сна ниже порога.
        return self.df[self.df["sleep_hours"] < threshold][["date", "day_of_week", "sleep_hours", "is_weekend"]]