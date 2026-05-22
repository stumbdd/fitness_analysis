import time
import functools
import logging
from typing import Iterator
import pandas as pd

# ---------- Генераторы ----------
def chunked_csv_reader(filepath: str, chunksize: int = 10) -> Iterator[pd.DataFrame]:
    
    # Генератор, читающий CSV по частям (чанкам).
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize, parse_dates=["date"]):
        yield chunk


def low_sleep_generator(df: pd.DataFrame, threshold: float = 6.0) -> Iterator[pd.Series]:
    
    # Генератор, выдающий строки (записи), где продолжительность сна ниже порога.
    
    for _, row in df.iterrows():
        if row["sleep_hours"] < threshold:
            yield row


# ---------- Декораторы ----------
def timeit(func):
    #Декоратор для замера времени выполнения функции.
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"[timeit] {func.__name__} выполнена за {elapsed:.4f} сек")
        return result
    return wrapper


def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"[log] Вызов {func.__name__} с args={args} kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper