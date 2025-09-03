import os
import random
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def calc_time_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create time feature dataframe compatible with Kronos tokenizer."""
    df = pd.DataFrame(index=dt_index)
    df["minute"] = 0
    df["hour"] = 0
    df["weekday"] = dt_index.weekday
    df["day"] = dt_index.day
    df["month"] = dt_index.month
    return df[["minute", "hour", "weekday", "day", "month"]]


def annualized_return(excess_returns: Iterable[float]) -> float:
    arr = np.array(list(excess_returns))
    if len(arr) == 0:
        return 0.0
    prod = np.prod(1 + arr)
    return prod ** (252 / len(arr)) - 1


def information_ratio(excess_returns: Iterable[float]) -> float:
    arr = np.array(list(excess_returns))
    if arr.std() == 0:
        return 0.0
    return arr.mean() / arr.std() * math.sqrt(252)


def max_drawdown(nav: Iterable[float]) -> float:
    arr = np.array(list(nav))
    cum_max = np.maximum.accumulate(arr)
    drawdown = (arr - cum_max) / cum_max
    return drawdown.min()


def plot_nav(nav: pd.Series, bench: pd.Series, path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(nav.index, nav.values, label="Portfolio")
    plt.plot(bench.index, bench.values, label="Benchmark")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_excess_nav(excess_nav: pd.Series, path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(excess_nav.index, excess_nav.values, label="Excess")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Excess Return")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
