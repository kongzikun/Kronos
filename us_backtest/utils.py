import os
import random
import json
import math
from typing import Iterable, Tuple, Dict

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


# ---------------------
# Accuracy diagnostics
# ---------------------

def realized_forward_avg_return(prices: pd.DataFrame, H: int) -> pd.DataFrame:
    """Compute realized forward average returns over horizon H for each (date, ticker).

    Return DataFrame indexed by date with columns as tickers, same shape as signals.
    Formula: (mean(close[t+1..t+H]) - close[t]) / close[t]
    """
    close = prices["close"].unstack("ticker").sort_index()
    shifted = [close.shift(-i) for i in range(1, H + 1)]
    avg_future = sum(shifted) / H
    ret = (avg_future - close) / close
    return ret


def _flatten_align(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Align two wide DataFrames by index/columns and return flattened arrays without NaNs."""
    idx = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    aa = a.loc[idx, cols].values.ravel()
    bb = b.loc[idx, cols].values.ravel()
    mask = (~np.isnan(aa)) & (~np.isnan(bb))
    return aa[mask], bb[mask]


def _rowwise_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation for two aligned Series; returns NaN if insufficient points."""
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return float("nan")
    a = xy.iloc[:, 0].values
    b = xy.iloc[:, 1].values
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rowwise_rank_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation computed via Pearson on ranks."""
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return float("nan")
    ra = xy.iloc[:, 0].rank(method="average")
    rb = xy.iloc[:, 1].rank(method="average")
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_accuracy_metrics(signals: pd.DataFrame, prices: pd.DataFrame, H: int) -> Dict[str, float]:
    """Compute accuracy metrics of model signals vs realized forward returns.

    - signals: DataFrame indexed by date with ticker columns (predicted forward avg return)
    - prices: MultiIndex [date, ticker] with 'close' column used to build realized labels
    - H: forward horizon length
    Returns a dict of metrics and saves detailed artifacts handled by caller.
    """
    y_true = realized_forward_avg_return(prices, H)
    # Align
    s_aligned = signals.reindex_like(y_true)
    y_aligned = y_true.loc[s_aligned.index, s_aligned.columns]

    # Global regression errors
    yhat, y = _flatten_align(s_aligned, y_aligned)
    mae = float(np.mean(np.abs(yhat - y))) if y.size else float("nan")
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2))) if y.size else float("nan")
    # Global correlation
    corr = float(np.corrcoef(yhat, y)[0, 1]) if y.size and np.std(yhat) > 0 and np.std(y) > 0 else float("nan")

    # Directional accuracy (sign match)
    sign_mask = (yhat != 0) & (y != 0)
    da = float(np.mean(np.sign(yhat[sign_mask]) == np.sign(y[sign_mask]))) if sign_mask.any() else float("nan")

    # Daily IC / RankIC across tickers
    ic_list = []
    ric_list = []
    for dt in s_aligned.index.intersection(y_aligned.index):
        ic = _rowwise_corr(s_aligned.loc[dt], y_aligned.loc[dt])
        ric = _rowwise_rank_corr(s_aligned.loc[dt], y_aligned.loc[dt])
        ic_list.append(ic)
        ric_list.append(ric)
    ic_arr = np.array(ic_list, dtype=float)
    ric_arr = np.array(ric_list, dtype=float)
    ic_mean = float(np.nanmean(ic_arr)) if ic_arr.size else float("nan")
    ric_mean = float(np.nanmean(ric_arr)) if ric_arr.size else float("nan")
    # t-stats (simple)
    def t_stat(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if arr.size < 3 or np.std(arr) == 0:
            return float("nan")
        return float(np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(arr.size)))

    ic_t = t_stat(ic_arr)
    ric_t = t_stat(ric_arr)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Corr_all": corr,
        "DA": da,
        "IC": ic_mean,
        "IC_t": ic_t,
        "RankIC": ric_mean,
        "RankIC_t": ric_t,
    }
