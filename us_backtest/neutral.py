from __future__ import annotations

import numpy as np
import pandas as pd


def compute_size_proxy(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute a time-varying size proxy from rolling average dollar amount.

    prices: MultiIndex [date, ticker] with columns including 'amount' (close*volume).
    Returns a wide DataFrame indexed by date and columns=tickers with log-rolling-mean(amount).
    """
    df = prices.copy()
    if "amount" not in df.columns:
        if {"close", "volume"}.issubset(df.columns):
            df["amount"] = df["close"] * df["volume"]
        else:
            raise ValueError("prices must contain 'amount' or both 'close' and 'volume'")

    wide_amt = df["amount"].unstack("ticker").sort_index()
    roll = wide_amt.rolling(window=window, min_periods=max(5, window // 2)).mean()
    size = np.log(roll + 1e-8)
    return size


def neutralize_by_size(signals: pd.DataFrame, size_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally neutralize signals by (z-scored) size each day.

    Uses a simple per-day scalar regression: signal = alpha + beta * size_z + eps -> residuals as neutralized signal.
    Returns a wide DataFrame aligned to signals.
    """
    # Align indices
    idx = signals.index.intersection(size_df.index)
    cols = signals.columns.intersection(size_df.columns)
    S = signals.loc[idx, cols].copy()
    Z = size_df.loc[idx, cols].copy()

    out = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for dt in S.index:
        x = S.loc[dt]
        z = Z.loc[dt]
        valid = ~(x.isna() | z.isna())
        if valid.sum() < 3:
            continue
        zcs = (z[valid] - z[valid].mean()) / (z[valid].std() + 1e-12)
        if zcs.std() == 0:
            resid = x[valid] - x[valid].mean()
        else:
            cov = np.cov(zcs.values, x[valid].values, bias=True)[0, 1]
            beta = cov / (zcs.var() + 1e-12)
            resid = x[valid] - beta * zcs
            resid = resid - resid.mean()  # center
        out.loc[dt, valid.index[valid]] = resid
    return out


def ic_by_size_quantile(signals: pd.DataFrame, labels: pd.DataFrame, size_df: pd.DataFrame, q: int = 5) -> dict:
    """Compute daily IC/RankIC within size quantile groups, then average and return means and t-stats.

    Grouping is based on per-ticker median size over the period.
    """
    # Aggregate size per ticker
    size_median = size_df.median(axis=0).dropna()
    # Assign quantiles
    bins = pd.qcut(size_median.rank(method="first"), q=q, labels=False)
    groups = {g: set(size_median.index[bins == g]) for g in range(q)}

    def row_corr(a: pd.Series, b: pd.Series) -> float:
        xy = pd.concat([a, b], axis=1).dropna()
        if len(xy) < 3 or xy.iloc[:, 0].std() == 0 or xy.iloc[:, 1].std() == 0:
            return np.nan
        return float(np.corrcoef(xy.iloc[:, 0], xy.iloc[:, 1])[0, 1])

    def row_rankcorr(a: pd.Series, b: pd.Series) -> float:
        xy = pd.concat([a, b], axis=1).dropna()
        if len(xy) < 3:
            return np.nan
        ra = xy.iloc[:, 0].rank()
        rb = xy.iloc[:, 1].rank()
        if ra.std() == 0 or rb.std() == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])

    results = {}
    for g, names in groups.items():
        if len(names) < 3:
            results[f"Q{g+1}_IC"] = np.nan
            results[f"Q{g+1}_IC_t"] = np.nan
            results[f"Q{g+1}_RankIC"] = np.nan
            results[f"Q{g+1}_RankIC_t"] = np.nan
            continue
        ic_list, ric_list = [], []
        for dt in signals.index.intersection(labels.index):
            s = signals.loc[dt, signals.columns.intersection(names)]
            y = labels.loc[dt, labels.columns.intersection(names)]
            ic_list.append(row_corr(s, y))
            ric_list.append(row_rankcorr(s, y))
        ic_arr = np.array(ic_list, dtype=float)
        ric_arr = np.array(ric_list, dtype=float)
        def t_stat(arr):
            arr = arr[~np.isnan(arr)]
            if arr.size < 3 or np.std(arr, ddof=1) == 0:
                return np.nan
            return float(np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(arr.size)))
        results[f"Q{g+1}_IC"] = float(np.nanmean(ic_arr)) if ic_arr.size else np.nan
        results[f"Q{g+1}_IC_t"] = t_stat(ic_arr)
        results[f"Q{g+1}_RankIC"] = float(np.nanmean(ric_arr)) if ric_arr.size else np.nan
        results[f"Q{g+1}_RankIC_t"] = t_stat(ric_arr)
    return results

