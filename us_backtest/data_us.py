from typing import List

import numpy as np
import pandas as pd
import time
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from utils import calc_time_features


SP500_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V"
]


def get_sp500_tickers(date: str | None = None) -> List[str]:
    """Fetch S&P500 constituents from Wikipedia, fallback to yfinance or a static list."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        symbols = tables[0]["Symbol"].tolist()
        symbols = [s.replace(".", "-") for s in symbols]
        if symbols:
            return symbols
    except Exception:
        pass

    try:
        return yf.tickers_sp500()
    except Exception:
        return SP500_FALLBACK


def download_ohlcv(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for tickers using yfinance."""
    dfs: List[pd.DataFrame] = []
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        subset = tickers[i : i + chunk_size]
        while True:
            try:
                chunk = yf.download(
                    subset,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
                break
            except YFRateLimitError:
                time.sleep(60)

        if chunk.empty:
            continue
        if isinstance(chunk.columns, pd.MultiIndex):
            chunk = chunk.stack(level=0, future_stack=True).rename_axis(
                index=["date", "ticker"]
            ).reset_index()
        else:  # single ticker
            chunk = chunk.reset_index()
            chunk["ticker"] = subset[0]
        dfs.append(chunk)

    data = pd.concat(dfs, ignore_index=True)
    data.columns = [c.lower().replace(" ", "_") for c in data.columns]
    data["amount"] = data["close"] * data["volume"]
    data = data.set_index(["date", "ticker"])[
        ["open", "high", "low", "close", "volume", "amount"]
    ]
    data = data.sort_index()
    return data


def prepare_windows(df: pd.DataFrame, lookback: int, horizon: int, batch_size: int = 64):
    """Yield sliding windows for each ticker for inference."""
    for ticker in df.index.get_level_values(1).unique():
        tdf = df.xs(ticker, level=1)
        tdf = tdf.dropna()
        arr = tdf.values
        dates = tdf.index
        time_arr = calc_time_features(dates).values
        n = len(tdf)
        if n < lookback + horizon:
            continue
        for start in range(lookback - 1, n - horizon, batch_size):
            end = min(n - horizon, start + batch_size)
            batch_x, batch_x_stamp, batch_y_stamp, signal_dates = [], [], [], []
            for t in range(start, end):
                s = t - lookback + 1
                batch_x.append(arr[s : s + lookback])
                batch_x_stamp.append(time_arr[s : s + lookback])
                batch_y_stamp.append(time_arr[t + 1 : t + 1 + horizon])
                signal_dates.append(dates[t])
            yield ticker, np.array(batch_x), np.array(batch_x_stamp), np.array(batch_y_stamp), signal_dates
