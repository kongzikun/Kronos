from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .utils import calc_time_features
except Exception:  # pragma: no cover
    from utils import calc_time_features


SP500_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V"
]


def get_sp500_tickers(date: Optional[str] = None) -> List[str]:
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
    """Download OHLCV data for tickers using yfinance.

    - Fetches per-ticker to avoid brittle MultiIndex parsing and partial failures.
    - Silently skips tickers with no data; raises with a clear message if all fail.
    - Ensures a 'close' column exists (falls back to 'adj_close' if needed).
    """
    frames = []
    failed: List[str] = []

    for t in tickers:
        try:
            tdf = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by=None,
                threads=False,
            )
        except Exception:
            failed.append(t)
            continue

        if tdf is None or tdf.empty:
            failed.append(t)
            continue

        tdf = tdf.copy()
        tdf.index.name = "date"
        # Normalize column names
        tdf.columns = [str(c).lower().replace(" ", "_") for c in tdf.columns]

        # Ensure required columns
        if "close" not in tdf.columns:
            if "adj_close" in tdf.columns:
                tdf["close"] = tdf["adj_close"]
            else:
                # No usable price; skip this ticker
                failed.append(t)
                continue
        if "volume" not in tdf.columns:
            tdf["volume"] = 0.0

        tdf["ticker"] = t
        tdf["amount"] = tdf["close"].astype(float) * tdf["volume"].astype(float)
        keep_cols = [c for c in ["open", "high", "low", "close", "volume", "amount", "ticker"] if c in tdf.columns]
        tdf = tdf[keep_cols].reset_index().set_index(["date", "ticker"]).sort_index()
        frames.append(tdf)

    if not frames:
        raise RuntimeError(
            "No OHLCV data downloaded for any ticker. Possible reasons: network blocked, rate limited by Yahoo, or invalid symbols.\n"
            f"Tried: {tickers[:10]}{'...' if len(tickers) > 10 else ''}"
        )

    data = pd.concat(frames, axis=0).sort_index()
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
