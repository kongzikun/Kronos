from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import os
from pathlib import Path
from datetime import datetime
try:
    from pandas_datareader import data as pdr  # optional: Stooq fallback
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False
import time

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


def download_ohlcv(tickers: List[str], start: str, end: str, rate_limit_sec: float = 0.5) -> pd.DataFrame:
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

        # Rate limit to avoid Yahoo 429
        if rate_limit_sec and rate_limit_sec > 0:
            time.sleep(rate_limit_sec)

    if not frames:
        raise RuntimeError(
            "No OHLCV data downloaded for any ticker. Possible reasons: network blocked, rate limited by Yahoo, or invalid symbols.\n"
            f"Tried: {tickers[:10]}{'...' if len(tickers) > 10 else ''}"
        )

    data = pd.concat(frames, axis=0).sort_index()
    return data


def _normalize_ohlcv_df(df: pd.DataFrame, ticker: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Normalize an OHLCV DataFrame to MultiIndex [date, ticker] with lowercase columns.

    Accepts typical columns: Date/date, Open/High/Low/Close/Adj Close/Volume. Returns None if unusable.
    """
    if df is None or df.empty:
        return None
    df = df.copy()
    if 'date' not in [c.lower() for c in df.columns] and not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse any 'Date' column
        for c in df.columns:
            if str(c).lower() == 'date':
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        # Maybe the index is already datetime-like
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None
    df.index.name = 'date'
    # Lowercase and underscore
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    # Ensure close
    if 'close' not in df.columns:
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        else:
            return None
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    if ticker is None:
        # Try infer ticker from a column
        if 'ticker' in df.columns:
            df = df.reset_index().set_index(['date', 'ticker']).sort_index()
        else:
            return None
    else:
        df['ticker'] = ticker
        df = df.reset_index().set_index(['date', 'ticker']).sort_index()
    df['amount'] = df['close'].astype(float) * df['volume'].astype(float)
    keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount'] if c in df.columns]
    return df[keep_cols]


def load_offline_ohlcv(path: str, tickers: Optional[List[str]], start: str, end: str) -> pd.DataFrame:
    """Load OHLCV from local CSV/Parquet files.

    Supports two layouts:
      1) Directory with per-ticker files: AAPL.csv or AAPL.parquet, etc.
      2) Single file with columns including 'date' and 'ticker'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Offline data path does not exist: {path}")

    frames: List[pd.DataFrame] = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    if p.is_dir():
        files = list(p.glob('*.csv')) + list(p.glob('*.parquet'))
        if not files:
            raise RuntimeError(f"No CSV/Parquet files found in {path}")
        # If tickers not provided, infer from filenames
        file_map = {f.stem.upper(): f for f in files}
        if not tickers:
            tickers = sorted(file_map.keys())
        for t in tickers:
            f = file_map.get(t.upper())
            if f is None:
                continue
            try:
                if f.suffix.lower() == '.csv':
                    df = pd.read_csv(f)
                else:
                    df = pd.read_parquet(f)
            except Exception:
                continue
            ndf = _normalize_ohlcv_df(df, ticker=t)
            if ndf is None or ndf.empty:
                continue
            ndf = ndf.loc[(ndf.index.get_level_values(0) >= start_dt) & (ndf.index.get_level_values(0) <= end_dt)]
            frames.append(ndf)
    else:
        # Single file
        if p.suffix.lower() == '.csv':
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        ndf = _normalize_ohlcv_df(df, ticker=None)
        if ndf is None or ndf.empty:
            raise RuntimeError("Offline file missing required columns (date,ticker,close[or adj_close]).")
        # Filter tickers if provided
        if tickers:
            idx = ndf.index
            ndf = ndf.loc[idx.get_level_values(1).isin([t.upper() for t in tickers])]
        ndf = ndf.loc[(ndf.index.get_level_values(0) >= start_dt) & (ndf.index.get_level_values(0) <= end_dt)]
        frames.append(ndf)

    if not frames:
        raise RuntimeError("No offline OHLCV data matched requested tickers and date range.")
    return pd.concat(frames, axis=0).sort_index()


def download_ohlcv_stooq(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Stooq via pandas-datareader.

    Note: Stooq uses descending dates; we normalize and stack to MultiIndex.
    """
    if not _HAS_PDR:
        raise RuntimeError("pandas_datareader not installed; install to use data_source=stooq")
    frames = []
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    for t in tickers:
        try:
            df = pdr.DataReader(t, 'stooq', start=start_dt, end=end_dt)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.sort_index()
        df = df.rename(columns=str.lower)
        df.index.name = 'date'
        if 'close' not in df.columns:
            continue
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        df['ticker'] = t
        df['amount'] = df['close'].astype(float) * df['volume'].astype(float)
        keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'ticker'] if c in df.columns]
        df = df[keep_cols].reset_index().set_index(['date', 'ticker']).sort_index()
        frames.append(df)
    if not frames:
        raise RuntimeError("No OHLCV data from Stooq for requested tickers/date range")
    return pd.concat(frames, axis=0).sort_index()


def infer_tickers_from_offline(path: str) -> List[str]:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    files = list(p.glob('*.csv')) + list(p.glob('*.parquet'))
    return sorted(set(f.stem.upper() for f in files))


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
