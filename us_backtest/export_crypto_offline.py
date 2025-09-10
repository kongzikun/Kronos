import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd


def crypto_universe_longlist() -> List[str]:
    """Return a generous list of Yahoo Finance crypto tickers.

    We intentionally include >70 symbols to ensure at least 50 succeed; the
    exporter will keep the first N successes.
    """
    return [
        # Mega/Large caps
        "BTC-USD","ETH-USD","USDT-USD","BNB-USD","SOL-USD","USDC-USD","XRP-USD","DOGE-USD","ADA-USD","TRX-USD",
        "TON-USD","SHIB-USD","AVAX-USD","DOT-USD","LTC-USD","BCH-USD","LINK-USD","MATIC-USD","UNI7083-USD","ATOM-USD",
        "XLM-USD","ICP-USD","NEAR-USD","APT-USD","FIL-USD","ARB-USD","SUI-USD","OP-USD","ETC-USD","HBAR-USD",
        "IMX-USD","ALGO-USD","EGLD-USD","AAVE-USD","FLOW-USD","VET-USD","MKR-USD","RUNE-USD","STX-USD","GRT-USD",
        "LDO-USD","TIA-USD","SAND-USD","AXS-USD","DYDX-USD","ENS-USD","PEPE-USD","KAVA-USD","XMR-USD","EOS-USD",
        "KLAY-USD","ZEC-USD","IOTA-USD","XTZ-USD","NEO-USD","THETA-USD","KSM-USD","ONE-USD","ZIL-USD","GMT-USD",
        "CHZ-USD","CRV-USD","CVX-USD","SNX-USD","CAKE-USD","RNDR-USD","INJ-USD","FTM-USD","AR-USD","QNT-USD",
        "GMX-USD","ROSE-USD","CFX-USD","FLR-USD","COMP-USD","XDC-USD","WOO-USD","FTT-USD","PYTH-USD","JTO-USD",
    ]


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Best-effort daily OHLCV fetcher.

    1) Try Binance public API (symbol like BTCUSDT) if ticker endswith 'USDT' or looks like a Binance symbol.
    2) Fallback to Yahoo Finance (ticker like BTC-USD) if needed.
    """
    import pandas as pd
    import requests

    def _try_binance(sym: str) -> pd.DataFrame:
        base = "https://api.binance.com/api/v3/klines"
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        # Add one day to include end date fully
        end_ms = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)
        params = {"symbol": sym, "interval": "1d", "startTime": start_ms, "endTime": end_ms}
        r = requests.get(base, params=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        cols = [
            "open_time","open","high","low","close","volume","close_time","quote_asset_volume",
            "number_of_trades","taker_buy_base","taker_buy_quote","ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
        keep = ["date","open","high","low","close","volume","amount"]
        df = df[keep].dropna()
        df = df.set_index("date").sort_index()
        return df

    def _try_yahoo(sym: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception:
            return pd.DataFrame()
        end_ex = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(sym, start=start, end=end_ex, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        })
        if "close" not in df.columns:
            if "adj_close" in df.columns:
                df["close"] = df["adj_close"]
            else:
                return pd.DataFrame()
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
        keep = [c for c in ["open","high","low","close","volume","amount"] if c in df.columns]
        return df[keep].dropna(how="any")

    # Route by naming convention
    df = pd.DataFrame()
    if ticker.endswith("USDT"):
        df = _try_binance(ticker)
    if df.empty and ticker.endswith("-USD"):
        df = _try_yahoo(ticker)
    return df


def main():
    ap = argparse.ArgumentParser(description="Export offline OHLCV for top crypto symbols from Yahoo Finance")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-08-30")
    ap.add_argument("--out_dir", default="offline_data/crypto_20230101_20250830")
    ap.add_argument("--max", type=int, default=50, help="Max number of symbols to export")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tickers = crypto_universe_longlist()
    successes = []
    for t in tickers:
        try:
            # Prefer Binance pair name if available; try both forms
            binance_sym = t.replace("-USD", "USDT") if t.endswith("-USD") else t
            df = download_one(binance_sym, args.start, args.end)
            if df.empty:
                df = download_one(t, args.start, args.end)
            if df.empty or len(df) < 50:
                continue
            # Save parquet as <TICKER>.parquet; filename stem is used as ticker
            stem = (binance_sym if not df.empty else t).upper()
            fp = out / f"{stem}.parquet"
            if "date" not in df.columns:
                df = df.reset_index().rename(columns={"index": "date"})
            df.to_parquet(fp, index=False)
            successes.append(stem)
            if len(successes) >= args.max:
                break
        except Exception:
            continue

    # Write tickers list
    with open(out / "tickers.txt", "w") as f:
        for t in successes:
            f.write(t + "\n")

    print(f"Exported {len(successes)} symbols to {str(out)}")


if __name__ == "__main__":
    main()
