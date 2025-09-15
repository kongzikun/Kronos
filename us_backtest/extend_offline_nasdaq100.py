import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_tickers(src_dir: Path) -> List[str]:
    files = list(src_dir.glob("*.parquet")) + list(src_dir.glob("*.csv"))
    return sorted({f.stem.upper() for f in files})


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    # Normalize
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])  # type: ignore
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df.index.name = "date"
    return df


def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly normalize yfinance columns to [open, high, low, close, adj_close, volume]."""
    if isinstance(df.columns, pd.MultiIndex):
        # Try to pick the semantic name from any level
        want = {"open": None, "high": None, "low": None, "close": None, "adj close": None, "volume": None}
        for col in df.columns:
            parts = [str(x).strip().lower() for x in (col if isinstance(col, tuple) else (col,))]
            for k in list(want.keys()):
                if k in parts:
                    want[k] = col
        new_df = pd.DataFrame(index=df.index)
        for k, src in want.items():
            if src is not None:
                name = k.replace(" ", "_")
                new_df[name] = df[src]
        df = new_df
    else:
        # Single-level columns
        mapping = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
        cols = {}
        for k in ["open", "high", "low", "close", "adj_close", "volume"]:
            if k in mapping:
                cols[k] = mapping[k]
        if cols:
            df = df[list(cols.values())]
            df.columns = list(cols.keys())
    # Ensure minimum set
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    # Keep canonical order if present
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep]


def download_earlier(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by=None,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize index/columns
    df.index.name = "date"
    df = _normalize_yf_columns(df)
    return df


def main():
    p = argparse.ArgumentParser(description="Extend NASDAQ-100 offline data earlier while keeping constituents fixed")
    p.add_argument("--src_dir", required=True, help="Existing offline dir (e.g., nasdaq100_20230101_20250830)")
    p.add_argument("--dst_dir", required=True, help="Output dir (e.g., nasdaq100_20200101_20250830)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2025-08-30")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = p.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(src)
    print(f"Tickers inferred from {src}: {len(tickers)}")

    saved = 0
    for t in tickers:
        # Load existing
        src_file_parquet = src / f"{t}.parquet"
        src_file_csv = src / f"{t}.csv"
        src_file = src_file_parquet if src_file_parquet.exists() else src_file_csv
        if not src_file.exists():
            print(f"[warn] missing source for {t}")
            continue
        try:
            existing = load_existing(src_file)
        except Exception as e:
            print(f"[warn] load failed for {t}: {e}")
            continue

        # Download earlier range (up to the first date in existing)
        first_date = existing.index.min() if len(existing) else None
        dl_end = first_date.strftime("%Y-%m-%d") if first_date is not None else args.end
        try:
            earlier = download_earlier(t, args.start, dl_end)
        except Exception as e:
            print(f"[warn] download failed for {t}: {e}")
            earlier = pd.DataFrame()

        # Combine
        frames = []
        if not earlier.empty:
            frames.append(earlier)
        if not existing.empty:
            frames.append(existing[[c for c in ["open", "high", "low", "close", "volume"] if c in existing.columns]])
        if not frames:
            print(f"[warn] no data for {t}")
            continue
        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        merged["amount"] = merged["close"].astype(float) * merged["volume"].astype(float)

        # Save
        out = dst / f"{t}.{ 'parquet' if args.format=='parquet' else 'csv' }"
        try:
            if args.format == "parquet":
                merged.to_parquet(out)
            else:
                merged.to_csv(out)
            saved += 1
        except Exception as e:
            print(f"[warn] save failed for {t}: {e}")
            continue

    if saved == 0:
        raise SystemExit("No files saved; check network and dependencies (pyarrow for parquet)")
    print(f"Saved {saved} files under {dst}")


if __name__ == "__main__":
    main()
