import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

try:
    # Prefer package-relative imports
    from .data_us import (
        download_ohlcv,
        download_ohlcv_stooq,
    )
except Exception:  # pragma: no cover
    from data_us import (
        download_ohlcv,
        download_ohlcv_stooq,
    )


def get_nasdaq100_tickers() -> List[str]:
    """Fetch NASDAQ-100 tickers from Wikipedia with basic cleaning.

    Falls back to a small core subset if parsing fails.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        candidates = []
        for df in tables:
            cols = {str(c).lower() for c in df.columns}
            if "ticker" in cols or "symbol" in cols:
                col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
                if col is not None:
                    tickers = df[col].astype(str).str.upper().tolist()
                    candidates.extend(tickers)
        cleaned = []
        for t in candidates:
            t = t.strip().split(" ")[0]
            t = t.replace(".", "-")
            if t and t.isascii():
                cleaned.append(t)
        # Deduplicate, light filter
        cleaned = [t for t in dict.fromkeys(cleaned) if 1 <= len(t) <= 6 and t not in {"TICKER", "SYMBOL"}]
        if len(cleaned) < 50:
            raise RuntimeError("Parsed too few tickers from Wikipedia")
        return sorted(cleaned)
    except Exception:
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "COST", "TSLA", "PEP",
            "NFLX", "ADBE", "AMD", "INTC", "CMCSA", "TXN", "QCOM", "CSCO", "AMAT", "PDD",
        ]


def save_ticker_file(df: pd.DataFrame, out_dir: Path, ticker: str, fmt: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.{ 'parquet' if fmt=='parquet' else 'csv' }"
    # df is expected indexed by date with columns including open/high/low/close/volume/amount
    if fmt == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)


def build_archive(out_dir: Path, archive: str) -> Path:
    if archive == "none":
        return out_dir
    if archive == "tar.gz":
        import tarfile
        tar_path = out_dir.with_suffix("")  # strip if any
        tar_path = out_dir.parent / f"{out_dir.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        return tar_path
    if archive == "zip":
        import shutil
        zip_path = out_dir.parent / f"{out_dir.name}.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)
        return zip_path
    raise ValueError(f"Unknown archive type: {archive}")


def main():
    parser = argparse.ArgumentParser(description="Export offline OHLCV files for NASDAQ-100 (or custom)")
    parser.add_argument("--index", default="nasdaq100", choices=["nasdaq100", "custom"], help="Universe to export")
    parser.add_argument("--tickers", default="", help="Comma-separated tickers when index=custom")
    parser.add_argument("--tickers_file", default="", help="Optional file with tickers, one per line")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-08-30")
    parser.add_argument("--data_source", choices=["stooq", "yahoo"], default="stooq")
    parser.add_argument("--yf_rate_limit", type=float, default=0.5, help="Sleep seconds between Yahoo requests")
    parser.add_argument("--out_root", default="offline_data", help="Root dir to store files")
    parser.add_argument("--name", default="nasdaq100_20230101_20250830", help="Subdirectory name under out_root")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--archive", choices=["tar.gz", "zip", "none"], default="tar.gz")
    args = parser.parse_args()

    # Resolve tickers
    tickers: List[str]
    if args.tickers_file:
        with open(args.tickers_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    elif args.index == "nasdaq100":
        tickers = get_nasdaq100_tickers()
    else:
        if not args.tickers:
            raise SystemExit("index=custom requires --tickers or --tickers_file")
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(f"Tickers: {len(tickers)} symbols")

    # Download prices
    if args.data_source == "stooq":
        prices = download_ohlcv_stooq(tickers, args.start, args.end)
    else:
        prices = download_ohlcv(tickers, args.start, args.end, rate_limit_sec=args.yf_rate_limit)

    # Save per-ticker files
    out_dir = Path(args.out_root) / args.name
    saved = 0
    for t in sorted(prices.index.get_level_values(1).unique()):
        try:
            tdf = prices.xs(t, level=1)
            # Keep canonical columns
            cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in tdf.columns]
            if not cols:
                continue
            save_ticker_file(tdf[cols], out_dir, t, args.format)
            saved += 1
        except Exception as e:
            print(f"[warn] Save failed for {t}: {e}")
            continue

    if saved == 0:
        raise SystemExit("No files saved; check data availability and tickers")
    archive_path = build_archive(out_dir, args.archive)
    print(f"Saved {saved} files under {out_dir}")
    print(f"Archive: {archive_path}")


if __name__ == "__main__":
    main()

