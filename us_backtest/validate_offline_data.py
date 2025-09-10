import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# Reuse normalizer from data_us
try:
    from .data_us import _normalize_ohlcv_df
except Exception:  # pragma: no cover
    from data_us import _normalize_ohlcv_df


def _load_one_file(fp: Path, ticker_hint: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp)
        else:
            df = pd.read_parquet(fp)
    except Exception:
        return None
    tick = ticker_hint or fp.stem.upper()
    try:
        ndf = _normalize_ohlcv_df(df, ticker=tick)
    except Exception:
        return None
    return ndf


def _sanity_ohlc(ndf: pd.DataFrame) -> Dict[str, int]:
    cols = {c for c in ["open", "high", "low", "close"] if c in ndf.columns}
    negatives = 0
    invalid_hilo = 0
    if cols:
        for c in cols:
            negatives += int((ndf[c] < 0).sum())
        if {"high", "low", "open", "close"}.issubset(ndf.columns):
            hi = ndf["high"]
            lo = ndf["low"]
            oc_max = ndf[["open", "close"]].max(axis=1)
            oc_min = ndf[["open", "close"]].min(axis=1)
            invalid_hilo = int(((hi < oc_max) | (lo > oc_min)).sum())
    return {"negatives": negatives, "invalid_hilo": invalid_hilo}


def _calc_gaps(dates: pd.DatetimeIndex) -> int:
    if dates.empty:
        return 0
    ds = pd.DatetimeIndex(dates.sort_values().unique())
    diffs = pd.Series(ds).diff().dropna().dt.days.values
    # count days where gap > 1 day; this will overcount holidays but is a good indicator
    return int((diffs > 1).sum())


def validate_offline_dir(path: str, out_dir: Optional[str]) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    files: List[Path] = []
    if p.is_dir():
        files = [*p.glob("*.csv"), *p.glob("*.parquet")]
        if not files:
            raise RuntimeError(f"No CSV/Parquet files found under {path}")
    else:
        files = [p]

    records = []
    bad_files = []

    for fp in sorted(files):
        ndf = _load_one_file(fp)
        if ndf is None or ndf.empty:
            bad_files.append(fp.name)
            continue
        idx_dates = ndf.index.get_level_values(0)
        ticker = ndf.index.get_level_values(1)[0] if len(ndf.index) > 0 else fp.stem.upper()
        row_count = len(ndf)
        uniq_dates = idx_dates.nunique()
        min_dt = idx_dates.min()
        max_dt = idx_dates.max()
        # missing values per column
        na_counts = {f"na_{c}": int(ndf[c].isna().sum()) for c in ndf.columns}
        sanity = _sanity_ohlc(ndf)
        gaps = _calc_gaps(idx_dates)
        is_sorted = bool(idx_dates.is_monotonic_increasing)

        rec = {
            "file": fp.name,
            "ticker": ticker,
            "rows": row_count,
            "days": uniq_dates,
            "date_min": str(min_dt) if pd.notna(min_dt) else None,
            "date_max": str(max_dt) if pd.notna(max_dt) else None,
            "gaps": gaps,
            "sorted": is_sorted,
        }
        rec.update(na_counts)
        rec.update(sanity)
        records.append(rec)

    report = pd.DataFrame.from_records(records)
    out_base = out_dir or os.path.join("outputs", "offline_validation")
    os.makedirs(out_base, exist_ok=True)
    # Name by directory
    tag = Path(path).stem
    csv_path = os.path.join(out_base, f"{tag}_report.csv")
    report.sort_values(["ticker", "file"]).to_csv(csv_path, index=False)

    # Aggregate summary
    summary = {
        "path": path,
        "files_total": len(files),
        "files_ok": int(len(records)),
        "files_bad": int(len(bad_files)),
        "bad_files": bad_files,
        "tickers": int(report["ticker"].nunique()) if not report.empty else 0,
        "rows_total": int(report["rows"].sum()) if not report.empty else 0,
        "avg_gaps": float(report["gaps"].mean()) if not report.empty else 0.0,
        "files_with_na": int(((report.filter(like="na_") > 0).any(axis=1)).sum()) if not report.empty else 0,
        "files_with_invalid_ohlc": int((report["invalid_hilo"] > 0).sum()) if not report.empty else 0,
    }
    pd.Series(summary).to_json(os.path.join(out_base, f"{tag}_summary.json"), indent=2)
    return csv_path


def main():
    ap = argparse.ArgumentParser(description="Validate offline OHLCV data files (CSV/Parquet)")
    ap.add_argument("--path", required=True, help="Directory with per-ticker files or a single file")
    ap.add_argument("--out_dir", default=None, help="Output directory for the validation report")
    args = ap.parse_args()
    out = validate_offline_dir(args.path, args.out_dir)
    print(f"Validation report saved to: {out}")


if __name__ == "__main__":
    main()
