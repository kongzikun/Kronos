import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

CURR = os.path.dirname(os.path.abspath(__file__))
if CURR not in sys.path:
    sys.path.append(CURR)
ROOT = os.path.dirname(CURR)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils import set_seed, realized_forward_avg_return, compute_accuracy_metrics, ensure_dir, save_json
from data_us import download_ohlcv, prepare_windows, _normalize_ohlcv_df
from kronos_infer import load_model, predict_batch
from backtest_topk_dropn import backtest
from sector_utils import select_sector_tickers
from neutral import compute_size_proxy, neutralize_by_size, ic_by_size_quantile


def main():
    p = argparse.ArgumentParser(description="Kronos US backtest - industry + size neutral experiment")
    p.add_argument("--prefer_sector", default="Information Technology")
    p.add_argument("--n_tickers", type=int, default=20)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-08-31")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--cost_bps", type=float, default=10)
    p.add_argument("--H", type=int, default=5)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--min_hold", type=int, default=5)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--out_dir", default="outputs/us_backtest_industry_neutral")
    p.add_argument("--offline_dir", default=None, help="Optional directory or file for offline OHLCV data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tickers = select_sector_tickers(args.n_tickers, args.prefer_sector)

    # Prefer Stooq (pandas-datareader) to avoid Yahoo rate/SSL issues
    def _download_stooq(tickers, start, end):
        try:
            from pandas_datareader import data as pdr
        except Exception:
            return None
        frames = []
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        for t in tickers:
            try:
                tried = []
                for sym in (t, f"{t}.US"):
                    tried.append(sym)
                    try:
                        sdf = pdr.DataReader(sym, "stooq", start_dt, end_dt)
                        if sdf is None or sdf.empty:
                            continue
                        sdf = sdf.sort_index()
                        ndf = _normalize_ohlcv_df(sdf, ticker=t)
                        if ndf is not None and not ndf.empty:
                            frames.append(ndf)
                            break
                    except Exception:
                        continue
            except Exception:
                continue
        if frames:
            return pd.concat(frames).sort_index()
        return None

    prices = None
    if args.offline_dir:
        try:
            from data_us import load_offline_ohlcv
            prices = load_offline_ohlcv(args.offline_dir, tickers, args.start, args.end)
        except Exception:
            prices = None
    if prices is None or prices.empty:
        prices = _download_stooq(tickers, args.start, args.end)
    if prices is None or prices.empty:
        prices = download_ohlcv(tickers, args.start, args.end, rate_limit_sec=0.2)
    tokenizer, model = load_model(device)

    # Signals
    signal_records = []
    for ticker, x, x_stamp, y_stamp, dates in prepare_windows(prices, args.lookback, args.H):
        preds = predict_batch(tokenizer, model, x, x_stamp, y_stamp, device, args.samples, 1.0, 0.9)
        mean_close = preds[:, :, 3].mean(axis=1)
        close_t = x[:, -1, 3]
        ret = (mean_close - close_t) / close_t
        df_sig = pd.DataFrame({"date": dates, "ticker": ticker, "signal": ret})
        signal_records.append(df_sig)
    if not signal_records:
        raise SystemExit("No signals generated. Check data availability.")
    signals = pd.concat(signal_records).set_index(["date", "ticker"]).unstack("ticker")["signal"].sort_index()

    # Benchmark: try S&P500; if unavailable (offline), use equal-weight close of universe as proxy
    try:
        bench = download_ohlcv(["^GSPC"], args.start, args.end).xs("^GSPC", level=1)["close"]
    except Exception:
        bench = prices["close"].unstack("ticker").mean(axis=1)

    # Base backtest
    base_dir = os.path.join(args.out_dir, "base")
    summary_base = backtest(
        signals=signals,
        prices=prices,
        benchmark=bench,
        out_dir=base_dir,
        k=args.k,
        n=args.n,
        min_hold=args.min_hold,
        cost=args.cost_bps / 10000,
    )

    # Labels and accuracy
    ensure_dir(base_dir)
    labels = realized_forward_avg_return(prices, args.H).reindex_like(signals)
    labels.to_parquet(f"{base_dir}/labels.parquet")
    acc_base = compute_accuracy_metrics(signals, prices, args.H)
    save_json(acc_base, f"{base_dir}/accuracy.json")

    # Size proxy and groupwise IC/RankIC by size
    size_df = compute_size_proxy(prices, window=20).reindex_like(signals)
    size_groups = ic_by_size_quantile(signals, labels, size_df, q=5)
    save_json(size_groups, f"{base_dir}/ic_by_size.json")

    # Neutralize by size and backtest again
    neut_signals = neutralize_by_size(signals, size_df)
    neut_dir = os.path.join(args.out_dir, "neutralized")
    summary_neut = backtest(
        signals=neut_signals,
        prices=prices,
        benchmark=bench,
        out_dir=neut_dir,
        k=args.k,
        n=args.n,
        min_hold=args.min_hold,
        cost=args.cost_bps / 10000,
    )
    acc_neut = compute_accuracy_metrics(neut_signals, prices, args.H)
    save_json(acc_neut, f"{neut_dir}/accuracy.json")

    # Print brief summaries
    print("Industry (same-sector) 20 names — Base vs Size-Neutralized")
    print(f"Universe: {len(tickers)} tickers from {args.prefer_sector} (or fallback)\n")
    print("Base:")
    print(summary_base)
    print("Accuracy:")
    print(acc_base)
    print("\nNeutralized (size):")
    print(summary_neut)
    print("Accuracy:")
    print(acc_neut)
    print("\nIC by size quantile (base signals):")
    print(size_groups)


if __name__ == "__main__":
    main()
