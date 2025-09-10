import argparse
import os
import sys
from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Local imports (support running as module or script)
try:
    from .utils import (
        set_seed,
        realized_forward_avg_return,
        compute_accuracy_metrics,
        ensure_dir,
        save_json,
    )
    from .data_us import (
        get_sp500_tickers,
        download_ohlcv,
        load_offline_ohlcv,
        download_ohlcv_stooq,
        prepare_windows,
        infer_tickers_from_offline,
    )
    from .kronos_infer import load_model, predict_batch
    from .backtest_topk_dropn import backtest
except Exception:  # pragma: no cover - direct script fallback
    from utils import (
        set_seed,
        realized_forward_avg_return,
        compute_accuracy_metrics,
        ensure_dir,
        save_json,
    )
    from data_us import (
        get_sp500_tickers,
        download_ohlcv,
        load_offline_ohlcv,
        download_ohlcv_stooq,
        prepare_windows,
        infer_tickers_from_offline,
    )
    from kronos_infer import load_model, predict_batch
    from backtest_topk_dropn import backtest


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Grid search for US backtest (portfolio params) with fixed cost")
    # Data/model args
    p.add_argument("--data_source", choices=["offline", "yahoo", "stooq"], default="offline")
    p.add_argument("--data_path", default="offline_data/nasdaq100_20230101_20250830")
    p.add_argument("--universe", default="sp500")
    p.add_argument("--tickers_file", default="")
    p.add_argument("--start", default="2024-07-01")
    p.add_argument("--end", default="2025-08-30")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--model_path", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Inference (fixed across grid to amortize cost)
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)

    # Grid (portfolio params only)
    p.add_argument("--k_list", type=str, default="10,25,50")
    p.add_argument("--n_list", type=str, default="2,5,10")
    p.add_argument("--min_hold_list", type=str, default="2,5,10")
    p.add_argument("--cost_bps", type=float, default=10.0, help="Fixed trading cost in bps for all runs")
    p.add_argument("--objective", choices=["IR", "AER"], default="IR")

    # Output
    p.add_argument("--out_dir", default="outputs/us_gridsearch")

    args = p.parse_args()
    set_seed(args.seed)

    # Device selection
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.out_dir)

    # Resolve tickers
    if args.tickers_file:
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    elif args.universe == "sp500":
        if args.data_source == "offline" and args.data_path:
            tickers = infer_tickers_from_offline(args.data_path)
            if not tickers:
                raise SystemExit("offline mode with universe=sp500 requires data_path containing per-ticker files")
        else:
            tickers = get_sp500_tickers()
    else:
        tickers = [t.strip() for t in args.universe.split(",") if t.strip()]

    # Load prices
    if args.data_source == "offline":
        prices = load_offline_ohlcv(args.data_path, tickers, args.start, args.end)
    elif args.data_source == "yahoo":
        prices = download_ohlcv(tickers, args.start, args.end, rate_limit_sec=0.4)
    else:
        prices = download_ohlcv_stooq(tickers, args.start, args.end)

    # Benchmark (best effort)
    try:
        if args.data_source == "yahoo":
            bench = download_ohlcv(["^GSPC"], args.start, args.end).xs("^GSPC", level=1)["close"]
        elif args.data_source == "stooq":
            bench = download_ohlcv_stooq(["^SPX"], args.start, args.end).xs("^SPX", level=1)["close"]
        else:
            raise RuntimeError("No online benchmark in offline mode")
    except Exception:
        # Equal-weight proxy in offline or failure cases
        bench = prices["close"].unstack("ticker").mean(axis=1)

    # Load model
    tokenizer, model = load_model(
        device,
        tokenizer_path=(args.tokenizer_path or None),
        model_path=(args.model_path or None),
    )

    # CUDA smoke test fallback
    if device.type == "cuda":
        try:
            _ = torch.randn(8, device=device)[:].clip_(-1, 1)
            torch.cuda.synchronize()
        except Exception:
            device = torch.device("cpu")
            tokenizer, model = load_model(device)

    # 1) Generate signals once (fixed inference params)
    sig_records = []
    total_est = 0
    for tk in prices.index.get_level_values(1).unique():
        tdf = prices.xs(tk, level=1).dropna()
        n = len(tdf)
        if n >= (args.lookback + args.H):
            span = (n - args.H) - (args.lookback - 1)
            if span > 0:
                total_est += span // max(1, args.stride)

    with tqdm(total=total_est, desc="Generating signals (once)", dynamic_ncols=True) as pbar:
        for ticker, x, x_stamp, y_stamp, dates in prepare_windows(prices, args.lookback, args.H, stride=args.stride):
            preds = predict_batch(tokenizer, model, x, x_stamp, y_stamp, device, args.samples, 1.0, 0.9)
            mean_close = preds[:, :, 3].mean(axis=1)
            close_t = x[:, -1, 3]
            ret = (mean_close - close_t) / close_t
            df_sig = pd.DataFrame({"date": dates, "ticker": ticker, "signal": ret})
            sig_records.append(df_sig)
            pbar.update(len(dates))

    if not sig_records:
        raise SystemExit("No signals generated — check data & params.")
    signals = pd.concat(sig_records).set_index(["date", "ticker"]).unstack("ticker")["signal"].sort_index()
    labels = realized_forward_avg_return(prices, args.H).reindex_like(signals)
    signals.to_parquet(os.path.join(args.out_dir, "signals.parquet"))
    labels.to_parquet(os.path.join(args.out_dir, "labels.parquet"))

    # 2) Grid over portfolio params only
    k_list = parse_int_list(args.k_list)
    n_list = parse_int_list(args.n_list)
    mh_list = parse_int_list(args.min_hold_list)
    combos: List[Tuple[int, int, int]] = list(product(k_list, n_list, mh_list))
    rows = []

    for (k, n, mh) in tqdm(combos, desc="Backtesting grid", dynamic_ncols=True):
        run_dir = os.path.join(args.out_dir, f"k{k}_n{n}_mh{mh}")
        summary = backtest(
            signals=signals,
            prices=prices,
            benchmark=bench,
            out_dir=run_dir,
            k=k,
            n=n,
            min_hold=mh,
            cost=args.cost_bps / 10000.0,
        )
        # Accuracy as supporting info (not optimization target unless desired)
        acc = compute_accuracy_metrics(signals, prices, args.H)
        save_json(acc, os.path.join(run_dir, "accuracy.json"))
        rows.append({
            "k": k,
            "n": n,
            "min_hold": mh,
            "cost_bps": args.cost_bps,
            **summary,
        })

    df = pd.DataFrame(rows)
    df.sort_values(args.objective, ascending=False, inplace=True)
    out_csv = os.path.join(args.out_dir, "grid_summary.csv")
    df.to_csv(out_csv, index=False)

    # Print top-5
    top5 = df.head(5)
    print("Top-5 combos (sorted by {}):".format(args.objective))
    print(top5)
    print(f"Saved full grid results to: {out_csv}")


if __name__ == "__main__":
    main()

