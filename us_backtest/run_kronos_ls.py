import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
    from .backtest_longshort import backtest_longshort
except Exception:  # pragma: no cover
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
    from backtest_longshort import backtest_longshort


def main():
    p = argparse.ArgumentParser(description="Kronos long-short backtest (offline-friendly; works for crypto)")
    # Data
    p.add_argument("--data_source", choices=["offline", "yahoo", "stooq"], default="offline")
    p.add_argument("--data_path", default="offline_data/crypto_20230101_20250830")
    p.add_argument("--universe", default="custom")
    p.add_argument("--tickers_file", default="", help="Optional file with tickers list (one per line)")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-08-30")

    # Inference
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--model_path", default="")

    # Portfolio (long-short)
    p.add_argument("--k_long", type=int, default=10)
    p.add_argument("--k_short", type=int, default=10)
    p.add_argument("--n_long", type=int, default=3)
    p.add_argument("--n_short", type=int, default=3)
    p.add_argument("--min_hold", type=int, default=5)
    p.add_argument("--cost_bps", type=float, default=10.0)

    p.add_argument("--out_dir", default="outputs/crypto_ls_backtest")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve tickers
    if args.tickers_file:
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    elif args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        # Infer from offline dir or parse --universe CSV
        if args.data_source == "offline":
            cand = infer_tickers_from_offline(args.data_path)
            if cand:
                tickers = cand
            else:
                tickers = [t.strip() for t in args.universe.split(',') if t.strip()]
        else:
            tickers = [t.strip() for t in args.universe.split(',') if t.strip()]

    # Fetch prices
    if args.data_source == "offline":
        prices = load_offline_ohlcv(args.data_path, tickers, args.start, args.end)
    elif args.data_source == "yahoo":
        prices = download_ohlcv(tickers, args.start, args.end, rate_limit_sec=0.4)
    else:
        prices = download_ohlcv_stooq(tickers, args.start, args.end)

    # Benchmark: use EW universe proxy by default (crypto has no SPX)
    bench = prices["close"].unstack("ticker").mean(axis=1)

    # Load model
    tokenizer, model = load_model(
        device,
        tokenizer_path=(args.tokenizer_path or None),
        model_path=(args.model_path or None),
    )
    if device.type == "cuda":
        try:
            _ = torch.randn(8, device=device)[:].clip_(-1, 1)
            torch.cuda.synchronize()
        except Exception:
            device = torch.device("cpu")
            tokenizer, model = load_model(device)

    # Generate signals once
    sig_records = []
    total_est = 0
    for tk in prices.index.get_level_values(1).unique():
        tdf = prices.xs(tk, level=1).dropna()
        n = len(tdf)
        if n >= (args.lookback + args.H):
            span = (n - args.H) - (args.lookback - 1)
            if span > 0:
                total_est += span // max(1, args.stride)

    pbar = tqdm(total=total_est, desc="Generating signals (LS)", dynamic_ncols=True)
    for ticker, x, x_stamp, y_stamp, dates in prepare_windows(prices, args.lookback, args.H, stride=args.stride):
        preds = predict_batch(tokenizer, model, x, x_stamp, y_stamp, device, args.samples, 1.0, 0.9)
        mean_close = preds[:, :, 3].mean(axis=1)
        close_t = x[:, -1, 3]
        ret = (mean_close - close_t) / close_t
        df_sig = pd.DataFrame({"date": dates, "ticker": ticker, "signal": ret})
        sig_records.append(df_sig)
        pbar.update(len(dates))
    pbar.close()

    if not sig_records:
        raise SystemExit("No signals generated — check data & params.")
    signals = pd.concat(sig_records).set_index(["date", "ticker"]).unstack("ticker")["signal"].sort_index()

    # Backtest long-short
    summary = backtest_longshort(
        signals=signals,
        prices=prices,
        benchmark=bench,
        out_dir=args.out_dir,
        k_long=args.k_long,
        k_short=args.k_short,
        n_long=args.n_long,
        n_short=args.n_short,
        min_hold=args.min_hold,
        cost=args.cost_bps / 10000.0,
    )

    # Accuracy
    labels = realized_forward_avg_return(prices, args.H).reindex_like(signals)
    labels.to_parquet(os.path.join(args.out_dir, "labels.parquet"))
    acc = compute_accuracy_metrics(signals, prices, args.H)
    save_json(acc, os.path.join(args.out_dir, "accuracy.json"))

    print("Crypto Long-Short Backtest finished.")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe size: {len(signals.columns)}")
    print(f"AER (excess): {summary.get('AER', float('nan')):.4f}")
    print(f"IR: {summary.get('IR', float('nan')):.3f}")
    print(f"Vol (ann): {summary.get('vol', float('nan')):.3f}")
    print(f"Win rate: {summary.get('win_rate', float('nan')):.3f}")
    print(f"Max DD: {summary.get('max_drawdown', float('nan')):.3f}")
    print(f"Turnover: {summary.get('turnover', float('nan')):.3f}")


if __name__ == "__main__":
    main()

