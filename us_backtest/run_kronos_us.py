import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# Prefer package-relative imports (python -m us_backtest.run_kronos_us); fallback for direct script run
try:
    from .utils import (
        set_seed,
        realized_forward_avg_return,
        compute_accuracy_metrics,
        ensure_dir,
        save_json,
    )
    from .data_us import get_sp500_tickers, download_ohlcv, prepare_windows
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
    from data_us import get_sp500_tickers, download_ohlcv, prepare_windows
    from kronos_infer import load_model, predict_batch
    from backtest_topk_dropn import backtest


def main():
    parser = argparse.ArgumentParser(description="Kronos US backtest")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--cost_bps", type=float, default=15)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--min_hold", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--out_dir", default="outputs/us_backtest")
    parser.add_argument("--yf_rate_limit", type=float, default=0.5, help="Sleep seconds between Yahoo requests to reduce 429s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device selection")
    parser.add_argument("--tokenizer_path", default="", help="Optional local path to KronosTokenizer weights")
    parser.add_argument("--model_path", default="", help="Optional local path to Kronos model weights")
    # Data source options
    parser.add_argument("--data_source", choices=["yahoo", "offline", "stooq"], default="yahoo",
                        help="Price source: yahoo (yfinance), offline (CSV/Parquet), stooq (pandas-datareader)")
    parser.add_argument("--data_path", default="", help="Path to offline data dir or file when data_source=offline")
    parser.add_argument("--tickers_file", default="", help="Optional file with tickers (one per line); overrides --universe when set")
    args = parser.parse_args()

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
        if args.data_source == "offline" and args.data_path:
            try:
                from .data_us import infer_tickers_from_offline
            except Exception:
                from data_us import infer_tickers_from_offline
            tickers = infer_tickers_from_offline(args.data_path)
            if not tickers:
                raise SystemExit("offline mode with universe=sp500 requires data_path containing per-ticker files")
        else:
            tickers = get_sp500_tickers()
    else:
        tickers = [t.strip() for t in args.universe.split(",") if t.strip()]

    # Fetch prices according to data_source
    if args.data_source == "yahoo":
        prices = download_ohlcv(tickers, args.start, args.end, rate_limit_sec=args.yf_rate_limit)
    elif args.data_source == "offline":
        try:
            from .data_us import load_offline_ohlcv
        except Exception:
            from data_us import load_offline_ohlcv
        prices = load_offline_ohlcv(args.data_path, tickers, args.start, args.end)
    elif args.data_source == "stooq":
        try:
            from .data_us import download_ohlcv_stooq
        except Exception:
            from data_us import download_ohlcv_stooq
        prices = download_ohlcv_stooq(tickers, args.start, args.end)
    else:
        raise SystemExit(f"Unknown data_source: {args.data_source}")
    # Load model (supports local paths via CLI or env)
    tokenizer, model = load_model(
        device,
        tokenizer_path=(args.tokenizer_path or None),
        model_path=(args.model_path or None),
    )

    # If CUDA is selected but kernels are incompatible (e.g., arch too new),
    # detect via a simple GPU op and fallback to CPU automatically.
    if device.type == "cuda":
        try:
            x_smoke = torch.randn(8, device=device)
            x_smoke = torch.clip(x_smoke, -1, 1)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[warn] CUDA smoke test failed ({e}); falling back to CPU.")
            device = torch.device("cpu")
            tokenizer, model = load_model(
                device,
                tokenizer_path=(args.tokenizer_path or None),
                model_path=(args.model_path or None),
            )

    signal_records = []
    for ticker, x, x_stamp, y_stamp, dates in prepare_windows(prices, args.lookback, args.H):
        preds = predict_batch(tokenizer, model, x, x_stamp, y_stamp, device, args.samples, 1.0, 0.9)
        # preds shape: [batch, H, features]; average close over horizon as per paper
        mean_close = preds[:, :, 3].mean(axis=1)
        close_t = x[:, -1, 3]
        ret = (mean_close - close_t) / close_t
        df_sig = pd.DataFrame({"date": dates, "ticker": ticker, "signal": ret})
        signal_records.append(df_sig)

    if not signal_records:
        raise SystemExit("No signals generated. Check data availability.")

    signal_df = pd.concat(signal_records).set_index(["date", "ticker"]).unstack("ticker")["signal"].sort_index()

    # Benchmark: try ^GSPC; if unavailable (network blocked or rate-limited),
    # fall back to equal-weighted universe close as a proxy.
    try:
        if args.data_source == "yahoo":
            bench = download_ohlcv(["^GSPC"], args.start, args.end).xs("^GSPC", level=1)["close"]
        elif args.data_source == "stooq":
            try:
                from .data_us import download_ohlcv_stooq
            except Exception:
                from data_us import download_ohlcv_stooq
            # Stooq symbol for S&P 500 index can vary; '^SPX' or '^US500' often not available.
            # Fall back to EW proxy if fetch fails.
            bench = download_ohlcv_stooq(["^SPX"], args.start, args.end).xs("^SPX", level=1)["close"]
        else:
            raise RuntimeError("No online benchmark in offline mode")
    except Exception:
        print("[warn] Failed to fetch benchmark. Using equal-weight universe proxy as benchmark.")
        bench = prices["close"].unstack("ticker").mean(axis=1)

    summary = backtest(
        signals=signal_df,
        prices=prices,
        benchmark=bench,
        out_dir=args.out_dir,
        k=args.k,
        n=args.n,
        min_hold=args.min_hold,
        cost=args.cost_bps / 10000,
    )

    # Accuracy diagnostics vs realized future returns
    ensure_dir(args.out_dir)
    labels_df = realized_forward_avg_return(prices, args.H).reindex_like(signal_df)
    labels_df.to_parquet(f"{args.out_dir}/labels.parquet")
    acc = compute_accuracy_metrics(signal_df, prices, args.H)
    save_json(acc, f"{args.out_dir}/accuracy.json")

    # Console summary
    print("US Backtest finished.")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe size: {len(signal_df.columns)}")
    print(f"AER (excess): {summary.get('AER', float('nan')):.4f}")
    print(f"IR: {summary.get('IR', float('nan')):.3f}")
    print(f"Vol (ann): {summary.get('vol', float('nan')):.3f}")
    print(f"Win rate: {summary.get('win_rate', float('nan')):.3f}")
    print(f"Max DD: {summary.get('max_drawdown', float('nan')):.3f}")
    print(f"Turnover: {summary.get('turnover', float('nan')):.3f}")
    print("-- Forecast accuracy (pred vs realized forward avg return) --")
    print(f"MAE: {acc.get('MAE', float('nan')):.4f}")
    print(f"RMSE: {acc.get('RMSE', float('nan')):.4f}")
    print(f"Corr_all: {acc.get('Corr_all', float('nan')):.3f}")
    print(f"DA: {acc.get('DA', float('nan')):.3f}")
    print(f"IC: {acc.get('IC', float('nan')):.3f} (t={acc.get('IC_t', float('nan')):.2f})")
    print(f"RankIC: {acc.get('RankIC', float('nan')):.3f} (t={acc.get('RankIC_t', float('nan')):.2f})")


if __name__ == "__main__":
    main()
