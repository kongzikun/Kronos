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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = args.universe.split(",")

    prices = download_ohlcv(tickers, args.start, args.end, rate_limit_sec=args.yf_rate_limit)
    tokenizer, model = load_model(device)

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
        bench = download_ohlcv(["^GSPC"], args.start, args.end).xs("^GSPC", level=1)["close"]
    except Exception:
        print("[warn] Failed to fetch ^GSPC from Yahoo. Using equal-weight universe proxy as benchmark.")
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
