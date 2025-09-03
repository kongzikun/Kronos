import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
# Ensure repository root is importable (for `model` package)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import set_seed
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = args.universe.split(",")

    prices = download_ohlcv(tickers, args.start, args.end)
    tokenizer, model = load_model(device)

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

    signal_df = pd.concat(signal_records).set_index(["date", "ticker"]).unstack("ticker")["signal"].sort_index()

    bench = download_ohlcv(["^GSPC"], args.start, args.end).xs("^GSPC", level=1)["close"]

    backtest(
        signals=signal_df,
        prices=prices,
        benchmark=bench,
        out_dir=args.out_dir,
        k=args.k,
        n=args.n,
        min_hold=args.min_hold,
        cost=args.cost_bps / 10000,
    )


if __name__ == "__main__":
    main()
