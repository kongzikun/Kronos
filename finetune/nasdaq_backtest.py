import os
import sys
import argparse
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

# Ensure finetune root is in path to import qlib_test helpers and model
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.append(ROOT_DIR)

from finetune.config import Config
from finetune.qlib_test import QlibTestDataset, generate_predictions


def get_nasdaq100_tickers() -> list:
    # Pull current constituents from Wikipedia
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        candidates = []
        for df in tables:
            cols = {c.lower() for c in df.columns}
            if "ticker" in cols or "symbol" in cols:
                col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
                if col is not None:
                    tickers = df[col].astype(str).str.upper().tolist()
                    candidates.extend(tickers)
        # Clean possible footnotes and dots
        cleaned = []
        for t in candidates:
            t = t.strip().split(" ")[0]
            t = t.replace(".", "-")  # yfinance uses '-' for tickers like BRK.B
            if t and t.isascii() and t.isalnum() or ("-" in t):
                cleaned.append(t)
        cleaned = sorted(list(dict.fromkeys(cleaned)))
        # Filter out common non-ticker artifacts
        cleaned = [t for t in cleaned if len(t) <= 6 and t not in {"TICKER", "SYMBOL"}]
        # In case parsing fails, fallback to a core subset
        if len(cleaned) < 20:
            raise RuntimeError("Parsed too few tickers from Wikipedia")
        return cleaned
    except Exception:
        # Minimal fallback set
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "COST", "TSLA", "PEP",
            "NFLX", "ADBE", "AMD", "INTC", "CMCSA", "TXN", "QCOM", "CSCO", "AMAT", "PDD",
        ]


def download_ohlcv(tickers: list, start: str, end: str) -> dict:
    # yfinance end is exclusive; extend by 1 day
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(tickers=tickers, start=start, end=end_dt.strftime("%Y-%m-%d"), auto_adjust=False, group_by="column")
    # If single ticker, yfinance returns a regular DataFrame
    data = {}
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns MultiIndex with level 0 = Price field, level 1 = Ticker
        lvl0 = df.columns.levels[0].tolist()
        lvl1 = df.columns.levels[1].tolist()
        # Iterate by ticker and extract columns per field safely
        for t in tickers:
            cols = []
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                key = (field, t)
                if key in df.columns:
                    cols.append(key)
            if len(cols) < 5:
                continue
            sub = df[cols]
            x = pd.DataFrame({
                "open": sub[("Open", t)],
                "high": sub[("High", t)],
                "low": sub[("Low", t)],
                "close": sub[("Close", t)],
                "vol": sub[("Volume", t)],
            })
            x["amt"] = (x[["open", "high", "low", "close"]].mean(axis=1)) * x["vol"].fillna(0)
            x = x.dropna()
            if len(x) >= 150:
                data[t] = x
    else:
        # Single ticker path
        x = pd.DataFrame({
            "open": df["Open"],
            "high": df["High"],
            "low": df["Low"],
            "close": df["Close"],
            "vol": df["Volume"],
        })
        x["amt"] = (x[["open", "high", "low", "close"]].mean(axis=1)) * x["vol"].fillna(0)
        x = x.dropna()
        data[tickers[0]] = x
    return data


def build_test_data(raw: dict, test_start: str, test_end: str, feature_list: list) -> dict:
    test_data = {}
    for t, df in raw.items():
        sub = df.copy()
        sub.index.name = "datetime"
        # keep only required features
        sub = sub[feature_list]
        mask = (sub.index >= test_start) & (sub.index <= test_end)
        sub = sub[mask]
        if len(sub) > 0:
            test_data[t] = sub
    return test_data


def simple_topk_backtest(pred_df: pd.DataFrame, close_df: pd.DataFrame, topk: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Align indices/columns
    pred_df, close_df = pred_df.align(close_df, join="inner", axis=None)
    # Next-day simple returns
    fwd_ret = close_df.shift(-1) / close_df - 1.0
    fwd_ret = fwd_ret.loc[pred_df.index]

    # For each date, select top-K by prediction and compute equal-weight returns
    def row_topk_ret(row):
        # row is predictions for a single date
        top_cols = row.nlargest(topk).index
        return fwd_ret.loc[row.name, top_cols].mean(skipna=True)

    port_ret = pred_df.apply(row_topk_ret, axis=1)
    port_ret = port_ret.dropna()
    # Build result dataframes similar to qlib_test
    report_df = pd.DataFrame({
        "cum_return_w_cost": port_ret.cumsum(),
    })
    return report_df, port_ret


def get_benchmark_series(start: str, end: str) -> pd.Series:
    # Use QQQ as NASDAQ-100 ETF
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    bench = yf.download(tickers=["QQQ"], start=start, end=end_dt.strftime("%Y-%m-%d"), auto_adjust=False, group_by="column")
    if isinstance(bench.columns, pd.MultiIndex):
        close = bench[("Close", "QQQ")].rename("Close")
    else:
        close = bench["Close"]
    ret = close.pct_change()
    return ret


def main():
    parser = argparse.ArgumentParser(description="NASDAQ-100 backtest using Kronos pre-trained models")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--signal", type=str, default="mean", choices=["mean", "last", "max", "min"], help="Which signal to backtest")
    parser.add_argument("--topk", type=int, default=50, help="Portfolio size")
    parser.add_argument("--samples", type=int, default=1, help="Sampling paths per step (inference)")
    args = parser.parse_args()

    cfg = Config()
    # Use pre-trained models from HF Hub if finetuned not available
    tokenizer_path = cfg.finetuned_tokenizer_path if os.path.exists(cfg.finetuned_tokenizer_path) else "NeoQuasar/Kronos-Tokenizer-base"
    model_path = cfg.finetuned_predictor_path if os.path.exists(cfg.finetuned_predictor_path) else "NeoQuasar/Kronos-small"

    # Date ranges
    test_start, test_end = cfg.test_time_range
    bt_start, bt_end = cfg.backtest_time_range

    # 1) Get tickers and download OHLCV
    print("Fetching NASDAQ-100 tickers...")
    tickers = get_nasdaq100_tickers()
    print(f"Tickers loaded: {len(tickers)}")
    # pad lookback
    start_for_dl = (pd.to_datetime(test_start) - pd.Timedelta(days=cfg.lookback_window + 10)).strftime("%Y-%m-%d")
    print(f"Downloading OHLCV from {start_for_dl} to {bt_end}...")
    raw = download_ohlcv(tickers, start_for_dl, bt_end)

    # 2) Build test_data compatible with QlibTestDataset
    print("Building test dataset for inference...")
    test_data = build_test_data(raw, test_start, test_end, cfg.feature_list)

    # 3) Run predictions
    run_config = {
        'device': args.device,
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'max_context': cfg.max_context,
        'pred_len': cfg.predict_window,
        'clip': cfg.clip,
        'T': cfg.inference_T,
        'top_k': cfg.inference_top_k,
        'top_p': cfg.inference_top_p,
        'sample_count': int(args.samples),
        'batch_size': min(256, cfg.backtest_batch_size),
    }

    # Output dir for caching
    out_dir = os.path.join(ROOT_DIR, "outputs", "backtest_results", "nasdaq_backtest")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f"predictions_s{args.samples}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached predictions from {cache_path}...")
        with open(cache_path, "rb") as f:
            preds = pickle.load(f)
    else:
        print("Generating predictions...")
        preds = generate_predictions(run_config, test_data)
        with open(cache_path, "wb") as f:
            pickle.dump(preds, f)
        print(f"Saved predictions cache to {cache_path}")
    pred_df = preds[args.signal]

    # 4) Prepare close price matrix aligned to prediction dates
    print("Preparing close price matrix for backtest...")
    close_dict = {}
    for t, df in raw.items():
        close_dict[t] = df["close"]
    close_df = pd.DataFrame(close_dict).sort_index()
    # Restrict to the backtest window
    mask = (close_df.index >= bt_start) & (close_df.index <= bt_end)
    close_df = close_df[mask]
    pred_df = pred_df.loc[(pred_df.index >= bt_start) & (pred_df.index <= bt_end)]

    # 5) Simple top-K backtest
    print("Running simple top-K backtest...")
    report_df, port_daily_ret = simple_topk_backtest(pred_df, close_df, topk=args.topk)

    # 6) Benchmark
    bench_ret = get_benchmark_series(bt_start, bt_end)
    bench_ret = bench_ret.loc[report_df.index]
    cum_bench = bench_ret.cumsum()

    # 7) Plot results
    os.makedirs(os.path.join(ROOT_DIR, "figures"), exist_ok=True)
    fig_path = os.path.join(ROOT_DIR, "figures", "nasdaq_backtest_result.png")
    plt.figure(figsize=(12, 6))
    plt.plot(report_df.index, report_df["cum_return_w_cost"], label=f"Strategy ({args.signal}, top{args.topk})")
    plt.plot(cum_bench.index, cum_bench.values, label="QQQ Benchmark", linestyle="--", color="black")
    plt.title("Cumulative Return (Simple Top-K) vs QQQ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"Saved figure to {fig_path}")

    # 8) Save raw results
    with open(os.path.join(out_dir, "predictions.pkl"), "wb") as f:
        pickle.dump(preds, f)
    report_df.to_csv(os.path.join(out_dir, "report_cum.csv"))
    port_daily_ret.rename("return").to_csv(os.path.join(out_dir, "portfolio_daily_return.csv"))
    cum_bench.rename("bench").to_csv(os.path.join(out_dir, "benchmark_cum.csv"))
    print(f"Results saved under {out_dir}")


if __name__ == "__main__":
    main()
