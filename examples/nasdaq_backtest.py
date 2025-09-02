import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yfinance as yf

# add repository root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor


def main():
    """Backtest Kronos-base on NASDAQ Composite (^IXIC) up to August 2025."""
    # configuration
    lookback = 512  # maximum context length for Kronos-base
    pred_len = 100  # number of days to backtest
    end_date = pd.Timestamp("2025-08-31")
    start_date = end_date - pd.DateOffset(years=3)

    # 1. Load historical data
    df = yf.download("^IXIC", start=start_date, end=end_date, interval="1d")
    df = df.droplevel(1, axis=1)
    df = df.reset_index()
    df.rename(
        columns={
            "Date": "timestamps",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df = df[["timestamps", "open", "high", "low", "close", "volume"]]

    # 2. Prepare backtest windows
    hist_df = df.iloc[-(lookback + pred_len):].reset_index(drop=True)
    x_df = hist_df.iloc[:lookback, 1:]
    x_timestamp = hist_df.iloc[:lookback, 0]
    y_timestamp = hist_df.iloc[lookback:, 0]
    actual_df = hist_df.iloc[lookback:, 1:]

    # 3. Load Kronos-base model and tokenizer
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # 4. Predict future K-lines within backtest window
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    # 5. Plot comparison between ground truth and prediction
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Close price
    axes[0].plot(hist_df["timestamps"], hist_df["close"], label="Ground Truth", color="blue")
    axes[0].plot(y_timestamp, pred_df["close"], label="Prediction", color="red")
    axes[0].set_ylabel("Close Price")
    axes[0].legend()

    # Volume
    axes[1].plot(hist_df["timestamps"], hist_df["volume"], label="Ground Truth", color="blue")
    axes[1].plot(y_timestamp, pred_df["volume"], label="Prediction", color="red")
    axes[1].set_ylabel("Volume")
    axes[1].legend()

    # Time axis formatting
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "figures" / "nasdaq_backtest.png"
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    main()
