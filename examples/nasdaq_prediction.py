import sys
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor


def main():
    """Forecast NASDAQ Composite (^IXIC) for the next 3 months using Kronos-base."""
    # configuration
    lookback = 512  # maximum context length for Kronos-base
    pred_len = 60   # ~3 months of trading days

    # 1. Load historical daily data from Yahoo Finance
    df = yf.download("^IXIC", period="3y", interval="1d")
    df = df.droplevel(1, axis=1)
    df = df.reset_index()
    df.rename(columns={
        "Date": "timestamps",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }, inplace=True)
    df = df[["timestamps", "open", "high", "low", "close", "volume"]]

    # prepare lookback window
    x_df = df.iloc[-lookback:, 1:]
    x_timestamp = df.iloc[-lookback:, 0]
    last_date = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len))

    # 2. Load Kronos-base model and tokenizer
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # 3. Forecast future K-lines
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

    print("Forecasted Data Head:")
    print(pred_df.head())

    # 4. Combine historical and predicted data
    hist_df = x_df.copy()
    hist_df.index = pd.DatetimeIndex(x_timestamp)
    pred_df = pred_df[["open", "high", "low", "close", "volume"]]
    pred_df.index = pd.DatetimeIndex(y_timestamp)
    combined_df = pd.concat([hist_df, pred_df])

    # 5. Plot candlestick chart
    plot_df = combined_df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="charles",
        title="NASDAQ Composite (^IXIC) Forecast - Next 3 Months",
    )


if __name__ == "__main__":
    main()
