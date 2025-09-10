import argparse
import os
import sys
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional

import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
except Exception:
    torch = None

from model import Kronos, KronosTokenizer, KronosPredictor


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_btc_hourly(period: str = '60d', csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch BTC-USD hourly OHLCV via yfinance using `period` and `interval='1h'`.
    If `csv_path` is provided, load from CSV with columns [date,open,high,low,close,volume].
    Returns DataFrame indexed by naive UTC timestamps.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
            df = df.set_index('date')
        df = df.sort_index()
        # normalize columns
        cols = {c.lower(): c for c in df.columns}
        required = ['open', 'high', 'low', 'close', 'volume']
        for r in required:
            if r not in [c.lower() for c in df.columns]:
                raise ValueError(f"CSV missing required column: {r}")
        df = df.rename(columns={cols.get('open', 'open'): 'open',
                                cols.get('high', 'high'): 'high',
                                cols.get('low', 'low'): 'low',
                                cols.get('close', 'close'): 'close',
                                cols.get('volume', 'volume'): 'volume'})
        return df[['open','high','low','close','volume']]

    try:
        import yfinance as yf
        data = yf.download('BTC-USD', period=period, interval='1h', progress=False)
        if data is None or data.empty:
            raise RuntimeError("Empty data returned from yfinance.")
        data = data.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        data = data[['open','high','low','close','volume']].dropna()
        # yfinance may return tz-aware index; convert to naive UTC
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC').tz_localize(None)
        return data
    except Exception as e:
        print("Error fetching hourly data from yfinance.", file=sys.stderr)
        print("Provide --csv_path with hourly OHLCV [date,open,high,low,close,volume]", file=sys.stderr)
        raise


def plot_hourly_forecast(hist_close: pd.Series, pred_close: pd.Series, out_path: str, hist_hours: int = 72) -> None:
    """Plot the last `hist_hours` of history and the next horizon prediction as continuation."""
    # keep last hist_hours from history
    hist_close = hist_close.iloc[-hist_hours:]
    plt.figure(figsize=(9, 4))
    plt.plot(hist_close.index, hist_close.values, label='History (Close)', color='C0')
    # connect last history point to first forecast for better continuity
    concat_idx = hist_close.index.tolist() + pred_close.index.tolist()
    concat_vals = hist_close.values.tolist() + pred_close.values.tolist()
    plt.plot(pred_close.index, pred_close.values, label='Forecast (Close)', color='C3', lw=2)
    plt.scatter(pred_close.index, pred_close.values, color='C3', s=10)
    plt.title('BTC-USD 10H Forecast (Kronos)')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Forecast next 10 hours of BTC-USD using Kronos and plot results.')
    parser.add_argument('--period', type=str, default='60d', help='yfinance period for hourly bars (e.g., 30d, 60d, 730d)')
    parser.add_argument('--lookback', type=int, default=480, help='context length (<=512)')
    parser.add_argument('--H', type=int, default=10, help='forecast horizon hours')
    parser.add_argument('--csv_path', type=str, default=None, help='optional hourly CSV path [date,open,high,low,close,volume]')
    parser.add_argument('--out_dir', type=str, default=None, help='optional artifacts output dir')

    args = parser.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = args.out_dir or os.path.join('.', 'artifacts', f'btc_10h_{stamp}')
    ensure_dir(out_dir)

    # Load hourly data
    df = fetch_btc_hourly(period=args.period, csv_path=args.csv_path)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df['amount'] = (df['volume'] * df['close']).fillna(0.0)

    # Select lookback window
    if len(df) < args.lookback + 1:
        print(f"Not enough data: need >= {args.lookback+1} rows, have {len(df)}", file=sys.stderr)
        sys.exit(1)
    x_df = df.iloc[-args.lookback:]

    last_ts = x_df.index[-1]
    # y timestamps are next H hourly steps in UTC
    y_index = pd.date_range(start=last_ts + timedelta(hours=1), periods=args.H, freq='H')

    # Load models
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass
    device = ("cuda:0" if cuda_available else "cpu")

    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    except Exception as e:
        print("Failed to load Kronos models from HF Hub. Ensure internet or local cache.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(1)

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # Predict next H hours
    pred_df = predictor.predict(
        df=x_df[['open','high','low','close','volume','amount']].copy(),
        x_timestamp=x_df.index.to_series(),
        y_timestamp=pd.Series(y_index),
        pred_len=args.H,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=False
    )

    # Save forecast CSV
    pred_path = os.path.join(out_dir, 'forecast_10h.csv')
    pred_df.to_csv(pred_path)

    # Plot
    hist_close = df['close']
    plot_path = os.path.join(out_dir, 'btc_10h_forecast.png')
    plot_hourly_forecast(hist_close, pred_df['close'], plot_path, hist_hours=72)

    # Save run meta
    meta = {
        'model_tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
        'model_base': 'NeoQuasar/Kronos-base',
        'lookback': args.lookback,
        'H': args.H,
        'period': args.period,
        'device': device,
        'out_dir': out_dir
    }
    with open(os.path.join(out_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(out_dir)


if __name__ == '__main__':
    main()

