# Kronos US Backtest

This directory provides a one-click script to reproduce a Kronos-style backtest on the S&P 500 universe.

## Requirements

- Python >= 3.10
- `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `yfinance`, `huggingface_hub`
- Kronos pretrained weights: ensure both are available locally so
  `KronosTokenizer.from_pretrained` and `Kronos.from_pretrained` can resolve them. If not found, the script will print a clear instruction and exit.

  Recommended sources/paths:
  - Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`
  - Model: `NeoQuasar/Kronos-base`

  You can either:
  - Let Hugging Face cache them automatically (requires network access), or
  - Manually download and place under a local directory, then set `HF_HOME` or pass the exact path to `from_pretrained` if you modify the code.
  - Project root contains `model/kronos.py` with the classes needed; weights are still required.

## Usage

```bash
python us_backtest/run_kronos_us.py \
    --universe sp500 \
    --start 2015-01-01 \
    --end 2025-08-31 \
    --k 50 --n 5 --cost_bps 15 \
    --H 10 --lookback 90 --min_hold 5 --samples 20
```

All outputs are saved under `outputs/us_backtest/`:

- `signals.parquet` – daily expected returns per ticker
- `trades.parquet` – executed trades and costs
- `nav.csv` – portfolio, benchmark and excess NAV
- `summary.json` – AER, IR, volatility, win rate, max drawdown and turnover
- `cum_return.png` / `cum_excess_return.png` – performance plots

Set the `--universe` argument to a comma-separated list of tickers to run on a custom universe.

Notes:
- Signals are formed at t close and traded at t+1 open to avoid look-ahead.
- Z-score normalization is fit per-lookback window and clipped to [-5, 5].
- Inference uses Monte Carlo sampling (N=20 by default) with temperature=1.0 and top-p=0.9; trajectories are averaged.
