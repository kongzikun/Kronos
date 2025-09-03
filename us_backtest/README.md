# Kronos US Backtest

This directory provides a one-click script to reproduce a Kronos-style backtest on the S&P 500 universe.

## Requirements

- Python >= 3.10
- `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `yfinance`
- Kronos pretrained weights: place `Kronos-base` and `Kronos-Tokenizer-base` in your HuggingFace cache or local path so that
  `KronosTokenizer.from_pretrained` and `Kronos.from_pretrained` can load them.
  Download from: https://huggingface.co/NeoQuasar/Kronos-base

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

## Run on Google Colab

```python
# clone and install
!git clone <repo-url>
%cd Kronos
!pip install -r requirements.txt

# launch backtest
!python us_backtest/run_kronos_us.py --universe sp500 \
    --start 2015-01-01 --end 2025-08-31 \
    --k 50 --n 5 --cost_bps 15 \
    --H 10 --lookback 90 --min_hold 5 --samples 20
```
