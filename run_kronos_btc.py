import argparse
import os
import sys
import math
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional

# Determinism
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

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    print("scipy is required for correlation metrics. Please install: pip install scipy", file=sys.stderr)
    raise

try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
    )
except Exception as e:
    print("scikit-learn is required for accuracy metrics. Please install: pip install scikit-learn", file=sys.stderr)
    raise

# Kronos imports (as specified)
from model import Kronos, KronosTokenizer, KronosPredictor


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_btc_data(start: str, end: Optional[str] = None, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch BTC-USD daily OHLCV via yfinance. If network blocked or yfinance missing,
    read from csv_path (must contain columns: date,open,high,low,close,volume).
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        df = df.sort_index()
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        required = ['open', 'high', 'low', 'close', 'volume']
        for r in required:
            if r not in [c.lower() for c in df.columns]:
                raise ValueError(f"CSV missing required column: {r}")
        # Align to lowercase names
        df = df.rename(columns={cols.get('open', 'open'): 'open',
                                cols.get('high', 'high'): 'high',
                                cols.get('low', 'low'): 'low',
                                cols.get('close', 'close'): 'close',
                                cols.get('volume', 'volume'): 'volume'})
        return df[['open','high','low','close','volume']]

    # Try yfinance fetch
    try:
        import yfinance as yf
        end_use = end if end and end != 'auto' else datetime.utcnow().strftime('%Y-%m-%d')
        data = yf.download('BTC-USD', start=start, end=end_use, interval='1d', progress=False)
        if data is None or data.empty:
            raise RuntimeError("Empty data returned from yfinance.")
        data = data.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        data = data[['open','high','low','close','volume']].dropna()
        return data
    except Exception as e:
        print("Error fetching data from yfinance (network likely blocked).", file=sys.stderr)
        print("Please supply a CSV with columns [date,open,high,low,close,volume] using --csv_path.", file=sys.stderr)
        raise


def compute_labels_avg5(close: pd.Series, step: int = 5) -> pd.Series:
    """
    Computes realized next-5-day average return:
      y_true_avg5(t) = ( mean(close[t+1..t+5]) - close[t] ) / close[t]
    step=5 for non-overlap; step=1 for overlap.
    """
    future_mean = (
        sum(close.shift(-k) for k in range(1, 6)) / 5.0
    )
    y = (future_mean - close) / close
    # For non-overlap primary, we'll later subsample with step=5 in metric eval
    if step == 5:
        # Drop indices without full future window
        valid = ~pd.concat([close.shift(-k) for k in range(1, 6)], axis=1).isna().any(axis=1)
        y = y.where(valid)
    return y


def t_stat_from_r(r: float, n: int) -> float:
    if n < 3 or np.isnan(r):
        return np.nan
    denom = max(1e-12, 1 - r**2)
    return r * math.sqrt(max(0, n - 2) / denom)


def compute_accuracy_metrics(mu_hat: pd.Series, y_true: pd.Series, segment_name: str, eps: float = 0.005) -> Dict[str, float]:
    """
    Aligns and computes metrics: MAE, RMSE, R2, IC, IC_t, RankIC, RankIC_t,
    DA_raw, DA_eps, DA_eps_adapt, BA, ROC_AUC, PR_AUC, and confusion matrix counts.
    """
    # Build a simple 2-column DataFrame with explicit names to avoid MultiIndex and 2D values
    s = pd.concat([mu_hat, y_true], axis=1)
    s.columns = ['mu_hat', 'y_true']
    s = s.dropna()
    if len(s) == 0:
        return {k: np.nan for k in [
            'MAE','RMSE','R2','IC','IC_t','RankIC','RankIC_t','DA_raw','DA_eps','DA_eps_adapt','BA','ROC_AUC','PR_AUC',
            'TP','FP','FN','TN','N']}

    # Ensure 1-D arrays
    y = np.asarray(s['y_true'].values).reshape(-1)
    p = np.asarray(s['mu_hat'].values).reshape(-1)
    n = len(s)

    # Regression metrics
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(mean_squared_error(y, p))
    r2 = r2_score(y, p) if n >= 2 else np.nan
    ic = np.corrcoef(y, p)[0, 1] if n >= 2 else np.nan
    try:
        rank_ic, _ = spearmanr(y, p)
    except Exception:
        rank_ic = np.nan
    ic_t = t_stat_from_r(ic, n)
    rank_ic_t = t_stat_from_r(rank_ic, n)

    # Directional metrics
    y_sign = np.sign(y)
    p_sign = np.sign(p)
    da_raw = np.mean((p_sign == y_sign) & (y_sign != 0)) if n > 0 else np.nan

    # EPS thresholded accuracy
    mask_eps = np.abs(y) >= eps
    da_eps = np.mean((p_sign[mask_eps] == y_sign[mask_eps]) & (y_sign[mask_eps] != 0)) if mask_eps.any() else np.nan

    # Adaptive EPS
    # Rolling std over 60 samples of y_true, scaled by 0.25; floor at 0.002
    y_series = pd.Series(y, index=s.index)
    roll_std = y_series.rolling(60, min_periods=20).std()
    eps_adapt_series = np.maximum(0.002, 0.25 * roll_std)
    mask_adapt = np.abs(y_series) >= eps_adapt_series
    sel = mask_adapt.values
    da_eps_adapt = np.mean((p_sign[sel] == y_sign[sel]) & (y_sign[sel] != 0)) if sel.any() else np.nan

    # Balanced Accuracy and Confusion Matrix (threshold at 0)
    y_bin = (y > 0).astype(int)
    p_bin = (p > 0).astype(int)
    try:
        ba = balanced_accuracy_score(y_bin, p_bin)
        tn, fp, fn, tp = confusion_matrix(y_bin, p_bin, labels=[0, 1]).ravel()
    except Exception:
        ba = np.nan
        tn = fp = fn = tp = np.nan

    # ROC-AUC and PR-AUC (directional classification; score = mu_hat)
    try:
        roc_auc = roc_auc_score(y_bin, p) if len(np.unique(y_bin)) > 1 else np.nan
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_bin, p) if len(np.unique(y_bin)) > 1 else np.nan
    except Exception:
        pr_auc = np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'IC': ic,
        'IC_t': ic_t,
        'RankIC': rank_ic,
        'RankIC_t': rank_ic_t,
        'DA_raw': da_raw,
        'DA_eps': da_eps,
        'DA_eps_adapt': da_eps_adapt,
        'BA': ba,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'TP': float(tp) if not np.isnan(tp) else np.nan,
        'FP': float(fp) if not np.isnan(fp) else np.nan,
        'FN': float(fn) if not np.isnan(fn) else np.nan,
        'TN': float(tn) if not np.isnan(tn) else np.nan,
        'N': float(n)
    }


def make_calibration(mu_hat: pd.Series, y_true: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    s = pd.concat([mu_hat, y_true], axis=1)
    s.columns = ['mu_hat', 'y_true']
    s = s.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=['bin', 'mu_hat_min', 'mu_hat_max', 'count', 'y_true_mean'])
    try:
        s['bin'] = pd.qcut(s['mu_hat'], q=n_bins, duplicates='drop')
    except Exception:
        # Fallback to equal-width bins
        s['bin'] = pd.cut(s['mu_hat'], bins=n_bins)
    df = s.groupby('bin').agg(
        mu_hat_min=('mu_hat', 'min'),
        mu_hat_max=('mu_hat', 'max'),
        count=('y_true', 'count'),
        y_true_mean=('y_true', 'mean')
    ).reset_index(drop=False)
    df['bin'] = df['bin'].astype(str)
    return df


def plot_scatter(mu_hat: pd.Series, y_true: pd.Series, out_path: str) -> None:
    s = pd.concat([mu_hat, y_true], axis=1)
    s.columns = ['mu_hat', 'y_true']
    s = s.dropna()
    if len(s) == 0:
        return
    p = np.asarray(s['mu_hat'].values).reshape(-1)
    y = np.asarray(s['y_true'].values).reshape(-1)
    r = np.corrcoef(y, p)[0, 1] if len(s) > 1 else np.nan
    r2 = r2_score(y, p) if len(s) > 1 else np.nan
    coef = np.polyfit(p, y, 1)
    xline = np.linspace(min(p), max(p), 100)
    yline = coef[0] * xline + coef[1]
    plt.figure(figsize=(6, 5))
    plt.scatter(p, y, alpha=0.5, s=12)
    plt.plot(xline, yline, color='red', lw=2, label=f"OLS fit")
    plt.title("Forecast vs Realized (avg 5d return)")
    plt.xlabel("mu_hat")
    plt.ylabel("y_true")
    plt.legend()
    plt.text(0.02, 0.98, f"R^2 = {r2:.3f}\nPearson r = {r:.3f}", transform=plt.gca().transAxes,
             ha='left', va='top', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.7))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(df_bins: pd.DataFrame, out_path: str) -> None:
    if df_bins.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(df_bins)), df_bins['y_true_mean'], color='C0')
    for i, cnt in enumerate(df_bins['count']):
        plt.text(i, df_bins['y_true_mean'].iloc[i], str(int(cnt)), ha='center', va='bottom', fontsize=8)
    plt.xticks(range(len(df_bins)), [f"{i+1}" for i in range(len(df_bins))])
    plt.xlabel("mu_hat decile")
    plt.ylabel("Realized mean avg5 return")
    plt.title("Calibration by mu_hat decile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: Tuple[int, int, int, int], out_path: str, title: str) -> None:
    tn, fp, fn, tp = cm
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    table_data = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1, 2)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_equity_vs_bh(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df['equity_strat'], label='Strategy', lw=1.5)
    plt.plot(df.index, df['equity_bh'], label='Buy & Hold', lw=1.5)
    plt.legend()
    plt.title("Equity Curve vs Buy & Hold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drawdown(df: pd.DataFrame, out_path: str) -> None:
    eq = df['equity_strat']
    rolling_max = eq.cummax()
    dd = (eq / rolling_max) - 1.0
    plt.figure(figsize=(8, 3))
    plt.fill_between(df.index, dd, 0, color='C3', alpha=0.5)
    plt.title("Strategy Drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_muhat_ic(ic: float, ic_t: float, rank_ic: float, rank_ic_t: float, out_path: str) -> None:
    plt.figure(figsize=(5, 4))
    vals = [ic, rank_ic]
    labels = ['IC', 'RankIC']
    plt.bar(labels, vals, color=['C0','C1'])
    plt.title("Correlation Diagnostics (OOS, non-overlap)")
    for i, (v, t) in enumerate(zip(vals, [ic_t, rank_ic_t])):
        y = v if np.isfinite(v) else 0.0
        t_str = f"{t:.2f}" if np.isfinite(t) else "nan"
        v_str = f"{v:.3f}" if np.isfinite(v) else "nan"
        plt.text(i, y, f"{v_str}\n(t={t_str})", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_position_and_mu(df: pd.DataFrame, out_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df.index, df['mu_hat'], color='C0', label='mu_hat', lw=1.2)
    ax1.set_ylabel('mu_hat', color='C0')
    ax2 = ax1.twinx()
    ax2.step(df.index, df['position'], where='post', color='C2', label='position')
    ax2.set_ylabel('position', color='C2')
    plt.title("mu_hat and Position")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_strategy_s1(mu_hat: pd.Series, close: pd.Series, q_hi_thr: float, q_lo_thr: float, min_hold: int, cost_bp: float) -> pd.DataFrame:
    # Ensure 1-D Series inputs
    if isinstance(mu_hat, pd.DataFrame):
        if mu_hat.shape[1] == 1:
            mu_hat = mu_hat.iloc[:, 0]
        else:
            mu_hat = mu_hat.squeeze()
            if not isinstance(mu_hat, pd.Series):
                mu_hat = pd.Series(mu_hat, index=close.index)
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            # Prefer column named 'close' if available
            if 'close' in close.columns:
                close = close['close']
            else:
                close = close.iloc[:, 0]
        else:
            close = close.squeeze()
            if not isinstance(close, pd.Series):
                close = pd.Series(close)

    idx = mu_hat.index
    position = pd.Series(0.0, index=idx)
    in_pos = False
    hold = 0
    for t in range(len(idx)):
        mu = mu_hat.iloc[t]
        if not in_pos:
            # Enter long when mu_hat >= q_hi
            if mu >= q_hi_thr:
                in_pos = True
                hold = 0
        else:
            hold += 1
            # Exit when mu_hat <= q_lo and min hold satisfied
            if (mu <= q_lo_thr) and (hold >= min_hold):
                in_pos = False
                hold = 0
        position.iloc[t] = 1.0 if in_pos else 0.0

    # Returns
    ret_1d = pd.Series(close.pct_change().fillna(0.0), index=close.index)
    strat_gross = position.shift(1).fillna(0.0) * ret_1d
    # Transaction costs on position changes
    pos_change = position.diff().fillna(position.iloc[0])
    turn_cost = (abs(pos_change) * (cost_bp / 10000.0))
    strat_net = strat_gross - turn_cost
    equity_strat = (1 + strat_net).cumprod()
    equity_bh = (1 + ret_1d).cumprod()

    df = pd.DataFrame({
        'position': position,
        'ret_1d': ret_1d,
        'strat_ret_gross': strat_gross,
        'trade_cost': -turn_cost,
        'strat_ret_net': strat_net,
        'equity_strat': equity_strat,
        'equity_bh': equity_bh,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Run Kronos BTC forecasting, strategy S1, and diagnostics.")
    parser.add_argument('--eps', type=float, default=0.005)
    parser.add_argument('--is_start', type=str, default='2017-01-01')
    parser.add_argument('--is_end', type=str, default='2021-12-31')
    parser.add_argument('--oos_start', type=str, default='2022-01-01')
    parser.add_argument('--oos_end', type=str, default='auto')
    parser.add_argument('--lookback', type=int, default=480)
    parser.add_argument('--H', type=int, default=5)
    parser.add_argument('--q_hi', type=float, default=0.60)
    parser.add_argument('--q_lo', type=float, default=0.57)
    parser.add_argument('--min_hold', type=int, default=2)
    parser.add_argument('--cost_bp', type=float, default=10.0)
    parser.add_argument('--csv_path', type=str, default=None, help='Optional local CSV fallback for OHLCV')

    args = parser.parse_args()

    # Artifact directory
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    art_dir = os.path.join('.', 'artifacts', stamp)
    ensure_dir(art_dir)

    # Load data
    try:
        df = fetch_btc_data(start='2015-01-01', end=args.oos_end, csv_path=args.csv_path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        print("Aborting. Provide --csv_path if internet is blocked.", file=sys.stderr)
        sys.exit(1)

    df = df.copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    # Amount proxy
    df['amount'] = (df['volume'] * df['close']).fillna(0.0)

    # Prepare model
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
        print("Error loading Kronos models from Hugging Face (network likely blocked or cache missing).", file=sys.stderr)
        print("Exception: ", str(e), file=sys.stderr)
        print("Aborting. Please run once with internet or provide local HF cache.", file=sys.stderr)
        sys.exit(1)

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # Rolling forecasts of next H days; compute mu_hat
    lookback = args.lookback
    H = args.H
    closes = df['close']
    dates = df.index
    mu_hat = pd.Series(index=dates, dtype=float)

    # Forecast loop
    for i in range(lookback, len(df) - H):
        hist = df.iloc[i - lookback:i]
        x_timestamp = hist.index.to_series()
        future_idx = df.index[i+1:i+1+H]
        if len(future_idx) < H:
            continue
        try:
            pred_df = predictor.predict(
                df=hist[['open','high','low','close','volume','amount']].copy(),
                x_timestamp=x_timestamp,
                y_timestamp=future_idx.to_series(),
                pred_len=H,
                T=1.0,
                top_k=0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            # mean of next H closes
            mean_future_close = pred_df['close'].iloc[:H].mean()
            curr_close = df['close'].iloc[i]
            mu_hat.iloc[i] = (mean_future_close - curr_close) / curr_close
        except Exception as e:
            # Skip failed dates; leave NaN
            continue

    # Labels
    y_true_overlap = compute_labels_avg5(closes, step=1)
    y_true_nonoverlap = compute_labels_avg5(closes, step=5)

    # Strategy thresholds from IS mu_hat distribution (non-overlap by default); use all mu_hat in IS
    is_mask = (dates >= pd.to_datetime(args.is_start)) & (dates <= pd.to_datetime(args.is_end))
    oos_mask = (dates >= pd.to_datetime(args.oos_start)) & (dates <= (pd.to_datetime(args.oos_end) if args.oos_end != 'auto' else dates.max()))

    q_hi_thr = mu_hat.loc[is_mask].quantile(args.q_hi)
    q_lo_thr = mu_hat.loc[is_mask].quantile(args.q_lo)

    # Run strategy over OOS+IS span where mu_hat exists
    strat_df = run_strategy_s1(mu_hat, closes, q_hi_thr, q_lo_thr, args.min_hold, args.cost_bp)

    # Collect daily series dataframe
    out_df = pd.DataFrame(index=dates)
    out_df['close'] = closes
    out_df['ret_1d'] = out_df['close'].pct_change()
    out_df['mu_hat'] = mu_hat
    out_df['q_hi'] = q_hi_thr
    out_df['q_lo'] = q_lo_thr
    out_df = out_df.join(strat_df[['position','strat_ret_gross','trade_cost','strat_ret_net','equity_strat','equity_bh']])
    out_df['y_true_avg5_nonoverlap'] = y_true_nonoverlap
    out_df['y_true_avg5_overlap'] = y_true_overlap

    # Metrics (IS, OOS, ALL) - using non-overlap primary; overlap secondary in long report
    def subsample_nonoverlap(series: pd.Series) -> pd.Series:
        # Subsample every H-th point starting from the first index with label
        s = series.dropna()
        if s.empty:
            return s
        # Align by calendar to reduce look-ahead bias: pick every H-th index
        return s.iloc[::H]

    segs = {
        'IS': is_mask,
        'OOS': oos_mask,
        'ALL': pd.Series(True, index=dates)
    }

    summary_rows = []
    acc_long_rows = []

    for name, mask in segs.items():
        mu_seg = mu_hat.loc[mask]
        y_seg_non = y_true_nonoverlap.loc[mask]
        y_seg_ovr = y_true_overlap.loc[mask]

        # Primary (non-overlap subsample)
        mu_non = subsample_nonoverlap(mu_seg)
        y_non = subsample_nonoverlap(y_seg_non)
        metrics_non = compute_accuracy_metrics(mu_non, y_non, name, eps=args.eps)
        summary_rows.append({'segment': name, **{k: metrics_non.get(k, np.nan) for k in [
            'MAE','RMSE','R2','IC','IC_t','RankIC','RankIC_t','DA_raw','DA_eps','DA_eps_adapt','BA','ROC_AUC','PR_AUC']}})
        for k, v in metrics_non.items():
            acc_long_rows.append({'segment': name, 'metric': k, 'value': v, 'notes': 'nonoverlap'})

        # Secondary (overlap)
        metrics_ovr = compute_accuracy_metrics(mu_seg, y_seg_ovr, name, eps=args.eps)
        for k, v in metrics_ovr.items():
            acc_long_rows.append({'segment': name, 'metric': k, 'value': v, 'notes': 'overlap'})

    summary_df = pd.DataFrame(summary_rows)
    acc_long_df = pd.DataFrame(acc_long_rows)

    # Save timeseries parquet and reports
    ts_path = os.path.join(art_dir, 'daily_timeseries.parquet')
    out_df.to_parquet(ts_path)

    # Append or create summary.csv
    sum_path = os.path.join(art_dir, 'summary.csv')
    summary_df.to_csv(sum_path, index=False)

    acc_path = os.path.join(art_dir, 'accuracy_report.csv')
    acc_long_df.to_csv(acc_path, index=False)

    # Plots
    plot_equity_vs_bh(out_df, os.path.join(art_dir, 'equity_vs_bh.png'))
    plot_drawdown(out_df, os.path.join(art_dir, 'drawdown.png'))

    # Use ALL, non-overlap for scatter/calibration/confusion/muhat_ic
    mu_all_non = subsample_nonoverlap(mu_hat)
    y_all_non = subsample_nonoverlap(y_true_nonoverlap)
    plot_scatter(mu_all_non, y_all_non, os.path.join(art_dir, 'forecast_scatter.png'))
    calib_df = make_calibration(mu_all_non, y_all_non, n_bins=10)
    plot_calibration(calib_df, os.path.join(art_dir, 'calibration_by_decile.png'))

    # Confusion matrix and DA/BA for ALL
    m_all = compute_accuracy_metrics(mu_all_non, y_all_non, 'ALL', eps=args.eps)
    tn = int(m_all.get('TN') if not math.isnan(m_all.get('TN', np.nan)) else 0)
    fp = int(m_all.get('FP') if not math.isnan(m_all.get('FP', np.nan)) else 0)
    fn = int(m_all.get('FN') if not math.isnan(m_all.get('FN', np.nan)) else 0)
    tp = int(m_all.get('TP') if not math.isnan(m_all.get('TP', np.nan)) else 0)
    title = f"Confusion Matrix | DA={m_all.get('DA_raw', np.nan):.3f}, BA={m_all.get('BA', np.nan):.3f}"
    plot_confusion_matrix((tn, fp, fn, tp), os.path.join(art_dir, 'confusion_matrix.png'), title)

    # IC plots (OOS non-overlap)
    m_oos = next((r for r in summary_rows if r['segment'] == 'OOS'), None)
    if m_oos:
        plot_muhat_ic(m_oos.get('IC', np.nan), m_oos.get('IC_t', np.nan), m_oos.get('RankIC', np.nan), m_oos.get('RankIC_t', np.nan),
                      os.path.join(art_dir, 'muhat_ic.png'))

    # Position and mu plot
    plot_position_and_mu(out_df, os.path.join(art_dir, 'position_and_mu.png'))

    # Write a short metadata JSON
    meta = {
        'model_tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
        'model_base': 'NeoQuasar/Kronos-base',
        'H': H,
        'lookback': lookback,
        'q_hi': args.q_hi,
        'q_lo': args.q_lo,
        'MIN_HOLD_DAYS': args.min_hold,
        'COST_BP': args.cost_bp,
        'IS': [args.is_start, args.is_end],
        'OOS': [args.oos_start, args.oos_end],
        'artifacts_dir': art_dir
    }
    with open(os.path.join(art_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(art_dir)


if __name__ == '__main__':
    main()
