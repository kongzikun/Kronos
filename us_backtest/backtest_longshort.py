from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .utils import (
        annualized_return,
        information_ratio,
        max_drawdown,
        plot_nav,
        plot_excess_nav,
        ensure_dir,
        save_json,
    )
except Exception:  # pragma: no cover
    from utils import (
        annualized_return,
        information_ratio,
        max_drawdown,
        plot_nav,
        plot_excess_nav,
        ensure_dir,
        save_json,
    )


@dataclass
class Position:
    shares: float  # positive for long, negative for short
    days: int


def backtest_longshort(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    out_dir: str,
    k_long: int = 10,
    k_short: int = 10,
    n_long: int = 3,
    n_short: int = 3,
    min_hold: int = 5,
    cost: float = 0.0010,
) -> Dict[str, float]:
    """Simple long-short top/bottom-k backtest with drop-n constraints.

    Mechanics:
    - Signals at close t, trade at open t+1.
    - Allocate at most ~0.5 NAV to new longs, ~0.5 NAV to new shorts on any trade day (budgeted equally across new names).
    - Transaction cost as fraction per traded notional (both sides).
    - NAV computed as cash + sum(shares * close).
    """
    ensure_dir(out_dir)

    dates = sorted(prices.index.get_level_values(0).unique())
    long_pos: Dict[str, Position] = {}
    short_pos: Dict[str, Position] = {}
    cash = 1.0
    nav_list: List[float] = []
    trade_records = []
    turnover_list: List[float] = []
    prev_value = 1.0

    bench_prices = benchmark.reindex(dates).ffill()

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        open_prices = prices.xs(date, level=0)["open"]
        close_prices = prices.xs(date, level=0)["close"]
        sig_prev = signals.loc[prev_date].dropna() if prev_date in signals.index else pd.Series(dtype=float)
        ranking = sig_prev.sort_values(ascending=False)
        desired_long = set(ranking.head(k_long).index)
        desired_short = set(ranking.tail(k_short).index)

        # Aging
        for p in list(long_pos.values()) + list(short_pos.values()):
            p.days += 1

        # Close long positions not in desired set after min_hold
        long_closes = [t for t in list(long_pos.keys()) if (t not in desired_long) and (long_pos[t].days >= min_hold)]
        long_closes.sort(key=lambda x: ranking.get(x, -np.inf))
        # Close shorts not in desired set after min_hold
        short_closes = [t for t in list(short_pos.keys()) if (t not in desired_short) and (short_pos[t].days >= min_hold)]
        short_closes.sort(key=lambda x: -ranking.get(x, np.inf))

        # Execute closes at open
        day_turnover = 0.0
        for tk in long_closes:
            if tk not in open_prices:
                continue
            price = float(open_prices[tk])
            sh = long_pos[tk].shares
            amt = sh * price
            fee = amt * cost
            cash += amt - fee
            day_turnover += abs(amt)
            trade_records.append({"date": date, "ticker": tk, "side": "long", "action": "sell", "shares": sh, "price": price, "amount": amt, "cost": fee})
            del long_pos[tk]
        for tk in short_closes:
            if tk not in open_prices:
                continue
            price = float(open_prices[tk])
            sh = short_pos[tk].shares  # negative
            amt = abs(sh) * price
            fee = amt * cost
            cash -= amt + fee  # buy to cover
            day_turnover += abs(amt)
            trade_records.append({"date": date, "ticker": tk, "side": "short", "action": "cover", "shares": sh, "price": price, "amount": -amt, "cost": fee})
            del short_pos[tk]

        # Determine new entries (limit by n_long/n_short)
        new_longs = [tk for tk in ranking.index if (tk in desired_long) and (tk not in long_pos) and (tk in open_prices)]
        new_shorts = [tk for tk in ranking.index[::-1] if (tk in desired_short) and (tk not in short_pos) and (tk in open_prices)]
        new_longs = new_longs[: max(0, min(n_long, k_long - len(long_pos)))]
        new_shorts = new_shorts[: max(0, min(n_short, k_short - len(short_pos)))]

        # Budget: roughly half of NAV for long entries, half for short entries
        # Compute provisional NAV at open using close_{t-1} for existing positions
        value_prev_close = cash
        for tk, pos in long_pos.items():
            if tk in close_prices:
                value_prev_close += pos.shares * float(close_prices[tk])
        for tk, pos in short_pos.items():
            if tk in close_prices:
                value_prev_close += pos.shares * float(close_prices[tk])  # shares negative

        # Allocate budgets equally among new names
        long_budget_each = (0.5 * max(0.0, value_prev_close)) / max(1, len(new_longs)) if new_longs else 0.0
        short_budget_each = (0.5 * max(0.0, value_prev_close)) / max(1, len(new_shorts)) if new_shorts else 0.0

        for tk in new_longs:
            price = float(open_prices[tk])
            amt = long_budget_each
            sh = amt / price if price > 0 else 0.0
            fee = amt * cost
            cash -= amt + fee
            day_turnover += abs(amt)
            long_pos[tk] = Position(shares=sh, days=0)
            trade_records.append({"date": date, "ticker": tk, "side": "long", "action": "buy", "shares": sh, "price": price, "amount": amt, "cost": fee})

        for tk in new_shorts:
            price = float(open_prices[tk])
            amt = short_budget_each
            sh = -(amt / price) if price > 0 else 0.0
            fee = amt * cost
            cash += amt - fee  # proceeds from short sale
            day_turnover += abs(amt)
            short_pos[tk] = Position(shares=sh, days=0)
            trade_records.append({"date": date, "ticker": tk, "side": "short", "action": "short", "shares": sh, "price": price, "amount": amt, "cost": fee})

        # End-of-day valuation at close
        value = cash
        for tk, pos in long_pos.items():
            if tk in close_prices:
                value += pos.shares * float(close_prices[tk])
        for tk, pos in short_pos.items():
            if tk in close_prices:
                value += pos.shares * float(close_prices[tk])
        nav_list.append(value)
        turnover_list.append(day_turnover / max(prev_value, 1e-12))
        prev_value = value

    nav_series = pd.Series(nav_list, index=dates[1:])
    bench_series = bench_prices.loc[nav_series.index] / bench_prices.loc[nav_series.index[0]]
    excess_nav = nav_series / bench_series

    nav_df = pd.DataFrame({"nav": nav_series, "bench_nav": bench_series, "excess_nav": excess_nav})
    nav_df.to_csv(f"{out_dir}/nav.csv")

    plot_nav(nav_series, bench_series, f"{out_dir}/cum_return.png")
    plot_excess_nav(excess_nav, f"{out_dir}/cum_excess_return.png")

    trades_df = pd.DataFrame(trade_records)
    trades_df.to_parquet(f"{out_dir}/trades.parquet")

    # Portfolio daily returns
    daily_ret = nav_series.pct_change().dropna()
    # Excess returns should compound via ratio of NAVs, not difference of returns
    # This avoids invalid negative factors that can lead to NaN when annualizing.
    excess_nav_ret = excess_nav.pct_change().dropna()

    turnover_series = pd.Series(turnover_list, index=dates[1:])
    turnover_mean = float(turnover_series.loc[nav_series.index].mean()) if len(nav_series) > 0 else 0.0
    summary = {
        # Crypto runs 24/7 → use 365 periods/year
        "AER": annualized_return(excess_nav_ret, periods_per_year=365),
        "IR": information_ratio(excess_nav_ret, periods_per_year=365),
        "vol": daily_ret.std() * np.sqrt(365),
        "win_rate": float((excess_nav_ret > 0).mean()),
        "max_drawdown": float(max_drawdown(nav_series)),
        "turnover": turnover_mean,
    }
    save_json(summary, f"{out_dir}/summary.json")
    return summary
