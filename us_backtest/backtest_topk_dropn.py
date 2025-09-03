from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

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
    shares: float
    days: int


def backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    out_dir: str,
    k: int = 50,
    n: int = 5,
    min_hold: int = 5,
    cost: float = 0.0015,
) -> None:
    ensure_dir(out_dir)

    dates = sorted(prices.index.get_level_values(0).unique())
    positions: Dict[str, Position] = {}
    cash = 1.0
    nav_list: List[float] = []
    trade_records = []
    turnover_list: List[float] = []

    bench_prices = benchmark.reindex(dates).fillna(method="ffill")
    bench_nav = [1.0]

    prev_value = 1.0

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]
        open_prices = prices.xs(date, level=0)["open"]
        close_prices = prices.xs(date, level=0)["close"]
        signal_prev = signals.loc[prev_date].dropna() if prev_date in signals.index else pd.Series(dtype=float)
        ranking = signal_prev.sort_values(ascending=False)
        topk = set(ranking.head(k).index)

        for pos in positions.values():
            pos.days += 1

        sell_candidates = [t for t in positions if (t not in topk) and positions[t].days >= min_hold]
        sell_candidates.sort(key=lambda x: ranking.get(x, -np.inf))
        sells = sell_candidates[:n]

        for tk in sells:
            if tk not in open_prices:
                continue
            price = open_prices[tk]
            share = positions[tk].shares
            amount = share * price
            cost_pay = amount * cost
            cash += amount - cost_pay
            trade_records.append({"date": date, "ticker": tk, "action": "sell", "shares": share, "price": price, "amount": amount, "cost": cost_pay})
            del positions[tk]

        buys = []
        for tk in ranking.index:
            if tk in positions:
                continue
            if tk not in open_prices:
                continue
            buys.append(tk)
            if len(buys) >= min(n, k - len(positions)):
                break

        buy_cash = cash / max(len(buys), 1)
        for tk in buys:
            price = open_prices[tk]
            amount = buy_cash
            share = amount / price
            cost_pay = amount * cost
            cash -= amount + cost_pay
            positions[tk] = Position(share, 0)
            trade_records.append({"date": date, "ticker": tk, "action": "buy", "shares": share, "price": price, "amount": amount, "cost": cost_pay})

        value = cash
        for tk, pos in positions.items():
            if tk in close_prices:
                value += pos.shares * close_prices[tk]
        nav_list.append(value)
        turnover = (sum(abs(rec["amount"]) for rec in trade_records if rec["date"] == date) / prev_value)
        turnover_list.append(turnover)
        prev_value = value

        bench_ret = bench_prices.loc[date] / bench_prices.loc[dates[0]]
        bench_nav.append(bench_ret)

    nav_series = pd.Series(nav_list, index=dates[1:])
    bench_series = pd.Series(bench_nav[1:], index=dates[1:])
    excess_nav = nav_series / bench_series

    nav_df = pd.DataFrame({"nav": nav_series, "bench_nav": bench_series, "excess_nav": excess_nav})
    nav_df.to_csv(f"{out_dir}/nav.csv")

    plot_nav(nav_series, bench_series, f"{out_dir}/cum_return.png")
    plot_excess_nav(excess_nav, f"{out_dir}/cum_excess_return.png")

    trades_df = pd.DataFrame(trade_records)
    trades_df.to_parquet(f"{out_dir}/trades.parquet")

    daily_ret = nav_series.pct_change().dropna()
    bench_ret = bench_series.pct_change().dropna()
    excess_ret = daily_ret - bench_ret
    summary = {
        "AER": annualized_return(excess_ret),
        "IR": information_ratio(excess_ret),
        "vol": daily_ret.std() * np.sqrt(252),
        "win_rate": float((excess_ret > 0).mean()),
        "max_drawdown": float(max_drawdown(nav_series)),
        "turnover": float(np.mean(turnover_list)),
    }
    save_json(summary, f"{out_dir}/summary.json")

    signals.to_parquet(f"{out_dir}/signals.parquet")
