"""
Reproducible worked example: the legacy BacktestEngine on a frozen universe.

For every symbol in a frozen universe file this script runs the swing-indicator
strategy (strategy/indicators.py) through backtest/engine.py with the same engine
defaults that `main.py --engine legacy` uses, computes a per-symbol buy-and-hold
baseline over the same bars, and writes:

    results/<tag>.csv      one row per symbol (strategy metrics + buy-and-hold)
    results/summary.md     run metadata, aggregate table, per-symbol table, caveats

Default run (the one reported in README.md):

    python examples/frozen_universe_backtest.py

which is equivalent to

    python examples/frozen_universe_backtest.py \
        --universe-file universe/universe_mean_reversion_manual_50.txt \
        --start 2021-01-04 --end 2024-12-31 --tag frozen50_2021_2024

Market data comes from yfinance through data.fetcher.DataFetcher (batched
downloads). Each symbol's daily OHLCV is cached as CSV under data_cache/ so that
re-runs are offline and deterministic; pass --no-cache to force a fresh download.
Note that --end is inclusive here (yfinance's own `end` argument is exclusive).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backtest.engine import BacktestEngine
from data.fetcher import DataFetcher
from strategy.indicators import SwingTradingIndicators
from universe.constructor import UniverseConstructor

DEFAULT_UNIVERSE = os.path.join(PROJECT_ROOT, "universe", "universe_mean_reversion_manual_50.txt")
FALLBACK_UNIVERSE = os.path.join(PROJECT_ROOT, "universe", "universe_mean_reversion_manual.txt")

STRATEGY_COLS = [
    "total_return_pct", "win_rate_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades",
    "exposure_pct", "avg_holding_days", "in_market_days", "in_market_sharpe_ratio",
]
BH_COLS = ["bh_total_return_pct", "bh_sharpe_ratio", "bh_max_drawdown_pct"]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def _cache_path(cache_dir: str, symbol: str, start: str, end: str) -> str:
    return os.path.join(cache_dir, f"{symbol}_{start}_{end}_1d.csv")


def _normalize_daily(df: pd.DataFrame, end_inclusive: date) -> pd.DataFrame:
    """Make a fetched/cached frame look like DataFetcher daily output and clip to the window."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out[out["date"] <= end_inclusive]
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out


def load_prices(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: Optional[str],
    batch_size: int,
    retry_wait: float,
) -> Dict[str, pd.DataFrame]:
    """Return {symbol: OHLCV frame} using the local CSV cache first, then yfinance."""
    end_inclusive = datetime.strptime(end, "%Y-%m-%d").date()
    end_exclusive = (end_inclusive + timedelta(days=1)).strftime("%Y-%m-%d")

    data: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for sym in symbols:
            path = _cache_path(cache_dir, sym, start, end)
            if os.path.exists(path):
                try:
                    cached = pd.read_csv(path)
                    if not cached.empty and "close" in cached.columns:
                        data[sym] = _normalize_daily(cached, end_inclusive)
                        continue
                except Exception as exc:  # corrupt cache file: refetch
                    print(f"  cache read failed for {sym} ({exc}); refetching")
            missing.append(sym)
        print(f"Cache: {len(data)} symbols loaded from {cache_dir}, {len(missing)} to download")
    else:
        missing = list(symbols)

    fetcher = DataFetcher()
    if missing:
        fetched = fetcher.fetch_multiple_symbols(missing, start, end_exclusive, interval="1d", batch_size=batch_size)
        still_missing = []
        for sym in missing:
            df = fetched.get(sym)
            if df is None or df.empty:
                still_missing.append(sym)
                continue
            data[sym] = _normalize_daily(df, end_inclusive)

        # One per-symbol retry pass after a cooldown; Yahoo sometimes returns empty batches when throttled.
        if still_missing:
            print(f"\n{len(still_missing)} symbols empty after batch download; per-symbol retry in {retry_wait:.0f}s")
            time.sleep(retry_wait)
            for sym in still_missing:
                try:
                    df = fetcher.fetch_stock_data(sym, start, end_exclusive)
                    if df is not None and not df.empty:
                        data[sym] = _normalize_daily(df, end_inclusive)
                        print(f"  recovered {sym}: {len(data[sym])} bars")
                except Exception as exc:
                    print(f"  {sym}: {exc}")
                time.sleep(1.0)

        if cache_dir:
            for sym in missing:
                if sym in data and not data[sym].empty:
                    data[sym].to_csv(_cache_path(cache_dir, sym, start, end), index=False)

    return data


# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------
def run_symbol(symbol: str, data: pd.DataFrame, initial_capital: float, position_size: float) -> dict:
    """Strategy metrics for one symbol plus a buy-and-hold baseline over the same bars."""
    indicators = SwingTradingIndicators(data)
    signals = indicators.get_signals_df()

    engine = BacktestEngine(
        data=data,
        signals=signals,
        initial_capital=initial_capital,
        position_size=position_size,
    )
    engine.run()
    m = engine.get_metrics()

    close = data["close"].astype(float).reset_index(drop=True)
    bh_equity = initial_capital * close / close.iloc[0]
    bh_returns = close.pct_change().dropna()

    row = {
        "symbol": symbol,
        "bars": int(len(data)),
        "first_date": str(data["date"].iloc[0]),
        "last_date": str(data["date"].iloc[-1]),
        "total_return_pct": float(m["total_return_pct"]),
        "win_rate_pct": float(m["win_rate_pct"]),
        "sharpe_ratio": float(m["sharpe_ratio"]),
        "max_drawdown_pct": float(m["max_drawdown_pct"]),
        "total_trades": int(m["total_trades"]),
        "profit_factor": float(m["profit_factor"]),
        "exposure_pct": float(m["exposure_pct"]),
        "avg_holding_days": float(m["avg_holding_days"]),
        "in_market_days": int(m["in_market_days"]),
        "in_market_sharpe_ratio": float(m["in_market_sharpe_ratio"]),
        "in_market_sortino_ratio": float(m["in_market_sortino_ratio"]),
        "bh_total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0),
        "bh_sharpe_ratio": float(engine._calculate_sharpe(bh_returns)),
        "bh_max_drawdown_pct": float(engine._calculate_max_drawdown(bh_equity)),
    }
    row["beats_buy_and_hold"] = bool(row["total_return_pct"] > row["bh_total_return_pct"])
    row["_engine"] = engine  # for parameter reporting; dropped before writing
    return row


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Minimal GitHub-flavoured markdown table (avoids a dependency on tabulate)."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def aggregate_table(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and median across symbols. Buy-and-hold is always invested, so its in-market Sharpe is its Sharpe."""
    bh_exposure = pd.Series(100.0, index=results.index)
    rows = []
    for label, s_col, b_col in [
        ("Total return (%)", "total_return_pct", "bh_total_return_pct"),
        ("Sharpe ratio (account)", "sharpe_ratio", "bh_sharpe_ratio"),
        ("Sharpe ratio (in-market)", "in_market_sharpe_ratio", "bh_sharpe_ratio"),
        ("Max drawdown (%)", "max_drawdown_pct", "bh_max_drawdown_pct"),
        ("Exposure (% of bars)", "exposure_pct", bh_exposure),
        ("Avg holding (bars)", "avg_holding_days", None),
        ("Win rate (%)", "win_rate_pct", None),
        ("Trades per symbol", "total_trades", None),
    ]:
        if b_col is None:
            bh = None
        elif isinstance(b_col, str):
            bh = results[b_col]
        else:
            bh = b_col
        rows.append(
            {
                "metric": label,
                "strategy mean": results[s_col].mean(),
                "strategy median": results[s_col].median(),
                "buy-and-hold mean": bh.mean() if bh is not None else np.nan,
                "buy-and-hold median": bh.median() if bh is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    path: str,
    results: pd.DataFrame,
    args: argparse.Namespace,
    universe_path: str,
    requested: List[str],
    missing: List[str],
    engine: BacktestEngine,
    csv_name: str,
    run_started: datetime,
    elapsed_s: float,
) -> None:
    import yfinance

    agg = aggregate_table(results)
    n = len(results)
    beats = int(results["beats_buy_and_hold"].sum())
    zero_trade = results.loc[results["total_trades"] == 0, "symbol"].tolist()

    cmd = (
        "python examples/frozen_universe_backtest.py "
        f"--universe-file {os.path.relpath(universe_path, PROJECT_ROOT)} "
        f"--start {args.start} --end {args.end} --capital {args.capital:g} "
        f"--position-size {args.position_size:g} --tag {args.tag}"
    )

    per_symbol = results[
        [
            "symbol", "bars", "total_trades", "total_return_pct", "win_rate_pct", "sharpe_ratio",
            "in_market_sharpe_ratio", "exposure_pct", "avg_holding_days", "max_drawdown_pct",
        ]
        + BH_COLS
    ].copy()

    lines = [
        f"# Frozen-universe backtest: {args.tag}",
        "",
        f"Run started {run_started.strftime('%Y-%m-%d %H:%M UTC')}; wall time {elapsed_s:.0f}s.",
        "",
        "## Command",
        "",
        "```bash",
        cmd,
        "```",
        "",
        "## Setup",
        "",
        f"- Universe file: `{os.path.relpath(universe_path, PROJECT_ROOT)}` ({len(requested)} symbols requested, "
        f"{n} with data{', ' + str(len(missing)) + ' missing: ' + ', '.join(missing) if missing else ''})",
        f"- Window: {args.start} to {args.end} inclusive, daily bars (Yahoo `Close`; split-adjusted, "
        "not dividend-adjusted)",
        "- Engine: `backtest/engine.py` (legacy), one long position per symbol at a time, "
        f"{engine.position_size:.0%} of current capital per trade, commission {engine.commission:.2%} per side, "
        f"fixed stop loss {engine.stop_loss_pct:.0%}, trailing stop {'on' if engine.trailing_stop else 'off'}, "
        f"halt new entries at {engine.max_drawdown_pct:.0%} portfolio drawdown, max holding 20 bars, "
        "take-profit at max(3%, 2x stop distance), exit on neutral signal after 5 bars",
        "- Signals: `strategy/indicators.py` `SwingTradingIndicators` (needs a 200-bar SMA, so roughly the first "
        "200 bars of each series produce no signals)",
        f"- Initial capital per symbol: {args.capital:,.0f}; each symbol is backtested independently",
        f"- Buy-and-hold baseline: fully invested in the symbol from the first to the last bar of the same window, "
        "same Sharpe/drawdown formulas as the engine (risk-free rate "
        f"{engine.risk_free_rate:.0%}, 252-day annualisation)",
        "- Two Sharpe columns for the strategy: `sharpe_ratio` is on daily account equity over every bar; "
        "`in_market_sharpe_ratio` is on the daily return of the capital at risk over the bars a position was held "
        "(`in_market_days` of them), see `BacktestEngine._in_market_returns`. Buy-and-hold is always invested, so "
        "its single Sharpe is shown in both rows of the aggregate table. `exposure_pct` is the share of bars with "
        "a position held at the close; `avg_holding_days` is the mean bars per trade",
        f"- Software: Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, "
        f"yfinance {yfinance.__version__}",
        "",
        "## Aggregate",
        "",
        df_to_markdown(agg),
        "",
        f"- Symbols where strategy total return > buy-and-hold total return: {beats} of {n}",
        f"- Symbols with zero trades: {len(zero_trade)}" + (f" ({', '.join(zero_trade)})" if zero_trade else ""),
        "",
        "## Per symbol",
        "",
        f"Full table with more columns: `results/{csv_name}`.",
        "",
        df_to_markdown(per_symbol),
        "",
        "## What these numbers do and do not show",
        "",
        f"- They show how this specific rule set behaved on these {n} tickers over this window with the engine's "
        "default risk settings. Each symbol is an isolated account; there is no portfolio-level capital sharing.",
        "- Strategy returns are on total account equity while only 10% of capital is at risk per trade, so a "
        "like-for-like comparison with a fully invested buy-and-hold is not possible from this table alone. "
        "Buy-and-hold is reported as context, not as a benchmark the strategy was designed to beat.",
        "- No slippage, no spread, no borrow or financing, fills at the daily close, 0.1% commission per side.",
        "- The account Sharpe is the engine's formula: (annualised mean daily return - 2% risk-free) / annualised "
        "volatility of account equity. An account that sits mostly in cash earns roughly 0%, so its excess return "
        "is about -2% divided by a very small volatility; that is why the account Sharpe values are large and "
        "negative even where the account did not lose money. Read them as 'rarely invested', not 'large losses'. "
        "The in-market Sharpe removes the idle days and the cash fraction and is the column to compare with "
        "buy-and-hold's Sharpe; it is estimated from only `in_market_days` observations per symbol (about a "
        "tenth of the bars), so it is noisy, and it is NaN for a symbol with fewer than two in-market bars.",
        "- The universe is a hand-picked list of large, long-lived names and ETFs, written after the test window "
        "with knowledge of which names survived. That limits but does not remove survivorship bias: it is not a "
        "point-in-time constituent list.",
        "- Everything is in-sample. Parameters were not tuned on this run, but there is no held-out period, "
        "no walk-forward, and no statistical test, so this is not evidence of out-of-sample edge.",
        "- Yahoo data can be revised; re-running without the local cache may produce slightly different bars.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe-file", default=DEFAULT_UNIVERSE, help="frozen universe .txt/.csv (default: 50-symbol)")
    p.add_argument("--start", default="2021-01-04", help="first bar date, YYYY-MM-DD (default 2021-01-04)")
    p.add_argument("--end", default="2024-12-31", help="last bar date, inclusive (default 2024-12-31)")
    p.add_argument("--capital", type=float, default=100000.0, help="initial capital per symbol (default 100000)")
    p.add_argument("--position-size", type=float, default=0.1, help="fraction of capital per trade (default 0.1)")
    p.add_argument("--tag", default=None, help="output stem; default frozen50_<startyear>_<endyear> for the 50 list")
    p.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"), help="output directory")
    p.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, "data_cache"), help="OHLCV CSV cache directory")
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write the OHLCV cache")
    p.add_argument("--batch-size", type=int, default=5, help="tickers per yfinance download call (default 5)")
    p.add_argument("--retry-wait", type=float, default=15.0, help="seconds to wait before per-symbol retries")
    p.add_argument("--max-symbols", type=int, default=None, help="only run the first N symbols (smoke testing)")
    args = p.parse_args(argv)

    if args.tag is None:
        stem = os.path.splitext(os.path.basename(args.universe_file))[0]
        prefix = "frozen50" if stem.endswith("_50") else stem.replace("universe_", "")
        args.tag = f"{prefix}_{args.start[:4]}_{args.end[:4]}"
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_started = datetime.now(timezone.utc)
    t0 = time.time()

    universe_path = os.path.abspath(args.universe_file)
    symbols = [s.upper().strip() for s in UniverseConstructor.load_universe(universe_path)]
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    print(f"Universe: {len(symbols)} symbols from {os.path.relpath(universe_path, PROJECT_ROOT)}")
    print(f"Window:   {args.start} to {args.end} (inclusive)\n")

    cache_dir = None if args.no_cache else args.cache_dir
    prices = load_prices(symbols, args.start, args.end, cache_dir, args.batch_size, args.retry_wait)
    missing = [s for s in symbols if s not in prices or prices[s].empty]
    if missing:
        print(f"\nNo data for {len(missing)} symbols: {', '.join(missing)}")

    rows = []
    for i, sym in enumerate(symbols, 1):
        if sym in missing:
            continue
        try:
            row = run_symbol(sym, prices[sym], args.capital, args.position_size)
        except Exception as exc:
            print(f"[{i}/{len(symbols)}] {sym}: FAILED ({exc})")
            missing.append(sym)
            continue
        rows.append(row)
        print(
            f"[{i}/{len(symbols)}] {sym:6s} bars={row['bars']:4d} trades={row['total_trades']:3d} "
            f"ret={row['total_return_pct']:7.2f}% win={row['win_rate_pct']:6.2f}% sharpe={row['sharpe_ratio']:6.2f} "
            f"in-mkt sharpe={row['in_market_sharpe_ratio']:6.2f} exp={row['exposure_pct']:5.1f}% "
            f"mdd={row['max_drawdown_pct']:6.2f}% | B&H ret={row['bh_total_return_pct']:8.2f}%"
        )

    if not rows:
        print("No symbols produced results; nothing written.")
        return 1

    engine = rows[-1].pop("_engine")
    for r in rows:
        r.pop("_engine", None)
    results = pd.DataFrame(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_name = f"{args.tag}.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    results.round(4).to_csv(csv_path, index=False)
    summary_path = os.path.join(args.out_dir, "summary.md")
    write_summary(
        summary_path, results, args, universe_path, symbols, missing, engine, csv_name, run_started, time.time() - t0
    )

    print("\nAggregate (strategy vs buy-and-hold):")
    print(aggregate_table(results).round(2).to_string(index=False))
    print(f"\nWrote {os.path.relpath(csv_path, PROJECT_ROOT)} and {os.path.relpath(summary_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
