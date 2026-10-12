"""
Figures for the frozen-universe worked example.

Reads results/<tag>.csv (written by examples/frozen_universe_backtest.py) and the
per-symbol OHLCV cache under data_cache/, re-runs the legacy engine offline for
every symbol in the CSV, checks that the recomputed metrics match the CSV, and
writes four PNG figures (150 dpi) under results/figures/:

    fig1_equity_curves.png           strategy vs buy-and-hold equity, 3 best and 3 worst symbols
    fig2_total_return_by_symbol.png  per-symbol total return, strategy vs buy-and-hold, all symbols
    fig3_trade_returns_and_risk.png  pooled per-trade return histogram; max drawdown vs total return
    fig4_portfolio_drawdown.png      equal-weight portfolio drawdown, strategy vs buy-and-hold

plus figure_stats.json with the numbers quoted in the README captions.

Usage:
    python examples/plot_frozen_universe_results.py
    python examples/plot_frozen_universe_results.py --symbols SPY QQQ AAPL MSFT XOM META

No network access: the cache must already contain the symbols in the CSV (run the
backtest example first).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))

from frozen_universe_backtest import _cache_path, _normalize_daily  # noqa: E402

from backtest.engine import BacktestEngine  # noqa: E402
from strategy.indicators import SwingTradingIndicators  # noqa: E402

# Palette (validated with the dataviz reference validator, light surface).
C_STRATEGY = "#2a78d6"  # categorical slot 1, blue
C_BH = "#eb6834"  # categorical slot 2, orange
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DPI = 150

LABEL_STRATEGY = "Strategy (10% of capital per trade)"
LABEL_BH = "Buy-and-hold (fully invested)"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def rerun_from_cache(
    symbols: List[str], start: str, end: str, cache_dir: str, capital: float, position_size: float
) -> Dict[str, dict]:
    """Re-run the legacy engine offline for each symbol; return equity series, buy-and-hold equity and trades."""
    end_date = pd.Timestamp(end).date()
    out: Dict[str, dict] = {}
    for sym in symbols:
        path = _cache_path(cache_dir, sym, start, end)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run examples/frozen_universe_backtest.py first")
        data = _normalize_daily(pd.read_csv(path), end_date)
        signals = SwingTradingIndicators(data).get_signals_df()
        engine = BacktestEngine(data=data, signals=signals, initial_capital=capital, position_size=position_size)
        engine.run()
        idx = pd.to_datetime(engine.results.index)
        equity = pd.Series(engine.results["equity"].to_numpy(dtype=float), index=idx, name=sym)
        close = data["close"].astype(float).to_numpy()
        bh_equity = pd.Series(capital * close / close[0], index=idx, name=sym)
        trades = engine.get_trades()
        out[sym] = {
            "equity": equity,
            "bh_equity": bh_equity,
            "trade_returns": trades["return_pct"].astype(float).to_numpy() if not trades.empty else np.array([]),
            "total_return_pct": float(engine.get_metrics()["total_return_pct"]),
        }
    return out


def check_matches_csv(results: pd.DataFrame, runs: Dict[str, dict]) -> None:
    """The figures must describe the committed CSV; refuse to draw if the offline re-run disagrees."""
    bad = []
    for _, r in results.iterrows():
        recomputed = runs[r["symbol"]]["total_return_pct"]
        if abs(recomputed - float(r["total_return_pct"])) > 1e-3:
            bad.append(f"{r['symbol']}: csv {r['total_return_pct']:.4f} vs rerun {recomputed:.4f}")
    if bad:
        raise RuntimeError("Re-run from cache does not match the CSV:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------------------
# Styling helpers
# --------------------------------------------------------------------------------------
def style_axes(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def new_figure(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    return fig


def legend(ax_or_fig, **kw):
    leg = ax_or_fig.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kw)
    return leg


def pct(x, _pos=None) -> str:
    return f"{x:+.0f}%" if abs(x) >= 10 else f"{x:+.1f}%"


def usd_k(x, _pos=None) -> str:
    return f"{x / 1000:,.0f}k"


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_equity_curves(results: pd.DataFrame, runs: Dict[str, dict], symbols: List[str], window: str, path: str):
    n = len(symbols)
    ncols = 3 if n > 2 else n
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows + 0.9), facecolor=SURFACE, squeeze=False)
    by_sym = results.set_index("symbol")

    for ax, sym in zip(axes.flat, symbols):
        eq = runs[sym]["equity"]
        bh = runs[sym]["bh_equity"]
        ax.plot(eq.index, eq.values, color=C_STRATEGY, linewidth=1.6, label=LABEL_STRATEGY)
        ax.plot(bh.index, bh.values, color=C_BH, linewidth=1.6, label=LABEL_BH)
        ax.axhline(eq.iloc[0], color=BASELINE, linewidth=0.8, linestyle=(0, (3, 3)))
        style_axes(ax)
        s_ret = by_sym.loc[sym, "total_return_pct"]
        b_ret = by_sym.loc[sym, "bh_total_return_pct"]
        ax.set_title(f"{sym}   strategy {s_ret:+.2f}%   buy-and-hold {b_ret:+.1f}%", fontsize=9, loc="left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(usd_k))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Account value (USD, start 100k)", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Date", fontsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, fontsize=8, labelcolor=INK_2,
               bbox_to_anchor=(0.005, 0.965))
    fig.suptitle(
        f"Equity per symbol, {window}: three best (top) and three worst (bottom) by strategy total return",
        x=0.01, ha="left", fontsize=11, color=INK, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def fig_total_return_bars(results: pd.DataFrame, window: str, path: str):
    df = results.sort_values("total_return_pct", ascending=True).reset_index(drop=True)
    n = len(df)
    fig = new_figure(9, 0.26 * n + 1.6)
    ax = fig.add_subplot(111)
    y = np.arange(n)
    h = 0.38
    ax.barh(y + h / 2, df["total_return_pct"], height=h, color=C_STRATEGY, label=LABEL_STRATEGY)
    ax.barh(y - h / 2, df["bh_total_return_pct"], height=h, color=C_BH, label=LABEL_BH)
    ax.axvline(0, color=INK_2, linewidth=0.9)
    style_axes(ax, grid_axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(df["symbol"], fontsize=7.5, color=INK_2)
    ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
    ax.set_xlabel("Total return over the window (%)", fontsize=8)
    ax.set_title(
        f"Total return per symbol, {window}, sorted by strategy return (best at top)",
        fontsize=11, loc="left", pad=26,
    )
    # Legend in the title band, above the axes, so it can never cover a bar.
    legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def fig_trades_and_risk(results: pd.DataFrame, runs: Dict[str, dict], window: str, path: str) -> dict:
    pooled = np.concatenate([runs[s]["trade_returns"] for s in results["symbol"]])
    med = float(np.median(pooled))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    # Left: pooled per-trade returns
    lo, hi = math.floor(pooled.min()), math.ceil(pooled.max())
    bins = np.arange(lo, hi + 0.5, 0.5)
    ax1.hist(pooled, bins=bins, color=C_STRATEGY, edgecolor=SURFACE, linewidth=0.6)
    ax1.axvline(0, color=BASELINE, linewidth=0.8)
    ax1.axvline(med, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ymax = ax1.get_ylim()[1]
    ax1.text(med, ymax * 0.97, f"  median {med:+.2f}%", color=INK_2, fontsize=8, va="top", ha="left")
    style_axes(ax1)
    ax1.set_xlabel("Per-trade return, entry to exit (%)", fontsize=8)
    ax1.set_ylabel("Number of trades", fontsize=8)
    ax1.set_title(f"All {len(pooled)} trades across {len(results)} symbols", fontsize=10, loc="left")

    # Right: max drawdown vs total return, both series
    ax2.scatter(results["bh_max_drawdown_pct"], results["bh_total_return_pct"], s=34, color=C_BH,
                edgecolors=SURFACE, linewidths=1.2, label=LABEL_BH, zorder=3)
    ax2.scatter(results["max_drawdown_pct"], results["total_return_pct"], s=34, color=C_STRATEGY,
                edgecolors=SURFACE, linewidths=1.2, label=LABEL_STRATEGY, zorder=4)
    ax2.axhline(0, color=INK_2, linewidth=0.9)
    style_axes(ax2, grid_axis="both")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(pct))
    ax2.set_xlabel("Maximum drawdown of account equity (%)", fontsize=8)
    ax2.set_ylabel("Total return (%)", fontsize=8)
    ax2.set_title("Per symbol: drawdown vs total return", fontsize=10, loc="left")
    legend(ax2, loc="upper left")

    fig.suptitle(f"Trade outcomes and risk, {window}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return {
        "trades_pooled": int(len(pooled)),
        "trade_return_median_pct": round(med, 3),
        "trade_return_mean_pct": round(float(pooled.mean()), 3),
        "trade_return_min_pct": round(float(pooled.min()), 3),
        "trade_return_max_pct": round(float(pooled.max()), 3),
        "trades_positive": int((pooled > 0).sum()),
        "trades_nonpositive": int((pooled <= 0).sum()),
    }


def fig_portfolio_drawdown(results: pd.DataFrame, runs: Dict[str, dict], window: str, path: str) -> Optional[dict]:
    symbols = list(results["symbol"])
    ref = runs[symbols[0]]["equity"].index
    misaligned = [s for s in symbols if not runs[s]["equity"].index.equals(ref)]
    if misaligned:
        print(f"Skipping portfolio drawdown figure: {len(misaligned)} symbols are not on the same dates "
              f"({', '.join(misaligned[:8])}{'...' if len(misaligned) > 8 else ''})")
        return None

    strat = pd.concat([runs[s]["equity"] for s in symbols], axis=1).mean(axis=1)
    bh = pd.concat([runs[s]["bh_equity"] for s in symbols], axis=1).mean(axis=1)

    def drawdown(eq: pd.Series) -> pd.Series:
        return (eq / eq.cummax() - 1.0) * 100.0

    dd_s, dd_b = drawdown(strat), drawdown(bh)

    fig = new_figure(11, 4.2)
    ax = fig.add_subplot(111)
    ax.fill_between(dd_b.index, dd_b.values, 0, color=C_BH, alpha=0.18, linewidth=0)
    ax.plot(dd_b.index, dd_b.values, color=C_BH, linewidth=1.6, label=LABEL_BH)
    ax.fill_between(dd_s.index, dd_s.values, 0, color=C_STRATEGY, alpha=0.25, linewidth=0)
    ax.plot(dd_s.index, dd_s.values, color=C_STRATEGY, linewidth=1.6, label=LABEL_STRATEGY)
    ax.axhline(0, color=INK_2, linewidth=0.9)
    style_axes(ax)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Drawdown from running peak (%)", fontsize=8)
    ax.set_title(
        f"Equal-weight portfolio of all {len(symbols)} symbols, {window}: drawdown of portfolio equity",
        fontsize=11, loc="left", pad=12,
    )
    legend(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return {
        "ew_strategy_total_return_pct": round(float(strat.iloc[-1] / strat.iloc[0] * 100 - 100), 3),
        "ew_strategy_max_drawdown_pct": round(float(-dd_s.min()), 3),
        "ew_bh_total_return_pct": round(float(bh.iloc[-1] / bh.iloc[0] * 100 - 100), 3),
        "ew_bh_max_drawdown_pct": round(float(-dd_b.min()), 3),
        "ew_bh_max_drawdown_date": str(dd_b.idxmin().date()),
        "ew_strategy_max_drawdown_date": str(dd_s.idxmin().date()),
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(PROJECT_ROOT, "results", "frozen50_2021_2024.csv"))
    p.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, "data_cache"))
    p.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results", "figures"))
    p.add_argument("--capital", type=float, default=100000.0, help="must match the backtest run (default 100000)")
    p.add_argument("--position-size", type=float, default=0.1, help="must match the backtest run (default 0.1)")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="symbols for the equity-curve small multiple (default: 3 best + 3 worst by strategy return)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    results = pd.read_csv(args.results)
    start, end = str(results["first_date"].min()), str(results["last_date"].max())
    window = f"{start} to {end}"
    symbols = list(results["symbol"])
    print(f"{len(symbols)} symbols in {os.path.relpath(args.results, PROJECT_ROOT)}; window {window}")

    runs = rerun_from_cache(symbols, start, end, args.cache_dir, args.capital, args.position_size)
    check_matches_csv(results, runs)
    print("Offline re-run matches the CSV for every symbol")

    ranked = results.sort_values("total_return_pct", ascending=False)["symbol"].tolist()
    if args.symbols:
        panel_symbols = [s.upper() for s in args.symbols]
        unknown = [s for s in panel_symbols if s not in runs]
        if unknown:
            raise SystemExit(f"--symbols not in results: {', '.join(unknown)}")
    else:
        panel_symbols = ranked[:3] + ranked[-3:]

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {
        "source_csv": os.path.relpath(args.results, PROJECT_ROOT),
        "window": window,
        "symbols": len(symbols),
        "best_three_by_strategy_return": ranked[:3],
        "worst_three_by_strategy_return": ranked[-3:],
        "equity_panel_symbols": panel_symbols,
        "strategy_total_return_min_pct": round(float(results["total_return_pct"].min()), 3),
        "strategy_total_return_max_pct": round(float(results["total_return_pct"].max()), 3),
        "bh_total_return_min_pct": round(float(results["bh_total_return_pct"].min()), 3),
        "bh_total_return_max_pct": round(float(results["bh_total_return_pct"].max()), 3),
        "strategy_max_drawdown_max_pct": round(float(results["max_drawdown_pct"].max()), 3),
        "bh_max_drawdown_min_pct": round(float(results["bh_max_drawdown_pct"].min()), 3),
        "bh_negative_symbols": results.loc[results["bh_total_return_pct"] < 0, "symbol"].tolist(),
    }

    p1 = os.path.join(args.out_dir, "fig1_equity_curves.png")
    fig_equity_curves(results, runs, panel_symbols, window, p1)
    p2 = os.path.join(args.out_dir, "fig2_total_return_by_symbol.png")
    fig_total_return_bars(results, window, p2)
    p3 = os.path.join(args.out_dir, "fig3_trade_returns_and_risk.png")
    stats.update(fig_trades_and_risk(results, runs, window, p3))
    p4 = os.path.join(args.out_dir, "fig4_portfolio_drawdown.png")
    port = fig_portfolio_drawdown(results, runs, window, p4)
    if port:
        stats.update(port)
    else:
        stats["portfolio_drawdown_figure"] = "skipped: per-symbol equity series not on identical dates"

    with open(os.path.join(args.out_dir, "figure_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Figures written to {os.path.relpath(args.out_dir, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
