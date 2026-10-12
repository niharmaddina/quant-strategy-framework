"""
Walk-forward evaluation of the swing strategy on the frozen 50-symbol universe.

The committed frozen-universe results are a single in-sample run with the default
parameters. This script asks the next question: if the exposed parameters had been
tuned on the first half of the window, would the choice have held up on the second
half? It splits the cached 2021-01-04 to 2024-12-31 window into

    in-sample       2021-01-04 .. 2022-12-31   (503 bars per symbol)
    out-of-sample   2023-01-03 .. 2024-12-31   (502 bars per symbol)

and runs four stages, each on every symbol of the universe as an isolated account:

  0. Sensitivity probe (in-sample only). Every parameter that the strategy or the
     engine exposes is varied one at a time from the default so the reader can see
     which ones move the result at all. This is how the grid axes were chosen; the
     table is written into the report.
  1. Grid (in-sample). GRID below: rsi_oversold x max_holding_days x stop_loss_pct,
     18 configurations including the default.
  2. Selection. The configuration with the highest mean per-symbol in-sample total
     return among those with at least MIN_TRADES_PER_SYMBOL pooled in-sample trades
     per symbol on average (100 for the 50-symbol universe; ties: higher median). The
     rule is fixed in this file before any out-of-sample number is computed; the
     threshold was set below the default's in-sample trade count so the default is
     admissible.
  3. Out-of-sample. The default configuration, the selected configuration and
     buy-and-hold are run on the out-of-sample bars with a fresh 100,000 per symbol.
     For transparency the out-of-sample result of every grid configuration is also
     recorded (walk_forward_grid.csv), which is what makes the in-sample vs
     out-of-sample transfer figure possible; the headline comparison uses only the
     configuration picked in stage 2.

Indicators are computed on each symbol's full series and then sliced, so the
out-of-sample period starts with warmed-up moving averages (a live trader would have
had that history). Every indicator in strategy/indicators.py is backward-looking
(rolling windows, EMA with adjust=False, shift/diff), so for the in-sample prefix
this is identical to computing on the in-sample bars alone; tests/test_walk_forward.py
checks that. Each period is a fresh account: no capital or open position is carried
across the boundary.

Outputs (all relative to the repo root):

    results/walk_forward.csv          per-symbol out-of-sample metrics for default, selected, buy-and-hold
    results/walk_forward_grid.csv     one row per grid configuration: in-sample and out-of-sample aggregates
    results/walk_forward.md           setup, probe, grid, selection, out-of-sample tables, caveats
    results/figures/fig5_walk_forward.png

Usage:

    python examples/walk_forward.py            # the run reported in README.md

Market data comes from the same per-symbol CSV cache as examples/frozen_universe_backtest.py
(data_cache/); run that example first if the cache is empty, otherwise this script is offline.
"""

from __future__ import annotations

import argparse
import itertools
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))

from frozen_universe_backtest import DEFAULT_UNIVERSE, df_to_markdown, load_prices  # noqa: E402

from backtest.engine import BacktestEngine  # noqa: E402
from strategy.indicators import SwingTradingIndicators  # noqa: E402
from universe.constructor import UniverseConstructor  # noqa: E402

# --------------------------------------------------------------------------------------
# Protocol (fixed before any out-of-sample number is computed)
# --------------------------------------------------------------------------------------
IS_START, IS_END = "2021-01-04", "2022-12-31"
OOS_START, OOS_END = "2023-01-03", "2024-12-31"

STRATEGY_KEYS = ("rsi_oversold", "rsi_overbought", "bb_period", "bb_std_dev")
ENGINE_KEYS = ("max_holding_days", "stop_loss_pct")

DEFAULT_CONFIG: Dict[str, float] = {"rsi_oversold": 35.0, "max_holding_days": 20, "stop_loss_pct": 0.05}

# Stage 1 grid: 3 x 3 x 2 = 18 configurations. Chosen from the stage-0 probe: the Bollinger
# parameters and the RSI overbought threshold do not change a single in-sample number (the
# strategy multiplies those flags by trend/momentum/price-action filters that never coincide
# with a lower-band touch), so they are left out.
GRID: Dict[str, List[float]] = {
    "rsi_oversold": [35.0, 45.0, 55.0],
    "max_holding_days": [5, 10, 20],
    "stop_loss_pct": [0.03, 0.05],
}

# Stage 0 probe: one parameter at a time, everything else at the default.
PROBE: Dict[str, List[float]] = {
    "rsi_oversold": [30.0, 45.0, 55.0],
    "rsi_overbought": [60.0, 70.0],
    "bb_period": [10, 30],
    "bb_std_dev": [1.0, 1.5, 2.5],
    "max_holding_days": [5, 10, 40],
    "stop_loss_pct": [0.03, 0.08],
}

# Stage 2 selection: minimum pooled in-sample trades, as a multiple of the number of symbols
# (2 per symbol on average = 100 trades for the 50-symbol universe).
MIN_TRADES_PER_SYMBOL = 2.0

SYMBOL_METRICS = [
    "total_return_pct", "total_trades", "win_rate_pct", "exposure_pct", "avg_holding_days",
    "sharpe_ratio", "in_market_sharpe_ratio", "max_drawdown_pct",
]


# --------------------------------------------------------------------------------------
# Split and run helpers (unit-tested in tests/test_walk_forward.py)
# --------------------------------------------------------------------------------------
def period_masks(
    dates: Iterable, is_start: str, is_end: str, oos_start: str, oos_end: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks selecting the in-sample and out-of-sample bars; both inclusive, strictly ordered."""
    if not pd.Timestamp(is_start) <= pd.Timestamp(is_end) < pd.Timestamp(oos_start) <= pd.Timestamp(oos_end):
        raise ValueError("Need is_start <= is_end < oos_start <= oos_end")
    d = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    is_mask = ((d >= pd.Timestamp(is_start)) & (d <= pd.Timestamp(is_end))).to_numpy()
    oos_mask = ((d >= pd.Timestamp(oos_start)) & (d <= pd.Timestamp(oos_end))).to_numpy()
    if not is_mask.any():
        raise ValueError(f"No bars in the in-sample window {is_start}..{is_end}")
    if not oos_mask.any():
        raise ValueError(f"No bars in the out-of-sample window {oos_start}..{oos_end}")
    return is_mask, oos_mask


def split_config(config: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Separate a flat configuration into SwingTradingIndicators kwargs and BacktestEngine kwargs."""
    unknown = set(config) - set(STRATEGY_KEYS) - set(ENGINE_KEYS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    strategy = {k: config[k] for k in STRATEGY_KEYS if k in config}
    engine = {k: config[k] for k in ENGINE_KEYS if k in config}
    return strategy, engine


def slice_period(data: pd.DataFrame, signals: pd.DataFrame, mask: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cut data and full-series signals to one period; both are re-indexed from 0 so the engine aligns them."""
    if len(data) != len(signals) or len(mask) != len(data):
        raise ValueError("data, signals and mask must have the same length")
    return data.loc[mask].reset_index(drop=True), signals.loc[mask, ["signal"]].reset_index(drop=True)


def config_label(config: Dict[str, float]) -> str:
    parts = []
    for k in ("rsi_oversold", "rsi_overbought", "bb_period", "bb_std_dev", "max_holding_days", "stop_loss_pct"):
        if k in config:
            v = config[k]
            if k == "stop_loss_pct":
                parts.append(f"stop {v:.0%}")
            elif k == "max_holding_days":
                parts.append(f"hold {int(v)}")
            elif k == "rsi_oversold":
                parts.append(f"rsi<{v:g}")
            elif k == "rsi_overbought":
                parts.append(f"rsi>{v:g}")
            elif k == "bb_period":
                parts.append(f"bb{int(v)}")
            else:
                parts.append(f"bb{v:g}sd")
    return ", ".join(parts)


class SignalCache:
    """Signals depend only on the strategy kwargs, so compute them once per (symbol, strategy kwargs)."""

    def __init__(self, prices: Dict[str, pd.DataFrame]):
        self.prices = prices
        self._cache: Dict[Tuple[str, Tuple], pd.DataFrame] = {}

    def get(self, symbol: str, strategy_kwargs: Dict[str, float]) -> pd.DataFrame:
        key = (symbol, tuple(sorted(strategy_kwargs.items())))
        if key not in self._cache:
            self._cache[key] = SwingTradingIndicators(self.prices[symbol], **strategy_kwargs).get_signals_df()
        return self._cache[key]


def run_config(
    prices: Dict[str, pd.DataFrame],
    config: Dict[str, float],
    mask_by_symbol: Dict[str, np.ndarray],
    capital: float,
    position_size: float,
    signal_cache: Optional[SignalCache] = None,
) -> pd.DataFrame:
    """Per-symbol engine metrics for one configuration over the bars selected by mask_by_symbol."""
    strategy_kwargs, engine_kwargs = split_config(config)
    cache = signal_cache or SignalCache(prices)
    rows = []
    for symbol, data in prices.items():
        signals = cache.get(symbol, strategy_kwargs)
        d, s = slice_period(data, signals, mask_by_symbol[symbol])
        engine = BacktestEngine(d, s, initial_capital=capital, position_size=position_size, **engine_kwargs)
        engine.run()
        m = engine.get_metrics()
        row = {"symbol": symbol, "bars": int(len(d))}
        row.update({k: float(m[k]) if k not in ("total_trades",) else int(m[k]) for k in SYMBOL_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def buy_and_hold(
    prices: Dict[str, pd.DataFrame], mask_by_symbol: Dict[str, np.ndarray], capital: float
) -> pd.DataFrame:
    """Fully invested from the first to the last bar of the period, engine formulas for Sharpe and drawdown."""
    rows = []
    for symbol, data in prices.items():
        d = data.loc[mask_by_symbol[symbol]].reset_index(drop=True)
        close = d["close"].astype(float)
        formulas = BacktestEngine(d[["close"]], pd.DataFrame({"signal": np.zeros(len(d), dtype=int)}))
        rows.append(
            {
                "symbol": symbol,
                "bars": int(len(d)),
                "total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0),
                "sharpe_ratio": float(formulas._calculate_sharpe(close.pct_change().dropna())),
                "max_drawdown_pct": float(formulas._calculate_max_drawdown(capital * close / close.iloc[0])),
            }
        )
    return pd.DataFrame(rows)


def summarize(per_symbol: pd.DataFrame) -> Dict[str, float]:
    """Universe-level aggregates of a per-symbol metrics frame."""
    out = {
        "mean_return_pct": float(per_symbol["total_return_pct"].mean()),
        "median_return_pct": float(per_symbol["total_return_pct"].median()),
        "symbols_positive": int((per_symbol["total_return_pct"] > 0).sum()),
        "trades": int(per_symbol["total_trades"].sum()) if "total_trades" in per_symbol else 0,
    }
    for col, name in [
        ("exposure_pct", "mean_exposure_pct"),
        ("avg_holding_days", "mean_holding_days"),
        ("sharpe_ratio", "mean_sharpe_ratio"),
        ("in_market_sharpe_ratio", "mean_in_market_sharpe_ratio"),
        ("max_drawdown_pct", "mean_max_drawdown_pct"),
        ("win_rate_pct", "mean_win_rate_pct"),
    ]:
        out[name] = float(per_symbol[col].mean()) if col in per_symbol else np.nan
    return out


def grid_configs(grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def select_best(grid_results: pd.DataFrame, min_trades: int) -> int:
    """
    Index (label) of the selected row: highest is_mean_return_pct among rows with is_trades >= min_trades,
    ties broken by higher is_median_return_pct, then by row order. Raises if nothing qualifies.
    """
    ok = grid_results[grid_results["is_trades"] >= min_trades]
    if ok.empty:
        raise ValueError(f"No configuration reached {min_trades} pooled in-sample trades")
    ranked = ok.sort_values(["is_mean_return_pct", "is_median_return_pct"], ascending=[False, False], kind="stable")
    return ranked.index[0]


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def make_figure(
    grid_results: pd.DataFrame,
    oos_summary: pd.DataFrame,
    selected_idx: int,
    default_idx: int,
    n_symbols: int,
    path: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    from plot_frozen_universe_results import BASELINE, C_BH, C_STRATEGY, DPI, INK, INK_2, MUTED, SURFACE, style_axes

    C_SELECTED = "#1baf7a"  # categorical slot 3 (aqua); slots 1-3 validate all-pairs, labels added for contrast
    SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # steps 100..700
    DIV_RED, DIV_MID, DIV_BLUE = "#e34948", "#f0efec", "#2a78d6"

    fig, (ax_grid, ax_xfer, ax_bars) = plt.subplots(
        1, 3, figsize=(15, 5.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]}
    )

    # ---- Panel A: in-sample grid as a heatmap (rows: hold x stop, columns: rsi threshold) ----
    g = grid_results.copy()
    g["row"] = [f"hold {int(h)}, stop {s:.0%}" for h, s in zip(g["max_holding_days"], g["stop_loss_pct"])]
    g["col"] = [f"RSI < {r:g}" for r in g["rsi_oversold"]]
    row_order = [f"hold {int(h)}, stop {s:.0%}" for s in GRID["stop_loss_pct"] for h in GRID["max_holding_days"]]
    col_order = [f"RSI < {r:g}" for r in GRID["rsi_oversold"]]
    mat = g.pivot(index="row", columns="col", values="is_mean_return_pct").loc[row_order, col_order]
    vals = mat.to_numpy(dtype=float)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin < 0 < vmax:
        cmap = LinearSegmentedColormap.from_list("div", [DIV_RED, DIV_MID, DIV_BLUE])
        lim = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    else:
        # One-signed: a single-hue ramp, darker = higher return (less negative when everything is below zero).
        cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax_grid.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    ax_grid.set_xticks(range(len(col_order)))
    ax_grid.set_xticklabels(col_order, fontsize=8, color=INK_2)
    ax_grid.set_yticks(range(len(row_order)))
    ax_grid.set_yticklabels(row_order, fontsize=8, color=INK_2)
    ax_grid.tick_params(length=0)
    for spine in ax_grid.spines.values():
        spine.set_visible(False)
    # 2px surface gaps between cells
    ax_grid.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax_grid.grid(which="minor", color=SURFACE, linewidth=2)
    ax_grid.tick_params(which="minor", length=0)
    cell_ink = {}
    for (i, j), v in np.ndenumerate(vals):
        rgba = cmap(norm(v))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        cell_ink[(i, j)] = SURFACE if lum < 0.5 else INK
        ax_grid.text(j, i - 0.06, f"{v:+.2f}%", ha="center", va="center", fontsize=8, color=cell_ink[(i, j)])

    def outline(idx, lw, label):
        """Ink outline over a surface-coloured ring so it reads on light and dark cells alike."""
        r = grid_results.loc[idx]
        i = row_order.index(f"hold {int(r['max_holding_days'])}, stop {r['stop_loss_pct']:.0%}")
        j = col_order.index(f"RSI < {r['rsi_oversold']:g}")
        ax_grid.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SURFACE, linewidth=lw + 2.5,
                                    zorder=5))
        ax_grid.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=lw, zorder=6))
        ax_grid.text(j, i + 0.44, label, ha="center", va="bottom", fontsize=6.5, color=cell_ink[(i, j)], zorder=7)

    outline(default_idx, 1.2, "default")
    if selected_idx != default_idx:
        outline(selected_idx, 2.2, "selected")
    else:
        ax_grid.text(0.0, 1.02, "selected = default", transform=ax_grid.transAxes, fontsize=8, color=INK_2)
    cbar = fig.colorbar(im, ax=ax_grid, fraction=0.046, pad=0.03, format=plt.FuncFormatter(lambda v, _p: f"{v:+.2f}%"))
    cbar.ax.tick_params(labelsize=7, colors=MUTED, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label("Mean in-sample total return per symbol (%)", fontsize=7.5, color=INK_2)
    ax_grid.set_title(f"In-sample grid, {IS_START} to {IS_END}\n{len(grid_results)} configurations, "
                      f"selection outlined", fontsize=9.5, loc="left", color=INK)

    # ---- Panel B: does in-sample rank transfer? ----
    x = grid_results["is_mean_return_pct"].to_numpy(dtype=float)
    y = grid_results["oos_mean_return_pct"].to_numpy(dtype=float)
    ax_xfer.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax_xfer.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax_xfer.scatter(x, y, s=36, color=MUTED, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                    label="Other grid configurations")
    for idx, color, label in [(default_idx, C_STRATEGY, "Default"), (selected_idx, C_SELECTED, "Selected in-sample")]:
        ax_xfer.scatter([x[idx]], [y[idx]], s=90, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=4,
                        label=f"{label}: {config_label(grid_results.loc[idx, list(GRID)].to_dict())}")
        ax_xfer.annotate(label, (x[idx], y[idx]), xytext=(7, 5), textcoords="offset points", fontsize=8, color=INK)
    rho = pd.Series(x).corr(pd.Series(y), method="spearman")
    style_axes(ax_xfer, grid_axis="both")
    for axis in (ax_xfer.xaxis, ax_xfer.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:+.2f}%"))
    ax_xfer.set_xlabel(f"In-sample mean return per symbol ({IS_START[:4]}-{IS_END[:4]})", fontsize=8)
    ax_xfer.set_ylabel(f"Out-of-sample mean return per symbol ({OOS_START[:4]}-{OOS_END[:4]})", fontsize=8)
    ax_xfer.set_title(f"Each configuration in-sample vs out-of-sample\nSpearman rank correlation {rho:+.2f}",
                      fontsize=9.5, loc="left", color=INK)
    ax_xfer.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="best")

    # ---- Panel C: out-of-sample comparison bars ----
    series = [("default", "Default configuration", C_STRATEGY), ("selected", "Selected in-sample", C_SELECTED),
              ("buy_and_hold", "Buy-and-hold (fully invested)", C_BH)]
    stats = [("mean_return_pct", "Mean"), ("median_return_pct", "Median")]
    xs = np.arange(len(stats))
    w = 0.24
    for k, (key, label, color) in enumerate(series):
        vals_k = [float(oos_summary.loc[key, col]) for col, _ in stats]
        bars = ax_bars.bar(xs + (k - 1) * (w + 0.03), vals_k, width=w, color=color, label=label, zorder=3)
        for b, v in zip(bars, vals_k):
            ax_bars.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -0.6), f"{v:+.2f}%",
                         ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK_2)
    ax_bars.axhline(0, color=INK_2, linewidth=0.9, zorder=2)
    style_axes(ax_bars)
    ax_bars.set_xticks(xs)
    ax_bars.set_xticklabels([lab for _, lab in stats], fontsize=8, color=INK_2)
    ax_bars.set_xlabel(f"Per-symbol total return across {n_symbols} symbols", fontsize=8)
    ax_bars.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:+.0f}%"))
    ax_bars.set_ylabel("Out-of-sample total return (%)", fontsize=8)
    ymin, ymax = ax_bars.get_ylim()
    ax_bars.set_ylim(min(ymin, -3), max(ymax, 1) * 1.5)  # headroom for the legend above the tallest bar
    ax_bars.set_title(f"Out-of-sample, {OOS_START} to {OOS_END}\nfresh 100,000 per symbol, {n_symbols} symbols",
                      fontsize=9.5, loc="left", color=INK)
    ax_bars.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left", ncol=1)

    fig.suptitle("Walk-forward evaluation: parameters chosen in-sample, judged out-of-sample",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return float(rho)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def write_report(
    path: str,
    args: argparse.Namespace,
    universe_rel: str,
    n_symbols: int,
    bars_is: int,
    bars_oos: int,
    probe: pd.DataFrame,
    grid_results: pd.DataFrame,
    default_idx: int,
    selected_idx: int,
    oos_summary: pd.DataFrame,
    per_symbol: pd.DataFrame,
    rho: float,
    min_trades: int,
    run_started: datetime,
    elapsed_s: float,
) -> None:
    import yfinance

    sel = grid_results.loc[selected_idx]
    dflt = grid_results.loc[default_idx]
    sel_cfg = {k: sel[k] for k in GRID}
    sel_label = config_label(sel_cfg)
    dflt_label = config_label(DEFAULT_CONFIG)
    n_qual = int(grid_results["qualifies"].sum())

    if probe.empty:
        probe_section = "Stage 0 was skipped for this run (`--skip-probe`)."
    else:
        probe_md = probe.copy()
        probe_md["value"] = probe_md["value"].map(lambda v: "" if pd.isna(v) else f"{v:g}")
        probe_cols = ["parameter", "value", "is_mean_return_pct", "is_median_return_pct", "is_trades",
                      "is_mean_in_market_sharpe_ratio", "changes_result"]
        probe_section = df_to_markdown(probe_md[probe_cols], floatfmt="{:.3f}")

    grid_md = grid_results.sort_values("is_mean_return_pct", ascending=False).copy()
    grid_md["configuration"] = [config_label({k: r[k] for k in GRID}) for _, r in grid_md.iterrows()]
    grid_md["note"] = ["selected" if i == selected_idx else ("default" if i == default_idx else "")
                       for i in grid_md.index]
    grid_md["oos_rank"] = grid_results["oos_mean_return_pct"].rank(ascending=False, method="min").astype(int)
    grid_cols = ["configuration", "is_mean_return_pct", "is_median_return_pct", "is_trades", "is_symbols_positive",
                 "is_mean_in_market_sharpe_ratio", "qualifies", "oos_mean_return_pct", "oos_median_return_pct",
                 "oos_trades", "oos_rank", "note"]

    oos_md = oos_summary.copy()
    oos_md.insert(0, "configuration", [
        f"default ({dflt_label})", f"selected in-sample ({sel_label})", "buy-and-hold"])
    for c in ("symbols_positive", "beats_buy_and_hold", "trades"):
        oos_md[c] = oos_md[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
    oos_cols = ["configuration", "mean_return_pct", "median_return_pct", "symbols_positive", "beats_buy_and_hold",
                "trades", "mean_exposure_pct", "mean_holding_days", "mean_sharpe_ratio",
                "mean_in_market_sharpe_ratio", "mean_max_drawdown_pct"]

    sym_cols = ["symbol", "default_total_return_pct", "default_total_trades", "selected_total_return_pct",
                "selected_total_trades", "bh_total_return_pct", "default_in_market_sharpe_ratio",
                "selected_in_market_sharpe_ratio", "bh_sharpe_ratio"]

    oos_sel_rank = int(grid_md.loc[selected_idx, "oos_rank"])
    oos_dflt_rank = int(grid_md.loc[default_idx, "oos_rank"])
    diff_oos = float(oos_summary.loc["selected", "mean_return_pct"] - oos_summary.loc["default", "mean_return_pct"])
    diff_is = float(sel["is_mean_return_pct"] - dflt["is_mean_return_pct"])
    sel_better_oos = int((per_symbol["selected_total_return_pct"] > per_symbol["default_total_return_pct"]).sum())

    if selected_idx == default_idx:
        verdict = ("In-sample selection returned the default configuration, so the walk-forward adds no "
                   "information about transfer beyond the default's own out-of-sample result.")
    else:
        direction = "kept" if diff_oos > 0 else "did not keep"
        verdict = (
            f"In-sample selection picked `{sel_label}`, which beat the default by {diff_is:+.3f} percentage "
            f"points of mean return in-sample. Out-of-sample it {direction} that edge: {diff_oos:+.3f} points "
            f"versus the default ({oos_summary.loc['selected', 'mean_return_pct']:+.3f}% vs "
            f"{oos_summary.loc['default', 'mean_return_pct']:+.3f}%), better than the default on "
            f"{sel_better_oos} of {n_symbols} symbols, and ranked {oos_sel_rank} of {len(grid_results)} grid "
            f"configurations out-of-sample (the default ranked {oos_dflt_rank}). Across all "
            f"{len(grid_results)} configurations the Spearman rank correlation between in-sample and "
            f"out-of-sample mean return is {rho:+.2f}."
        )

    cmd = (
        f"python examples/walk_forward.py --universe-file {universe_rel} --capital {args.capital:g} "
        f"--position-size {args.position_size:g}"
    )
    lines = [
        "# Walk-forward evaluation: frozen 50-symbol universe",
        "",
        f"Run started {run_started.strftime('%Y-%m-%d %H:%M UTC')}; wall time {elapsed_s:.0f}s.",
        "",
        "## Command",
        "",
        "```bash",
        "python examples/walk_forward.py",
        "# equivalent to:",
        f"# {cmd}",
        "```",
        "",
        "## Setup",
        "",
        f"- Universe: `{universe_rel}`, {n_symbols} symbols, each an isolated account with "
        f"{args.capital:,.0f} initial capital per period and {args.position_size:.0%} of capital per trade",
        f"- In-sample: {IS_START} to {IS_END} inclusive, {bars_is} bars per symbol; out-of-sample: {OOS_START} to "
        f"{OOS_END} inclusive, {bars_oos} bars per symbol. Daily bars from the same `data_cache/` CSVs as "
        "`results/frozen50_2021_2024.csv` (Yahoo `Close`, split-adjusted, not dividend-adjusted)",
        "- Signals: `strategy/indicators.py` `SwingTradingIndicators`, computed on each symbol's full series and "
        "then cut to the period, so the out-of-sample period starts with warmed-up indicators; all indicators are "
        "backward-looking, so the in-sample slice is identical to an in-sample-only computation. The in-sample "
        "period still contains the 200-bar SMA warm-up at its start, during which no signals fire",
        "- Engine: `backtest/engine.py` with the `main.py --engine legacy` defaults except where a grid axis "
        "overrides them (0.1% commission per side, trailing stop off, take-profit at max(3%, 2x stop distance), "
        "exit on a neutral signal after 5 bars, new entries halted at a 20% drawdown)",
        "- Grid: `rsi_oversold` x `max_holding_days` x `stop_loss_pct` = "
        + " x ".join(str(len(v)) for v in GRID.values())
        + f" = {len(grid_results)} configurations: "
        + "; ".join(f"`{k}` in {{{', '.join(f'{v:g}' for v in vs)}}}" for k, vs in GRID.items()),
        f"- Selection rule (fixed in the script before any out-of-sample number was computed): highest mean "
        f"per-symbol in-sample total return among configurations with at least {min_trades} pooled in-sample "
        f"trades ({MIN_TRADES_PER_SYMBOL:g} per symbol on average; the default has {int(dflt['is_trades'])}), "
        f"ties broken by higher median. {n_qual} of {len(grid_results)} configurations qualify",
        "- Buy-and-hold: fully invested in the symbol over the out-of-sample bars, engine formulas for Sharpe "
        "(2% risk-free, 252-day annualisation) and drawdown",
        f"- Software: Python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__}, "
        f"yfinance {yfinance.__version__}",
        "",
        "## Stage 0: which parameters matter (in-sample, one at a time)",
        "",
        "Each exposed parameter varied alone from the default; `changes_result` is whether any in-sample number "
        "differs from the default row.",
        "",
        probe_section,
        "",
        "The Bollinger parameters and the RSI overbought threshold do not change a single number: the strategy "
        "multiplies the Bollinger and RSI flags by trend, momentum and price-action filters, and a close at or "
        "below the lower band almost never coincides with those filters being true, while RSI overbought only "
        "feeds sell signals that never meet an open long. The grid therefore varies the RSI oversold threshold "
        "(upwards, since 30 and 35 are indistinguishable), the maximum holding period and the stop loss.",
        "",
        "## Stage 1 and 2: in-sample grid and selection",
        "",
        f"Sorted by in-sample mean return. `oos_rank` is the configuration's rank by out-of-sample mean return "
        f"among the {len(grid_results)} configurations (1 = best), shown for the transfer check; it played no "
        "part in the selection. Machine-readable: `results/walk_forward_grid.csv`.",
        "",
        df_to_markdown(grid_md[grid_cols], floatfmt="{:.3f}"),
        "",
        f"Selected: `{sel_label}` (in-sample mean {sel['is_mean_return_pct']:+.3f}%, median "
        f"{sel['is_median_return_pct']:+.3f}%, {int(sel['is_trades'])} trades). Default: `{dflt_label}` "
        f"(in-sample mean {dflt['is_mean_return_pct']:+.3f}%, median {dflt['is_median_return_pct']:+.3f}%, "
        f"{int(dflt['is_trades'])} trades).",
        "",
        "## Stage 3: out-of-sample",
        "",
        f"{OOS_START} to {OOS_END}, fresh {args.capital:,.0f} per symbol. Means and medians are across the "
        f"{n_symbols} symbols; `beats_buy_and_hold` counts symbols where the configuration's total return exceeds "
        "buy-and-hold in the same symbol. Per-symbol rows: `results/walk_forward.csv`.",
        "",
        df_to_markdown(oos_md[oos_cols], floatfmt="{:.3f}"),
        "",
        "### What it shows",
        "",
        verdict,
        "",
        "Both strategy configurations remain close to flat out-of-sample while buy-and-hold in the same "
        f"symbols returned {oos_summary.loc['buy_and_hold', 'mean_return_pct']:+.2f}% on average; "
        "re-tuning the exposed thresholds does not change the character of the result reported in "
        "`results/summary.md`.",
        "",
        "### Per symbol (out-of-sample)",
        "",
        df_to_markdown(per_symbol[sym_cols]),
        "",
        "## Caveats",
        "",
        "- One split, one selection. A single in-sample/out-of-sample cut on four years of daily data is the "
        "minimum honest protocol, not a statistical test; with 18 configurations and small return differences "
        "the in-sample winner is partly noise, which is exactly what the out-of-sample column is there to show.",
        "- The grid is small by design and only covers thresholds the code exposes. The exits that dominate "
        "trade length (neutral-signal exit after 5 bars, take-profit rule) are fixed engine behaviour and were "
        "not tuned.",
        "- Each period is an isolated account per symbol: no portfolio aggregation, no slippage or spread, fills "
        "at the daily close, dividends excluded from buy-and-hold.",
        "- The universe is the same hand-picked list of survivors as the main run; the out-of-sample period is "
        "out-of-sample for the parameters, not for the choice of symbols.",
        "- Yahoo data can be revised; with `data_cache/` present the run is offline and reproduces these files "
        "exactly.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe-file", default=DEFAULT_UNIVERSE, help="frozen universe file (default: 50-symbol)")
    p.add_argument("--capital", type=float, default=100000.0, help="initial capital per symbol per period")
    p.add_argument("--position-size", type=float, default=0.1, help="fraction of capital per trade (default 0.1)")
    p.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"))
    p.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, "data_cache"))
    p.add_argument("--max-symbols", type=int, default=None, help="only the first N symbols (smoke testing)")
    p.add_argument("--skip-probe", action="store_true", help="skip stage 0 (the probe table is then omitted)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_started = datetime.now(timezone.utc)
    t0 = time.time()

    universe_path = os.path.abspath(args.universe_file)
    universe_rel = os.path.relpath(universe_path, PROJECT_ROOT)
    symbols = [s.upper().strip() for s in UniverseConstructor.load_universe(universe_path)]
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    prices = load_prices(symbols, IS_START, OOS_END, args.cache_dir, batch_size=5, retry_wait=15.0)
    missing = [s for s in symbols if s not in prices or prices[s].empty]
    if missing:
        print(f"No data for {len(missing)} symbols, skipped: {', '.join(missing)}")
    prices = {s: prices[s] for s in symbols if s not in missing}
    if not prices:
        print("No price data; nothing to do")
        return 1

    is_masks, oos_masks = {}, {}
    for s, d in prices.items():
        is_masks[s], oos_masks[s] = period_masks(d["date"], IS_START, IS_END, OOS_START, OOS_END)
    bars_is = int(np.median([m.sum() for m in is_masks.values()]))
    bars_oos = int(np.median([m.sum() for m in oos_masks.values()]))
    print(f"{len(prices)} symbols; in-sample {bars_is} bars, out-of-sample {bars_oos} bars (median per symbol)")

    cache = SignalCache(prices)

    # Stage 0: probe
    probe_rows = []
    if not args.skip_probe:
        base = run_config(prices, DEFAULT_CONFIG, is_masks, args.capital, args.position_size, cache)
        base_sum = summarize(base)
        probe_rows.append({"parameter": "(default)", "value": np.nan, **base_sum, "changes_result": False})
        for key, values in PROBE.items():
            for v in values:
                cfg = dict(DEFAULT_CONFIG)
                cfg[key] = v
                res = run_config(prices, cfg, is_masks, args.capital, args.position_size, cache)
                changed = not np.allclose(
                    res[["total_return_pct", "total_trades"]].to_numpy(dtype=float),
                    base[["total_return_pct", "total_trades"]].to_numpy(dtype=float),
                )
                probe_rows.append({"parameter": key, "value": v, **summarize(res), "changes_result": changed})
                print(f"probe {key}={v:g}: IS mean {probe_rows[-1]['mean_return_pct']:+.3f}% "
                      f"trades {probe_rows[-1]['trades']} changed={changed}")
    probe = pd.DataFrame(probe_rows)
    if not probe.empty:
        probe = probe.rename(columns={
            "mean_return_pct": "is_mean_return_pct", "median_return_pct": "is_median_return_pct",
            "trades": "is_trades", "mean_in_market_sharpe_ratio": "is_mean_in_market_sharpe_ratio",
        })

    # Stage 1: grid in-sample (and, recorded for the transfer check only, out-of-sample)
    configs = grid_configs(GRID)
    grid_rows = []
    for i, cfg in enumerate(configs):
        is_res = run_config(prices, cfg, is_masks, args.capital, args.position_size, cache)
        oos_res = run_config(prices, cfg, oos_masks, args.capital, args.position_size, cache)
        is_sum, oos_sum = summarize(is_res), summarize(oos_res)
        row = dict(cfg)
        row.update({f"is_{k}": v for k, v in is_sum.items()})
        row.update({f"oos_{k}": v for k, v in oos_sum.items()})
        grid_rows.append(row)
        print(f"grid {i + 1}/{len(configs)} {config_label(cfg):28s} IS mean {is_sum['mean_return_pct']:+.3f}% "
              f"trades {is_sum['trades']:4d} | OOS mean {oos_sum['mean_return_pct']:+.3f}% "
              f"trades {oos_sum['trades']:4d}")
    grid_results = pd.DataFrame(grid_rows)
    min_trades = int(round(MIN_TRADES_PER_SYMBOL * len(prices)))
    grid_results["qualifies"] = grid_results["is_trades"] >= min_trades

    # Stage 2: selection (in-sample only)
    default_idx = next(i for i, c in enumerate(configs) if all(np.isclose(c[k], DEFAULT_CONFIG[k]) for k in c))
    selected_idx = int(select_best(grid_results, min_trades))
    grid_results["is_default"] = [i == default_idx for i in grid_results.index]
    grid_results["selected"] = [i == selected_idx for i in grid_results.index]
    selected_cfg = configs[selected_idx]
    print(f"\nSelected in-sample: {config_label(selected_cfg)} (default: {config_label(DEFAULT_CONFIG)})")

    # Stage 3: out-of-sample for default, selected and buy-and-hold
    oos_default = run_config(prices, DEFAULT_CONFIG, oos_masks, args.capital, args.position_size, cache)
    oos_selected = run_config(prices, selected_cfg, oos_masks, args.capital, args.position_size, cache)
    oos_bh = buy_and_hold(prices, oos_masks, args.capital)

    per_symbol = oos_default.rename(columns={c: f"default_{c}" for c in SYMBOL_METRICS})
    per_symbol = per_symbol.merge(
        oos_selected.drop(columns=["bars"]).rename(columns={c: f"selected_{c}" for c in SYMBOL_METRICS}), on="symbol"
    )
    per_symbol = per_symbol.merge(
        oos_bh.drop(columns=["bars"]).rename(columns={c: f"bh_{c}" for c in ["total_return_pct", "sharpe_ratio",
                                                                              "max_drawdown_pct"]}), on="symbol"
    )
    per_symbol.insert(1, "period", f"{OOS_START}..{OOS_END}")

    summaries = {}
    for key, frame in [("default", oos_default), ("selected", oos_selected), ("buy_and_hold", oos_bh)]:
        s = summarize(frame)
        if key == "buy_and_hold":
            s["beats_buy_and_hold"] = np.nan
            s["trades"] = np.nan
        else:
            beats = frame["total_return_pct"].to_numpy() > oos_bh["total_return_pct"].to_numpy()
            s["beats_buy_and_hold"] = int(beats.sum())
        summaries[key] = s
    oos_summary = pd.DataFrame(summaries).T
    oos_summary.index.name = "configuration"

    # Outputs
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    per_symbol.round(4).to_csv(os.path.join(args.out_dir, "walk_forward.csv"), index=False)
    grid_results.round(4).to_csv(os.path.join(args.out_dir, "walk_forward_grid.csv"), index=False)
    fig_path = os.path.join(fig_dir, "fig5_walk_forward.png")
    rho = make_figure(grid_results, oos_summary, selected_idx, default_idx, len(prices), fig_path)
    write_report(
        os.path.join(args.out_dir, "walk_forward.md"), args, universe_rel, len(prices), bars_is, bars_oos, probe,
        grid_results, default_idx, selected_idx, oos_summary, per_symbol, rho, min_trades, run_started,
        time.time() - t0,
    )

    print("\nOut-of-sample summary:")
    print(oos_summary[["mean_return_pct", "median_return_pct", "symbols_positive", "beats_buy_and_hold", "trades",
                       "mean_exposure_pct", "mean_in_market_sharpe_ratio"]].astype(float).round(3).to_string())
    print(f"Spearman(in-sample, out-of-sample) over {len(configs)} configurations: {rho:+.3f}")
    print(f"Wrote results/walk_forward.csv, results/walk_forward_grid.csv, results/walk_forward.md, "
          f"results/figures/fig5_walk_forward.png ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
