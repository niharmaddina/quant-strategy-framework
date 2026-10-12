"""
VectorBT-backed backtesting engine.

Design goals:
- Keep the existing project interface (`.run()`, `.results`, `.trades`, `.get_metrics()`).
- Avoid hard dependency at import time: vectorbt is imported lazily inside `run()`.
- Produce result columns expected by `BacktestVisualizer`: price/signal/equity/returns/cumulative_returns.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _align_data_and_signals(
    data: pd.DataFrame, signals: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (df_indexed, signal_series_aligned_to_df_index)."""
    if data.empty:
        raise ValueError("Data DataFrame is empty")
    if signals.empty:
        raise ValueError("Signals DataFrame is empty")
    if "close" not in data.columns:
        raise ValueError("Data must contain 'close' column")
    if "signal" not in signals.columns:
        raise ValueError("Signals must contain 'signal' column")

    df = data.copy()
    sig = signals.copy()

    if "date" in df.columns:
        df = df.set_index("date")

    if "date" in sig.columns:
        sig = sig.set_index("date")
    else:
        # positional align if possible, else reindex and fill flat
        if len(sig) == len(df):
            sig.index = df.index
        else:
            sig = sig.reindex(df.index, fill_value=0)

    signal_series = sig["signal"].fillna(0).astype(int)
    return df, signal_series


def _infer_freq(index: pd.Index) -> Optional[str]:
    """Best-effort frequency string for VectorBT (e.g. '1D')."""
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return None
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if freq is None:
        return None
    # VectorBT expects things like "1D", "1H" etc; infer_freq returns those for common cases.
    return str(freq)


class VectorBTBacktestEngine:
    """
    Backtesting engine powered by vectorbt.

    Notes:
    - This engine intentionally mirrors the legacy engine's public surface so it can be used
      by `main.py` and `BacktestVisualizer` without changes.
    - It does NOT attempt to exactly replicate the legacy engine's discretionary exits
      (max holding days / trend reversal / etc). Instead, it primarily respects:
      - entry/exit signals
      - fees
      - optional stop-loss / trailing stop if supported by the installed vectorbt version
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        position_size: float = 0.1,
        commission: float = 0.001,
        stop_loss_pct: float = 0.05,
        trailing_stop: bool = False,
        trailing_stop_pct: float = 0.03,
        max_drawdown_pct: float = 0.20,  # kept for API parity (not enforced here)
        risk_free_rate: float = 0.02,
    ):
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")
        if not 0 < position_size <= 1:
            raise ValueError(f"Position size must be between 0 and 1, got {position_size}")
        if commission < 0:
            raise ValueError(f"Commission must be non-negative, got {commission}")
        if not 0 < stop_loss_pct < 1:
            raise ValueError(f"Stop loss must be between 0 and 1, got {stop_loss_pct}")
        if not 0 < trailing_stop_pct < 1:
            raise ValueError(f"Trailing stop must be between 0 and 1, got {trailing_stop_pct}")
        if not 0 < max_drawdown_pct <= 1:
            raise ValueError(f"Max drawdown must be between 0 and 1, got {max_drawdown_pct}")

        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = float(initial_capital)
        self.position_size = float(position_size)
        self.commission = float(commission)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trailing_stop = bool(trailing_stop)
        self.trailing_stop_pct = float(trailing_stop_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.risk_free_rate = float(risk_free_rate)

        self.results: Optional[pd.DataFrame] = None
        self.trades: pd.DataFrame = pd.DataFrame()
        self.metrics: Dict = {}
        self._pf = None  # underlying vectorbt Portfolio

    def run(self) -> pd.DataFrame:
        """Run VectorBT backtest and populate `.results`, `.trades`, `.metrics`."""
        try:
            import vectorbt as vbt  # type: ignore
        except Exception as e:
            raise ImportError(
                "VectorBT is not installed. Install it with `pip install -U vectorbt` "
                "or run with `--engine legacy`."
            ) from e

        df, signal_series = _align_data_and_signals(self.data, self.signals)
        close = df["close"].astype(float)

        # Entries/exits derived from your existing signal convention.
        entries = (signal_series == 1)
        exits = (signal_series == -1)

        # Build kwargs in a version-tolerant way.
        import inspect

        sig = inspect.signature(vbt.Portfolio.from_signals)
        supported = set(sig.parameters.keys())

        kwargs = {
            "init_cash": self.initial_capital,
            "fees": self.commission,
            # Position sizing: prefer percent sizing if supported.
            "size": self.position_size,
            "size_type": "percent",
            "freq": _infer_freq(close.index),
            # Stops (if supported by installed vectorbt)
            "sl_stop": self.stop_loss_pct,
            # trailing stop variants differ across versions; we pass whichever exists
            "tsl_stop": self.trailing_stop_pct if self.trailing_stop else None,
            "sl_trail": True if self.trailing_stop else None,
        }

        # Drop Nones and unsupported kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None and k in supported}

        # Some vectorbt versions don't support percent sizing via size_type='percent'
        # so we fall back to "all-in" sizing if we get an error.
        try:
            pf = vbt.Portfolio.from_signals(close, entries, exits, **filtered_kwargs)
        except Exception:
            fallback_kwargs = dict(filtered_kwargs)
            fallback_kwargs.pop("size_type", None)
            # "all-in" sizing is most version-compatible via size=np.inf (if supported).
            if "size" in supported:
                fallback_kwargs["size"] = np.inf
            pf = vbt.Portfolio.from_signals(close, entries, exits, **fallback_kwargs)

        self._pf = pf

        # Equity curve
        equity = None
        for attr in ("value", "value"):
            try:
                candidate = getattr(pf, attr)
                equity = candidate() if callable(candidate) else candidate
                break
            except Exception:
                continue
        if equity is None:
            # last resort: derive from cash + asset value if available
            try:
                equity = pf.value()
            except Exception as e:
                raise RuntimeError("Unable to extract equity curve from VectorBT Portfolio") from e

        equity = pd.Series(equity, index=close.index, name="equity").astype(float)

        results = pd.DataFrame(index=close.index)
        results["price"] = close
        results["signal"] = signal_series.reindex(close.index).fillna(0).astype(int)
        results["equity"] = equity
        results["returns"] = results["equity"].pct_change()
        results["cumulative_returns"] = (1 + results["returns"].fillna(0)).cumprod() - 1

        self.results = results
        self.trades = self._extract_trades(pf)
        self._calculate_metrics()
        return results

    def _extract_trades(self, pf) -> pd.DataFrame:
        """Best-effort extraction of trades into the legacy-like schema."""
        try:
            trades_obj = getattr(pf, "trades", None)
            if trades_obj is None:
                return pd.DataFrame()

            # VectorBT commonly provides `records_readable` (property or method).
            rec = getattr(trades_obj, "records_readable", None)
            if rec is None:
                return pd.DataFrame()
            rec_df = rec() if callable(rec) else rec
            if rec_df is None or len(rec_df) == 0:
                return pd.DataFrame()
            rec_df = pd.DataFrame(rec_df).copy()

            # Try to map common columns; keep robust across versions.
            colmap = {}
            for src, dst in [
                ("Entry Timestamp", "entry_date"),
                ("Exit Timestamp", "exit_date"),
                ("Entry Price", "entry_price"),
                ("Exit Price", "exit_price"),
                ("Size", "shares"),
                ("PnL", "pnl"),
                ("Return", "return_pct"),
            ]:
                if src in rec_df.columns:
                    colmap[src] = dst
            trades = rec_df.rename(columns=colmap)

            # Normalize return to percentage if it looks like decimal
            if "return_pct" in trades.columns:
                try:
                    if trades["return_pct"].abs().max() <= 2.0:
                        trades["return_pct"] = trades["return_pct"] * 100.0
                except Exception:
                    pass

            # Keep only columns the rest of the repo expects.
            wanted = ["entry_date", "exit_date", "entry_price", "exit_price", "shares", "pnl", "return_pct"]
            keep = [c for c in wanted if c in trades.columns]
            trades = trades[keep].copy()

            # Provide an exit_reason column for compatibility (VectorBT doesn't map 1:1)
            trades["exit_reason"] = "signal_or_stop"
            return trades
        except Exception:
            return pd.DataFrame()

    def _calculate_metrics(self):
        """Calculate a legacy-compatible metrics dict from `.results` / `.trades`."""
        if self.results is None or self.results.empty:
            self.metrics = {}
            return

        equity = self.results["equity"]
        returns = self.results["returns"].dropna()

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if len(equity) > 1 else 0.0
        total_trades = len(self.trades) if not self.trades.empty else 0

        if total_trades > 0 and "pnl" in self.trades.columns:
            winning = self.trades[self.trades["pnl"] > 0]
            losing = self.trades[self.trades["pnl"] <= 0]
            win_rate = (len(winning) / total_trades) * 100
            avg_win = float(winning["pnl"].mean()) if len(winning) else 0.0
            avg_loss = float(losing["pnl"].mean()) if len(losing) else 0.0
            profit_factor = (
                abs(winning["pnl"].sum() / losing["pnl"].sum())
                if len(losing) and float(losing["pnl"].sum()) != 0.0
                else 0.0
            )
            if "return_pct" in self.trades.columns:
                avg_return_per_trade = float(self.trades["return_pct"].mean())
            else:
                avg_return_per_trade = 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_return_per_trade = 0.0

        sharpe_ratio = self._calculate_sharpe(returns) if len(returns) else 0.0
        sortino_ratio = self._calculate_sortino(returns) if len(returns) else 0.0
        max_drawdown = self._calculate_max_drawdown(equity)
        calmar_ratio = self._calculate_calmar(total_return, max_drawdown)
        volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) else 0.0

        self.metrics = {
            "total_return_pct": float(total_return),
            "final_capital": float(equity.iloc[-1]),
            "total_trades": int(total_trades),
            "win_rate_pct": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "avg_return_per_trade_pct": float(avg_return_per_trade),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown_pct": float(max_drawdown),
            "volatility_pct": float(volatility),
            # keep legacy keys present (exit reason breakdown not available here)
            "stop_loss_trades": 0,
            "trailing_stop_trades": 0,
            "signal_trades": int(total_trades),
            "max_drawdown_trades": 0,
        }

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        if len(returns) == 0:
            return 0.0
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        std_dev = returns.std()
        if std_dev == 0 or np.isnan(std_dev):
            return 0.0
        excess_returns = returns.mean() * 252 - float(risk_free_rate)
        sharpe = excess_returns / (std_dev * np.sqrt(252))
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        return float(sharpe)

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        if len(returns) == 0:
            return 0.0
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = downside.std() * np.sqrt(252)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        excess_return = returns.mean() * 252 - float(risk_free_rate)
        sortino = excess_return / downside_std
        if np.isnan(sortino) or np.isinf(sortino):
            return 0.0
        return float(sortino)

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        if len(equity) == 0:
            return 0.0
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return float(abs(drawdown.min()) * 100)

    def _calculate_calmar(self, total_return_pct: float, max_drawdown_pct: float) -> float:
        if max_drawdown_pct == 0:
            return 0.0
        total_return = total_return_pct / 100.0
        max_drawdown = max_drawdown_pct / 100.0
        calmar = total_return / max_drawdown
        if np.isnan(calmar) or np.isinf(calmar):
            return 0.0
        return float(calmar)

    def get_metrics(self) -> Dict:
        return self.metrics.copy()

    def print_metrics(self):
        if not self.metrics:
            print("No metrics available. Run backtest first.")
            return
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS (VectorBT)")
        print("=" * 50)
        print(f"Total Return:        {self.metrics['total_return_pct']:.2f}%")
        print(f"Final Capital:       ${self.metrics['final_capital']:,.2f}")
        print(f"Total Trades:        {self.metrics['total_trades']}")
        print(f"Win Rate:            {self.metrics['win_rate_pct']:.2f}%")
        print(f"Sharpe Ratio:        {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {self.metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown:        {self.metrics['max_drawdown_pct']:.2f}%")
        print("=" * 50 + "\n")

    def get_trades(self) -> pd.DataFrame:
        return self.trades.copy() if not self.trades.empty else pd.DataFrame()


