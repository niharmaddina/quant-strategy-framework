import pandas as pd
from typing import Any

from .utils import get_logger

logger = get_logger(__name__)


def summarize_stats(stats: pd.Series | dict[str, Any]):
    """Print key stats in a neat way."""
    s = pd.Series(stats)
    keys = [
        "Start",
        "End",
        "Duration",
        "Equity Final [$]",
        "Return [%]",
        "Buy & Hold Return [%]",
        "Sharpe Ratio",
        "Max. Drawdown [%]",
        "Win Rate [%]",
        "Avg. Trade",
        "# Trades",
    ]
    logger.info("===== Backtest Summary =====")
    for k in keys:
        if k in s.index:
            logger.info(f"{k:25s}: {s[k]}")


def extract_equity_curve(stats: Any) -> pd.Series | None:
    """
    Backtesting.py can return extra data if you pass return_data=True.
    This is a hook if you want to extend for plotting/equity curve analysis later.
    """
    if isinstance(stats, dict) and "equity_curve" in stats:
        ec = stats["equity_curve"]
        if isinstance(ec, pd.DataFrame) and "Equity" in ec.columns:
            return ec["Equity"]
    return None
