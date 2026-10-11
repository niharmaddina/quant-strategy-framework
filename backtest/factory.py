"""
Backtest engine factory.

Keeps the legacy engine intact for tests, while allowing swapping to VectorBT.
"""

from __future__ import annotations

from typing import Literal

from backtest.engine import BacktestEngine

EngineName = Literal["legacy", "vectorbt"]


def create_backtest_engine(engine: EngineName, **kwargs):
    """
    Create a backtest engine instance.

    Args:
        engine: "legacy" or "vectorbt"
        **kwargs: forwarded to the engine constructor
    """
    if engine == "legacy":
        return BacktestEngine(**kwargs)

    if engine == "vectorbt":
        from backtest.vectorbt_engine import VectorBTBacktestEngine

        return VectorBTBacktestEngine(**kwargs)

    raise ValueError(f"Unknown engine: {engine}")


