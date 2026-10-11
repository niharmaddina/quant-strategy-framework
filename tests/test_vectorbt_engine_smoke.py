import pytest
import pandas as pd
import numpy as np


pytest.importorskip("vectorbt", reason="vectorbt not installed")


def test_vectorbt_engine_runs_smoke():
    from backtest.vectorbt_engine import VectorBTBacktestEngine

    dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
    prices = 100 + np.cumsum(np.random.randn(60))
    data = pd.DataFrame({"date": dates, "close": prices})

    signals = pd.DataFrame({"date": dates, "signal": np.zeros(60)})
    signals.loc[5, "signal"] = 1
    signals.loc[20, "signal"] = -1

    eng = VectorBTBacktestEngine(data=data, signals=signals, initial_capital=100000, position_size=0.1)
    res = eng.run()

    assert "equity" in res.columns
    assert "price" in res.columns
    assert "signal" in res.columns
    assert "returns" in res.columns
    assert isinstance(eng.get_metrics(), dict)


