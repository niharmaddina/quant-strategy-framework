"""
Unit tests for the exposure, holding-period and in-market risk metrics that
BacktestEngine reports for mostly-idle accounts (see the class docstring and
_calculate_metrics / _in_market_returns in backtest/engine.py).

Everything here is synthetic and offline. Prices are chosen so that neither the
3% take-profit nor the 5% stop loss can fire unless a test wants them to.
"""

import json

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine


def make_data(prices):
    dates = pd.date_range("2023-01-02", periods=len(prices), freq="B")
    return pd.DataFrame({"date": dates, "close": np.asarray(prices, dtype=float)})


def make_signals(n, buys=(), sells=(), hold_from=None):
    """signal=1 on `buys`, -1 on `sells`; if hold_from is given, signal=1 on every bar from there on."""
    s = np.zeros(n, dtype=int)
    if hold_from is not None:
        s[hold_from:] = 1
    s[list(buys)] = 1
    s[list(sells)] = -1
    return pd.DataFrame({"signal": s})


def one_trade_in_idle_account():
    """300 flat bars; one 5-bar trade (entered bar 100, sold bar 105) with +0.5% per bar while held."""
    prices = np.full(300, 100.0)
    prices[101:106] = [100.5, 101.0, 101.5, 102.0, 102.5]
    prices[106:] = 102.5
    data = make_data(prices)
    signals = make_signals(300, buys=[100], sells=[105])
    engine = BacktestEngine(data, signals, initial_capital=100000, position_size=0.1, commission=0.001)
    engine.run()
    return engine


class TestIdleAccount:
    @pytest.mark.fast
    def test_never_trades(self):
        data = make_data(np.linspace(100, 120, 60))
        engine = BacktestEngine(data, make_signals(60))
        engine.run()
        m = engine.get_metrics()

        assert m["total_trades"] == 0
        assert m["exposure_pct"] == 0.0
        assert m["in_market_days"] == 0
        assert np.isnan(m["avg_holding_days"])
        assert np.isnan(m["in_market_sharpe_ratio"])
        assert np.isnan(m["in_market_sortino_ratio"])
        # Account-level ratios keep the historical 0.0-when-undefined convention.
        assert m["sharpe_ratio"] == 0.0
        assert m["sortino_ratio"] == 0.0
        assert (engine.results["position"] == 0).all()
        assert not engine.results["in_market"].any()

    @pytest.mark.fast
    def test_export_writes_null_for_undefined_ratios(self, tmp_path):
        engine = BacktestEngine(make_data(np.linspace(100, 120, 60)), make_signals(60))
        engine.run()
        path = tmp_path / "metrics.json"
        engine.export_metrics_to_json(str(path))
        loaded = json.loads(path.read_text())
        assert loaded["in_market_sharpe_ratio"] is None
        assert loaded["avg_holding_days"] is None
        assert loaded["exposure_pct"] == 0.0


class TestMixedAccount:
    """One short trade inside a long idle stretch: the case the metrics were added for."""

    @pytest.mark.fast
    def test_exposure_and_holding_period(self):
        engine = one_trade_in_idle_account()
        m = engine.get_metrics()
        trades = engine.get_trades()

        assert m["total_trades"] == 1
        assert trades.iloc[0]["exit_reason"] == "signal"
        assert trades.iloc[0]["bars_held"] == 5
        assert m["avg_holding_days"] == 5.0
        # Held at the close of bars 100..104 -> 5 of 300 bars.
        assert m["exposure_pct"] == pytest.approx(5 / 300 * 100)
        # Returns of bars 101..105 were earned with the position open.
        assert m["in_market_days"] == 5
        assert engine.results["in_market"].to_numpy().nonzero()[0].tolist() == [101, 102, 103, 104, 105]

    @pytest.mark.fast
    def test_in_market_returns_are_returns_on_capital_at_risk(self):
        engine = one_trade_in_idle_account()
        r = engine._in_market_returns()
        shares = engine.get_trades().iloc[0]["shares"]
        assert shares == 100
        # Bars 101..104: plain price change over the position's value at the previous close.
        expected = [0.5 / 100.0, 0.5 / 100.5, 0.5 / 101.0, 0.5 / 101.5]
        np.testing.assert_allclose(r.to_numpy()[:4], expected, rtol=1e-12)
        # Exit bar 105: price change minus the exit commission, over the position value at bar 104.
        exit_commission = shares * 102.5 * 0.001
        expected_exit = (shares * 0.5 - exit_commission) / (shares * 102.0)
        assert r.to_numpy()[4] == pytest.approx(expected_exit, rel=1e-12)

    @pytest.mark.fast
    def test_account_sharpe_is_large_negative_while_in_market_sharpe_is_positive(self):
        """The account made money (+0.25% minus commission) but sat in cash 98% of the time."""
        m = one_trade_in_idle_account().get_metrics()

        assert m["total_return_pct"] > 0
        assert m["sharpe_ratio"] < -5  # -2% risk-free over near-zero account volatility
        assert m["in_market_sharpe_ratio"] > 0
        assert np.isfinite(m["in_market_sharpe_ratio"])
        # All five in-market returns are positive, so the downside deviation is undefined -> NaN, not 0.
        assert np.isnan(m["in_market_sortino_ratio"])

    @pytest.mark.fast
    def test_exposure_identity_over_several_trades(self):
        """sum(bars_held) == in_market_days == exposure * bars when every trade exits before the end."""
        n = 200
        data = make_data(np.full(n, 100.0))
        # Keep the signal at 1 while holding so the engine's "neutral signal after 5 bars" exit cannot fire;
        # the three trades then close on their sell signals after 3, 7 and 11 bars.
        signals = make_signals(n, sells=[13, 67, 131])
        for start, stop in ((10, 13), (60, 67), (120, 131)):
            signals.loc[start:stop - 1, "signal"] = 1
        engine = BacktestEngine(data, signals, commission=0.0)
        engine.run()
        m = engine.get_metrics()
        trades = engine.get_trades()

        assert trades["bars_held"].tolist() == [3, 7, 11]
        assert m["in_market_days"] == 21
        assert m["exposure_pct"] == pytest.approx(21 / n * 100)
        assert m["avg_holding_days"] == pytest.approx(7.0)
        # Flat price and zero commission: every in-market return is exactly zero -> undefined ratio.
        assert np.isnan(m["in_market_sharpe_ratio"])


class TestFullyInvestedAccount:
    @pytest.fixture
    def oscillating_prices(self):
        # +/-1% around 100: never reaches the 3% take-profit or the 5% stop.
        i = np.arange(40)
        return 100.0 * (1.0 + 0.01 * np.sin(i / 3.0))

    @pytest.mark.fast
    def test_in_market_equals_whole_window(self, oscillating_prices):
        n = len(oscillating_prices)
        data = make_data(oscillating_prices)
        engine = BacktestEngine(
            data, make_signals(n, hold_from=0), position_size=0.99, commission=0.001, max_holding_days=10_000
        )
        engine.run()
        m = engine.get_metrics()
        trades = engine.get_trades()

        assert len(trades) == 1
        assert trades.iloc[0]["exit_reason"] == "end_of_period"
        assert trades.iloc[0]["bars_held"] == n - 1
        assert m["avg_holding_days"] == n - 1
        assert m["exposure_pct"] == 100.0
        assert m["in_market_days"] == n - 1

        # With no intermediate commissions the return on capital at risk is the stock's own return.
        stock_returns = pd.Series(oscillating_prices).pct_change().dropna().to_numpy()
        np.testing.assert_allclose(engine._in_market_returns().to_numpy(), stock_returns, rtol=1e-9)

        # 99% invested: the in-market Sharpe is the account Sharpe up to the 1% cash dilution.
        assert m["in_market_sharpe_ratio"] == pytest.approx(m["sharpe_ratio"], abs=0.05)
        assert m["in_market_sortino_ratio"] == pytest.approx(m["sortino_ratio"], abs=0.05)


class TestRatioHelpers:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(pd.DataFrame({"close": [100.0, 101.0, 102.0]}), pd.DataFrame({"signal": [0, 0, 0]}))

    @pytest.mark.fast
    def test_undefined_sentinel(self, engine):
        for bad in (pd.Series(dtype=float), pd.Series([0.01]), pd.Series([0.0, 0.0, 0.0])):
            assert engine._calculate_sharpe(bad) == 0.0
            assert np.isnan(engine._calculate_sharpe(bad, undefined=np.nan))
            assert engine._calculate_sortino(bad) == 0.0
            assert np.isnan(engine._calculate_sortino(bad, undefined=np.nan))
        # No negative returns: Sortino has no downside deviation.
        up_only = pd.Series([0.01, 0.02, 0.005])
        assert engine._calculate_sortino(up_only) == 0.0
        assert np.isnan(engine._calculate_sortino(up_only, undefined=np.nan))

    @pytest.mark.fast
    def test_sortino_uses_engine_risk_free_rate(self):
        """Regression: _calculate_sortino used to hard-code a 2% default instead of self.risk_free_rate."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0])
        downside_std = returns[returns < 0].std() * np.sqrt(252)
        for rf in (0.0, 0.02, 0.05):
            engine = BacktestEngine(
                pd.DataFrame({"close": [100.0, 101.0]}), pd.DataFrame({"signal": [0, 0]}), risk_free_rate=rf
            )
            expected = (returns.mean() * 252 - rf) / downside_std
            assert engine._calculate_sortino(returns) == pytest.approx(expected, rel=1e-12)
            assert engine._calculate_sharpe(returns) == pytest.approx(
                (returns.mean() * 252 - rf) / (returns.std() * np.sqrt(252)), rel=1e-12
            )


class TestMaxHoldingDaysParameter:
    @pytest.mark.fast
    def test_custom_cap_forces_exit(self):
        n = 30
        data = make_data(np.full(n, 100.0))
        engine = BacktestEngine(data, make_signals(n, hold_from=3), commission=0.0, max_holding_days=4)
        engine.run()
        first = engine.get_trades().iloc[0]
        assert first["exit_reason"] == "max_holding_period"
        assert first["bars_held"] == 4

    @pytest.mark.fast
    def test_default_is_twenty(self):
        engine = BacktestEngine(make_data(np.full(5, 100.0)), make_signals(5))
        assert engine.max_holding_days == 20

    @pytest.mark.fast
    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            BacktestEngine(make_data(np.full(5, 100.0)), make_signals(5), max_holding_days=0)
