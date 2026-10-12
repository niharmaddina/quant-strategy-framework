"""
Smoke tests for the walk-forward split logic in examples/walk_forward.py, on synthetic data.

No network: prices are a seeded random walk on business days. The tests cover the
period masks, the "no look-ahead" property that makes slicing full-series signals
legitimate, the configuration split, the selection rule and one small end-to-end run.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import walk_forward as wf  # noqa: E402

from strategy.indicators import SwingTradingIndicators  # noqa: E402


def synthetic_prices(symbol: str, start="2021-01-04", end="2024-12-31") -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates))))
    spread = np.abs(rng.normal(0, 0.6, len(dates)))
    df = pd.DataFrame(
        {
            "date": dates.date,
            "open": close + rng.normal(0, 0.3, len(dates)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
        }
    )
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df


class TestPeriodMasks:
    @pytest.mark.fast
    def test_disjoint_ordered_and_inclusive(self):
        data = synthetic_prices("AAA")
        is_mask, oos_mask = wf.period_masks(data["date"], wf.IS_START, wf.IS_END, wf.OOS_START, wf.OOS_END)
        dates = pd.to_datetime(data["date"])

        assert is_mask.dtype == bool and oos_mask.dtype == bool
        assert len(is_mask) == len(oos_mask) == len(data)
        assert not (is_mask & oos_mask).any()
        assert dates[is_mask].min() == pd.Timestamp("2021-01-04")
        assert dates[is_mask].max() <= pd.Timestamp("2022-12-31")
        assert dates[oos_mask].min() >= pd.Timestamp("2023-01-03")
        assert dates[oos_mask].max() == pd.Timestamp("2024-12-31")
        assert dates[is_mask].max() < dates[oos_mask].min()
        # The only bars in neither period lie strictly between the two windows (here: 2023-01-02).
        gap = dates[~(is_mask | oos_mask)]
        assert ((gap > pd.Timestamp(wf.IS_END)) & (gap < pd.Timestamp(wf.OOS_START))).all()
        assert is_mask.sum() + oos_mask.sum() + len(gap) == len(data)

    @pytest.mark.fast
    def test_rejects_overlapping_or_misordered_windows(self):
        dates = synthetic_prices("AAA")["date"]
        with pytest.raises(ValueError):
            wf.period_masks(dates, "2021-01-04", "2023-06-30", "2023-01-03", "2024-12-31")
        with pytest.raises(ValueError):
            wf.period_masks(dates, "2023-01-03", "2024-12-31", "2021-01-04", "2022-12-31")

    @pytest.mark.fast
    def test_rejects_empty_period(self):
        dates = synthetic_prices("AAA", end="2022-06-30")["date"]
        with pytest.raises(ValueError, match="out-of-sample"):
            wf.period_masks(dates, wf.IS_START, wf.IS_END, wf.OOS_START, wf.OOS_END)


class TestNoLookAhead:
    @pytest.mark.fast
    def test_in_sample_slice_of_full_series_signals_equals_in_sample_only_signals(self):
        """Every indicator is backward-looking, so computing on the full series and cutting the prefix
        gives the same in-sample signals as computing on the in-sample bars alone."""
        data = synthetic_prices("BBB")
        is_mask, _ = wf.period_masks(data["date"], wf.IS_START, wf.IS_END, wf.OOS_START, wf.OOS_END)

        full = SwingTradingIndicators(data).get_signals_df()
        prefix_only = SwingTradingIndicators(data.loc[is_mask].reset_index(drop=True)).get_signals_df()
        sliced = full.loc[is_mask].reset_index(drop=True)

        assert sliced["signal"].tolist() == prefix_only["signal"].tolist()
        np.testing.assert_allclose(sliced["confidence"].to_numpy(), prefix_only["confidence"].to_numpy(), atol=1e-9)

    @pytest.mark.fast
    def test_out_of_sample_slice_keeps_warm_indicators(self):
        """Cutting the suffix keeps the 200-bar SMA history: signals can fire from the first OOS bar region,
        whereas an OOS-only computation has a 200-bar dead zone."""
        data = synthetic_prices("CCC")
        _, oos_mask = wf.period_masks(data["date"], wf.IS_START, wf.IS_END, wf.OOS_START, wf.OOS_END)
        full = SwingTradingIndicators(data).get_indicators_df()
        oos_only = SwingTradingIndicators(data.loc[oos_mask].reset_index(drop=True)).get_indicators_df()

        assert full.loc[oos_mask, "sma_200"].notna().all()
        assert oos_only["sma_200"].iloc[:199].isna().all()


class TestConfigHelpers:
    @pytest.mark.fast
    def test_split_config(self):
        strategy, engine = wf.split_config({"rsi_oversold": 45.0, "max_holding_days": 10, "stop_loss_pct": 0.03})
        assert strategy == {"rsi_oversold": 45.0}
        assert engine == {"max_holding_days": 10, "stop_loss_pct": 0.03}
        with pytest.raises(ValueError):
            wf.split_config({"take_profit": 0.1})

    @pytest.mark.fast
    def test_grid_configs_enumerates_product_and_contains_default(self):
        configs = wf.grid_configs(wf.GRID)
        assert len(configs) == int(np.prod([len(v) for v in wf.GRID.values()]))
        assert len(configs) <= 24
        assert any(all(np.isclose(c[k], wf.DEFAULT_CONFIG[k]) for k in c) for c in configs)
        assert len({tuple(sorted(c.items())) for c in configs}) == len(configs)

    @pytest.mark.fast
    def test_select_best_applies_min_trades_then_mean_then_median(self):
        grid = pd.DataFrame(
            {
                "is_mean_return_pct": [0.9, 0.3, 0.3, -0.1],
                "is_median_return_pct": [0.5, 0.1, 0.2, 0.0],
                "is_trades": [40, 150, 120, 200],
            }
        )
        # Row 0 has the best mean but too few trades; rows 1 and 2 tie on mean, row 2 wins on median.
        assert wf.select_best(grid, min_trades=100) == 2
        assert wf.select_best(grid, min_trades=10) == 0
        with pytest.raises(ValueError):
            wf.select_best(grid, min_trades=1000)


class TestEndToEnd:
    @pytest.mark.fast
    def test_run_config_and_buy_and_hold_on_two_synthetic_symbols(self):
        prices = {s: synthetic_prices(s) for s in ("AAA", "BBB")}
        masks = {
            s: wf.period_masks(d["date"], wf.IS_START, wf.IS_END, wf.OOS_START, wf.OOS_END) for s, d in prices.items()
        }
        is_masks = {s: m[0] for s, m in masks.items()}
        oos_masks = {s: m[1] for s, m in masks.items()}

        cache = wf.SignalCache(prices)
        res_is = wf.run_config(prices, wf.DEFAULT_CONFIG, is_masks, 100000.0, 0.1, cache)
        res_oos = wf.run_config(prices, {"rsi_oversold": 55.0, "max_holding_days": 5, "stop_loss_pct": 0.03},
                                oos_masks, 100000.0, 0.1, cache)
        bh = wf.buy_and_hold(prices, oos_masks, 100000.0)

        for frame, mask_set in ((res_is, is_masks), (res_oos, oos_masks)):
            assert list(frame["symbol"]) == ["AAA", "BBB"]
            assert set(wf.SYMBOL_METRICS) <= set(frame.columns)
            assert frame["bars"].tolist() == [int(mask_set[s].sum()) for s in ("AAA", "BBB")]
            assert frame["total_return_pct"].notna().all()
            assert (frame["exposure_pct"].between(0, 100)).all()
        assert bh["bars"].tolist() == res_oos["bars"].tolist()
        # Buy-and-hold total return is just last/first over the OOS bars.
        d = prices["AAA"].loc[oos_masks["AAA"], "close"]
        assert bh.loc[bh["symbol"] == "AAA", "total_return_pct"].item() == pytest.approx(
            (d.iloc[-1] / d.iloc[0] - 1) * 100
        )
        # Signals were computed once per (symbol, strategy kwargs): two symbols x two distinct rsi thresholds.
        assert len(cache._cache) == 4

        summary = wf.summarize(res_is)
        assert summary["trades"] == int(res_is["total_trades"].sum())
        assert summary["mean_return_pct"] == pytest.approx(res_is["total_return_pct"].mean())
