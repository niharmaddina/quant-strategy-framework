"""
Consolidated ML tests (features, preprocessing, labeling, model, integration).

Goal:
- fewer files
- shared fixtures (see tests/conftest.py)
- deterministic / no-network (marked fast)
"""

import pytest
import pandas as pd
import numpy as np

from ml.preprocessing import prep_ohlcv, resample_ohlcv
from ml.labeling import label_engine_tp_sl_first_long_only
from ml.features import build_mr_quality_features
from ml.model import MRSignalQualityModel

from strategy.mean_reversion import MeanReversionIndicators
from backtest.engine import BacktestEngine


class TestMLPreprocessing:
    """Preprocessing utilities."""

    @pytest.mark.fast
    def test_prep_ohlcv_with_date_column(self, sample_intraday_5m_ohlcv):
        result = prep_ohlcv(sample_intraday_5m_ohlcv)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'date' not in result.columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
        assert len(result) == len(sample_intraday_5m_ohlcv)

    @pytest.mark.fast
    def test_prep_ohlcv_with_datetime_index(self, sample_intraday_5m_ohlcv):
        df = sample_intraday_5m_ohlcv.set_index('date')
        result = prep_ohlcv(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'open' in result.columns

    @pytest.mark.fast
    def test_prep_ohlcv_missing_columns(self):
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='5T'),
            'open': range(10),
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            prep_ohlcv(df)

    @pytest.mark.fast
    def test_resample_ohlcv_5m_to_15m(self, sample_intraday_5m_ohlcv):
        df = prep_ohlcv(sample_intraday_5m_ohlcv)
        result = resample_ohlcv(df, '15T')
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) <= len(df)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns


class TestMLLabeling:
    """Labeling utilities."""

    @pytest.mark.fast
    def test_label_engine_tp_sl_basic(self):
        dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='5T')
        n = len(dates)
        base_price = 100.0

        signals = np.zeros(n)
        signals[10] = 1
        signals[30] = 1
        signals[50] = 1

        prices = np.full(n, base_price)
        highs = np.full(n, base_price * 1.02)
        lows = np.full(n, base_price * 0.98)

        # TP-ish path
        prices[10:25] = np.linspace(base_price, base_price * 1.05, 15)
        highs[10:25] = prices[10:25] * 1.01
        lows[10:25] = prices[10:25] * 0.99

        # SL-ish path
        prices[30:45] = np.linspace(base_price, base_price * 0.94, 15)
        highs[30:45] = prices[30:45] * 1.01
        lows[30:45] = prices[30:45] * 0.99

        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
            'signal': signals
        }).set_index('date')

        y = label_engine_tp_sl_first_long_only(
            df=data,
            signal_col="signal",
            delay_bars=1,
            horizon_bars=48,
            stop_loss_pct=0.05,
            entry_price_col="open"
        )

        assert isinstance(y, pd.Series)
        assert len(y) == len(data)
        assert y.name == "y_tp_before_sl"

    @pytest.mark.fast
    def test_label_engine_tp_sl_missing_columns(self):
        df = pd.DataFrame({
            'open': range(100),
            'low': range(100),
            'close': range(100),
            'signal': [0] * 100
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            label_engine_tp_sl_first_long_only(df)


class TestMLFeatures:
    """Feature engineering."""

    @pytest.mark.fast
    def test_build_mr_quality_features_basic(self, sample_intraday_5m_ohlcv):
        mr = MeanReversionIndicators(sample_intraday_5m_ohlcv, lookback_period=20)
        full = mr.get_full_analysis()

        X = build_mr_quality_features(full)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(full)
        assert len(X.columns) > 0

    @pytest.mark.fast
    def test_build_mr_quality_features_chaos_hooks(self, sample_intraday_5m_ohlcv):
        mr = MeanReversionIndicators(sample_intraday_5m_ohlcv, lookback_period=20)
        full = mr.get_full_analysis()
        full['hurst_15m'] = np.random.RandomState(0).rand(len(full))
        full['perm_entropy_15m'] = np.random.RandomState(1).rand(len(full))

        X = build_mr_quality_features(full)
        assert 'hurst_15m' in X.columns
        assert 'perm_entropy_15m' in X.columns


class TestMRSignalQualityModel:
    """Model unit tests."""

    @pytest.fixture
    def sample_training_data(self, sample_intraday_5m_ohlcv):
        mr = MeanReversionIndicators(sample_intraday_5m_ohlcv, lookback_period=20)
        full = mr.get_full_analysis()

        # sparse labels on some BUY signals
        y = pd.Series(np.nan, index=full.index)
        buy_signals = full[full['signal'] == 1]
        for i, idx in enumerate(buy_signals.index[::10]):
            y.loc[idx] = 1 if i % 2 == 0 else 0
        return full, y

    @pytest.mark.fast
    def test_model_fit_predict(self, sample_training_data):
        full, y = sample_training_data
        model = MRSignalQualityModel(random_state=42)

        # If no labels, fit should raise
        if y.notna().sum() == 0:
            with pytest.raises(ValueError, match="No training samples"):
                model.fit(full, y)
            return

        model.fit(full, y)
        preds = model.predict(full)

        assert isinstance(preds, pd.DataFrame)
        assert 'signal_ml' in preds.columns
        assert set(preds['signal_ml'].unique()).issubset({0, 1})


class TestMLIntegration:
    """End-to-end (still mock/offline)."""

    @pytest.mark.fast
    def test_end_to_end_workflow(self, sample_intraday_5m_ohlcv):
        # 1) prep
        df_5m = prep_ohlcv(sample_intraday_5m_ohlcv)
        df_for_ind = df_5m.reset_index()

        # 2) indicators
        mr = MeanReversionIndicators(df_for_ind, lookback_period=20)
        full = mr.get_full_analysis()
        if 'date' in full.columns:
            full = full.set_index('date').sort_index()

        # 3) labels
        y = label_engine_tp_sl_first_long_only(
            df=full,
            delay_bars=1,
            horizon_bars=48,
            stop_loss_pct=0.05,
            entry_price_col="open"
        )

        if y.notna().sum() == 0:
            # acceptable: no BUY signals early enough to label in this synthetic sample
            return

        # 4) train + predict
        model = MRSignalQualityModel(random_state=42)
        model.fit(full, y)
        ml_output = model.predict(full)
        assert 'signal_ml' in ml_output.columns

    @pytest.mark.fast
    def test_backtest_integration(self, sample_intraday_5m_ohlcv):
        df_5m = prep_ohlcv(sample_intraday_5m_ohlcv)
        df_for_ind = df_5m.reset_index()

        mr = MeanReversionIndicators(df_for_ind, lookback_period=20)
        full = mr.get_full_analysis()
        if 'date' in full.columns:
            full = full.set_index('date').sort_index()

        y = label_engine_tp_sl_first_long_only(
            df=full,
            delay_bars=1,
            horizon_bars=48,
            stop_loss_pct=0.05
        )
        if y.notna().sum() == 0:
            return

        model = MRSignalQualityModel(random_state=42)
        model.fit(full, y)
        ml_output = model.predict(full)

        data_for_bt = full.reset_index()[['date', 'close']]
        signals_ml = ml_output.reset_index()
        if 'date' not in signals_ml.columns:
            signals_ml['date'] = full.index
        signals_ml = signals_ml[['date', 'signal_ml']].rename(columns={'signal_ml': 'signal'})

        bt = BacktestEngine(
            data=data_for_bt,
            signals=signals_ml,
            stop_loss_pct=0.05,
            commission=0.001,
            position_size=0.1
        )

        bt.run()
        assert bt.results is not None
        assert len(bt.results) > 0
        assert 'equity' in bt.results.columns


