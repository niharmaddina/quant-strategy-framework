"""
Fast unit tests for SwingTradingIndicators.
"""

import pandas as pd
import pytest

from strategy.indicators import SwingTradingIndicators


class TestSwingTradingIndicators:
    """Test suite for SwingTradingIndicators."""
    
    @pytest.mark.fast
    def test_init_with_valid_data(self, sample_price_data):
        """Test initialization with valid data."""
        indicators = SwingTradingIndicators(sample_price_data)
        
        assert indicators is not None
        assert not indicators.data.empty
        assert not indicators.signals.empty
    
    @pytest.mark.fast
    def test_calculate_all_indicators(self, sample_price_data):
        """Test that all indicators are calculated."""
        indicators = SwingTradingIndicators(sample_price_data)
        
        data = indicators.get_indicators_df()
        
        # Check key indicators exist
        assert 'rsi' in data.columns
        assert 'macd' in data.columns
        assert 'sma_20' in data.columns
        assert 'bb_upper' in data.columns
        assert 'stoch_k' in data.columns
        assert 'atr' in data.columns
    
    @pytest.mark.fast
    def test_signal_generation(self, sample_price_data):
        """Test signal generation."""
        indicators = SwingTradingIndicators(sample_price_data)
        signals = indicators.get_signals_df()
        
        assert 'signal' in signals.columns
        assert 'confidence' in signals.columns
        assert 'bullish_strength' in signals.columns
        assert 'bearish_strength' in signals.columns
        
        # Check signal values are valid
        assert signals['signal'].isin([-1, 0, 1]).all()
        assert (signals['confidence'] >= 0).all()
        assert (signals['confidence'] <= 1).all()
    
    @pytest.mark.fast
    def test_get_current_signals(self, sample_price_data):
        """Test getting current signals."""
        indicators = SwingTradingIndicators(sample_price_data)
        current = indicators.get_current_signals()
        
        assert 'signal' in current
        assert 'confidence' in current
        assert 'price' in current
        assert current['signal'] in ['BULLISH', 'BEARISH', 'NEUTRAL']
    
    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        incomplete_data = pd.DataFrame({'close': [100, 101, 102]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            SwingTradingIndicators(incomplete_data)
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        empty_data = pd.DataFrame()
        
        # Empty DataFrame will fail on missing columns check first
        with pytest.raises(ValueError, match="Missing required columns|DataFrame is empty"):
            SwingTradingIndicators(empty_data)
    
    @pytest.mark.fast
    def test_rsi_calculation(self, sample_price_data):
        """Test RSI calculation doesn't have division by zero."""
        indicators = SwingTradingIndicators(sample_price_data)
        data = indicators.get_indicators_df()
        
        # RSI should be between 0 and 100
        rsi_values = data['rsi'].dropna()
        if len(rsi_values) > 0:
            assert (rsi_values >= 0).all()
            assert (rsi_values <= 100).all()
            assert not rsi_values.isna().any()



class TestSwingTradingIndicatorParameters:
    """Keyword thresholds added for walk-forward evaluation; the defaults must reproduce the shipped rules."""

    @pytest.mark.fast
    def test_defaults_match_explicit_defaults(self, sample_price_data):
        a = SwingTradingIndicators(sample_price_data).get_full_analysis()
        b = SwingTradingIndicators(
            sample_price_data, rsi_oversold=35.0, rsi_overbought=65.0, bb_period=20, bb_std_dev=2.0
        ).get_full_analysis()
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.fast
    def test_rsi_threshold_changes_flags_monotonically(self, sample_price_data):
        loose = SwingTradingIndicators(sample_price_data, rsi_oversold=50.0).get_signals_df()
        default = SwingTradingIndicators(sample_price_data).get_signals_df()
        strict = SwingTradingIndicators(sample_price_data, rsi_oversold=20.0).get_signals_df()
        assert loose['rsi_oversold'].sum() >= default['rsi_oversold'].sum() >= strict['rsi_oversold'].sum()
        assert loose['rsi_oversold'].sum() > strict['rsi_oversold'].sum()

    @pytest.mark.fast
    def test_bollinger_width_scales_with_std_dev(self, sample_price_data):
        narrow = SwingTradingIndicators(sample_price_data, bb_std_dev=1.0).get_indicators_df()
        wide = SwingTradingIndicators(sample_price_data, bb_std_dev=2.0).get_indicators_df()
        # width = 2 * k * std / middle, so doubling k doubles the width; the middle band is unchanged
        pd.testing.assert_series_equal(wide['bb_width'], 2.0 * narrow['bb_width'], check_names=False, rtol=1e-9)
        pd.testing.assert_series_equal(wide['bb_middle'], narrow['bb_middle'])

    @pytest.mark.fast
    def test_invalid_parameters_raise(self, sample_price_data):
        with pytest.raises(ValueError):
            SwingTradingIndicators(sample_price_data, rsi_oversold=70.0, rsi_overbought=65.0)
        with pytest.raises(ValueError):
            SwingTradingIndicators(sample_price_data, bb_period=1)
        with pytest.raises(ValueError):
            SwingTradingIndicators(sample_price_data, bb_std_dev=0.0)
