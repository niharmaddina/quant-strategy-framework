"""
Pytest configuration and shared fixtures for fast testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_price_data():
    """Generate realistic sample price data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 252 trading days (1 year)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    dates = dates[:252]  # Exactly 252 trading days
    
    # Generate realistic price movement
    n = len(dates)
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Geometric random walk
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(n) * 0.5,
        'high': prices + abs(np.random.randn(n) * 1.5),
        'low': prices - abs(np.random.randn(n) * 1.5),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n)
    })
    
    # Ensure high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    data['open'] = data['open'].clip(lower=data['low'], upper=data['high'])
    
    return data


@pytest.fixture
def sample_signals(sample_price_data):
    """Generate sample trading signals."""
    signals = pd.DataFrame(index=sample_price_data.index)
    signals['signal'] = 0
    
    # Add some realistic signals (every ~20 days)
    signal_indices = list(range(20, len(signals), 20))
    for i, idx in enumerate(signal_indices[:10]):
        signals.loc[signals.index[idx], 'signal'] = 1 if i % 2 == 0 else -1
    
    return signals


@pytest.fixture
def mock_data_fetcher(monkeypatch):
    """Create a mock DataFetcher that returns sample data."""
    from data.fetcher import DataFetcher
    
    class MockDataFetcher(DataFetcher):
        def __init__(self):
            super().__init__()
            self._mock_data = None
        
        def set_mock_data(self, data):
            """Set mock data to return."""
            self._mock_data = data
        
        def fetch_stock_data(self, symbol, start_date, end_date, interval="1d"):
            """Return mock data instead of fetching from API."""
            # Validate inputs first (same as parent class)
            if not symbol or not isinstance(symbol, str):
                raise ValueError(f"Invalid symbol: {symbol}")
            
            if self._mock_data is not None:
                return self._mock_data.copy()
            # Generate default mock data
            return self._generate_default_mock_data(symbol, start_date, end_date)
        
        def _generate_default_mock_data(self, symbol, start_date, end_date):
            """Generate default mock data."""
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = dates[dates.weekday < 5][:252]  # Trading days only
            
            np.random.seed(hash(symbol) % 1000)  # Different seed per symbol
            n = len(dates)
            returns = np.random.randn(n) * 0.02
            prices = 100 * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'date': dates,
                'open': prices + np.random.randn(n) * 0.5,
                'high': prices + abs(np.random.randn(n) * 1.5),
                'low': prices - abs(np.random.randn(n) * 1.5),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n)
            })
            
            data['high'] = data[['high', 'close']].max(axis=1)
            data['low'] = data[['low', 'close']].min(axis=1)
            
            # Standardize column names (matching real fetcher behavior)
            data.index.name = 'date'
            return data
    
    return MockDataFetcher()


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration file."""
    import yaml
    
    config_file = tmp_path / "test_config.yaml"
    config = {
        'logging': {
            'level': 'DEBUG',
            'log_dir': str(tmp_path / 'logs'),
            'log_file': 'test.log',
            'console_output': False
        },
        'data': {
            'cache_dir': str(tmp_path / 'test_cache'),
            'cache_enabled': False,  # Disable cache for tests
            'cache_expiry_hours': 24,
            'rate_limit_delay': 0.0,  # No delay in tests
            'retry_attempts': 1,  # Fast failures
            'retry_delay': 0.0
        },
        'backtest': {
            'default_initial_capital': 100000,
            'default_position_size': 0.1,
            'default_commission': 0.001,
            'risk_free_rate': 0.02
        },
        'strategy': {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sma_short': 20,
            'sma_medium': 50,
            'sma_long': 200,
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'atr_period': 14,
            'signal_threshold': 3,
            'min_confidence': 0.6
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


@pytest.fixture
def sample_intraday_5m_ohlcv():
    """
    Deterministic 5-minute OHLCV data for ML/mean-reversion tests.
    Keeps ML tests fast and avoids repeating fixture logic across files.
    """
    dates = pd.date_range(start='2024-01-01 09:30', end='2024-01-05 16:00', freq='5T')
    np.random.seed(42)

    base_price = 100
    n = len(dates)
    returns = np.random.randn(n) * 0.001
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(n) * 0.1,
        'high': prices + abs(np.random.randn(n)) * 0.3,
        'low': prices - abs(np.random.randn(n)) * 0.3,
        'close': prices,
        'volume': np.random.randint(100000, 500000, n)
    })

    # Ensure high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    data['open'] = data['open'].clip(lower=data['low'], upper=data['high'])

    return data

