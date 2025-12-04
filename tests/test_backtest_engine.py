"""
Fast unit tests for BacktestEngine with mock data.
"""

import pytest
import pandas as pd
import numpy as np

# Import only what we need for tests (avoid visualizer import)
from backtest.engine import BacktestEngine


class TestBacktestEngine:
    """Test suite for BacktestEngine."""
    
    @pytest.mark.fast
    def test_init_with_valid_data(self, sample_price_data, sample_signals):
        """Test BacktestEngine initialization with valid data."""
        engine = BacktestEngine(
            data=sample_price_data,
            signals=sample_signals,
            initial_capital=100000,
            position_size=0.1,
            commission=0.001
        )
        
        assert engine.initial_capital == 100000
        assert engine.position_size == 0.1
        assert engine.commission == 0.001
    
    @pytest.mark.fast
    def test_run_backtest(self, sample_price_data, sample_signals):
        """Test running a backtest (fast with mock data)."""
        engine = BacktestEngine(
            data=sample_price_data,
            signals=sample_signals,
            initial_capital=100000,
            position_size=0.1
        )
        
        results = engine.run()
        
        assert isinstance(results, pd.DataFrame)
        assert 'equity' in results.columns
        assert 'returns' in results.columns
        assert len(results) > 0
    
    @pytest.mark.fast
    def test_risk_metrics_calculation(self, sample_price_data, sample_signals):
        """Test that new risk metrics are calculated."""
        engine = BacktestEngine(
            data=sample_price_data,
            signals=sample_signals,
            initial_capital=100000
        )
        
        engine.run()
        metrics = engine.get_metrics()
        
        # Check new metrics exist
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
    
    @pytest.mark.fast
    def test_export_functionality(self, sample_price_data, sample_signals, tmp_path):
        """Test export methods."""
        engine = BacktestEngine(
            data=sample_price_data,
            signals=sample_signals,
            initial_capital=100000
        )
        
        engine.run()
        
        # Test CSV export
        results_file = tmp_path / "test_results.csv"
        engine.export_results_to_csv(str(results_file))
        assert results_file.exists()
        
        # Test JSON export
        metrics_file = tmp_path / "test_metrics.json"
        engine.export_metrics_to_json(str(metrics_file))
        assert metrics_file.exists()
        
        # Test trades export
        trades = engine.get_trades()
        if not trades.empty:
            trades_file = tmp_path / "test_trades.csv"
            engine.export_trades_to_csv(str(trades_file))
            assert trades_file.exists()
    
    def test_empty_data_raises_error(self, sample_signals):
        """Test that empty data raises ValueError."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Data DataFrame is empty"):
            BacktestEngine(empty_data, sample_signals)
    
    def test_invalid_capital_raises_error(self, sample_price_data, sample_signals):
        """Test that invalid capital raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            BacktestEngine(sample_price_data, sample_signals, initial_capital=-1000)
    
    @pytest.mark.fast
    def test_sortino_calculation(self, sample_price_data, sample_signals):
        """Test Sortino ratio calculation."""
        engine = BacktestEngine(
            data=sample_price_data,
            signals=sample_signals,
            initial_capital=100000
        )
        
        engine.run()
        
        # Create test returns
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])
        sortino = engine._calculate_sortino(returns)
        
        assert isinstance(sortino, (int, float))
        assert not np.isnan(sortino)
        assert not np.isinf(sortino)
    
    @pytest.mark.fast
    def test_calmar_calculation(self, sample_price_data, sample_signals):
        """Test Calmar ratio calculation."""
        engine = BacktestEngine(
            data=sample_price_data,
            signals=sample_signals,
            initial_capital=100000
        )
        
        calmar = engine._calculate_calmar(10.0, 5.0)  # 10% return, 5% drawdown
        assert calmar == 2.0  # 10/5 = 2
        
        calmar_zero = engine._calculate_calmar(10.0, 0.0)
        assert calmar_zero == 0.0

