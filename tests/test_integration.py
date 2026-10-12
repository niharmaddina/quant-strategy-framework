"""
Fast integration tests using mock data.
Tests the full pipeline without network calls.
"""

import pytest

from backtest.engine import BacktestEngine
from strategy.indicators import SwingTradingIndicators

# Import visualizer only when needed (with try/except for optional dependency)
try:
    from backtest.visualizer import BacktestVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    BacktestVisualizer = None


@pytest.mark.fast
@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the full trading pipeline."""
    
    def test_full_pipeline_mock(self, mock_data_fetcher, sample_price_data):
        """Test complete pipeline with mock data (fast)."""
        # Set up mock data
        mock_data_fetcher.set_mock_data(sample_price_data)
        
        # Fetch data (uses mock)
        data = mock_data_fetcher.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
        assert len(data) > 0
        
        # Calculate indicators
        indicators = SwingTradingIndicators(data)
        signals = indicators.get_signals_df()
        assert not signals.empty
        
        # Run backtest
        backtest = BacktestEngine(
            data=data,
            signals=signals,
            initial_capital=100000,
            position_size=0.1
        )
        
        results = backtest.run()
        assert not results.empty
        
        # Get metrics
        metrics = backtest.get_metrics()
        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        
        # Test visualization (just check it doesn't crash) - skip if matplotlib not available
        if HAS_VISUALIZER:
            visualizer = BacktestVisualizer(backtest)
            fig = visualizer.plot_all()
            assert fig is not None
    
    @pytest.mark.fast
    def test_multiple_symbols_pipeline(self, mock_data_fetcher):
        """Test pipeline with multiple symbols using mock data."""
        symbols = ['AAPL', 'MSFT']
        
        results = {}
        for symbol in symbols:
            # Fetch (mock) data
            data = mock_data_fetcher.fetch_stock_data(symbol, '2023-01-01', '2023-12-31')
            
            # Calculate indicators
            indicators = SwingTradingIndicators(data)
            signals = indicators.get_signals_df()
            
            # Run backtest
            backtest = BacktestEngine(data=data, signals=signals, initial_capital=100000)
            backtest.run()
            
            results[symbol] = backtest.get_metrics()
        
        assert len(results) == 2
        assert all('total_return_pct' in m for m in results.values())

