# Quick Test Guide

## Run Tests in 3 Seconds

```bash
# Fast tests (no network, uses mocks)
pytest tests/ -m fast

# Or use the helper script
python tests/run_tests.py --fast
```

## Test with Real Data (When Needed)

```python
# In your code or test
from data.fetcher import DataFetcher

# Use real yfinance API
fetcher = DataFetcher(use_mock=False)
data = fetcher.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')

# Or use mock data (fast, no network)
fetcher = DataFetcher(use_mock=True)
data = fetcher.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
```

## Easy Parameter Testing

```python
# Test with different parameters
def test_with_params():
    from backtest.engine import BacktestEngine
    
    # Test different capital amounts
    for capital in [50000, 100000, 200000]:
        engine = BacktestEngine(
            data=sample_data,
            signals=sample_signals,
            initial_capital=capital,
            position_size=0.1
        )
        engine.run()
        metrics = engine.get_metrics()
        print(f"Capital {capital}: Return {metrics['total_return_pct']:.2f}%")
```

## Test Markers

- `@pytest.mark.fast` - Runs in < 1 second, uses mocks
- `@pytest.mark.slow` - May take longer
- `@pytest.mark.network` - Requires internet

## Run Specific Tests

```bash
# Single test
pytest tests/test_backtest_engine.py::TestBacktestEngine::test_run_backtest

# All tests in a file
pytest tests/test_backtest_engine.py

# Tests matching pattern
pytest tests/ -k "test_risk"
```

