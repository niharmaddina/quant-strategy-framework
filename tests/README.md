# Testing Guide

## Quick Start

### Run Fast Tests (No Network Required)
```bash
# Run all fast tests
pytest tests/ -m fast

# Or use the test runner
python tests/run_tests.py --fast
```

### Run All Tests
```bash
# Run all tests including network tests
pytest tests/

# Or
python tests/run_tests.py --all
```

### Run Specific Test File
```bash
pytest tests/test_backtest_engine.py -v
```

## Test Structure

- **Fast Tests** (`-m fast`): Use mock data, no network calls, run in < 1 second
- **Integration Tests** (`-m integration`): Test full pipeline with mocks
- **Network Tests**: Require real yfinance API (use sparingly)

## Using Mock Data

The test framework automatically uses mock data by default. To use real data in tests:

```python
# In your test
def test_with_real_data():
    fetcher = DataFetcher(use_mock=False)  # Use real API
    data = fetcher.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
```

## Test Configuration

Tests use mock data by default for speed. To configure:

1. **Use Mock Data (Default)**: Fast, no network
   ```python
   fetcher = DataFetcher(use_mock=True)
   ```

2. **Use Real Data**: Requires network
   ```python
   fetcher = DataFetcher(use_mock=False)
   ```

3. **Custom Mock Data**: Provide your own data generator
   ```python
   def my_mock_data(symbol, start, end):
       # Return custom DataFrame
       return pd.DataFrame(...)
   
   fetcher = DataFetcher(use_mock=True, mock_data_func=my_mock_data)
   ```

## Running Tests

```bash
# Fast tests only (recommended for development)
pytest tests/ -m fast -v

# All tests
pytest tests/ -v

# Specific test
pytest tests/test_backtest_engine.py::TestBacktestEngine::test_run_backtest -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Markers

- `@pytest.mark.fast` - Fast test, uses mocks
- `@pytest.mark.slow` - Slow test, may use network
- `@pytest.mark.integration` - Integration test
- `@pytest.mark.network` - Requires network

