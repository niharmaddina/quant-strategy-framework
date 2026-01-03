# Test Suite

Test suite for the quant strategy framework.

## Test Organization

### Core Test Files

- **`test_backtest_engine_comprehensive.py`** - Backtest engine tests
  - Initialization and validation
  - Trade execution
  - Stop loss and trailing stop
  - Performance metrics
  - Export functionality
  - Edge cases

- **`test_universe_comprehensive.py`** - Universe constructor tests
  - Sector classification
  - Metadata fetching
  - Filtering methods
  - Correlation analysis
  - Universe building
  - Scoring and ranking

- **`test_data_fetcher.py`** - Data fetcher unit tests
- **`test_indicators.py`** - Strategy indicators unit tests
- **`test_integration.py`** - End-to-end integration tests
- **`test_utils.py`** - Utility validation tests
- **`test_universe_simple.py`** - Quick universe smoke test (no API calls, script-style)

## Running Tests

### Run All Tests (Recommended)

```bash
# Run all fast tests (default)
python tests/run_all_tests.py

# Run all tests including slow ones
python tests/run_all_tests.py --all

# Run with coverage report
python tests/run_all_tests.py --coverage

# Run only simple universe tests (no API calls)
python tests/run_all_tests.py --simple
```

### Run Specific Test Files

```bash
# Run specific test file
pytest tests/test_backtest_engine_comprehensive.py -v

# Run specific test class
pytest tests/test_universe_comprehensive.py::TestSectorClassification -v

# Run specific test
pytest tests/test_backtest_engine_comprehensive.py::TestPerformanceMetrics::test_sharpe_ratio_calculation -v
```

### Run with Pytest Directly

```bash
# Run all fast tests
pytest tests/ -m fast -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.fast` - Fast tests using mocks (default)
- `@pytest.mark.slow` - Slow tests that may take longer
- `@pytest.mark.network` - Tests requiring network access
- `@pytest.mark.integration` - Integration tests

## Test Configuration

- **`conftest.py`** - Shared pytest fixtures
- **`pytest.ini`** - Pytest configuration
- **`run_all_tests.py`** - Unified test runner

## Notes

- Prefer running `-m fast` locally (no-network / deterministic).
- Network-dependent tests should be explicitly marked and run intentionally.
