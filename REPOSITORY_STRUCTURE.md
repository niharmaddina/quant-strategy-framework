# Repository Structure

This document describes the organized structure of the quant strategy framework repository.

## Directory Structure

```
quant-strategy-framework/
├── backtest/              # Backtesting engine and visualization
│   ├── __init__.py
│   ├── engine.py          # Core backtesting engine
│   └── visualizer.py      # Visualization tools
│
├── data/                  # Data fetching and management
│   ├── __init__.py
│   └── fetcher.py         # Data fetcher (yfinance wrapper)
│
├── docs/                  # Project documentation (keep short + current)
│   ├── README.md           # Docs index
│   ├── API_BATCHING_GUIDE.md
│   ├── CHART_ENGINE_GUIDE.md
│   ├── ROADMAP.md
│   ├── MEAN_REVERSION_STRATEGY.md
│   └── ML_REAL_DATA_GUIDE.md
│
├── examples/              # Example scripts and demos
│   ├── example_usage.py   # Basic usage examples
│   ├── strategy_comparison.py  # Strategy vs buy-and-hold comparison
│   └── universe_demo.py   # Universe construction demo
│
├── ml/                    # ML feature/model utilities (optional)
│   ├── __init__.py
│   ├── features.py
│   ├── labeling.py
│   ├── model.py
│   └── preprocessing.py
│
├── logs/                  # Log files (gitignored)
│   └── strategy.log
│
├── strategy/              # Trading strategy implementation
│   ├── __init__.py
│   ├── indicators.py      # Swing-style technical indicators and signals
│   └── mean_reversion.py  # Mean reversion indicators (supports intraday)
│
├── tests/                 # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py        # Shared pytest fixtures
│   ├── pytest.ini         # Pytest configuration
│   ├── run_all_tests.py   # Unified test runner (USE THIS)
│   ├── README.md          # Test documentation
│   │
│   └── test_*.py          # Test files
│       ├── test_backtest_engine_comprehensive.py  # 52 tests
│       ├── test_universe_comprehensive.py         # 39 tests
│       ├── test_data_fetcher.py
│       ├── test_indicators.py
│       ├── test_integration.py
│       ├── test_universe_simple.py
│       ├── test_utils.py
│       └── ...
│
├── universe/              # Universe construction
│   ├── __init__.py
│   ├── constructor.py     # Universe constructor class
│   ├── UNIVERSE_GUIDE.md  # Universe guide
│   ├── universe_simple.txt
│   └── universe_with_metadata.csv
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── validation.py      # Input validation utilities
│
├── main.py                # Main entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── README.md             # Main documentation
└── .gitignore            # Git ignore rules
```

## File Organization

### Core Modules
- **`backtest/`** - Backtesting engine with risk management, performance metrics, and visualization
- **`data/`** - Data fetching from Yahoo Finance with caching and error handling
- **`strategy/`** - Trading strategy with technical indicators and signal generation
- **`universe/`** - Universe construction with filtering, sector balancing, and correlation analysis
- **`utils/`** - Utility functions for validation and common operations

### Documentation
- **`docs/`** - Project documentation and improvement plans
- **`README.md`** - Main project documentation
- **`tests/README.md`** - Test suite documentation
- **`universe/UNIVERSE_GUIDE.md`** - Universe construction guide

### Examples
- **`examples/`** - Example scripts demonstrating framework usage
  - `example_usage.py` - Basic usage examples
  - `strategy_comparison.py` - Strategy performance comparison
  - `universe_demo.py` - Universe construction demo

### Tests
- **`tests/`** - Comprehensive test suite
  - `run_all_tests.py` - **Unified test runner (use this)**
  - Comprehensive test files with full coverage
  - Test documentation in `tests/docs/`

### Configuration
- **`config.yaml`** - Main configuration file
- **`requirements.txt`** - Python package dependencies
- **`.gitignore`** - Git ignore patterns

## Running Tests

### Unified Test Runner (Recommended)

```bash
# Run all fast tests (default)
python tests/run_all_tests.py

# Run all tests including slow ones
python tests/run_all_tests.py --all

# Run with coverage
python tests/run_all_tests.py --coverage

# Run only simple tests
python tests/run_all_tests.py --simple
```

### Direct Pytest

```bash
# Run all fast tests
pytest tests/ -m fast -v

# Run specific test file
pytest tests/test_backtest_engine_comprehensive.py -v
```

## Key Files

### Entry Points
- **`main.py`** - Main execution script for running strategies
- **`tests/run_all_tests.py`** - Unified test runner

### Core Functionality
- **`backtest/engine.py`** - Backtesting engine (674 lines)
- **`universe/constructor.py`** - Universe constructor (1610 lines)
- **`strategy/indicators.py`** - Technical indicators (518 lines)
- **`data/fetcher.py`** - Data fetching

### Examples
- **`examples/example_usage.py`** - Usage examples
- **`examples/strategy_comparison.py`** - Performance comparison
- **`examples/universe_demo.py`** - Universe demo

## Test Coverage

- **Backtest Engine**: 52 comprehensive tests
- **Universe Constructor**: 39 comprehensive tests
- **Data Fetcher**: Unit tests with mocking
- **Strategy Indicators**: Unit tests
- **Integration**: End-to-end tests
- **Utils**: Validation tests

## Notes

- All files are organized into logical modules
- Example scripts are in `examples/` folder
- Documentation is in `docs/` and module-specific guides (e.g. `universe/UNIVERSE_GUIDE.md`)
- Tests are comprehensive with full coverage
- One unified test runner for all tests

