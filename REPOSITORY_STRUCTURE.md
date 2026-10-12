# Repository Structure

This document describes the organized structure of the quant strategy framework repository.

## Directory Structure

```
quant-strategy-framework/
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/ci.yml   # ruff + fast pytest on push / pull request (Python 3.11, 3.12)
│
├── backtest/              # Backtesting engines and visualization
│   ├── __init__.py
│   ├── engine.py          # Legacy single-position backtesting engine (default for tests/results)
│   ├── vectorbt_engine.py # Optional VectorBT-backed engine (`--engine vectorbt`)
│   ├── factory.py         # Engine selection
│   ├── chart_engine.py    # Chart engine
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
│   ├── frozen_universe_backtest.py  # Reproducible frozen-universe run -> results/ (see README "Results")
│   ├── plot_frozen_universe_results.py  # Figures for the README from results/ + data_cache/ (offline)
│   ├── walk_forward.py    # In-sample grid -> selection -> out-of-sample evaluation (offline from data_cache/)
│   ├── strategy_comparison.py  # Strategy vs buy-and-hold comparison (single symbol, with chart)
│   ├── universe_demo.py   # Universe construction demo
│   └── ...                # mean-reversion, chart-engine and ML examples
│
├── results/               # Committed outputs of the frozen-universe example
│   ├── frozen50_2021_2024.csv
│   ├── summary.md
│   ├── walk_forward.csv   # per-symbol out-of-sample metrics: default, selected, buy-and-hold
│   ├── walk_forward_grid.csv  # one row per grid configuration, in-sample and out-of-sample aggregates
│   ├── walk_forward.md    # walk-forward report (probe, grid, selection, out-of-sample, caveats)
│   └── figures/           # fig1..fig5 PNGs (150 dpi) + figure_stats.json used by README captions
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
│       ├── test_risk_metrics.py                   # exposure / holding-period / in-market ratio tests
│       ├── test_walk_forward.py                   # walk-forward split / selection smoke tests (synthetic)
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
├── pyproject.toml         # Tool configuration (ruff)
├── Makefile               # make test / lint / fix / example / figures / walk-forward / clean
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # requirements.txt + ruff
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
- **`pyproject.toml`** - ruff configuration (E/F/I rules, 120 columns)
- **`requirements.txt`** / **`requirements-dev.txt`** - Python package dependencies (dev adds ruff)
- **`.github/workflows/ci.yml`** - CI: lint + fast tests
- **`.gitignore`** - Git ignore patterns (`results/*.csv` and `results/*.md` are deliberately tracked)

## Running Tests

### Make targets

```bash
make test   # fast suite, same command CI runs: python -m pytest tests -m "not network and not slow"
make lint   # ruff check .
make fix    # ruff check --fix .
```

### Unified Test Runner

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

