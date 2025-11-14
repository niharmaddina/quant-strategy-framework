# ONI Strategy M1

Mark 1 of ONI Capital's first quantitative trading strategy.

## Project Structure

This repository follows a modular, production-ready architecture designed for quantitative trading systems. The structure separates concerns between runnable applications, shared libraries, strategy implementations, research, and infrastructure.

```
├─ apps/                    # Top-level runnable services & CLIs
│  ├─ research-cli/         # CLI tools for researchers (data pulls, backtests)
│  ├─ backtester/           # Backtest engine service
│  ├─ live-trading/        # Live trading system
│  ├─ risk-engine/          # Risk + limits engine
│  └─ reporting/            # Daily/weekly PnL & risk reports
│
├─ libs/                    # Shared libraries (versioned, imported by apps)
│  ├─ core/                 # Core domain models: Orders, Positions, Instruments
│  ├─ data/                 # Data access layer: vendors, caching, ETL helpers
│  ├─ factors/              # Factor definitions, signals, features
│  ├─ execution/            # Execution logic, smart order routing, slippage models
│  ├─ risk/                 # Risk models, VaR, limits, exposure calcs
│  ├─ portfolio/            # Portfolio construction, optimization, rebalancing
│  └─ utils/                # Logging, config, common helpers
│
├─ strategies/              # Strategy-specific code (thin wrappers around libs)
│  ├─ stat_arb_equities/    # Statistical arbitrage strategy for equities
│  ├─ intraday_momentum/    # Intraday momentum strategy
│  └─ options_vol_arb/      # Options volatility arbitrage strategy
│
├─ research/                # Notebooks and experiments
│  ├─ notebooks/
│  │  ├─ data_exploration/  # Data exploration notebooks
│  │  ├─ feature_research/  # Feature research notebooks
│  │  └─ strategy_prototypes/ # Strategy prototype notebooks
│  └─ experiments/
│     ├─ exp_001_xxx/       # Individual experiment directories
│     ├─ exp_002_yyy/
│     └─ registry.json      # Simple index of experiments + results
│
├─ data_pipelines/          # Ingestion & ETL
│  ├─ dags/                 # Airflow / Prefect flows
│  ├─ jobs/                 # Standalone ETL scripts
│  └─ schemas/              # Data schemas, validation rules
│
├─ infra/                   # Infrastructure as code
│  ├─ docker/               # Dockerfiles for each app
│  ├─ k8s/                  # Kubernetes manifests/helm charts
│  ├─ terraform/            # Cloud infra (VPC, clusters, DBs)
│  └─ monitoring/           # Prometheus/Grafana configs, alerts
│
├─ config/                  # Central config, non-secret
│  ├─ environments/
│  │  ├─ dev.yaml           # Development environment config
│  │  ├─ staging.yaml       # Staging environment config
│  │  └─ prod.yaml          # Production environment config
│  ├─ logging.yaml           # Logging configuration
│  └─ risk_limits.yaml      # Risk limits configuration
│
├─ scripts/                 # One-off maintenance/dev scripts
│  ├─ sync_data.sh          # Data synchronization script
│  ├─ run_backtest_local.sh # Local backtest runner
│  └─ update_env.sh         # Environment update script
│
├─ tests/                   # Top-level test runner or cross-cutting tests
│  ├─ unit/                 # Unit tests
│  ├─ integration/          # Integration tests
│  └─ regression/           # Regression tests
│
├─ docs/                    # Documentation
│  ├─ architecture/         # Architecture documentation
│  ├─ runbooks/             # Operational runbooks (e.g., "How to restart live-trading")
│  └─ strategies/           # Strategy methodology docs & disclaimers
│
├─ .github/
│  ├─ workflows/            # CI: lint, tests, build, deploy
│  └─ ISSUE_TEMPLATE/       # Issue templates
│
├─ pyproject.toml           # Python project configuration
├─ Makefile                 # Common dev commands
├─ README.md                # This file
└─ LICENSE                  # License file
```

## Directory Descriptions

### `apps/`
Contains all runnable applications and services. Each app is a standalone service that can be deployed independently:
- **research-cli**: Command-line tools for researchers to pull data, run backtests, and perform analysis
- **backtester**: Backtesting engine service for historical strategy evaluation
- **live-trading**: Production live trading system
- **risk-engine**: Real-time risk and limits monitoring engine
- **reporting**: Automated daily/weekly PnL and risk reporting service

### `libs/`
Shared libraries that are imported by applications. These are versioned and provide reusable functionality:
- **core**: Core domain models (Orders, Positions, Instruments) - the foundation of the trading system
- **data**: Data access layer including vendor integrations, caching, and ETL helpers
- **factors**: Factor definitions, signal generation, and feature engineering
- **execution**: Execution logic, smart order routing, and slippage models
- **risk**: Risk models, VaR calculations, limits, and exposure calculations
- **portfolio**: Portfolio construction, optimization, and rebalancing logic
- **utils**: Common utilities including logging, configuration management, and helper functions

### `strategies/`
Strategy-specific implementations. These should be thin wrappers that compose functionality from `libs/`:
- Each strategy directory contains the specific logic, parameters, and configuration for that strategy
- Strategies should be modular and testable independently

### `research/`
Research and experimentation workspace:
- **notebooks/**: Jupyter notebooks organized by purpose (data exploration, feature research, prototypes)
- **experiments/**: Structured experiment directories with results and metadata tracked in `registry.json`

### `data_pipelines/`
Data ingestion and ETL infrastructure:
- **dags/**: Workflow definitions for Airflow/Prefect
- **jobs/**: Standalone ETL scripts
- **schemas/**: Data schemas and validation rules

### `infra/`
Infrastructure as code:
- **docker/**: Dockerfiles for containerizing applications
- **k8s/**: Kubernetes manifests and Helm charts
- **terraform/**: Cloud infrastructure definitions (VPC, clusters, databases)
- **monitoring/**: Prometheus/Grafana configurations and alert definitions

### `config/`
Centralized configuration files (non-secret):
- Environment-specific configurations
- Logging settings
- Risk limits and parameters

### `scripts/`
Utility scripts for development and maintenance tasks

### `tests/`
Test suite organized by test type:
- **unit/**: Fast, isolated unit tests
- **integration/**: Integration tests for component interactions
- **regression/**: Regression tests to prevent regressions

### `docs/`
Documentation:
- **architecture/**: System architecture and design documents
- **runbooks/**: Operational procedures and troubleshooting guides
- **strategies/**: Strategy methodology documentation and disclaimers

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment:
   - Copy and modify config files in `config/environments/` as needed

3. Run tests:
   ```bash
   make test
   ```

## Development

See `Makefile` for common development commands.

## License

See `LICENSE` file for details.
