# Roadmap (Short)

This file tracks *remaining* improvements. Anything already implemented should not live here,
except for the short "Done" list below, kept so the remaining items have context.

## Done (October 2026)

- Lint and CI: `pyproject.toml` (ruff, E/F/I, 120 columns), `.github/workflows/ci.yml` (ruff + fast
  pytest on Python 3.11 and 3.12 for pushes and pull requests), `Makefile` targets for test/lint/example.
- Reproducible worked example: `examples/frozen_universe_backtest.py` runs the legacy engine over the
  frozen 50-symbol universe for 2021-01-04 to 2024-12-31, with a per-symbol buy-and-hold baseline, and
  writes `results/frozen50_2021_2024.csv` and `results/summary.md` (reported in the README "Results"
  section). Downloaded OHLCV is cached per symbol as CSV under `data_cache/` so re-runs are offline.

## Data / reliability

- Move the per-symbol CSV cache used by `examples/frozen_universe_backtest.py` into `DataFetcher`
  (keyed by symbol/date range/interval) so `main.py` and the other examples benefit too
- Add a "cooldown + retry budget" that is consistent across all yfinance entrypoints

## Backtesting realism

- Add optional slippage model (e.g., bps + spread)
- Add explicit "mark-to-market" and multi-position portfolio support (currently single-position per symbol)
- Report risk metrics that make sense for a mostly-idle account: the engine's Sharpe subtracts a 2%
  risk-free rate from near-zero account returns, which produces large negative values for a strategy
  that is rarely invested (see `results/summary.md`); an exposure-adjusted or invested-period variant
  would be more informative

## Evaluation protocol

- Add a held-out window or walk-forward split to the worked example; the committed results are
  in-sample only

## Universe tooling

- Add a simple "point-in-time universe" loader interface (requires external data source)

## Documentation

- Keep docs short and aligned with `main.py` behavior and strategy modules.

## ML (follow-up clarity)

- Clarify ML logic/assumptions later:
  - Labeling rules + target definition (horizon/threshold)
  - Leakage checks + split strategy (time-based vs random; per-symbol constraints)
  - Intended workflow (training/validation) and how results should be interpreted
  - How/if ML signals are meant to integrate with the backtest engine
