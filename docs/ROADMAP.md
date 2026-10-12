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
- Risk metrics for a mostly-idle account: `BacktestEngine` now reports `exposure_pct`,
  `avg_holding_days`, `in_market_days`, `in_market_sharpe_ratio` and `in_market_sortino_ratio` next to
  the unchanged account-level ratios. The in-market ratios use the return on capital at risk over the
  bars a position was held and are NaN (not 0) when undefined; the rule and its reasoning are in the
  engine's class docstring. `max_holding_days` is a constructor argument (default 20, as before) and
  `SwingTradingIndicators` takes `rsi_oversold`, `rsi_overbought`, `bb_period`, `bb_std_dev` keywords
  with the shipped values as defaults. The frozen-universe CSV and summary carry the new columns.

## Data / reliability

- Move the per-symbol CSV cache used by `examples/frozen_universe_backtest.py` into `DataFetcher`
  (keyed by symbol/date range/interval) so `main.py` and the other examples benefit too
- Add a "cooldown + retry budget" that is consistent across all yfinance entrypoints

## Backtesting realism

- Add optional slippage model (e.g., bps + spread)
- Add explicit "mark-to-market" and multi-position portfolio support (currently single-position per symbol)
- Let idle cash earn the risk-free rate (or a configurable rate) so that the account-level Sharpe of a
  rarely invested account is not dominated by the -2% excess return of uninvested capital

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
