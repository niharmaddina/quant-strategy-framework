# Roadmap (Short)

This file tracks *remaining* improvements. Anything already implemented should not live here.

## Data / reliability

- Add optional local caching of downloaded OHLCV (disk cache keyed by symbol/date/interval)
- Add a “cooldown + retry budget” that is consistent across all yfinance entrypoints

## Backtesting realism

- Add optional slippage model (e.g., bps + spread)
- Add explicit “mark-to-market” and multi-position portfolio support (currently single-position per symbol)

## Universe tooling

- Add a simple “point-in-time universe” loader interface (requires external data source)

## Documentation

- Keep docs short and aligned with `main.py` behavior and strategy modules.

## ML (follow-up clarity)

- Clarify ML logic/assumptions later:
  - Labeling rules + target definition (horizon/threshold)
  - Leakage checks + split strategy (time-based vs random; per-symbol constraints)
  - Intended workflow (training/validation) and how results should be interpreted
  - How/if ML signals are meant to integrate with the backtest engine


