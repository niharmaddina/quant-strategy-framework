# PR: Initial framework drop (backtest + mean reversion + universe + ML utilities)

## Summary

This PR adds the current state of the quant strategy research framework, including backtesting, indicators/mean reversion, universe tooling, ML utilities, docs, and tests.

## Key components

- Backtesting: `backtest/`
- Data fetching: `data/`
- Strategies/indicators: `strategy/`
- Universe tooling + frozen universes: `universe/`
- ML utilities (optional): `ml/`
- Examples: `examples/`
- Docs: `docs/`
- Tests: `tests/`

## How to test

```bash
python tests/run_all_tests.py
```

## Follow-ups (explicit)

- **TODO (follow-up)**: Clarify ML logic/assumptions later:
  - Labeling rules + target definition (what horizon, what “positive” means)
  - Leakage checks and how splits are constructed
  - Intended training/validation workflow
  - How ML signals are meant to plug into backtests (what’s executed, what’s offline-only)


