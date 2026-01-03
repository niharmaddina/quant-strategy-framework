## Summary

- Consolidate current repository state into a single PR.

## What changed

- [ ] Backtesting engine + visualization
- [ ] Data fetching + batching guidance
- [ ] Strategy modules (swing + mean reversion)
- [ ] Universe tooling + frozen universe files
- [ ] ML utilities (features/labeling/model/preprocessing)
- [ ] Examples + docs + tests

## How to test

```bash
python tests/run_all_tests.py
```

## Notes / follow-ups

- **TODO (follow-up)**: Clarify ML logic/assumptions later (labeling rules, target definition, leakage checks, train/val split contract, and how ML signals are intended to integrate with the backtest engine).


