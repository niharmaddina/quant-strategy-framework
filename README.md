# Quant Strategy Framework

Backtesting, mean-reversion signals and frozen-universe tooling for swing-trading research in
Python. Built by Nihar Maddina and Tushar Gona (2025-2026); this is the maintained public home,
the original development repository was `tgona/oni_strat_m1`.

Status (October 2026): the fast test suite passes (147 tests, no network), `main.py` runs end to end
with the legacy engine on current `yfinance` (1.7), and the VectorBT engine is available behind
`--engine vectorbt` for anyone with VectorBT installed. Research tooling, not production trading.

This repo is a small research framework for:
- **Backtesting**: single-asset backtests with basic risk controls + metrics
- **Signals**: swing-style indicator signals and a dedicated mean-reversion indicator set
- **Universe management**: use a **frozen/manual universe** for backtests (minimize selection/survivorship-style bias)

## Quick start

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### VectorBT engine (optional but recommended)

VectorBT is best installed into a **dedicated virtual environment**, because it pins
older `numba/numpy` versions that can conflict with stacks like TensorFlow/JAX.

Install into your current venv (simple):

```bash
pip install -U vectorbt
```

Optional extras:

```bash
pip install -U "vectorbt[full]"
```

Or install a dedicated VectorBT venv for this repo (recommended):

```bash
python -m venv .venv_vectorbt
source .venv_vectorbt/bin/activate
pip install -r requirements-vectorbt.txt
```

### Run a backtest (single or multiple symbols)

```bash
python main.py --engine vectorbt --symbols AAPL MSFT --start 2021-01-02 --end 2026-01-02 --no-plot
```

Force legacy engine:

```bash
python main.py --engine legacy --symbols AAPL MSFT --start 2021-01-02 --end 2026-01-02 --no-plot
```

### Run using the frozen 50-symbol universe (recommended for backtests)

`--universe` now loads the manual/frozen universe file `universe/universe_mean_reversion_manual_50.txt` (no “optimized universe” metadata selection).

```bash
python main.py --universe --start 2021-01-02 --end 2026-01-02 --no-plot
```

To use a different frozen list:

```bash
python main.py --universe-file universe/universe_mean_reversion_manual.txt --start 2018-01-01 --end 2024-01-01 --no-plot
```

## Strategy modules

- **Swing indicators**: `strategy/indicators.py` (`SwingTradingIndicators`)
- **Mean reversion (granular)**: `strategy/mean_reversion.py` (`MeanReversionIndicators`)
  - Example: `examples/mean_reversion_example.py`

## Documentation

- `docs/MEAN_REVERSION_STRATEGY.md`
- `docs/API_BATCHING_GUIDE.md`
- `docs/CHART_ENGINE_GUIDE.md`
- `universe/UNIVERSE_GUIDE.md`

## Tests

Run the fast (no-network) suite:

```bash
python tests/run_all_tests.py
```

## Clear local state

To wipe generated artifacts (plots/logs/cache/pyc) without touching git history:

```bash
python -m utils.clean_local_state
```

## Notes / limitations

- Market data is from **yfinance** and can be rate-limited; prefer batching/caching and avoid “one call per symbol” loops when possible.
- This is research tooling, not production trading infrastructure.

## License

MIT, see `LICENSE`.

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.
