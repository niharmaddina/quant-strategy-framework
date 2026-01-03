# Quant Strategy Framework (Backtesting + Mean Reversion + Universe Tools)

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

### Run a backtest (single or multiple symbols)

```bash
python main.py --symbols AAPL MSFT --start 2021-01-02 --end 2026-01-02 --no-plot
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

## Notes / limitations

- Market data is from **yfinance** and can be rate-limited; prefer batching/caching and avoid “one call per symbol” loops when possible.
- This is research tooling, not production trading infrastructure.

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.
