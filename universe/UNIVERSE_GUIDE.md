# Universe Guide (Current)

This repo supports two distinct “universe” workflows:

- **Frozen / manual universes (recommended for backtests)**: avoids selection/look-ahead bias from using today’s metadata to choose tickers.
- **Dynamic / metadata-based universes (optional)**: useful for live/forward testing, but can bias historical backtests.

## Recommended: frozen universe for backtests

### Why

If you select tickers using *current* market cap, profitability, “quality”, etc., your backtest can inherit **survivorship/selection bias**. A frozen list doesn’t fully eliminate this (true point-in-time constituents require external data), but it avoids the common “today decided what we traded in 2021” failure mode.

### Built-in frozen universes

- `universe/universe_mean_reversion_manual_50.txt` (50 symbols)
- `universe/universe_mean_reversion_manual.txt` (smaller starter set)

### CLI usage

```bash
python main.py --universe --start 2021-01-02 --end 2026-01-02 --no-plot
```

Or explicitly:

```bash
python main.py --universe-file universe/universe_mean_reversion_manual_50.txt --start 2021-01-02 --end 2026-01-02 --no-plot
```

### File format

Universe files can be plain text with one ticker per line. Comments are allowed:

```text
# my universe
SPY
AAPL  # inline comments are ok
```

## Optional: manual sector-balanced universe (no metadata)

If you want a deterministic “few per sector” list without any API calls:

```python
from universe.constructor import UniverseConstructor

u = UniverseConstructor()
symbols = u.create_manual_sector_balanced_universe(
    target_size=30,
    stocks_per_sector=2,
    include_index_etfs=True,
)
```

## Optional: dynamic / metadata-based universes (use carefully)

`UniverseConstructor` still contains:

- Liquidity / volatility / quality filters (via `yfinance` metadata)
- Correlation de-duplication
- Scoring and ranking

These can be useful for **live/forward** workflows, but they can bias historical backtests because they use “current” fundamentals/metadata.

## Sector mapping

`universe/constructor.py` contains a curated `SECTOR_MAPPING` used for:
- sector lookup (`get_symbol_sector`)
- deterministic sector-balanced selection (manual mode)


