# Frozen-universe backtest: frozen50_2021_2024

Run started 2026-10-12 04:43 UTC; wall time 3s.

## Command

```bash
python examples/frozen_universe_backtest.py --universe-file universe/universe_mean_reversion_manual_50.txt --start 2021-01-04 --end 2024-12-31 --capital 100000 --position-size 0.1 --tag frozen50_2021_2024
```

## Setup

- Universe file: `universe/universe_mean_reversion_manual_50.txt` (50 symbols requested, 50 with data)
- Window: 2021-01-04 to 2024-12-31 inclusive, daily bars (Yahoo `Close`; split-adjusted, not dividend-adjusted)
- Engine: `backtest/engine.py` (legacy), one long position per symbol at a time, 10% of current capital per trade, commission 0.10% per side, fixed stop loss 5%, trailing stop off, halt new entries at 20% portfolio drawdown, max holding 20 bars, take-profit at max(3%, 2x stop distance), exit on neutral signal after 5 bars
- Signals: `strategy/indicators.py` `SwingTradingIndicators` (needs a 200-bar SMA, so roughly the first 200 bars of each series produce no signals)
- Initial capital per symbol: 100,000; each symbol is backtested independently
- Buy-and-hold baseline: fully invested in the symbol from the first to the last bar of the same window, same Sharpe/drawdown formulas as the engine (risk-free rate 2%, 252-day annualisation)
- Two Sharpe columns for the strategy: `sharpe_ratio` is on daily account equity over every bar; `in_market_sharpe_ratio` is on the daily return of the capital at risk over the bars a position was held (`in_market_days` of them), see `BacktestEngine._in_market_returns`. Buy-and-hold is always invested, so its single Sharpe is shown in both rows of the aggregate table. `exposure_pct` is the share of bars with a position held at the close; `avg_holding_days` is the mean bars per trade
- Software: Python 3.11.3, pandas 3.0.6, numpy 2.4.6, yfinance 1.7.0

## Aggregate

| metric | strategy mean | strategy median | buy-and-hold mean | buy-and-hold median |
|---|---|---|---|---|
| Total return (%) | -0.19 | -0.21 | 49.84 | 40.74 |
| Sharpe ratio (account) | -4.96 | -4.50 | 0.39 | 0.42 |
| Sharpe ratio (in-market) | -0.49 | -0.50 | 0.39 | 0.42 |
| Max drawdown (%) | 1.06 | 0.90 | 34.77 | 33.81 |
| Exposure (% of bars) | 6.18 | 6.32 | 100.00 | 100.00 |
| Avg holding (bars) | 6.67 | 6.29 |  |  |
| Win rate (%) | 41.50 | 40.00 |  |  |
| Trades per symbol | 9.32 | 10.00 |  |  |

- Symbols where strategy total return > buy-and-hold total return: 7 of 50
- Symbols with zero trades: 0

## Per symbol

Full table with more columns: `results/frozen50_2021_2024.csv`.

| symbol | bars | total_trades | total_return_pct | win_rate_pct | sharpe_ratio | in_market_sharpe_ratio | exposure_pct | avg_holding_days | max_drawdown_pct | bh_total_return_pct | bh_sharpe_ratio | bh_max_drawdown_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY | 1005 | 16 | 0.02 | 62.50 | -6.22 | 0.29 | 9.65 | 6.06 | 0.71 | 58.92 | 0.67 | 25.36 |
| QQQ | 1005 | 13 | 0.62 | 69.23 | -3.73 | 1.20 | 9.25 | 7.15 | 0.57 | 65.28 | 0.59 | 35.62 |
| DIA | 1005 | 4 | 0.38 | 50.00 | -10.83 | 3.30 | 3.08 | 7.75 | 0.19 | 40.74 | 0.54 | 21.90 |
| IWM | 1005 | 8 | -1.44 | 12.50 | -6.51 | -4.91 | 4.08 | 5.12 | 1.48 | 14.19 | 0.17 | 33.13 |
| XLK | 1005 | 15 | 0.56 | 40.00 | -3.34 | 0.99 | 9.15 | 6.13 | 0.78 | 81.80 | 0.66 | 34.02 |
| XLF | 1005 | 15 | -1.01 | 53.33 | -5.64 | -1.98 | 8.76 | 5.87 | 1.60 | 66.20 | 0.66 | 26.87 |
| XLV | 1005 | 5 | -0.43 | 20.00 | -9.03 | -2.51 | 2.89 | 5.80 | 0.77 | 21.80 | 0.28 | 16.06 |
| XLY | 1005 | 10 | 1.28 | 50.00 | -3.10 | 2.19 | 8.46 | 8.50 | 0.50 | 40.73 | 0.40 | 40.28 |
| XLP | 1005 | 4 | -0.16 | 50.00 | -11.14 | -1.12 | 2.79 | 7.00 | 0.43 | 17.82 | 0.23 | 17.81 |
| XLE | 1005 | 11 | -0.28 | 36.36 | -2.70 | -0.19 | 8.26 | 7.55 | 1.61 | 125.66 | 0.81 | 26.86 |
| XLI | 1005 | 10 | 0.11 | 40.00 | -7.32 | 0.62 | 6.47 | 6.50 | 0.25 | 52.48 | 0.60 | 22.67 |
| XLC | 1005 | 15 | -0.85 | 40.00 | -4.36 | -1.18 | 10.75 | 7.20 | 0.88 | 45.62 | 0.45 | 47.23 |
| XLRE | 1005 | 3 | -0.18 | 33.33 | -7.22 | -0.88 | 2.89 | 9.67 | 0.55 | 14.95 | 0.17 | 37.87 |
| XLB | 1005 | 7 | -0.69 | 28.57 | -7.57 | -2.84 | 4.08 | 5.86 | 1.07 | 17.35 | 0.20 | 25.70 |
| XLU | 1005 | 7 | 1.23 | 71.43 | -4.66 | 3.36 | 6.97 | 10.00 | 0.27 | 23.84 | 0.28 | 28.07 |
| AAPL | 1005 | 14 | 0.73 | 57.14 | -2.52 | 0.92 | 10.55 | 7.57 | 0.86 | 93.51 | 0.68 | 31.31 |
| MSFT | 1005 | 12 | -0.56 | 50.00 | -3.10 | -0.55 | 9.55 | 8.00 | 1.40 | 93.62 | 0.69 | 37.56 |
| CSCO | 1005 | 7 | 0.51 | 42.86 | -4.78 | 1.49 | 5.37 | 7.71 | 0.65 | 34.67 | 0.36 | 38.60 |
| ORCL | 1005 | 15 | 0.38 | 33.33 | -2.35 | 0.49 | 8.56 | 5.73 | 1.29 | 161.40 | 0.88 | 41.08 |
| JPM | 1005 | 12 | 1.05 | 58.33 | -3.54 | 1.72 | 10.25 | 8.58 | 0.63 | 90.44 | 0.71 | 40.65 |
| BAC | 1005 | 10 | -0.63 | 40.00 | -4.47 | -1.11 | 7.56 | 7.60 | 1.89 | 46.35 | 0.41 | 49.03 |
| WFC | 1005 | 9 | 0.47 | 55.56 | -3.16 | 0.90 | 6.37 | 7.11 | 0.94 | 136.50 | 0.78 | 38.66 |
| GS | 1005 | 9 | -0.69 | 33.33 | -3.74 | -1.07 | 6.17 | 6.89 | 1.60 | 116.08 | 0.79 | 33.99 |
| JNJ | 1005 | 3 | -0.24 | 33.33 | -7.79 | -1.34 | 2.09 | 7.00 | 0.39 | -7.59 | -0.17 | 22.94 |
| UNH | 1005 | 11 | -1.02 | 54.55 | -3.36 | -1.44 | 6.57 | 6.00 | 2.40 | 44.74 | 0.43 | 22.35 |
| PFE | 1005 | 5 | -0.59 | 20.00 | -3.20 | -1.24 | 2.89 | 5.80 | 1.55 | -27.93 | -0.28 | 59.51 |
| AMZN | 1005 | 14 | -1.44 | 35.71 | -3.12 | -1.53 | 8.26 | 5.93 | 1.63 | 37.69 | 0.35 | 56.15 |
| HD | 1005 | 10 | -1.62 | 20.00 | -5.36 | -3.70 | 5.87 | 5.90 | 2.26 | 47.39 | 0.44 | 35.95 |
| MCD | 1005 | 7 | -0.91 | 42.86 | -6.30 | -2.98 | 4.38 | 6.29 | 0.91 | 37.90 | 0.44 | 18.20 |
| PG | 1005 | 10 | -0.61 | 20.00 | -6.35 | -1.65 | 6.17 | 6.20 | 1.23 | 21.64 | 0.26 | 24.63 |
| KO | 1005 | 7 | 0.00 | 42.86 | -8.17 | 0.24 | 3.98 | 5.71 | 0.48 | 18.01 | 0.22 | 20.89 |
| PEP | 1005 | 11 | -0.87 | 54.55 | -5.65 | -1.96 | 7.06 | 6.45 | 0.96 | 5.40 | 0.04 | 22.86 |
| XOM | 1005 | 11 | -1.53 | 36.36 | -2.65 | -1.60 | 6.87 | 6.27 | 3.13 | 159.20 | 0.92 | 20.51 |
| CVX | 1005 | 10 | 0.44 | 50.00 | -2.72 | 0.77 | 5.67 | 5.70 | 0.82 | 70.98 | 0.58 | 26.68 |
| COP | 1005 | 11 | -0.45 | 54.55 | -2.75 | -0.45 | 6.87 | 6.27 | 1.42 | 150.62 | 0.80 | 33.49 |
| CAT | 1005 | 15 | -1.24 | 33.33 | -3.18 | -1.26 | 10.15 | 6.80 | 1.26 | 99.15 | 0.68 | 33.64 |
| HON | 1005 | 5 | -0.01 | 60.00 | -5.05 | 0.10 | 2.99 | 6.00 | 0.62 | 8.63 | 0.11 | 28.70 |
| UPS | 1005 | 1 | -1.03 | 0.00 | -4.47 | -8.04 | 0.40 | 4.00 | 1.03 | -23.02 | -0.19 | 47.16 |
| GOOGL | 1005 | 14 | -1.99 | 28.57 | -3.14 | -2.15 | 7.76 | 5.50 | 2.09 | 119.33 | 0.73 | 44.32 |
| META | 1005 | 18 | 2.24 | 38.89 | -1.09 | 1.35 | 10.85 | 6.06 | 1.30 | 117.71 | 0.62 | 76.74 |
| VZ | 1005 | 4 | -0.67 | 0.00 | -6.62 | -3.09 | 2.49 | 6.25 | 0.89 | -32.05 | -0.45 | 48.47 |
| AMT | 1005 | 4 | 0.47 | 50.00 | -4.37 | 1.91 | 2.59 | 6.50 | 0.52 | -16.48 | -0.12 | 48.07 |
| PLD | 1005 | 6 | 0.05 | 50.00 | -5.34 | 0.31 | 4.08 | 6.83 | 0.72 | 10.06 | 0.15 | 43.75 |
| O | 1005 | 4 | 0.16 | 75.00 | -9.45 | 1.60 | 2.09 | 5.25 | 0.27 | -7.92 | -0.11 | 38.32 |
| LIN | 1005 | 10 | -1.03 | 40.00 | -5.95 | -2.72 | 6.27 | 6.30 | 1.25 | 61.77 | 0.58 | 23.72 |
| SHW | 1005 | 12 | 1.49 | 66.67 | -2.99 | 2.28 | 9.35 | 7.83 | 1.08 | 40.96 | 0.39 | 42.86 |
| FCX | 1005 | 8 | -0.61 | 25.00 | -3.02 | -0.92 | 4.78 | 6.00 | 2.25 | 40.31 | 0.37 | 51.68 |
| NEE | 1005 | 6 | 0.89 | 50.00 | -4.53 | 2.72 | 4.78 | 8.00 | 0.43 | -3.41 | 0.03 | 47.17 |
| DUK | 1005 | 8 | 0.05 | 25.00 | -5.59 | 0.35 | 4.78 | 6.00 | 0.51 | 20.10 | 0.23 | 25.98 |
| SO | 1005 | 10 | 0.38 | 40.00 | -4.64 | 1.01 | 7.06 | 7.10 | 0.72 | 38.68 | 0.43 | 23.28 |

## What these numbers do and do not show

- They show how this specific rule set behaved on these 50 tickers over this window with the engine's default risk settings. Each symbol is an isolated account; there is no portfolio-level capital sharing.
- Strategy returns are on total account equity while only 10% of capital is at risk per trade, so a like-for-like comparison with a fully invested buy-and-hold is not possible from this table alone. Buy-and-hold is reported as context, not as a benchmark the strategy was designed to beat.
- No slippage, no spread, no borrow or financing, fills at the daily close, 0.1% commission per side.
- The account Sharpe is the engine's formula: (annualised mean daily return - 2% risk-free) / annualised volatility of account equity. An account that sits mostly in cash earns roughly 0%, so its excess return is about -2% divided by a very small volatility; that is why the account Sharpe values are large and negative even where the account did not lose money. Read them as 'rarely invested', not 'large losses'. The in-market Sharpe removes the idle days and the cash fraction and is the column to compare with buy-and-hold's Sharpe; it is estimated from only `in_market_days` observations per symbol (about a tenth of the bars), so it is noisy, and it is NaN for a symbol with fewer than two in-market bars.
- The universe is a hand-picked list of large, long-lived names and ETFs, written after the test window with knowledge of which names survived. That limits but does not remove survivorship bias: it is not a point-in-time constituent list.
- Everything is in-sample. Parameters were not tuned on this run, but there is no held-out period, no walk-forward, and no statistical test, so this is not evidence of out-of-sample edge.
- Yahoo data can be revised; re-running without the local cache may produce slightly different bars.
