# Walk-forward evaluation: frozen 50-symbol universe

Run started 2026-10-12 05:01 UTC; wall time 65s.

## Command

```bash
python examples/walk_forward.py
# equivalent to:
# python examples/walk_forward.py --universe-file universe/universe_mean_reversion_manual_50.txt --capital 100000 --position-size 0.1
```

## Setup

- Universe: `universe/universe_mean_reversion_manual_50.txt`, 50 symbols, each an isolated account with 100,000 initial capital per period and 10% of capital per trade
- In-sample: 2021-01-04 to 2022-12-31 inclusive, 503 bars per symbol; out-of-sample: 2023-01-03 to 2024-12-31 inclusive, 502 bars per symbol. Daily bars from the same `data_cache/` CSVs as `results/frozen50_2021_2024.csv` (Yahoo `Close`, split-adjusted, not dividend-adjusted)
- Signals: `strategy/indicators.py` `SwingTradingIndicators`, computed on each symbol's full series and then cut to the period, so the out-of-sample period starts with warmed-up indicators; all indicators are backward-looking, so the in-sample slice is identical to an in-sample-only computation. The in-sample period still contains the 200-bar SMA warm-up at its start, during which no signals fire
- Engine: `backtest/engine.py` with the `main.py --engine legacy` defaults except where a grid axis overrides them (0.1% commission per side, trailing stop off, take-profit at max(3%, 2x stop distance), exit on a neutral signal after 5 bars, new entries halted at a 20% drawdown)
- Grid: `rsi_oversold` x `max_holding_days` x `stop_loss_pct` = 3 x 3 x 2 = 18 configurations: `rsi_oversold` in {35, 45, 55}; `max_holding_days` in {5, 10, 20}; `stop_loss_pct` in {0.03, 0.05}
- Selection rule (fixed in the script before any out-of-sample number was computed): highest mean per-symbol in-sample total return among configurations with at least 100 pooled in-sample trades (2 per symbol on average; the default has 129), ties broken by higher median. 18 of 18 configurations qualify
- Buy-and-hold: fully invested in the symbol over the out-of-sample bars, engine formulas for Sharpe (2% risk-free, 252-day annualisation) and drawdown
- Software: Python 3.11.3, pandas 3.0.6, numpy 2.4.6, yfinance 1.7.0

## Stage 0: which parameters matter (in-sample, one at a time)

Each exposed parameter varied alone from the default; `changes_result` is whether any in-sample number differs from the default row.

| parameter | value | is_mean_return_pct | is_median_return_pct | is_trades | is_mean_in_market_sharpe_ratio | changes_result |
|---|---|---|---|---|---|---|
| (default) |  | -0.088 | -0.038 | 129 | -1.617 | False |
| rsi_oversold | 30 | -0.088 | -0.038 | 129 | -1.617 | False |
| rsi_oversold | 45 | -0.093 | -0.046 | 136 | -1.524 | True |
| rsi_oversold | 55 | -0.209 | -0.164 | 177 | -2.246 | True |
| rsi_overbought | 60 | -0.088 | -0.038 | 129 | -1.617 | False |
| rsi_overbought | 70 | -0.088 | -0.038 | 129 | -1.617 | False |
| bb_period | 10 | -0.088 | -0.038 | 129 | -1.617 | False |
| bb_period | 30 | -0.088 | -0.038 | 129 | -1.617 | False |
| bb_std_dev | 1 | -0.088 | -0.038 | 129 | -1.617 | False |
| bb_std_dev | 1.5 | -0.088 | -0.038 | 129 | -1.617 | False |
| bb_std_dev | 2.5 | -0.088 | -0.038 | 129 | -1.617 | False |
| max_holding_days | 5 | -0.105 | -0.040 | 189 | -1.890 | True |
| max_holding_days | 10 | -0.069 | -0.030 | 138 | -1.685 | True |
| max_holding_days | 40 | -0.088 | -0.038 | 129 | -1.617 | False |
| stop_loss_pct | 0.03 | -0.103 | -0.034 | 142 | -1.893 | True |
| stop_loss_pct | 0.08 | -0.082 | -0.034 | 125 | -1.488 | True |

The Bollinger parameters and the RSI overbought threshold do not change a single number: the strategy multiplies the Bollinger and RSI flags by trend, momentum and price-action filters, and a close at or below the lower band almost never coincides with those filters being true, while RSI overbought only feeds sell signals that never meet an open long. The grid therefore varies the RSI oversold threshold (upwards, since 30 and 35 are indistinguishable), the maximum holding period and the stop loss.

## Stage 1 and 2: in-sample grid and selection

Sorted by in-sample mean return. `oos_rank` is the configuration's rank by out-of-sample mean return among the 18 configurations (1 = best), shown for the transfer check; it played no part in the selection. Machine-readable: `results/walk_forward_grid.csv`.

| configuration | is_mean_return_pct | is_median_return_pct | is_trades | is_symbols_positive | is_mean_in_market_sharpe_ratio | qualifies | oos_mean_return_pct | oos_median_return_pct | oos_trades | oos_rank | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rsi<35, hold 10, stop 5% | -0.069 | -0.030 | 138 | 15 | -1.685 | True | -0.100 | -0.171 | 374 | 18 | selected |
| rsi<45, hold 10, stop 5% | -0.074 | -0.046 | 145 | 14 | -1.591 | True | -0.079 | -0.238 | 394 | 11 |  |
| rsi<35, hold 20, stop 5% | -0.088 | -0.038 | 129 | 14 | -1.617 | True | -0.098 | -0.151 | 337 | 16 | default |
| rsi<45, hold 20, stop 3% | -0.091 | -0.040 | 148 | 14 | -1.735 | True | -0.067 | -0.171 | 383 | 9 |  |
| rsi<45, hold 20, stop 5% | -0.093 | -0.046 | 136 | 14 | -1.524 | True | -0.076 | -0.171 | 357 | 10 |  |
| rsi<45, hold 10, stop 3% | -0.093 | -0.040 | 152 | 14 | -1.843 | True | -0.052 | -0.238 | 408 | 7 |  |
| rsi<45, hold 5, stop 3% | -0.102 | -0.050 | 198 | 10 | -1.968 | True | -0.066 | -0.201 | 557 | 8 |  |
| rsi<35, hold 5, stop 3% | -0.103 | -0.040 | 191 | 12 | -2.098 | True | -0.089 | -0.166 | 536 | 14 |  |
| rsi<35, hold 20, stop 3% | -0.103 | -0.034 | 142 | 14 | -1.893 | True | -0.094 | -0.151 | 363 | 15 |  |
| rsi<35, hold 5, stop 5% | -0.105 | -0.040 | 189 | 13 | -1.890 | True | -0.099 | -0.166 | 530 | 17 |  |
| rsi<35, hold 10, stop 3% | -0.106 | -0.030 | 146 | 15 | -2.004 | True | -0.080 | -0.173 | 388 | 12 |  |
| rsi<45, hold 5, stop 5% | -0.121 | -0.073 | 197 | 11 | -1.821 | True | -0.083 | -0.196 | 551 | 13 |  |
| rsi<55, hold 10, stop 5% | -0.190 | -0.144 | 186 | 10 | -2.288 | True | -0.012 | -0.122 | 497 | 2 |  |
| rsi<55, hold 20, stop 5% | -0.209 | -0.164 | 177 | 11 | -2.246 | True | -0.012 | -0.108 | 459 | 3 |  |
| rsi<55, hold 20, stop 3% | -0.219 | -0.262 | 189 | 11 | -2.674 | True | -0.016 | -0.073 | 487 | 4 |  |
| rsi<55, hold 10, stop 3% | -0.222 | -0.243 | 193 | 10 | -2.757 | True | -0.001 | -0.089 | 512 | 1 |  |
| rsi<55, hold 5, stop 3% | -0.225 | -0.196 | 239 | 10 | -2.713 | True | -0.035 | -0.113 | 670 | 6 |  |
| rsi<55, hold 5, stop 5% | -0.229 | -0.196 | 238 | 10 | -2.330 | True | -0.035 | -0.113 | 663 | 5 |  |

Selected: `rsi<35, hold 10, stop 5%` (in-sample mean -0.069%, median -0.030%, 138 trades). Default: `rsi<35, hold 20, stop 5%` (in-sample mean -0.088%, median -0.038%, 129 trades).

## Stage 3: out-of-sample

2023-01-03 to 2024-12-31, fresh 100,000 per symbol. Means and medians are across the 50 symbols; `beats_buy_and_hold` counts symbols where the configuration's total return exceeds buy-and-hold in the same symbol. Per-symbol rows: `results/walk_forward.csv`.

| configuration | mean_return_pct | median_return_pct | symbols_positive | beats_buy_and_hold | trades | mean_exposure_pct | mean_holding_days | mean_sharpe_ratio | mean_in_market_sharpe_ratio | mean_max_drawdown_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| default (rsi<35, hold 20, stop 5%) | -0.098 | -0.151 | 21 | 12 | 337 | 9.131 | 6.777 | -5.127 | -1.041 | 0.789 |
| selected in-sample (rsi<35, hold 10, stop 5%) | -0.100 | -0.171 | 20 | 12 | 374 | 9.402 | 6.181 | -5.077 | -1.139 | 0.806 |
| buy-and-hold | 35.040 | 12.551 | 37 |  |  |  |  | 0.512 |  | 21.135 |

### What it shows

In-sample selection picked `rsi<35, hold 10, stop 5%`, which beat the default by +0.019 percentage points of mean return in-sample. Out-of-sample it did not keep that edge: -0.003 points versus the default (-0.100% vs -0.098%), better than the default on 9 of 50 symbols, and ranked 18 of 18 grid configurations out-of-sample (the default ranked 16). Across all 18 configurations the Spearman rank correlation between in-sample and out-of-sample mean return is -0.64.

Both strategy configurations remain close to flat out-of-sample while buy-and-hold in the same symbols returned +35.04% on average; re-tuning the exposed thresholds does not change the character of the result reported in `results/summary.md`.

### Per symbol (out-of-sample)

| symbol | default_total_return_pct | default_total_trades | selected_total_return_pct | selected_total_trades | bh_total_return_pct | default_in_market_sharpe_ratio | selected_in_market_sharpe_ratio | bh_sharpe_ratio |
|---|---|---|---|---|---|---|---|---|
| SPY | 0.06 | 15 | 0.06 | 15 | 53.90 | 0.39 | 0.39 | 1.59 |
| QQQ | 0.64 | 12 | 0.97 | 14 | 93.30 | 1.27 | 1.71 | 1.83 |
| DIA | 0.38 | 4 | 0.37 | 5 | 28.46 | 3.30 | 3.21 | 0.99 |
| IWM | -1.33 | 7 | -1.33 | 7 | 27.43 | -5.02 | -5.02 | 0.60 |
| XLK | 0.40 | 14 | 0.40 | 14 | 88.64 | 0.83 | 0.83 | 1.56 |
| XLF | -0.89 | 14 | -0.89 | 14 | 40.82 | -1.82 | -1.82 | 1.08 |
| XLV | -0.19 | 3 | -0.19 | 3 | 1.56 | -2.62 | -2.62 | -0.05 |
| XLY | 1.18 | 9 | 1.14 | 11 | 74.74 | 2.26 | 2.19 | 1.45 |
| XLP | 0.09 | 2 | 0.09 | 2 | 5.84 | 1.53 | 1.53 | 0.13 |
| XLE | 0.23 | 2 | -0.13 | 3 | 1.49 | 2.49 | -1.10 | 0.04 |
| XLI | 0.13 | 9 | 0.14 | 10 | 33.88 | 0.72 | 0.73 | 0.96 |
| XLC | -0.85 | 15 | -0.32 | 17 | 99.16 | -1.18 | -0.38 | 1.94 |
| XLRE | 0.26 | 2 | 0.24 | 3 | 9.71 | 1.91 | 1.79 | 0.24 |
| XLB | 0.02 | 4 | 0.14 | 5 | 8.43 | 0.40 | 1.28 | 0.21 |
| XLU | 0.76 | 4 | 0.55 | 6 | 7.36 | 3.22 | 1.94 | 0.18 |
| AAPL | 0.42 | 12 | 0.40 | 13 | 100.22 | 0.75 | 0.72 | 1.65 |
| MSFT | -0.63 | 10 | -0.87 | 14 | 75.93 | -0.70 | -0.98 | 1.28 |
| CSCO | 0.39 | 6 | 0.36 | 7 | 23.49 | 1.25 | 1.20 | 0.54 |
| ORCL | 0.40 | 14 | 0.58 | 14 | 99.04 | 0.54 | 0.74 | 1.20 |
| JPM | 1.23 | 11 | 1.14 | 15 | 77.41 | 2.16 | 1.87 | 1.32 |
| BAC | -0.66 | 8 | -0.70 | 10 | 31.15 | -1.44 | -1.53 | 0.59 |
| WFC | 1.22 | 7 | 1.21 | 8 | 68.08 | 2.49 | 2.38 | 0.99 |
| GS | -0.65 | 7 | -0.69 | 8 | 65.39 | -1.14 | -1.20 | 1.06 |
| JNJ | -0.16 | 1 | -0.16 | 1 | -18.84 | -3.53 | -3.53 | -0.71 |
| UNH | -1.39 | 3 | -1.39 | 3 | -2.46 | -7.34 | -7.34 | -0.01 |
| PFE | -0.54 | 1 | -0.54 | 1 | -48.24 | -15.08 | -15.08 | -1.38 |
| AMZN | -1.13 | 13 | -1.13 | 13 | 155.64 | -1.24 | -1.24 | 1.63 |
| HD | -1.14 | 6 | -1.03 | 7 | 23.13 | -4.21 | -3.62 | 0.51 |
| MCD | -0.36 | 4 | -0.22 | 4 | 9.67 | -1.81 | -0.98 | 0.25 |
| PG | -0.57 | 6 | -0.57 | 6 | 10.61 | -4.47 | -4.47 | 0.28 |
| KO | -0.03 | 4 | -0.03 | 4 | -1.10 | -0.01 | -0.01 | -0.13 |
| PEP | -0.26 | 4 | -0.28 | 5 | -15.24 | -1.39 | -1.50 | -0.58 |
| XOM | -0.93 | 3 | -0.93 | 3 | 1.00 | -5.92 | -5.92 | 0.04 |
| CVX | -0.26 | 1 | -0.26 | 1 | -16.75 | -4.92 | -4.92 | -0.42 |
| COP | -0.38 | 2 | -0.38 | 2 | -12.33 | -5.45 | -5.45 | -0.21 |
| CAT | -0.93 | 14 | -0.96 | 15 | 51.86 | -0.95 | -0.97 | 0.83 |
| HON | -0.01 | 5 | -0.01 | 5 | 5.41 | 0.10 | 0.10 | 0.13 |
| UPS | -1.03 | 1 | -1.03 | 1 | -28.06 | -8.04 | -8.04 | -0.61 |
| GOOGL | -1.46 | 13 | -1.46 | 13 | 112.41 | -1.62 | -1.62 | 1.38 |
| META | 2.24 | 18 | 2.37 | 20 | 369.38 | 1.35 | 1.38 | 2.17 |
| VZ | -0.67 | 4 | -0.67 | 4 | -0.32 | -3.09 | -3.09 | 0.02 |
| AMT | 0.50 | 3 | 0.50 | 3 | -14.56 | 2.33 | 2.33 | -0.26 |
| PLD | -0.09 | 1 | -0.09 | 1 | -6.20 | -2.50 | -2.50 | -0.08 |
| O | 0.16 | 3 | 0.16 | 3 | -16.29 | 2.01 | 2.01 | -0.50 |
| LIN | -0.91 | 7 | -0.92 | 8 | 31.51 | -3.18 | -3.18 | 0.77 |
| SHW | 1.89 | 8 | 1.57 | 10 | 42.01 | 4.35 | 3.08 | 0.79 |
| FCX | -0.50 | 5 | -0.53 | 5 | 0.42 | -1.57 | -1.60 | 0.12 |
| NEE | 1.00 | 3 | 1.01 | 4 | -14.48 | 4.47 | 4.19 | -0.23 |
| DUK | -0.38 | 7 | -0.38 | 7 | 3.90 | -1.38 | -1.38 | 0.08 |
| SO | -0.14 | 6 | -0.35 | 7 | 14.49 | -0.54 | -1.43 | 0.36 |

## Caveats

- One split, one selection. A single in-sample/out-of-sample cut on four years of daily data is the minimum honest protocol, not a statistical test; with 18 configurations and small return differences the in-sample winner is partly noise, which is exactly what the out-of-sample column is there to show.
- The grid is small by design and only covers thresholds the code exposes. The exits that dominate trade length (neutral-signal exit after 5 bars, take-profit rule) are fixed engine behaviour and were not tuned.
- Each period is an isolated account per symbol: no portfolio aggregation, no slippage or spread, fills at the daily close, dividends excluded from buy-and-hold.
- The universe is the same hand-picked list of survivors as the main run; the out-of-sample period is out-of-sample for the parameters, not for the choice of symbols.
- Yahoo data can be revised; with `data_cache/` present the run is offline and reproduces these files exactly.
