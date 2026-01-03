# Mean Reversion Strategy with Granular Prices

## Overview

Enhanced mean reversion strategy with support for granular intraday price data (1m, 5m, 15m, 30m, 1h intervals). The strategy uses Z-scores, Bollinger Bands, and price distance from mean to identify mean reversion opportunities.

## Key Features

### 1. Granular Data Support

The `DataFetcher` now supports intraday intervals:
- **1m, 2m, 5m**: Limited to 7-60 days
- **15m, 30m, 60m, 90m, 1h**: Limited to 730 days
- **1d, 5d, 1wk, 1mo**: Full historical data

### 2. Mean Reversion Indicators

The `MeanReversionIndicators` class provides:

#### Core Indicators
- **Z-Score**: `(price - mean) / std`
  - Z > 2.0: Overbought (sell signal)
  - Z < -2.0: Oversold (buy signal)
  
- **Bollinger Bands**: Mean ± (std × multiplier)
  - Price at/below lower band: Oversold
  - Price at/above upper band: Overbought
  
- **Price Distance from Mean**: Percentage distance from rolling mean
- **Mean Reversion Strength**: Combined score (0-10) based on multiple factors

#### Signal Generation
- **Buy Signals**: Triggered when:
  - Z-score < -2.0 OR
  - Price ≤ Lower Bollinger Band OR
  - Distance from mean < -2%
  - AND minimum strength threshold met
  
- **Sell Signals**: Triggered when:
  - Z-score > 2.0 OR
  - Price ≥ Upper Bollinger Band OR
  - Distance from mean > 2%
  - AND minimum strength threshold met

### 3. Signal Quality Metrics

- **Confidence Score** (0-1): Based on signal strength and filters
- **Mean Reversion Strength** (0-10): Combines Z-score, distance, BB position, momentum
- **Expected Reversion %**: Estimated move back to mean
- **Risk/Reward Ratio**: Expected reward vs. risk

## Usage

### Basic Example

```python
from data.fetcher import DataFetcher
from strategy.mean_reversion import MeanReversionIndicators

# Fetch intraday data
fetcher = DataFetcher()
data = fetcher.fetch_stock_data(
    symbol="SPY",
    start_date="2024-01-01",
    end_date="2024-01-31",
    interval="1h"  # 1-hour intervals
)

# Calculate mean reversion indicators
mr = MeanReversionIndicators(
    data=data,
    lookback_period=20,  # 20 periods
    z_score_threshold=2.0,
    bb_std_dev=2.0
)

# Get current signals
signals = mr.get_current_signals()
print(f"Signal: {signals['signal']}")
print(f"Z-Score: {signals['z_score']:.2f}")
print(f"Confidence: {signals['confidence']:.2%}")
```

### Finding Opportunities

```python
# Get filtered opportunities
opportunities = mr.get_mean_reversion_opportunities(
    min_strength=5.0,  # Minimum strength (0-10)
    min_confidence=0.5  # Minimum confidence (0-1)
)

print(f"Found {len(opportunities)} opportunities")
```

### Running Backtest

```python
from backtest.engine import BacktestEngine

signals_df = mr.get_signals_df()

engine = BacktestEngine(
    data=data,
    signals=signals_df[['signal']],
    initial_capital=100000,
    position_size=0.1,
    stop_loss_pct=0.02,  # Tighter stop for intraday
    commission=0.001
)

results = engine.run()
metrics = engine.get_metrics()
```

## Strategy Parameters

### Lookback Period
- **1h data**: 20-40 periods (20-40 hours)
- **15m data**: 40-80 periods (10-20 hours)
- **5m data**: 60-120 periods (5-10 hours)

### Z-Score Threshold
- **Default**: 2.0 (2 standard deviations)
- **Tighter**: 1.5 (more signals, lower quality)
- **Stricter**: 2.5 (fewer signals, higher quality)

### Bollinger Bands
- **Default**: 2.0 standard deviations
- Adjust based on volatility regime

## Mean Reversion Strength Calculation

The strength score (0-10) combines:
- **Z-Score Magnitude** (30%): How extreme the deviation
- **Distance from Mean** (20%): Percentage distance
- **Bollinger Band Position** (30%): Position within bands
- **Momentum** (20%): Reversion potential

Additional bonuses:
- **Volume Confirmation**: +20% if volume > 1.2× average
- **High Volatility**: +10% if volatility > 1.2× average

## Signal Filters

Signals are filtered by:
1. **Minimum Strength**: Default 3.0/10
2. **Z-Score Direction**: Must be opposite of mean
3. **Bollinger Band Position**: Must be in extreme 20% of range
4. **Volume Confirmation**: Higher volume preferred
5. **Volatility Regime**: High volatility preferred

## Best Practices

### For Intraday Trading (1h, 15m, 5m)
- Use shorter lookback periods (20-60 periods)
- Tighter stop losses (1-2%)
- Focus on high-volume periods
- Consider market hours (avoid pre/post market)

### For Daily Data
- Use longer lookback periods (20-50 days)
- Standard stop losses (3-5%)
- More selective signals (higher strength threshold)

### Risk Management
- Always use stop losses
- Limit position size (5-10% per trade)
- Monitor volatility regime
- Exit when price returns to mean

## Example Scripts

See `examples/mean_reversion_example.py` for:
- 1-hour intraday mean reversion
- 15-minute mean reversion
- Granular price analysis (5-minute)

## Performance Considerations

- **Data Fetching**: Intraday data has API limits
  - 1m: Max 7 days
  - 5m: Max 60 days
  - 15m+: Max 730 days

- **Calculation Speed**: Mean reversion indicators are fast
  - Processes 1000+ data points in < 1 second

- **Signal Frequency**: Higher with shorter intervals
  - 1h: ~5-10 signals per day
  - 15m: ~10-20 signals per day
  - 5m: ~20-40 signals per day

## Integration with Existing Strategy

The mean reversion strategy can be used alongside the swing trading strategy:

```python
from strategy.indicators import SwingTradingIndicators
from strategy.mean_reversion import MeanReversionIndicators

# Swing trading for trend following
swing = SwingTradingIndicators(daily_data)

# Mean reversion for intraday opportunities
mr = MeanReversionIndicators(intraday_data)

# Combine signals based on your logic
```

## Next Steps

1. **Backtest Optimization**: Test different lookback periods and thresholds
2. **Multi-Timeframe**: Combine multiple timeframes for confirmation
3. **Sector Analysis**: Apply to sector ETFs for diversification
4. **Risk Management**: Implement dynamic position sizing based on volatility

