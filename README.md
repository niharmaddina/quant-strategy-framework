# Quantitative Trading Strategy & Backtesting Framework

A comprehensive quantitative trading strategy framework designed for **options swing trading** with manual execution. The system generates trading signals and indicators that traders can use to make informed options trading decisions.

## Features

- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands, Stochastic, ATR
- **Signal Generation**: Automated signal generation with confidence scoring
- **Backtesting Engine**: Full backtesting pipeline with comprehensive metrics
- **Universe Construction**: Guidance and tools for building focused trading universes
- **Visualization**: Equity curves, drawdown charts, and trade analysis
- **Options-Ready**: Designed for swing trading with options (signals for manual execution)

## Project Structure

```
oni_strat_m1/
├── data/              # Data fetching utilities
│   ├── __init__.py
│   └── fetcher.py     # yfinance data fetcher
├── universe/          # Universe construction
│   ├── __init__.py
│   └── constructor.py # Universe building and filtering
├── strategy/          # Trading strategy
│   ├── __init__.py
│   └── indicators.py  # Technical indicators and signals
├── backtest/          # Backtesting engine
│   ├── __init__.py
│   ├── engine.py      # Backtesting logic
│   └── visualizer.py  # Visualization tools
├── main.py            # Main execution script
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run backtest on default symbols (AAPL, MSFT):

```bash
python main.py
```

### Custom Symbols

```bash
python main.py --symbols AAPL MSFT TSLA NVDA
```

### Custom Date Range

```bash
python main.py --symbols AAPL --start 2023-01-01 --end 2024-01-01
```

### Use Balanced Universe

```bash
python main.py --universe
```

### Full Options

```bash
python main.py \
    --symbols AAPL MSFT TSLA \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --capital 100000 \
    --position-size 0.15 \
    --no-plot
```

## Universe Construction Guide

### Why Focus on Universe Selection?

For options swing trading, a well-constructed universe is critical:

1. **Reduces Complexity**: Fewer symbols = easier analysis and monitoring
2. **Improves Quality**: Focus on liquid, tradeable stocks
3. **Better Signals**: High-quality stocks produce more reliable signals
4. **Manageable Risk**: Easier to understand and control exposure

### Recommended Universe Criteria

#### 1. **Liquidity** (Most Important)
- **Options Volume**: > 100k daily contracts
- **Stock Volume**: > 1M average daily volume
- **Tight Spreads**: Low bid-ask spreads on options
- **High Open Interest**: Ensures you can enter/exit positions

#### 2. **Volatility**
- **IV > 20%**: Sufficient premium for options strategies
- **Not Too Volatile**: Avoid extreme volatility that's hard to predict
- **Swing Opportunities**: Enough movement for 3-7 day swings

#### 3. **Market Cap**
- **> $2B**: Reduces manipulation risk
- **Mid to Large Cap**: Better options liquidity
- **Established Companies**: More predictable price action

#### 4. **Sector Diversification**
- **3-5 Sectors**: Reduces correlation risk
- **Avoid Overlap**: Don't pick highly correlated stocks
- **Balance**: Mix of growth and value

#### 5. **Universe Size**
- **Start Small**: 10-20 stocks initially
- **Quality over Quantity**: Better to have 15 great stocks than 50 mediocre ones
- **Easy to Monitor**: You need to watch these manually

### Example Universe Construction

```python
from universe.constructor import UniverseConstructor

# Create universe
universe = UniverseConstructor()

# Add your preferred stocks
universe.add_symbol('AAPL')
universe.add_symbol('MSFT')
# ... etc

# Filter by liquidity
liquid_stocks = universe.filter_by_liquidity(min_volume=1_000_000)

# Get balanced universe
balanced = universe.create_sector_balanced_universe(stocks_per_sector=2)

# Export for later use
universe.export_universe('my_universe.txt')
```

### Pre-defined Universes

The framework includes:

- **Popular Options Stocks**: High-volume, liquid options stocks
- **Sector ETFs**: For broad market exposure
- **Balanced Universe**: Sector-diversified selection

## Strategy Overview

### Indicators Used

1. **RSI (Relative Strength Index)**
   - Oversold: < 30 (bullish signal)
   - Overbought: > 70 (bearish signal)

2. **MACD (Moving Average Convergence Divergence)**
   - Bullish crossover: MACD crosses above signal line
   - Bearish crossover: MACD crosses below signal line

3. **Moving Averages**
   - SMA 20/50 crossover
   - Price relative to 200 SMA for trend

4. **Bollinger Bands**
   - Oversold: Price touches lower band
   - Overbought: Price touches upper band

5. **Stochastic Oscillator**
   - Oversold: < 20
   - Overbought: > 80

6. **ATR (Average True Range)**
   - Measures volatility for position sizing

### Signal Generation

Signals are generated by combining multiple indicators:

- **Bullish Signal**: 3+ bullish indicators align
- **Bearish Signal**: 3+ bearish indicators align
- **Confidence Score**: 0-1 based on indicator agreement

### Using Signals for Options Trading

The strategy generates signals that you can use for manual options trading:

**Bullish Signal (Signal = 1)**
- Consider: Long calls, bull spreads, covered calls on existing positions
- Entry: When confidence > 0.6
- Exit: When signal flips to bearish or neutral

**Bearish Signal (Signal = -1)**
- Consider: Long puts, bear spreads, protective puts
- Entry: When confidence > 0.6
- Exit: When signal flips to bullish or neutral

**Neutral Signal (Signal = 0)**
- Wait for clearer direction
- Consider: Iron condors, straddles (if expecting volatility)

## Backtesting Metrics

The backtesting engine provides:

- **Total Return**: Overall percentage return
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized volatility
- **Average Return per Trade**: Mean return per trade

## Programmatic Usage

### Fetch Data

```python
from data.fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
```

### Calculate Indicators

```python
from strategy.indicators import SwingTradingIndicators

indicators = SwingTradingIndicators(data)
current_signal = indicators.get_current_signals()

print(f"Signal: {current_signal['signal']}")
print(f"Confidence: {current_signal['confidence']:.2%}")
```

### Run Backtest

```python
from backtest.engine import BacktestEngine

backtest = BacktestEngine(
    data=data,
    signals=indicators.get_signals_df(),
    initial_capital=100000,
    position_size=0.1
)

results = backtest.run()
backtest.print_metrics()
```

### Visualize Results

```python
from backtest.visualizer import BacktestVisualizer

visualizer = BacktestVisualizer(backtest)
fig = visualizer.plot_all()
fig.savefig('backtest_results.png')
```

## Output

The framework generates:

1. **Console Output**: 
   - Current signals for each symbol
   - Backtest metrics
   - Summary statistics

2. **Plots** (in `output/` directory):
   - Equity curves
   - Drawdown charts
   - Price charts with signals
   - Returns distribution

## Best Practices

### For Options Swing Trading

1. **Use Signals as Guidance**: Don't blindly follow - use as one input
2. **Check Options Chain**: Verify liquidity before trading
3. **Consider IV**: High IV = expensive options (may want to sell)
4. **Time Decay**: Swing trades should be 3-7 days typically
5. **Risk Management**: Never risk more than you can afford to lose

### For Universe Management

1. **Start Small**: Begin with 10-15 stocks
2. **Review Regularly**: Remove underperformers, add new opportunities
3. **Sector Balance**: Maintain diversification
4. **Liquidity First**: Always prioritize liquid options

### For Backtesting

1. **Use Realistic Parameters**: Include commissions, slippage
2. **Test Multiple Periods**: Bull markets, bear markets, sideways
3. **Out-of-Sample Testing**: Don't overfit to historical data
4. **Consider Market Regime**: Strategies work differently in different conditions

## Limitations & Considerations

- **Free Data**: Uses yfinance (may have rate limits)
- **Simplified Execution**: Assumes perfect fills (no slippage modeling)
- **No Options Pricing**: Doesn't model options prices, only stock signals
- **Manual Execution**: Signals are for manual review, not automated trading
- **Past Performance**: Backtests don't guarantee future results

## Next Steps

1. **Customize Indicators**: Modify `strategy/indicators.py` to add your own
2. **Refine Universe**: Build your own universe based on your criteria
3. **Add Options Data**: Integrate options chain analysis
4. **Risk Management**: Add position sizing rules, stop losses
5. **Paper Trading**: Test signals in real-time before live trading

## License

This project is for educational and research purposes.

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor.
