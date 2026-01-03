# Advanced Chart Engine Guide

## Overview

The `ChartEngine` is a comprehensive visualization system for backtest results, strategy comparisons, and mean reversion analysis. It provides professional-quality charts with extensive customization options.

## Features

### 1. Comprehensive Dashboards
- Equity curves with annotations
- Drawdown analysis
- Returns distribution
- Monthly returns heatmap
- Rolling Sharpe ratio
- Rolling volatility
- Trade analysis
- Price charts with signals

### 2. Strategy Comparisons
- Side-by-side equity curve comparison
- Returns comparison bar charts
- Drawdown comparison
- Metrics comparison tables
- Risk-return scatter plots

### 3. Mean Reversion Analysis
- Price with Bollinger Bands
- Z-score visualization
- Mean reversion strength
- Distance from mean
- Volume profile
- Signal timeline

### 4. Multi-Symbol Analysis
- Returns comparison across symbols
- Sharpe ratio comparison
- Win rate comparison
- Max drawdown comparison
- Risk-return profiles
- Comprehensive metrics tables

## Quick Start

### Basic Usage

```python
from backtest.engine import BacktestEngine
from backtest.chart_engine import ChartEngine

# Run your backtest
engine = BacktestEngine(data, signals, initial_capital=100000)
engine.run()

# Create comprehensive dashboard
chart_engine = ChartEngine()
fig = chart_engine.plot_comprehensive_dashboard(
    engine,
    title="My Strategy Analysis"
)

# Save the chart
fig.savefig("dashboard.png", dpi=150, bbox_inches='tight')
```

### Strategy Comparison

```python
# Run multiple strategies
strategies = {
    'Strategy A': engine_a,
    'Strategy B': engine_b
}

# Compare with benchmark
fig = chart_engine.plot_strategy_comparison(
    strategies,
    benchmark=buy_hold_engine,
    title="Strategy Comparison"
)
```

### Mean Reversion Visualization

```python
from strategy.mean_reversion import MeanReversionIndicators

# Calculate indicators
mr = MeanReversionIndicators(data, lookback_period=20)
indicators = mr.get_indicators_df()
signals = mr.get_signals_df()

# Visualize
fig = chart_engine.plot_mean_reversion_analysis(
    data,
    indicators,
    signals,
    title="Mean Reversion Analysis"
)
```

### Multi-Symbol Comparison

```python
# Prepare results for multiple symbols
symbol_results = {
    'AAPL': {'metrics': metrics_aapl, 'engine': engine_aapl},
    'MSFT': {'metrics': metrics_msft, 'engine': engine_msft},
    'GOOGL': {'metrics': metrics_googl, 'engine': engine_googl}
}

# Compare
fig = chart_engine.plot_multi_symbol_comparison(
    symbol_results,
    title="Multi-Symbol Performance"
)
```

## Chart Types

### 1. Comprehensive Dashboard (`plot_comprehensive_dashboard`)

Creates a 4x3 grid dashboard with:
- **Top Row (Full Width)**: Equity curve with profit/loss fill
- **Second Row**: Drawdown, Returns Distribution, Monthly Returns Heatmap
- **Third Row**: Rolling Sharpe, Rolling Volatility, Trade Analysis
- **Bottom Row (Full Width)**: Price chart with trading signals

**Features:**
- Annotated equity curve with final value
- Color-coded profit/loss areas
- Metrics text box in corner
- Professional styling

### 2. Strategy Comparison (`plot_strategy_comparison`)

Compares multiple strategies with:
- **Top Row (Full Width)**: Overlaid equity curves
- **Second Row**: Returns bar chart, Drawdown comparison
- **Third Row**: Metrics table, Risk-return scatter

**Use Cases:**
- Comparing different parameter sets
- Strategy vs. benchmark
- Multiple strategy variants

### 3. Mean Reversion Analysis (`plot_mean_reversion_analysis`)

Specialized charts for mean reversion:
- **Top Row (Full Width)**: Price with Bollinger Bands and signals
- **Second Row**: Z-score with thresholds, Mean reversion strength
- **Third Row**: Distance from mean, Volume profile
- **Bottom Row (Full Width)**: Signal timeline

**Features:**
- Bollinger Band visualization
- Z-score threshold lines
- Strength scoring visualization
- Volume confirmation

### 4. Multi-Symbol Comparison (`plot_multi_symbol_comparison`)

Compares performance across symbols:
- **Top Row**: Returns bar, Sharpe comparison, Win rate comparison
- **Bottom Row**: Max drawdown, Risk-return scatter, Metrics table

**Use Cases:**
- Portfolio analysis
- Symbol selection
- Performance ranking

## Customization

### Styling

```python
# Custom style
chart_engine = ChartEngine(style='seaborn-v0_8', figsize=(20, 12))

# Available styles:
# - 'seaborn-v0_8' (default)
# - 'seaborn-v0_8-darkgrid'
# - 'ggplot'
# - 'classic'
# - 'bmh'
```

### Color Scheme

The engine uses a professional color scheme:
- **Primary**: Blue (#2E86AB)
- **Secondary**: Purple (#A23B72)
- **Success**: Green (#06A77D)
- **Danger**: Red (#D00000)
- **Warning**: Orange (#FFB627)
- **Info**: Cyan (#06AED5)

### Saving Charts

```python
# High resolution
fig.savefig("chart.png", dpi=300, bbox_inches='tight')

# PDF format
fig.savefig("chart.pdf", bbox_inches='tight')

# SVG format (vector)
fig.savefig("chart.svg", bbox_inches='tight')
```

## Advanced Features

### Rolling Metrics

The dashboard includes rolling calculations:
- **Rolling Sharpe**: 30-period default (adjustable)
- **Rolling Volatility**: 30-period default (adjustable)

### Monthly Returns Heatmap

Automatically calculates and displays:
- Monthly returns as color-coded grid
- Year-over-year comparison
- Visual pattern identification

### Trade Analysis

Visualizes:
- Win/loss distribution
- P&L histogram
- Trade frequency

### Metrics Display

Automatically formats and displays:
- Total return
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Win rate
- Profit factor
- Total trades

## Best Practices

### 1. Data Preparation
- Ensure data has proper datetime index
- Include all required columns (close, open, high, low, volume)
- Handle missing data before visualization

### 2. Chart Organization
- Use descriptive titles
- Save charts with meaningful filenames
- Organize output directory structure

### 3. Performance
- For large datasets, consider downsampling for visualization
- Use appropriate DPI (150-300) for file size vs. quality balance

### 4. Presentation
- Use consistent styling across all charts
- Include context in titles
- Save in multiple formats (PNG for quick view, PDF for reports)

## Examples

See `examples/chart_engine_example.py` for complete working examples:
- Comprehensive dashboard
- Strategy comparison
- Mean reversion visualization
- Multi-symbol comparison

## Integration

The ChartEngine integrates seamlessly with:
- `BacktestEngine`: Direct visualization of results
- `SwingTradingIndicators`: Strategy signals
- `MeanReversionIndicators`: Mean reversion analysis
- `UniverseConstructor`: Multi-symbol analysis

## Troubleshooting

### Common Issues

1. **"No backtest results available"**
   - Ensure `engine.run()` has been called
   - Check that results DataFrame is not empty

2. **Missing columns error**
   - Verify data has required columns (equity, price, returns)
   - Check column names match expected format

3. **Date index issues**
   - Ensure index is datetime type
   - Use `pd.to_datetime()` if needed

4. **Memory issues with large datasets**
   - Downsample data before visualization
   - Use lower DPI for file saving
   - Close figures after saving

## API Reference

### ChartEngine Class

```python
ChartEngine(style='seaborn-v0_8', figsize=(16, 10))
```

**Methods:**
- `plot_comprehensive_dashboard(engine, title, save_path)`
- `plot_strategy_comparison(strategies, benchmark, title, save_path)`
- `plot_mean_reversion_analysis(data, indicators, signals, title, save_path)`
- `plot_multi_symbol_comparison(symbol_results, title, save_path)`

All methods return matplotlib Figure objects for further customization.

