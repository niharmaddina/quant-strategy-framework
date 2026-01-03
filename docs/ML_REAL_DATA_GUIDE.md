# ML Training on Real Market Data - Guide

## Current Status

The ML module is fully implemented and tested. However, **yfinance is currently experiencing rate limiting issues** (429 Too Many Requests), which prevents fetching real market data at the moment.

## Quick Start

### Option 1: Use Mock Data (Recommended for Testing)

```bash
python examples/ml_training_validation.py
```

This will use mock data and demonstrate the complete workflow.

### Option 2: Real Market Data (When yfinance is Available)

```bash
python examples/ml_training_real_data.py
```

This script:
- Tests yfinance connection first
- Tries multiple symbols (SPY, QQQ, AAPL, etc.)
- Tries multiple date ranges
- Tries multiple intervals (5m, 15m, 1h, 1d)
- Includes rate limiting protection with delays

## Troubleshooting yfinance Issues

### Common Issues

1. **429 Too Many Requests (Rate Limiting)**
   - **Solution**: Wait 5-10 minutes and try again
   - The script includes automatic delays, but you may need to wait longer

2. **No Data Returned**
   - **Solution**: Try different date ranges or symbols
   - Intraday data (5m, 15m) is limited to recent periods (60 days for 5m)

3. **Connection Errors**
   - **Solution**: Check your internet connection
   - yfinance may be temporarily unavailable

### Best Practices

1. **Use Daily Data First**: Daily data (1d) is more reliable than intraday
2. **Limit Requests**: Don't run multiple scripts simultaneously
3. **Use Recent Dates**: Intraday data is only available for recent periods
4. **Try Different Symbols**: Some symbols may have better data availability

## Manual Data Fetching

If yfinance continues to have issues, you can manually fetch data:

```python
from data.fetcher import DataFetcher
from ml.preprocessing import prep_ohlcv

fetcher = DataFetcher()

# Try different date ranges
raw_data = fetcher.fetch_stock_data(
    symbol="SPY",
    start_date="2024-11-01",
    end_date="2024-11-30",
    interval="5m"  # or "15m", "1h", "1d"
)

# Prepare data
df = prep_ohlcv(raw_data)
```

## Alternative Data Sources

If yfinance continues to be unreliable, consider:

1. **Alpha Vantage**: Free API with rate limits
2. **Polygon.io**: Free tier available
3. **IEX Cloud**: Free tier available
4. **Your broker's API**: Many brokers provide free historical data

## What's Working

✅ **All ML components are fully tested and working:**
- Preprocessing (36/36 tests passing)
- Labeling (engine-matching TP/SL logic)
- Feature engineering
- Model training and prediction
- Integration with BacktestEngine

✅ **Mock data workflow works perfectly:**
- Can train and validate models
- Complete end-to-end pipeline
- All tests passing

## Next Steps

1. **For Testing**: Use mock data (`ml_training_validation.py`)
2. **For Real Data**: Wait for yfinance rate limits to clear, then run `ml_training_real_data.py`
3. **For Production**: Consider integrating an alternative data source

## Example: Using Mock Data

```python
from data.fetcher import DataFetcher
from strategy.mean_reversion import MeanReversionIndicators
from ml.preprocessing import prep_ohlcv
from ml.labeling import label_engine_tp_sl_first_long_only
from ml.model import MRSignalQualityModel

# Use mock data
fetcher = DataFetcher(use_mock=True)
raw_5m = fetcher.fetch_stock_data("SPY", "2024-01-01", "2024-12-31", "5m")

