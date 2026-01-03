# API Batching + Rate Limit Guide (yfinance)

## What “batching” means in this repo

`DataFetcher.fetch_multiple_symbols()` batches **symbols into groups** and fetches each group using **one `yfinance.download()` call** (instead of one request per symbol). This is the only practical way to reduce request volume with Yahoo.

## Why it matters

- Yahoo/`yfinance` throttles aggressively.
- When throttled, you often see errors like “`Expecting value ...`” or “`possibly delisted; no timezone found`” even for valid tickers.
- Batching reduces the number of HTTP calls, but **does not guarantee** you won’t be throttled; it just makes it much less likely.

## Recommended usage

### Fetch multiple symbols (same date range)

## Basic Usage

### 1. Fetch Multiple Symbols (Same Date Range)

```python
from data.fetcher import DataFetcher

fetcher = DataFetcher(
    batch_delay=1.0,      # 1 second between batches
    rate_limit_delay=10.0  # cooldown before a single retry on empty responses
)

# Fetch multiple symbols with the same date range
symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
data_dict = fetcher.fetch_multiple_symbols(
    symbols=symbols,
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d",
    batch_size=5  # Process 5 symbols per batch
)

# Access individual symbol data
spy_data = data_dict["SPY"]
qqq_data = data_dict["QQQ"]
```

**Output:**
```
Fetching batch 1/1 (5 symbols)...
  ✓ SPY: 252 bars
  ✓ QQQ: 252 bars
  ✓ AAPL: 252 bars
  ✓ MSFT: 252 bars
  ✓ TSLA: 252 bars
```

### 2. Batch Fetch with Different Parameters

Use `fetch_batch()` for different symbols, dates, or intervals:

```python
from data.fetcher import DataFetcher

fetcher = DataFetcher()

# Prepare batch requests: (symbol, start_date, end_date, interval)
requests = [
    ("SPY", "2024-01-01", "2024-12-31", "1d"),
    ("QQQ", "2024-01-01", "2024-12-31", "1d"),
    ("SPY", "2024-11-01", "2024-11-30", "5m"),  # Different interval
    ("AAPL", "2024-10-01", "2024-10-31", "15m"),
]

# Fetch all in batches
results = fetcher.fetch_batch(requests, batch_size=3)

# Access results by key: "{symbol}_{start_date}_{end_date}_{interval}"
spy_daily = results["SPY_2024-01-01_2024-12-31_1d"]
spy_5m = results["SPY_2024-11-01_2024-11-30_5m"]
```

**Output:**
```
Processing batch 1/2 (3 requests)...
  ✓ SPY (1d): 252 bars
  ✓ QQQ (1d): 252 bars
  ✓ SPY (5m): 1320 bars
Processing batch 2/2 (1 requests)...
  ✓ AAPL (15m): 440 bars
```

## Configuration knobs that actually help

### DataFetcher Initialization

```python
fetcher = DataFetcher(
    use_mock=False,           # Use real API (False) or mock data (True)
    batch_delay=1.0,          # Seconds between batches (default: 1.0)
    rate_limit_delay=5.0      # Seconds to wait when rate limited (default: 5.0)
)
```

**Parameters:**
- `batch_delay`: delay between batches
- `rate_limit_delay`: cooldown when a batch returns empty (often throttling)

### Batch Size

Control how many requests are processed per batch:

```python
# Small batches (safer, slower)
data = fetcher.fetch_multiple_symbols(symbols, start, end, batch_size=2)

# Large batches (faster, riskier)
data = fetcher.fetch_multiple_symbols(symbols, start, end, batch_size=10)
```

**Recommendations:**
- **Conservative**: `batch_size=2-3` (safer for rate limits)
- **Balanced**: `batch_size=5` (default, good balance)
- **Aggressive**: `batch_size=10+` (faster but may hit rate limits)

## Rate limiting reality (important)

Yahoo rate limiting is not always a clean “429” response. In practice, `yfinance` can return empty frames or JSON parsing errors.

What we do:
- If a **batched download returns empty**, we wait `rate_limit_delay` seconds and retry once.
- If it’s still empty, we fall back to per-symbol fetch for that batch (best-effort).

What you should do:
- Keep batches modest (`batch_size=3–10`)
- Add cooldowns (`batch_delay`, `rate_limit_delay`)
- Avoid running multiple scripts in parallel
- Prefer **frozen universe files** for backtests so you’re not doing extra metadata calls

```python
fetcher = DataFetcher(rate_limit_delay=10.0)  # Wait 10s on rate limit

# If rate limited, automatically waits and retries
data = fetcher.fetch_stock_data("SPY", "2024-01-01", "2024-12-31")
```

### Error Handling

Failed requests return empty DataFrames, allowing processing to continue:

```python
results = fetcher.fetch_multiple_symbols(
    ["SPY", "INVALID", "QQQ"],
    "2024-01-01",
    "2024-12-31"
)

# Check for empty DataFrames
for symbol, data in results.items():
    if data.empty:
        print(f"⚠ {symbol}: Failed to fetch")
    else:
        print(f"✓ {symbol}: {len(data)} bars")
```

## Real-World Examples

### Example 1: ML Training Data Collection

```python
from data.fetcher import DataFetcher

fetcher = DataFetcher(batch_delay=2.0, rate_limit_delay=10.0)

# Collect training data for multiple symbols and intervals
symbols = ["SPY", "QQQ", "AAPL"]
intervals = ["5m", "15m", "1h"]
date_ranges = [
    ("2024-11-01", "2024-11-30"),
    ("2024-10-01", "2024-10-31"),
]

# Build batch requests
requests = []
for symbol in symbols:
    for start, end in date_ranges:
        for interval in intervals:
            requests.append((symbol, start, end, interval))

# Fetch all in batches
print(f"Fetching {len(requests)} requests in batches...")
results = fetcher.fetch_batch(requests, batch_size=3)

# Process results
for key, data in results.items():
    if len(data) > 50:  # Minimum data requirement
        print(f"✓ {key}: {len(data)} bars")
```

### Example 2: Universe Construction

```python
from data.fetcher import DataFetcher

fetcher = DataFetcher()

# Fetch data for entire universe
universe = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]

# Batch fetch all at once
all_data = fetcher.fetch_multiple_symbols(
    universe,
    start_date="2024-01-01",
    end_date="2024-12-31",
    interval="1d",
    batch_size=5
)

# Process each symbol
for symbol, data in all_data.items():
    if not data.empty:
        print(f"Processing {symbol}...")
        # Run indicators, backtest, etc.
```

### Main entrypoint note

`main.py` fetches data for multiple symbols via `fetch_multiple_symbols()`. If Yahoo is throttling hard, the safest workflow is:
- reduce universe size temporarily, or
- run fewer symbols at once, or
- wait and retry later

## Best Practices

### 1. **Adjust Delays Based on Usage**

```python
# For frequent use (be more conservative)
fetcher = DataFetcher(batch_delay=2.0, rate_limit_delay=10.0)

# For occasional use (can be more aggressive)
fetcher = DataFetcher(batch_delay=0.5, rate_limit_delay=5.0)
```

### 2. **Use Appropriate Batch Sizes**

```python
# Many symbols: smaller batches
fetcher.fetch_multiple_symbols(50_symbols, start, end, batch_size=3)

# Few symbols: larger batches
fetcher.fetch_multiple_symbols(5_symbols, start, end, batch_size=5)
```

### 3. **Handle Empty Results**

```python
results = fetcher.fetch_batch(requests)

# Filter out failed requests
successful = {
    key: data for key, data in results.items()
    if not data.empty and len(data) > 0
}
```

### 4. **Monitor Rate Limits**

```python
# If you see many rate limit warnings, increase delays
fetcher = DataFetcher(
    batch_delay=3.0,        # Longer delay between batches
    rate_limit_delay=15.0   # Longer wait on rate limit
)
```

## Troubleshooting

### Issue: Still Getting Rate Limited

**Solution**: Increase delays and reduce batch size
```python
fetcher = DataFetcher(batch_delay=3.0, rate_limit_delay=15.0)
results = fetcher.fetch_batch(requests, batch_size=2)
```

### Issue: Batching Too Slow

**Solution**: Reduce delays and increase batch size (if not rate limited)
```python
fetcher = DataFetcher(batch_delay=0.5, rate_limit_delay=5.0)
results = fetcher.fetch_batch(requests, batch_size=10)
```

### Issue: Some Requests Fail Silently

**Solution**: Check for empty DataFrames
```python
results = fetcher.fetch_batch(requests)

failed = [key for key, data in results.items() if data.empty]
if failed:
    print(f"⚠ Failed requests: {failed}")
```

## API Reference

### `fetch_multiple_symbols()`

```python
def fetch_multiple_symbols(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    batch_size: int = 5
) -> Dict[str, pd.DataFrame]
```

Fetches data for multiple symbols with the same date range and interval.

### `fetch_batch()`

```python
def fetch_batch(
    requests: List[Tuple[str, str, str, str]],
    batch_size: int = 5
) -> Dict[str, pd.DataFrame]
```

Fetches data for multiple symbol/date/interval combinations.

**Request format**: `(symbol, start_date, end_date, interval)`

**Return keys**: `"{symbol}_{start_date}_{end_date}_{interval}"`

## Summary

✅ **Use batching** to fetch multiple symbols efficiently  
✅ **Configure delays** based on your usage patterns  
✅ **Handle errors** by checking for empty DataFrames  
✅ **Monitor rate limits** and adjust delays as needed  

The batching system is now integrated throughout the codebase, making API calls more efficient and reliable!

