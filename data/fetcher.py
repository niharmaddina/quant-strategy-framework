"""
Data fetching utilities for stock and options data.
Uses yfinance for free market data.
Supports both real API calls and mock data for testing.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


class DataFetcher:
    """
    Fetches stock and options data using yfinance.
    
    Can be configured to use mock data for testing by setting use_mock=True
    or by providing a mock_data_func.
    
    Includes batching and rate limiting to avoid API throttling.
    """
    
    def __init__(
        self,
        use_mock: bool = False,
        mock_data_func: Optional[callable] = None,
        batch_delay: float = 1.0,
        rate_limit_delay: float = 5.0
    ):
        """
        Initialize data fetcher.
        
        Args:
            use_mock: If True, use mock data instead of real API calls
            mock_data_func: Optional function to generate mock data.
                          Signature: (symbol, start_date, end_date) -> pd.DataFrame
            batch_delay: Delay between batched requests in seconds (default: 1.0)
            rate_limit_delay: Delay when rate limited in seconds (default: 5.0)
        """
        self.use_mock = use_mock
        self.mock_data_func = mock_data_func
        self.batch_delay = batch_delay
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
    
    def _generate_mock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate mock data for testing.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with mock OHLCV data
        """
        if self.mock_data_func:
            return self.mock_data_func(symbol, start_date, end_date)
        
        # Default mock data generation
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5][:252]  # Trading days only
        
        import numpy as np
        np.random.seed(hash(symbol) % 1000)  # Different seed per symbol
        n = len(dates)
        returns = np.random.randn(n) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(n) * 0.5,
            'high': prices + abs(np.random.randn(n) * 1.5),
            'low': prices - abs(np.random.randn(n) * 1.5),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        
        data['high'] = data[['high', 'close']].max(axis=1)
        data['low'] = data[['low', 'close']].min(axis=1)
        data.index.name = 'date'
        return data
    
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data with support for granular intraday intervals.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval. Supported:
                - Daily: "1d", "5d", "1wk", "1mo"
                - Intraday: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"
                - Note: Intraday data limited to ~60 days for 1m, ~730 days for others
        
        Returns:
            DataFrame with OHLCV data and datetime index
        
        Raises:
            ValueError: If dates are invalid or data is empty
            ConnectionError: If network request fails
        """
        # Validate inputs first (before generating mock data)
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        # Use mock data if configured
        if self.use_mock:
            data = self._generate_mock_data(symbol, start_date, end_date)
            # Validate data quality
            self._validate_data_quality(data, symbol)
            return data
        
        try:
            # Validate date formats
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
        
        # Validate interval
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", 
                          "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {interval}. Valid: {valid_intervals}")
        
        # Check intraday data limitations
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
        if is_intraday:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days_diff = (end_dt - start_dt).days
            
            # yfinance limitations for intraday data
            if interval == "1m" and days_diff > 7:
                raise ValueError("1m interval limited to 7 days of data")
            elif interval in ["2m", "5m"] and days_diff > 60:
                raise ValueError(f"{interval} interval limited to 60 days of data")
            elif interval in ["15m", "30m", "60m", "90m", "1h"] and days_diff > 730:
                raise ValueError(f"{interval} interval limited to 730 days of data")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {symbol}: {e}")
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        # Standardize column names
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # For intraday data, keep datetime index; for daily, convert to date
        if is_intraday:
            # Keep datetime index for intraday
            data.index.name = 'datetime'
            data = data.reset_index()
            # Rename to 'date' for consistency
            if 'datetime' in data.columns:
                data = data.rename(columns={'datetime': 'date'})
        else:
            # Daily data: convert to date
            data.index.name = 'date'
            data = data.reset_index()
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date']).dt.date
        
        # Validate data quality
        self._validate_data_quality(data, symbol)
        
        return data
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str):
        """
        Validate data quality and log warnings.
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging
        """
        issues = []
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            for col, count in missing_counts.items():
                if count > 0:
                    issues.append(f"{col}: {count} missing values")
        
        # Check for zero/negative prices
        if 'close' in data.columns:
            zero_prices = (data['close'] <= 0).sum()
            if zero_prices > 0:
                issues.append(f"close: {zero_prices} zero/negative prices")
        
        # Check for zero/negative volumes
        if 'volume' in data.columns:
            zero_volumes = (data['volume'] <= 0).sum()
            if zero_volumes > 0:
                issues.append(f"volume: {zero_volumes} zero/negative volumes")
        
        # Check for duplicate dates
        if 'date' in data.columns:
            duplicates = data['date'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"date: {duplicates} duplicate dates")
        
        # Log warnings if issues found
        if issues:
            print(f"Warning: Data quality issues for {symbol}: {', '.join(issues)}")
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get summary statistics for data.
        
        Args:
            data: DataFrame with price data
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_days': len(data),
            'date_range': None,
            'price_range': None,
            'avg_volume': None,
            'missing_data': {}
        }
        
        if 'date' in data.columns:
            summary['date_range'] = {
                'start': str(data['date'].min()),
                'end': str(data['date'].max())
            }
        
        if 'close' in data.columns:
            summary['price_range'] = {
                'min': float(data['close'].min()),
                'max': float(data['close'].max()),
                'mean': float(data['close'].mean())
            }
        
        if 'volume' in data.columns:
            summary['avg_volume'] = float(data['volume'].mean())
        
        # Count missing values
        missing = data.isnull().sum()
        summary['missing_data'] = {col: int(count) for col, count in missing.items() if count > 0}
        
        return summary
    
    def fetch_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None
    ) -> Dict:
        """
        Fetch options chain data.
        
        Args:
            symbol: Stock ticker symbol
            expiration_date: Optional expiration date (YYYY-MM-DD)
        
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        
        Raises:
            ValueError: If symbol is invalid or expiration date format is wrong
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if expiration_date:
            try:
                datetime.strptime(expiration_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid expiration_date format. Use YYYY-MM-DD, got: {expiration_date}")
        
        if self.use_mock:
            # Return empty mock options data
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiration': None}
        
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiration': None}
            
            # Use specified expiration or nearest
            if expiration_date:
                exp_date = min(expirations, key=lambda x: abs(
                    (datetime.strptime(x, "%Y-%m-%d") - 
                     datetime.strptime(expiration_date, "%Y-%m-%d")).days
                ))
            else:
                exp_date = expirations[0]
            
            opt_chain = ticker.option_chain(exp_date)
            
            return {
                'calls': opt_chain.calls,
                'puts': opt_chain.puts,
                'expiration': exp_date
            }
        except Exception as e:
            # Return empty DataFrames instead of printing error
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiration': None, 'error': str(e)}
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        batch_size: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with batching and rate limiting.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (default: "1d")
            batch_size: Number of symbols to fetch in each batch (default: 5)
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict: Dict[str, pd.DataFrame] = {}

        # Validate interval (same set as fetch_stock_data)
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
                          "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {interval}. Valid: {valid_intervals}")

        # Use mock data if configured
        if self.use_mock:
            for symbol in symbols:
                try:
                    data = self._generate_mock_data(symbol, start_date, end_date)
                    self._validate_data_quality(data, symbol)
                    data_dict[symbol] = data
                except Exception:
                    data_dict[symbol] = pd.DataFrame()
            return data_dict

        # yfinance batching: one download call per batch (true batching)
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]

        def _standardize_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
            """Standardize yfinance output to our expected OHLCV schema with a 'date' column."""
            if df is None or df.empty:
                return pd.DataFrame()

            out = df.copy()

            # Standardize column names
            out.columns = [str(c).lower().replace(' ', '_') for c in out.columns]

            # yfinance commonly returns: open/high/low/close/adj_close/volume
            # Ensure we have the expected close column name
            if 'adj_close' in out.columns and 'close' not in out.columns:
                out = out.rename(columns={'adj_close': 'close'})

            out.index.name = 'date'
            out = out.reset_index()

            # Align with existing daily behavior: date as python date for daily bars
            if 'date' in out.columns:
                if is_intraday:
                    # keep full datetime-like values
                    out['date'] = pd.to_datetime(out['date'])
                else:
                    out['date'] = pd.to_datetime(out['date']).dt.date

            self._validate_data_quality(out, symbol)
            return out

        total_batches = (len(symbols) + batch_size - 1) // batch_size if batch_size else 1
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            print(f"Fetching batch {i//batch_size + 1}/{total_batches} ({len(batch)} symbols)...")

            try:
                # Use group_by='ticker' to make multi-ticker parsing easier
                raw = yf.download(
                    tickers=" ".join(batch),
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    # Yahoo throttles aggressively; threaded downloads can trigger blocks
                    threads=False,
                    progress=False,
                )
            except Exception as e:
                # If the batch fails, fall back to per-symbol fetch_stock_data (slower but resilient)
                print(f"  ⚠ Batch download failed ({e}); falling back to per-symbol fetch for this batch...")
                for symbol in batch:
                    try:
                        data_dict[symbol] = self.fetch_stock_data(symbol, start_date, end_date, interval)
                        print(f"  ✓ {symbol}: {len(data_dict[symbol])} bars")
                    except Exception as e2:
                        print(f"  ✗ Error fetching {symbol}: {e2}")
                        data_dict[symbol] = pd.DataFrame()
                if i + batch_size < len(symbols):
                    time.sleep(self.batch_delay)
                continue

            # Parse response
            # If Yahoo throttled us, yfinance may return an empty frame (and log "possibly delisted").
            # Retry once after a cooldown to reduce spurious failures.
            if raw is None or getattr(raw, "empty", False):
                print(f"  ⚠ Empty batch response; waiting {self.rate_limit_delay}s and retrying once...")
                time.sleep(self.rate_limit_delay)
                try:
                    raw = yf.download(
                        tickers=" ".join(batch),
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        group_by="ticker",
                        auto_adjust=False,
                        threads=False,
                        progress=False,
                    )
                except Exception as e:
                    print(f"  ✗ Retry failed ({e}); falling back to per-symbol fetch for this batch...")
                    raw = None

            if raw is None or getattr(raw, "empty", False):
                for symbol in batch:
                    try:
                        data_dict[symbol] = self.fetch_stock_data(symbol, start_date, end_date, interval)
                        print(f"  ✓ {symbol}: {len(data_dict[symbol])} bars")
                    except Exception as e2:
                        print(f"  ✗ Error fetching {symbol}: {e2}")
                        data_dict[symbol] = pd.DataFrame()
                if i + batch_size < len(symbols):
                    time.sleep(self.batch_delay)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                # Expected for multiple tickers
                for symbol in batch:
                    if symbol in raw.columns.get_level_values(0):
                        sym_df = raw[symbol].dropna(how='all')
                        data_dict[symbol] = _standardize_frame(sym_df, symbol)
                    else:
                        data_dict[symbol] = pd.DataFrame()
                    print(f"  ✓ {symbol}: {len(data_dict[symbol])} bars")
            else:
                # Single ticker batch: raw is a simple OHLCV frame
                symbol = batch[0]
                data_dict[symbol] = _standardize_frame(raw.dropna(how='all'), symbol)
                print(f"  ✓ {symbol}: {len(data_dict[symbol])} bars")

            # Delay between batches (except for the last batch)
            if i + batch_size < len(symbols):
                time.sleep(self.batch_delay)

        return data_dict
    
    def fetch_batch(
        self,
        requests: List[Tuple[str, str, str, str]],
        batch_size: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch data for multiple symbol/date/interval combinations.
        
        Args:
            requests: List of tuples (symbol, start_date, end_date, interval)
            batch_size: Number of requests to process in each batch (default: 5)
        
        Returns:
            Dictionary mapping "{symbol}_{start_date}_{end_date}_{interval}" to DataFrame
        
        Example:
            requests = [
                ("SPY", "2024-01-01", "2024-12-31", "1d"),
                ("QQQ", "2024-01-01", "2024-12-31", "1d"),
            ]
            data = fetcher.fetch_batch(requests)
        """
        data_dict = {}
        
        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(requests) + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} requests)...")
            
            for symbol, start_date, end_date, interval in batch:
                key = f"{symbol}_{start_date}_{end_date}_{interval}"
                try:
                    data = self.fetch_stock_data(symbol, start_date, end_date, interval)
                    data_dict[key] = data
                    print(f"  ✓ {symbol} ({interval}): {len(data)} bars")
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        print(f"  ⚠ Rate limited, waiting {self.rate_limit_delay}s...")
                        time.sleep(self.rate_limit_delay)
                        # Retry once
                        try:
                            data = self.fetch_stock_data(symbol, start_date, end_date, interval)
                            data_dict[key] = data
                            print(f"  ✓ {symbol} ({interval}): {len(data)} bars (after retry)")
                        except Exception as e2:
                            print(f"  ✗ Failed after retry: {e2}")
                            data_dict[key] = pd.DataFrame()
                    else:
                        print(f"  ✗ Error: {e}")
                        data_dict[key] = pd.DataFrame()
            
            # Delay between batches (except for the last batch)
            if i + batch_size < len(requests):
                time.sleep(self.batch_delay)
        
        return data_dict
