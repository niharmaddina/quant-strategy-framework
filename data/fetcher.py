"""
Data fetching utilities for stock and options data.
Uses yfinance for free market data.
Supports both real API calls and mock data for testing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


class DataFetcher:
    """
    Fetches stock and options data using yfinance.
    
    Can be configured to use mock data for testing by setting use_mock=True
    or by providing a mock_data_func.
    """
    
    def __init__(self, use_mock: bool = False, mock_data_func: Optional[callable] = None):
        """
        Initialize data fetcher.
        
        Args:
            use_mock: If True, use mock data instead of real API calls
            mock_data_func: Optional function to generate mock data.
                          Signature: (symbol, start_date, end_date) -> pd.DataFrame
        """
        self.use_mock = use_mock
        self.mock_data_func = mock_data_func
    
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
        Fetch historical stock data.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            DataFrame with OHLCV data
        
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
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {symbol}: {e}")
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        # Standardize column names
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        data.index.name = 'date'
        data = data.reset_index()
        
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
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data_dict = {}
        for symbol in symbols:
            try:
                data_dict[symbol] = self.fetch_stock_data(
                    symbol, start_date, end_date, interval
                )
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return data_dict
