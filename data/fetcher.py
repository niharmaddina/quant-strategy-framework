"""
Data fetching utilities for stock and options data.
Uses yfinance for free market data.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timedelta


class DataFetcher:
    """Fetches stock and options data using yfinance."""
    
    def __init__(self):
        self.cache = {}
    
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
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        # Standardize column names
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        data.index.name = 'date'
        data = data.reset_index()
        
        return data
    
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
        """
        ticker = yf.Ticker(symbol)
        
        try:
            expirations = ticker.options
            if not expirations:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            
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
            print(f"Error fetching options for {symbol}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
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

