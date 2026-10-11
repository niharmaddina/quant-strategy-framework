"""
Preprocessing utilities for OHLCV data.

Handles data preparation, resampling, and format standardization
for intraday data from yfinance.
"""

import pandas as pd
from typing import Optional


def prep_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare OHLCV DataFrame for analysis.
    
    Standardizes column names and sets datetime index.
    Works with DataFetcher output format (date column, OHLCV lowercase).
    
    Args:
        df: DataFrame with 'date' column and OHLCV columns
        
    Returns:
        DataFrame with datetime index and standardized columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = df.copy()
    
    # Handle case where 'date' is both index and column
    if df.index.name == 'date' and 'date' in df.columns:
        # Remove date column, use index
        df = df.drop(columns=['date'])
    
    # Convert date to datetime if it's not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')
    elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
        # Already has datetime index
        df = df.sort_index()
    else:
        raise ValueError("DataFrame must have 'date' column or datetime index")
    
    # Ensure standard OHLCV columns exist (lowercase)
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure volume exists (optional but preferred)
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    
    return df


def resample_ohlcv(
    df: pd.DataFrame,
    target_interval: str,
    source_interval: Optional[str] = None
) -> pd.DataFrame:
    """
    Resample OHLCV data to target interval.
    
    Args:
        df: DataFrame with datetime index and OHLCV columns
        target_interval: Target interval (e.g., '5min', '15min', '1H', '1D')
        source_interval: Optional source interval for validation
        
    Returns:
        Resampled DataFrame
        
    Examples:
        >>> df_5m = resample_ohlcv(df_1m, '5min')  # 1m to 5m
        >>> df_15m = resample_ohlcv(df_5m, '15min')  # 5m to 15m
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")
    
    # Map common interval strings to pandas frequency
    interval_map = {
        '1m': '1min',
        '2m': '2min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '60m': '60min',
        '1h': '1H',
        '1d': '1D',
    }
    
    # Convert target_interval if needed
    if target_interval in interval_map:
        target_interval = interval_map[target_interval]
    
    # Resample OHLCV properly
    resampled = pd.DataFrame(index=df.index)
    
    # Open: first value in period
    resampled['open'] = df['open'].resample(target_interval).first()
    
    # High: maximum value in period
    resampled['high'] = df['high'].resample(target_interval).max()
    
    # Low: minimum value in period
    resampled['low'] = df['low'].resample(target_interval).min()
    
    # Close: last value in period
    resampled['close'] = df['close'].resample(target_interval).last()
    
    # Volume: sum of volume in period
    if 'volume' in df.columns:
        resampled['volume'] = df['volume'].resample(target_interval).sum()
    else:
        resampled['volume'] = 0.0
    
    # Drop any rows with NaN (incomplete periods)
    resampled = resampled.dropna()
    
    return resampled

