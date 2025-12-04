"""
Technical indicators for swing trading strategy.
Generates signals for manual options trading.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class SwingTradingIndicators:
    """
    Technical indicators optimized for swing trading.
    Generates signals that traders can use to make manual options decisions.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.
        
        Args:
            data: DataFrame with columns: date, open, high, low, close, volume
        """
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("DataFrame is empty")
        
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        """Calculate all indicators."""
        df = self.data.copy()
        
        # Price-based indicators
        self._calculate_moving_averages(df)
        self._calculate_rsi(df)
        self._calculate_macd(df)
        self._calculate_bollinger_bands(df)
        self._calculate_stochastic(df)
        self._calculate_atr(df)
        
        # Generate signals
        self._generate_signals()
    
    def _calculate_moving_averages(self, df: pd.DataFrame):
        """Calculate moving averages."""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        self.data = df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14):
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50.0)  # Default to neutral RSI when loss is zero
        self.data = df
    
    def _calculate_macd(self, df: pd.DataFrame):
        """Calculate MACD."""
        if 'ema_12' not in df.columns:
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        if 'ema_26' not in df.columns:
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        self.data = df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands."""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # Avoid division by zero
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        denominator = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = (df['close'] - df['bb_lower']) / denominator
        df['bb_width'] = df['bb_width'].fillna(0.0)
        df['bb_position'] = df['bb_position'].fillna(0.5)  # Default to middle when range is zero
        self.data = df
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Avoid division by zero when high_max == low_min
        denominator = (high_max - low_min).replace(0, np.nan)
        df['stoch_k'] = 100 * ((df['close'] - low_min) / denominator)
        df['stoch_k'] = df['stoch_k'].fillna(50.0)  # Default to neutral when range is zero
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        self.data = df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        self.data = df
    
    def _generate_signals(self):
        """Generate trading signals based on indicators."""
        df = self.data.copy()
        signals = pd.DataFrame(index=df.index)
        
        # Signal 1: RSI Oversold/Overbought
        signals['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        signals['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # Signal 2: MACD Crossover
        signals['macd_bullish'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        signals['macd_bearish'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)
        
        # Signal 3: Moving Average Crossover
        signals['ma_bullish'] = (
            (df['sma_20'] > df['sma_50']) & 
            (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        ).astype(int)
        signals['ma_bearish'] = (
            (df['sma_20'] < df['sma_50']) & 
            (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
        ).astype(int)
        
        # Signal 4: Bollinger Band Breakout
        signals['bb_oversold'] = (df['close'] < df['bb_lower']).astype(int)
        signals['bb_overbought'] = (df['close'] > df['bb_upper']).astype(int)
        
        # Signal 5: Stochastic
        signals['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        signals['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        # Combined signal strength (0-5)
        signals['bullish_strength'] = (
            signals['rsi_oversold'] +
            signals['macd_bullish'] +
            signals['ma_bullish'] +
            signals['bb_oversold'] +
            signals['stoch_oversold']
        )
        
        signals['bearish_strength'] = (
            signals['rsi_overbought'] +
            signals['macd_bearish'] +
            signals['ma_bearish'] +
            signals['bb_overbought'] +
            signals['stoch_overbought']
        )
        
        # Overall signal
        signals['signal'] = 0
        signals.loc[signals['bullish_strength'] >= 3, 'signal'] = 1  # Buy signal
        signals.loc[signals['bearish_strength'] >= 3, 'signal'] = -1  # Sell signal
        
        # Signal confidence (0-1)
        signals['confidence'] = (
            np.maximum(signals['bullish_strength'], signals['bearish_strength']) / 5.0
        )
        
        self.signals = signals
    
    def get_current_signals(self) -> Dict:
        """
        Get the most recent trading signals.
        
        Returns:
            Dictionary with current signal information
        """
        if self.signals.empty:
            return {}
        
        latest = self.signals.iloc[-1]
        latest_data = self.data.iloc[-1]
        
        signal_type = "NEUTRAL"
        if latest['signal'] == 1:
            signal_type = "BULLISH"
        elif latest['signal'] == -1:
            signal_type = "BEARISH"
        
        return {
            'signal': signal_type,
            'signal_value': int(latest['signal']),
            'confidence': float(latest['confidence']),
            'bullish_strength': int(latest['bullish_strength']),
            'bearish_strength': int(latest['bearish_strength']),
            'price': float(latest_data['close']),
            'rsi': float(latest_data['rsi']) if not pd.isna(latest_data['rsi']) else None,
            'macd': float(latest_data['macd']) if not pd.isna(latest_data['macd']) else None,
            'date': latest_data.get('date', self.data.index[-1])
        }
    
    def get_indicators_df(self) -> pd.DataFrame:
        """Get DataFrame with all indicators."""
        return self.data.copy()
    
    def get_signals_df(self) -> pd.DataFrame:
        """Get DataFrame with all signals."""
        return self.signals.copy()
    
    def get_full_analysis(self) -> pd.DataFrame:
        """Get combined DataFrame with data, indicators, and signals."""
        return pd.concat([self.data, self.signals], axis=1)

