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
        self._calculate_market_regime(df)
        
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
        
        # Calculate volatility regime (high/low volatility)
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['volatility_regime'] = (df['atr'] > df['atr_ma'] * 1.2).astype(int)  # High volatility
        
        self.data = df
    
    def _calculate_market_regime(self, df: pd.DataFrame):
        """Detect market regime: trending vs ranging."""
        # Calculate ADX (Average Directional Index) for trend strength
        # Simplified ADX calculation
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Smooth the DMs
        plus_dm_smooth = plus_dm.rolling(window=14).mean()
        minus_dm_smooth = minus_dm.rolling(window=14).mean()
        
        # Calculate True Range (already have ATR)
        tr = df['atr'] * 14  # Approximate TR from ATR
        
        # Calculate DI+ and DI-
        di_plus = 100 * (plus_dm_smooth / tr.replace(0, np.nan))
        di_minus = 100 * (minus_dm_smooth / tr.replace(0, np.nan))
        
        # Simplified ADX (trend strength)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
        df['adx'] = dx.rolling(window=14).mean()
        df['adx'] = df['adx'].fillna(0)
        
        # Market regime: trending (ADX > 25) vs ranging (ADX < 25)
        df['trending_market'] = (df['adx'] > 25).astype(int)
        df['ranging_market'] = (df['adx'] < 20).astype(int)
        
        self.data = df
    
    def _generate_signals(self):
        """Generate trading signals based on indicators with improved logic."""
        df = self.data.copy()
        signals = pd.DataFrame(index=df.index)
        
        # ========================================================================
        # TREND FILTERS (Critical for swing trading success)
        # ========================================================================
        # Only buy in uptrends, only sell in downtrends
        signals['uptrend'] = (
            (df['close'] > df['sma_20']) & 
            (df['sma_20'] > df['sma_50']) & 
            (df['sma_50'] > df['sma_200'])
        ).astype(int)
        
        signals['downtrend'] = (
            (df['close'] < df['sma_20']) & 
            (df['sma_20'] < df['sma_50']) & 
            (df['sma_50'] < df['sma_200'])
        ).astype(int)
        
        # Momentum filter: price should be above/below key MAs
        signals['bullish_momentum'] = (
            (df['close'] > df['ema_12']) & 
            (df['ema_12'] > df['ema_26'])
        ).astype(int)
        
        signals['bearish_momentum'] = (
            (df['close'] < df['ema_12']) & 
            (df['ema_12'] < df['ema_26'])
        ).astype(int)
        
        # ========================================================================
        # VOLUME CONFIRMATION (Important for signal quality)
        # ========================================================================
        # Calculate volume moving average
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            signals['volume_confirmation'] = (df['volume'] > df['volume_sma'] * 1.2).astype(int)
        else:
            signals['volume_confirmation'] = 1  # No volume data, assume confirmation
        
        # ========================================================================
        # INDICATOR SIGNALS (Improved logic)
        # ========================================================================
        # Signal 1: RSI - Use for pullback entries in uptrends
        # Buy when RSI is oversold BUT in an uptrend (pullback buy)
        signals['rsi_oversold'] = (df['rsi'] < 35).astype(int)  # Slightly less strict
        signals['rsi_overbought'] = (df['rsi'] > 65).astype(int)  # Slightly less strict
        
        # Signal 2: MACD - Momentum confirmation
        signals['macd_bullish'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd_histogram'] > 0)  # Histogram positive = bullish momentum
        ).astype(int)
        signals['macd_bearish'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd_histogram'] < 0)  # Histogram negative = bearish momentum
        ).astype(int)
        
        # Signal 3: Moving Average - Trend confirmation
        signals['ma_bullish'] = (
            (df['sma_20'] > df['sma_50']) & 
            (df['close'] > df['sma_20'])  # Price above short-term MA
        ).astype(int)
        signals['ma_bearish'] = (
            (df['sma_20'] < df['sma_50']) & 
            (df['close'] < df['sma_20'])  # Price below short-term MA
        ).astype(int)
        
        # Signal 4: Bollinger Bands - Mean reversion in trends
        # Buy when price touches lower band in uptrend (pullback)
        signals['bb_oversold'] = (
            (df['close'] <= df['bb_lower']) & 
            signals['uptrend'] == 1  # Only in uptrends
        ).astype(int)
        # Sell when price touches upper band in downtrend
        signals['bb_overbought'] = (
            (df['close'] >= df['bb_upper']) & 
            signals['downtrend'] == 1  # Only in downtrends
        ).astype(int)
        
        # Signal 5: Stochastic - Entry timing
        signals['stoch_oversold'] = (
            (df['stoch_k'] < 25) & 
            (df['stoch_d'] < 25) & 
            signals['uptrend'] == 1  # Only in uptrends
        ).astype(int)
        signals['stoch_overbought'] = (
            (df['stoch_k'] > 75) & 
            (df['stoch_d'] > 75) & 
            signals['downtrend'] == 1  # Only in downtrends
        ).astype(int)
        
        # ========================================================================
        # MARKET REGIME FILTERS
        # ========================================================================
        # Prefer trending markets for swing trades (better directional moves)
        # In ranging markets, be more selective
        if 'trending_market' in df.columns:
            signals['market_regime_ok'] = (
                df['trending_market'] == 1  # Prefer trending markets
            ).astype(int)
        else:
            signals['market_regime_ok'] = 1  # Default to OK if not calculated
        
        # Volatility filter: avoid extremely high volatility (too risky)
        # and very low volatility (no movement)
        if 'volatility_regime' in df.columns:
            signals['volatility_ok'] = (
                (df['volatility_regime'] == 0) |  # Normal or low volatility
                (df['atr_pct'] < 5.0)  # Or ATR < 5% (manageable)
            ).astype(int)
        else:
            signals['volatility_ok'] = 1
        
        # ========================================================================
        # IMPROVED SIGNAL STRENGTH CALCULATION
        # ========================================================================
        # Base indicator strength (0-5)
        base_bullish = (
            signals['rsi_oversold'] +
            signals['macd_bullish'] +
            signals['ma_bullish'] +
            signals['bb_oversold'] +
            signals['stoch_oversold']
        )
        
        base_bearish = (
            signals['rsi_overbought'] +
            signals['macd_bearish'] +
            signals['ma_bearish'] +
            signals['bb_overbought'] +
            signals['stoch_overbought']
        )
        
        # Additional quality filters
        # Price action quality: price should be making higher lows in uptrend
        price_action_bullish = (
            (df['close'] > df['close'].shift(5)) &  # Higher than 5 days ago
            (df['low'] > df['low'].rolling(5).min().shift(1))  # Higher lows
        ).astype(int)
        
        price_action_bearish = (
            (df['close'] < df['close'].shift(5)) &  # Lower than 5 days ago
            (df['high'] < df['high'].rolling(5).max().shift(1))  # Lower highs
        ).astype(int)
        
        # Apply trend filter multiplier (critical!)
        # Only count signals that align with trend AND have quality price action
        signals['bullish_strength'] = (
            base_bullish * signals['uptrend'] * signals['bullish_momentum'] *
            signals['market_regime_ok'] * signals['volatility_ok'] *
            price_action_bullish +
            (signals['volume_confirmation'] * 0.5)  # Bonus for volume
        )
        
        signals['bearish_strength'] = (
            base_bearish * signals['downtrend'] * signals['bearish_momentum'] *
            signals['market_regime_ok'] * signals['volatility_ok'] *
            price_action_bearish +
            (signals['volume_confirmation'] * 0.5)  # Bonus for volume
        )
        
        # ========================================================================
        # SIGNAL GENERATION WITH SOPHISTICATED LOGIC
        # ========================================================================
        # Multi-tier signal generation for better quality
        
        signals['signal'] = 0
        
        # ========================================================================
        # BUY SIGNAL LOGIC (Multiple confirmation required)
        # ========================================================================
        # Tier 1: Strong signals (3+ indicators, perfect conditions)
        strong_buy_condition = (
            (signals['bullish_strength'] >= 3.0) &
            (signals['uptrend'] == 1) &
            (signals['bullish_momentum'] == 1) &
            (signals['market_regime_ok'] == 1) &
            (signals['volatility_ok'] == 1) &
            (price_action_bullish == 1)
        )
        
        # Tier 2: Good signals (2+ indicators, strong trend)
        strong_uptrend = (
            (signals['uptrend'] == 1) & 
            (signals['bullish_momentum'] == 1) &
            (df['close'] > df['sma_200']) &  # Above long-term trend
            (df['sma_20'] > df['sma_50'] * 1.02)  # MAs well separated
        )
        
        good_buy_condition = (
            (signals['bullish_strength'] >= 2.0) &
            strong_uptrend &
            (signals['market_regime_ok'] == 1) &
            (
                (signals['rsi_oversold'] == 1) |  # RSI pullback
                (signals['macd_bullish'] == 1)  # MACD confirmation
            )
        )
        
        # Tier 3: Pullback entries (weaker but in very strong trends)
        very_strong_uptrend = (
            strong_uptrend &
            (df['close'] > df['sma_200'] * 1.05) &  # Well above 200 SMA
            (df['sma_50'] > df['sma_200'])  # All MAs aligned
        )
        
        pullback_buy_condition = (
            (signals['bullish_strength'] >= 1.5) &
            very_strong_uptrend &
            (signals['rsi_oversold'] == 1) &  # Must have RSI pullback
            (signals['bb_oversold'] == 1) &  # And BB touch
            (signals['volatility_ok'] == 1)
        )
        
        # Combine all buy conditions
        buy_condition = strong_buy_condition | good_buy_condition | pullback_buy_condition
        signals.loc[buy_condition, 'signal'] = 1
        
        # ========================================================================
        # SELL SIGNAL LOGIC (Multiple confirmation required)
        # ========================================================================
        # Tier 1: Strong sell signals
        strong_sell_condition = (
            (signals['bearish_strength'] >= 3.0) &
            (signals['downtrend'] == 1) &
            (signals['bearish_momentum'] == 1) &
            (signals['market_regime_ok'] == 1) &
            (price_action_bearish == 1)
        )
        
        # Tier 2: Good sell signals
        strong_downtrend = (
            (signals['downtrend'] == 1) & 
            (signals['bearish_momentum'] == 1) &
            (df['close'] < df['sma_200']) &  # Below long-term trend
            (df['sma_20'] < df['sma_50'] * 0.98)  # MAs well separated
        )
        
        good_sell_condition = (
            (signals['bearish_strength'] >= 2.0) &
            strong_downtrend &
            (signals['market_regime_ok'] == 1) &
            (
                (signals['rsi_overbought'] == 1) |  # RSI overbought
                (signals['macd_bearish'] == 1)  # MACD confirmation
            )
        )
        
        # Combine all sell conditions
        sell_condition = strong_sell_condition | good_sell_condition
        signals.loc[sell_condition, 'signal'] = -1
        
        # If both bullish and bearish signals occur on same day, use the stronger one
        both_signals = (signals['signal'] == 1) & (signals['signal'] == -1)  # This won't happen, but handle edge case
        # Actually, check if we have conflicting signals
        has_buy = signals['signal'] == 1
        has_sell = signals['signal'] == -1
        # If both conditions were true, prefer the stronger one
        # (This is already handled by the conditions above, but keep for safety)
        
        # ========================================================================
        # SIGNAL CONFIDENCE (Sophisticated calculation)
        # ========================================================================
        # Multi-factor confidence score
        max_strength = np.maximum(signals['bullish_strength'], signals['bearish_strength'])
        trend_quality = np.maximum(signals['uptrend'], signals['downtrend'])
        
        # Market regime quality
        regime_quality = signals['market_regime_ok'] * 0.5 + signals['volatility_ok'] * 0.5
        
        # Price action quality
        price_quality = np.maximum(price_action_bullish, price_action_bearish)
        
        # Calculate confidence with weighted factors
        signals['confidence'] = (
            (max_strength / 6.0) * 0.40 +      # Indicator strength (40%)
            (trend_quality * 0.25) +            # Trend alignment (25%)
            (regime_quality * 0.15) +          # Market regime (15%)
            (price_quality * 0.10) +            # Price action (10%)
            (signals['volume_confirmation'] * 0.10)  # Volume (10%)
        )
        
        # Boost confidence for strong signals
        strong_signal_boost = (max_strength >= 3.0).astype(int) * 0.1
        signals['confidence'] = np.minimum(signals['confidence'] + strong_signal_boost, 1.0)
        
        # Reduce confidence for weak signals
        weak_signal_penalty = (max_strength < 1.5).astype(int) * 0.2
        signals['confidence'] = np.maximum(signals['confidence'] - weak_signal_penalty, 0.0)
        
        # Add risk/reward estimate
        if 'atr' in df.columns:
            # Estimate potential reward based on ATR
            atr_multiple = 2.0  # Target 2x ATR for swing trades
            signals['estimated_reward_pct'] = (df['atr'] / df['close']) * 100 * atr_multiple
            signals['estimated_risk_pct'] = 5.0  # 5% stop loss
            signals['risk_reward_ratio'] = signals['estimated_reward_pct'] / signals['estimated_risk_pct']
            
            # Adjust confidence based on risk/reward
            good_rr = (signals['risk_reward_ratio'] >= 1.5).astype(int) * 0.05
            signals['confidence'] = np.minimum(signals['confidence'] + good_rr, 1.0)
        else:
            signals['estimated_reward_pct'] = 0
            signals['estimated_risk_pct'] = 5.0
            signals['risk_reward_ratio'] = 0
        
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

