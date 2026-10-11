"""
Mean reversion strategy indicators optimized for granular intraday data.
Focuses on identifying overbought/oversold conditions and mean reversion opportunities.
"""

from typing import Dict
import numpy as np
import pandas as pd


class MeanReversionIndicators:
    """
    Mean reversion indicators optimized for intraday trading.
    
    Uses Z-score, Bollinger Bands, and price distance from mean to identify
    mean reversion opportunities with granular price data.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback_period: int = 20,
        z_score_threshold: float = 2.0,
        bb_std_dev: float = 2.0
    ):
        """
        Initialize mean reversion indicators.
        
        Args:
            data: DataFrame with columns: date, open, high, low, close, volume
            lookback_period: Period for calculating mean and standard deviation
            z_score_threshold: Z-score threshold for mean reversion signals (default: 2.0)
            bb_std_dev: Standard deviations for Bollinger Bands (default: 2.0)
        """
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("DataFrame is empty")
        
        if len(data) < lookback_period:
            raise ValueError(f"Data length ({len(data)}) must be >= lookback_period ({lookback_period})")
        
        self.data = data.copy()
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.bb_std_dev = bb_std_dev
        self.signals = pd.DataFrame(index=data.index)
        
        # Calculate all indicators
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        """Calculate all mean reversion indicators."""
        df = self.data.copy()
        
        # Core mean reversion indicators
        self._calculate_rolling_statistics(df)
        self._calculate_z_score(df)
        self._calculate_bollinger_bands(df)
        self._calculate_price_distance_from_mean(df)
        self._calculate_mean_reversion_strength(df)
        
        # Additional filters
        self._calculate_volume_profile(df)
        self._calculate_volatility_regime(df)
        
        # Generate signals
        self._generate_signals()
    
    def _calculate_rolling_statistics(self, df: pd.DataFrame):
        """Calculate rolling mean and standard deviation."""
        df['mean'] = df['close'].rolling(window=self.lookback_period).mean()
        df['std'] = df['close'].rolling(window=self.lookback_period).std()
        df['median'] = df['close'].rolling(window=self.lookback_period).median()
        
        # Fill initial NaN values with expanding window
        df['mean'] = df['mean'].fillna(df['close'].expanding().mean())
        df['std'] = df['std'].fillna(df['close'].expanding().std())
        df['median'] = df['median'].fillna(df['close'].expanding().median())
        
        self.data = df
    
    def _calculate_z_score(self, df: pd.DataFrame):
        """
        Calculate Z-score: (price - mean) / std.
        
        Z-score > threshold: Overbought (mean reversion sell signal)
        Z-score < -threshold: Oversold (mean reversion buy signal)
        """
        # Avoid division by zero
        denominator = df['std'].replace(0, np.nan)
        df['z_score'] = (df['close'] - df['mean']) / denominator
        df['z_score'] = df['z_score'].fillna(0.0)
        
        # Normalized Z-score (0 to 1 scale)
        # Map Z-score to 0-1 range: -3 to +3 maps to 0 to 1
        df['z_score_normalized'] = (df['z_score'] + 3) / 6
        df['z_score_normalized'] = df['z_score_normalized'].clip(0, 1)
        
        # Z-score percentiles
        df['z_score_percentile'] = df['z_score'].rolling(window=100, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        df['z_score_percentile'] = df['z_score_percentile'].fillna(0.5)
        
        self.data = df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame):
        """Calculate Bollinger Bands for mean reversion."""
        if 'mean' not in df.columns:
            df['mean'] = df['close'].rolling(window=self.lookback_period).mean()
        if 'std' not in df.columns:
            df['std'] = df['close'].rolling(window=self.lookback_period).std()
        
        df['bb_middle'] = df['mean']
        df['bb_upper'] = df['bb_middle'] + (df['std'] * self.bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['std'] * self.bb_std_dev)
        
        # Bollinger Band width (volatility measure)
        denominator = df['bb_middle'].replace(0, np.nan)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / denominator
        df['bb_width'] = df['bb_width'].fillna(0.0)
        
        # Position within bands (0 = lower band, 1 = upper band)
        band_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = (df['close'] - df['bb_lower']) / band_range
        df['bb_position'] = df['bb_position'].fillna(0.5)
        
        # Distance from bands
        df['distance_from_upper'] = (df['close'] - df['bb_upper']) / df['bb_middle']
        df['distance_from_lower'] = (df['bb_lower'] - df['close']) / df['bb_middle']
        
        self.data = df
    
    def _calculate_price_distance_from_mean(self, df: pd.DataFrame):
        """Calculate price distance from mean in percentage terms."""
        denominator = df['mean'].replace(0, np.nan)
        df['distance_from_mean_pct'] = ((df['close'] - df['mean']) / denominator) * 100
        df['distance_from_mean_pct'] = df['distance_from_mean_pct'].fillna(0.0)
        
        # Absolute distance
        df['abs_distance_from_mean_pct'] = df['distance_from_mean_pct'].abs()
        
        # Distance from median
        denominator_median = df['median'].replace(0, np.nan)
        df['distance_from_median_pct'] = ((df['close'] - df['median']) / denominator_median) * 100
        df['distance_from_median_pct'] = df['distance_from_median_pct'].fillna(0.0)
        
        self.data = df
    
    def _calculate_mean_reversion_strength(self, df: pd.DataFrame):
        """
        Calculate mean reversion strength score.
        
        Combines multiple factors:
        - Z-score magnitude
        - Distance from mean
        - Bollinger Band position
        - Price momentum (reversion potential)
        """
        # Base strength from Z-score (higher absolute Z-score = stronger signal)
        z_strength = df['z_score'].abs() / self.z_score_threshold
        z_strength = z_strength.clip(0, 2.0)  # Cap at 2x threshold
        
        # Distance strength (how far from mean)
        distance_strength = df['abs_distance_from_mean_pct'] / 5.0  # Normalize to 5% moves
        distance_strength = distance_strength.clip(0, 2.0)
        
        # Bollinger Band strength
        # Strong signal when price is at or beyond bands
        bb_strength = np.where(
            df['close'] <= df['bb_lower'],
            (df['bb_lower'] - df['close']) / df['std'].replace(0, np.nan),  # Below lower band
            np.where(
                df['close'] >= df['bb_upper'],
                (df['close'] - df['bb_upper']) / df['std'].replace(0, np.nan),  # Above upper band
                0.0  # Between bands
            )
        )
        bb_strength = pd.Series(bb_strength, index=df.index).fillna(0.0)
        
        # Momentum strength (how fast price is moving away from mean)
        # Negative momentum when moving away = stronger reversion potential
        price_change = df['close'].diff()
        mean_change = df['mean'].diff()
        momentum = price_change - mean_change
        
        # If price is above mean and momentum is negative = reversion potential
        # If price is below mean and momentum is positive = reversion potential
        momentum_strength = np.where(
            (df['close'] > df['mean']) & (momentum < 0),
            abs(momentum) / df['std'].replace(0, np.nan),  # Price above mean, moving down
            np.where(
                (df['close'] < df['mean']) & (momentum > 0),
                abs(momentum) / df['std'].replace(0, np.nan),  # Price below mean, moving up
                0.0
            )
        )
        momentum_strength = pd.Series(momentum_strength, index=df.index).fillna(0.0)
        
        # Combined mean reversion strength (0-10 scale)
        df['mean_reversion_strength'] = (
            z_strength * 3.0 +  # Z-score weight: 30%
            distance_strength * 2.0 +  # Distance weight: 20%
            bb_strength * 3.0 +  # BB weight: 30%
            momentum_strength * 2.0  # Momentum weight: 20%
        )
        
        # Normalize to 0-10 scale
        df['mean_reversion_strength'] = df['mean_reversion_strength'].clip(0, 10)
        
        self.data = df
    
    def _calculate_volume_profile(self, df: pd.DataFrame):
        """Calculate volume profile for mean reversion confirmation."""
        if 'volume' in df.columns:
            # Volume moving average
            df['volume_ma'] = df['volume'].rolling(window=self.lookback_period).mean()
            df['volume_ma'] = df['volume_ma'].fillna(df['volume'].expanding().mean())
            
            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            
            # Volume confirmation: higher volume on mean reversion moves
            df['volume_confirmation'] = (df['volume_ratio'] > 1.2).astype(int)
        else:
            df['volume_ma'] = 0
            df['volume_ratio'] = 1.0
            df['volume_confirmation'] = 1  # Assume confirmation if no volume data
        
        self.data = df
    
    def _calculate_volatility_regime(self, df: pd.DataFrame):
        """Calculate volatility regime for mean reversion filtering."""
        if 'std' not in df.columns:
            df['std'] = df['close'].rolling(window=self.lookback_period).std()
        
        # Volatility moving average
        df['volatility_ma'] = df['std'].rolling(window=self.lookback_period * 2).mean()
        df['volatility_ma'] = df['volatility_ma'].fillna(df['std'].expanding().mean())
        
        # Volatility ratio
        df['volatility_ratio'] = df['std'] / df['volatility_ma'].replace(0, np.nan)
        df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)
        
        # Volatility regime: high volatility = better mean reversion opportunities
        df['high_volatility'] = (df['volatility_ratio'] > 1.2).astype(int)
        df['low_volatility'] = (df['volatility_ratio'] < 0.8).astype(int)
        df['normal_volatility'] = (
            (df['volatility_ratio'] >= 0.8) & (df['volatility_ratio'] <= 1.2)
        ).astype(int)
        
        self.data = df
    
    def _generate_signals(self):
        """Generate mean reversion trading signals."""
        df = self.data.copy()
        signals = pd.DataFrame(index=df.index)
        
        # ========================================================================
        # MEAN REVERSION SIGNAL CONDITIONS
        # ========================================================================
        
        # Buy Signal: Oversold conditions (price below mean, expect reversion up)
        oversold_conditions = (
            (df['z_score'] < -self.z_score_threshold) |  # Z-score oversold
            (df['close'] <= df['bb_lower']) |  # At or below lower Bollinger Band
            (df['distance_from_mean_pct'] < -2.0)  # More than 2% below mean
        )
        
        # Additional buy filters
        buy_filters = (
            (df['mean_reversion_strength'] >= 3.0) &  # Minimum strength
            (df['z_score'] < 0) &  # Below mean
            (df['bb_position'] < 0.2)  # In lower 20% of Bollinger Band range
        )
        
        signals['buy_signal'] = (oversold_conditions & buy_filters).astype(int)
        
        # Sell Signal: Overbought conditions (price above mean, expect reversion down)
        overbought_conditions = (
            (df['z_score'] > self.z_score_threshold) |  # Z-score overbought
            (df['close'] >= df['bb_upper']) |  # At or above upper Bollinger Band
            (df['distance_from_mean_pct'] > 2.0)  # More than 2% above mean
        )
        
        # Additional sell filters
        sell_filters = (
            (df['mean_reversion_strength'] >= 3.0) &  # Minimum strength
            (df['z_score'] > 0) &  # Above mean
            (df['bb_position'] > 0.8)  # In upper 20% of Bollinger Band range
        )
        
        signals['sell_signal'] = (overbought_conditions & sell_filters).astype(int)
        
        # ========================================================================
        # SIGNAL STRENGTH AND CONFIDENCE
        # ========================================================================
        
        # Buy signal strength
        signals['buy_strength'] = np.where(
            signals['buy_signal'] == 1,
            df['mean_reversion_strength'] * (
                1.0 +  # Base strength
                (df['volume_confirmation'] * 0.2) +  # Volume bonus
                (df['high_volatility'] * 0.1)  # High volatility bonus
            ),
            0.0
        )
        
        # Sell signal strength
        signals['sell_strength'] = np.where(
            signals['sell_signal'] == 1,
            df['mean_reversion_strength'] * (
                1.0 +  # Base strength
                (df['volume_confirmation'] * 0.2) +  # Volume bonus
                (df['high_volatility'] * 0.1)  # High volatility bonus
            ),
            0.0
        )
        
        # Combined signal (1 = buy, -1 = sell, 0 = hold)
        signals['signal'] = 0
        signals.loc[signals['buy_signal'] == 1, 'signal'] = 1
        signals.loc[signals['sell_signal'] == 1, 'signal'] = -1
        
        # If both signals occur (shouldn't happen), use stronger one
        both_signals = (signals['buy_signal'] == 1) & (signals['sell_signal'] == 1)
        signals.loc[both_signals & (signals['buy_strength'] > signals['sell_strength']), 'signal'] = 1
        signals.loc[both_signals & (signals['sell_strength'] > signals['buy_strength']), 'signal'] = -1
        
        # Confidence score (0-1)
        max_strength = np.maximum(signals['buy_strength'], signals['sell_strength'])
        signals['confidence'] = np.minimum(max_strength / 10.0, 1.0)
        
        # Expected mean reversion distance (target)
        signals['expected_reversion_pct'] = np.where(
            signals['signal'] == 1,
            abs(df['distance_from_mean_pct']),  # Buy: expect move back to mean
            np.where(
                signals['signal'] == -1,
                abs(df['distance_from_mean_pct']),  # Sell: expect move back to mean
                0.0
            )
        )
        
        # Risk/Reward estimate
        # Risk: distance to stop loss (e.g., 1 std beyond current position)
        # Reward: distance to mean
        signals['estimated_risk_pct'] = df['std'] / df['close'] * 100  # 1 std as risk
        signals['estimated_reward_pct'] = signals['expected_reversion_pct']
        signals['risk_reward_ratio'] = np.where(
            signals['estimated_risk_pct'] > 0,
            signals['estimated_reward_pct'] / signals['estimated_risk_pct'],
            0.0
        )
        
        self.signals = signals
    
    def get_current_signals(self) -> Dict:
        """
        Get the most recent mean reversion signals.
        
        Returns:
            Dictionary with current signal information
        """
        if self.signals.empty:
            return {}
        
        latest = self.signals.iloc[-1]
        latest_data = self.data.iloc[-1]
        
        signal_type = "NEUTRAL"
        if latest['signal'] == 1:
            signal_type = "BUY (OVERSOLD)"
        elif latest['signal'] == -1:
            signal_type = "SELL (OVERBOUGHT)"
        
        return {
            'signal': signal_type,
            'signal_value': int(latest['signal']),
            'confidence': float(latest['confidence']),
            'buy_strength': float(latest['buy_strength']),
            'sell_strength': float(latest['sell_strength']),
            'z_score': float(latest_data['z_score']),
            'distance_from_mean_pct': float(latest_data['distance_from_mean_pct']),
            'bb_position': float(latest_data['bb_position']),
            'mean_reversion_strength': float(latest_data['mean_reversion_strength']),
            'price': float(latest_data['close']),
            'mean': float(latest_data['mean']),
            'expected_reversion_pct': float(latest['expected_reversion_pct']),
            'risk_reward_ratio': float(latest['risk_reward_ratio']),
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
    
    def get_mean_reversion_opportunities(
        self,
        min_strength: float = 5.0,
        min_confidence: float = 0.5
    ) -> pd.DataFrame:
        """
        Get mean reversion opportunities filtered by strength and confidence.
        
        Args:
            min_strength: Minimum mean reversion strength (0-10)
            min_confidence: Minimum confidence (0-1)
        
        Returns:
            DataFrame with filtered opportunities
        """
        full_df = self.get_full_analysis()
        
        opportunities = full_df[
            (full_df['signal'] != 0) &
            (full_df['mean_reversion_strength'] >= min_strength) &
            (full_df['confidence'] >= min_confidence)
        ].copy()
        
        return opportunities

