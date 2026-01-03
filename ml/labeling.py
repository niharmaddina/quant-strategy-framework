"""
Labeling utilities for training ML models.

Creates labels that match BacktestEngine TP/SL logic using intrabar high/low
for realistic training data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def label_engine_tp_sl_first_long_only(
    df: pd.DataFrame,
    signal_col: str = "signal",
    delay_bars: int = 1,
    horizon_bars: int = 48,
    stop_loss_pct: float = 0.05,
    commission: float = 0.001,
    entry_price_col: str = "open"
) -> pd.Series:
    """
    Label each BUY signal as 1 if TP is hit before SL.
    
    Engine-matching TP logic:
        stop_loss_price = entry * (1 - stop_loss_pct)
        profit_target = max(entry * 1.03, entry + (entry - stop_loss_price) * 2)
                      = max(entry * 1.03, entry * (1 + 2 * stop_loss_pct))
    
    Uses intrabar high/low for realistic labeling (more conservative than close-only).
    Only labels rows where signal == 1 (long-only).
    
    Args:
        df: DataFrame with OHLCV data, signal column, and datetime index
        signal_col: Name of signal column (default: "signal")
        delay_bars: Bars to wait before entry (default: 1 for 5m/15m/1h)
        horizon_bars: Maximum bars to look ahead for TP/SL (default: 48)
        stop_loss_pct: Stop loss percentage (must match BacktestEngine.stop_loss_pct)
        commission: Commission rate (optional, for future use)
        entry_price_col: Column to use for entry price (default: "open")
        
    Returns:
        Series with labels (1 = TP hit first, 0 = SL hit first, NaN = no signal)
        
    Notes:
        - Conservative tie-break: if both TP and SL touched in same bar, assume SL first
        - Only labels actual BUY signals (signal == 1)
        - Uses intrabar high/low for realistic training (more conservative than close-only)
    """
    n = len(df)
    y = np.full(n, np.nan)
    
    # Validate required columns
    required_cols = ['high', 'low', signal_col, entry_price_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    sig = df[signal_col].values.astype(int)
    high = df["high"].values
    low = df["low"].values
    entry_series = df[entry_price_col].values
    
    # Precompute TP multipliers
    # TP = max(3%, 2R) = max(entry * 1.03, entry * (1 + 2 * stop_loss_pct))
    tp_mult_2r = 1.0 + 2.0 * stop_loss_pct
    tp_mult_3p = 1.03
    
    # Process each signal
    for t in range(n - (delay_bars + horizon_bars + 1)):
        if sig[t] != 1:
            continue  # Only label BUY signals
        
        entry_idx = t + delay_bars
        entry = entry_series[entry_idx]
        
        if not np.isfinite(entry) or entry <= 0:
            continue
        
        # Calculate stop loss and take profit prices
        sl = entry * (1.0 - stop_loss_pct)
        tp = entry * max(tp_mult_3p, tp_mult_2r)
        
        # Check if TP or SL is hit first (using intrabar high/low)
        hit = 0.0
        for k in range(entry_idx, entry_idx + horizon_bars):
            if k >= n:
                break
            
            # Conservative: if both touched in same bar, assume SL first (worst-case)
            if low[k] <= sl and high[k] >= tp:
                hit = 0.0
                break
            
            # Stop loss hit
            if low[k] <= sl:
                hit = 0.0
                break
            
            # Take profit hit
            if high[k] >= tp:
                hit = 1.0
                break
        
        y[t] = hit
    
    return pd.Series(y, index=df.index, name="y_tp_before_sl")

