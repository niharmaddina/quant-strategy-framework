"""
ML Training on Real Market Data - Alternative Script

This script attempts to fetch real market data and train the ML model.
If yfinance is unavailable, it provides clear instructions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from strategy.mean_reversion import MeanReversionIndicators
from backtest.engine import BacktestEngine
from ml.preprocessing import prep_ohlcv
from ml.labeling import label_engine_tp_sl_first_long_only
from ml.model import MRSignalQualityModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def test_yfinance_connection():
    """Test if yfinance is working."""
    try:
        import yfinance as yf
        import time
        # Don't test with .info as it triggers rate limiting
        # Just try to fetch a small amount of historical data
        ticker = yf.Ticker("SPY")
        time.sleep(1)  # Small delay to avoid rate limiting
        data = ticker.history(period="5d", interval="1d")
        if len(data) > 0:
            return True, "Connection successful"
        else:
            return False, "Connection successful but no data returned"
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            return False, "Rate limited - please wait a few minutes and try again"
        return False, f"Connection failed: {error_msg}"


def main():
    """
    Train ML model on real market data.
    """
    print("=" * 70)
    print("ML Signal Quality Model - Real Market Data Training")
    print("=" * 70)
    
    # Test connection first
    print("\n0. Testing yfinance connection...")
    connected, msg = test_yfinance_connection()
    print(f"   {msg}")
    
    if not connected:
        print("\n⚠ yfinance appears to be unavailable.")
        print("   This is a known issue with yfinance - it can be unreliable.")
        print("\n   Options:")
        print("   1. Wait a few minutes and try again")
        print("   2. Use mock data: python examples/ml_training_validation.py")
        print("   3. Try a different data source")
        return
    
    # ========================================================================
    # 1. Fetch and prepare data
    # ========================================================================
    print("\n1. Fetching real market data...")
    fetcher = DataFetcher()
    
    # Try to fetch recent data
    # Use a date range that's likely to work
    symbols_to_try = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
    date_ranges = [
        ("2024-12-01", "2024-12-31"),
        ("2024-11-01", "2024-11-30"),
        ("2024-10-01", "2024-10-31"),
    ]
    intervals = ["5m", "15m", "1h", "1d"]
    
    raw_data = None
    used_symbol = None
    used_dates = None
    used_interval = None
    
    # Build batch request list
    batch_requests = []
    for interval in intervals:
        for start_date, end_date in date_ranges:
            for symbol in symbols_to_try:
                batch_requests.append((symbol, start_date, end_date, interval))
    
    print(f"   Prepared {len(batch_requests)} requests, fetching in batches...")
    
    # Fetch in batches
    batch_results = fetcher.fetch_batch(batch_requests, batch_size=3)
    
    # Find the first successful result with enough data
    raw_data = None
    for key, data in batch_results.items():
        if len(data) > 50:  # Need at least 50 bars for meaningful training
            # Parse key: "{symbol}_{start_date}_{end_date}_{interval}"
            parts = key.split("_")
            used_symbol = parts[0]
            used_dates = (parts[1], parts[2])
            used_interval = parts[3]
            raw_data = data
            print(f"   ✓ Success! Using {used_symbol} ({used_interval}): {len(raw_data)} bars")
            break
    
    if raw_data is None or len(raw_data) < 50:
        print("\n❌ Could not fetch sufficient real market data.")
        print("   This is common with yfinance - it can be unreliable.")
        print("\n   Recommendation: Use mock data for testing:")
        print("   python examples/ml_training_validation.py")
        return
    
    print(f"\n   Using: {used_symbol} | {used_interval} | {used_dates[0]} to {used_dates[1]}")
    
    # Prepare data
    df = prep_ohlcv(raw_data)
    df_for_indicators = df.reset_index()
    if 'date' not in df_for_indicators.columns:
        df_for_indicators['date'] = df.index
    
    # ========================================================================
    # 2. Calculate indicators
    # ========================================================================
    print("\n2. Calculating mean reversion indicators...")
    mr = MeanReversionIndicators(df_for_indicators, lookback_period=20)
    full = mr.get_full_analysis()
    
    # Put back on datetime index
    if 'date' in full.columns:
        full = full.set_index('date').sort_index()
    elif not isinstance(full.index, pd.DatetimeIndex):
        if len(full) == len(df):
            full.index = df.index
    
    print(f"   ✓ Calculated indicators for {len(full)} bars")
    print(f"   ✓ Generated {full['signal'].sum()} BUY signals")
    
    # ========================================================================
    # 3. Create labels
    # ========================================================================
    print("\n3. Creating engine-matching labels...")
    
    # Adjust horizon based on interval
    if used_interval == "5m":
        horizon_bars = 48  # 4 hours
    elif used_interval == "15m":
        horizon_bars = 16  # 4 hours
    elif used_interval == "1h":
        horizon_bars = 4  # 4 hours
    else:  # daily
        horizon_bars = 20  # 20 days
    
    y = label_engine_tp_sl_first_long_only(
        df=full,
        delay_bars=1,
        horizon_bars=horizon_bars,
        stop_loss_pct=0.05,
        entry_price_col="open"
    )
    
    labeled_count = y.notna().sum()
    positive_labels = (y == 1).sum()
    negative_labels = (y == 0).sum()
    
    print(f"   ✓ Labeled {labeled_count} signals")
    print(f"   ✓ Positive (TP hit): {positive_labels}")
    print(f"   ✓ Negative (SL hit): {negative_labels}")
    
    if labeled_count < 10:
        print(f"\n   ⚠ WARNING: Only {labeled_count} labeled signals found.")
        print("   This may not be enough for reliable training.")
        print("   Consider using more data or a different date range.")
    
    if labeled_count == 0:
        print("\n   ❌ No labeled signals found. Cannot train model.")
        return
    
    # ========================================================================
    # 4. Train model
    # ========================================================================
    print("\n4. Training ML model...")
    
    # Split data
    labeled_indices = y[y.notna()].index
    split_idx = int(len(labeled_indices) * 0.7)
    train_indices = labeled_indices[:split_idx]
    val_indices = labeled_indices[split_idx:]
    
    X_train = full.loc[train_indices]
    y_train = y.loc[train_indices]
    X_val = full.loc[val_indices]
    y_val = y.loc[val_indices]
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    
    model = MRSignalQualityModel(random_state=42)
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    
    # Feature importance
    print("\n   Top 10 most important features:")
    feature_importance = model.get_feature_importance()
    for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
        print(f"   {i:2d}. {feature:30s} {importance:.4f}")
    
    # ========================================================================
    # 5. Validate
    # ========================================================================
    print("\n5. Validating model...")
    predictions = model.predict(X_val)
    y_val_actual = y_val.astype(int)
    y_val_pred = (predictions['p_smooth'] > 0.5).astype(int)
    
    accuracy = accuracy_score(y_val_actual, y_val_pred)
    precision = precision_score(y_val_actual, y_val_pred, zero_division=0)
    recall = recall_score(y_val_actual, y_val_pred, zero_division=0)
    f1 = f1_score(y_val_actual, y_val_pred, zero_division=0)
    
    print(f"\n   Validation Metrics:")
    print(f"   {'='*50}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   {'='*50}")
    
    # ========================================================================
    # 6. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nData: {used_symbol} | {used_interval} | {used_dates[0]} to {used_dates[1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Validation Accuracy: {accuracy:.2%}")
    print(f"Validation F1 Score: {f1:.4f}")
    print("\n✓ Model is ready for use!")


if __name__ == "__main__":
    main()

