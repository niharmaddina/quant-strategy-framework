"""
ML Training and Validation Script

Trains the ML signal quality model and validates its performance.
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def main():
    """
    Train and validate ML model.
    """
    print("=" * 70)
    print("ML Signal Quality Model - Training and Validation")
    print("=" * 70)
    
    # ========================================================================
    # 1. Fetch and prepare data
    # ========================================================================
    print("\n1. Fetching data...")
    fetcher = DataFetcher()
    
    # Fetch real market data
    print("   Fetching real market data from yfinance...")
    from datetime import datetime, timedelta
    
    # Try multiple date ranges and intervals
    # yfinance intraday data can be unreliable, so we'll try multiple options
    date_ranges = [
        ("2024-12-01", "2024-12-31", "5m"),  # Most recent month
        ("2024-11-15", "2024-12-15", "5m"),  # Overlapping range
        ("2024-10-01", "2024-10-31", "5m"),  # October
        ("2024-01-01", "2024-12-31", "1d"),  # Daily data as fallback
    ]
    
    # Build batch request list
    batch_requests = []
    for start_date, end_date, interval in date_ranges:
        for symbol in ["SPY", "QQQ", "AAPL"]:
            batch_requests.append((symbol, start_date, end_date, interval))
    
    print(f"   Prepared {len(batch_requests)} requests, fetching in batches...")
    
    # Fetch in batches
    batch_results = fetcher.fetch_batch(batch_requests, batch_size=3)
    
    # Find the first successful result
    raw_5m = None
    used_interval = None
    for key, data in batch_results.items():
        if len(data) > 0:
            # Parse key: "{symbol}_{start_date}_{end_date}_{interval}"
            parts = key.split("_")
            symbol = parts[0]
            start_date = parts[1]
            end_date = parts[2]
            interval = parts[3]
            raw_5m = data
            used_interval = interval
            print(f"   ✓ Successfully fetched {len(raw_5m)} bars of real {interval} data for {symbol}")
            print(f"   Date range: {start_date} to {end_date}")
            break
    
    if raw_5m is None or len(raw_5m) == 0:
        print("\n" + "="*70)
        print("⚠ WARNING: Could not fetch real market data from yfinance")
        print("="*70)
        print("\nThis may be due to:")
        print("  - yfinance API issues or rate limiting")
        print("  - Network connectivity problems")
        print("  - Market data availability for the requested dates")
        print("\nOptions:")
        print("  1. Try again later (yfinance may be temporarily unavailable)")
        print("  2. Use mock data for testing (set use_mock=True in DataFetcher)")
        print("  3. Check your internet connection")
        print("  4. Try a different date range")
        print("\nFor now, you can run with mock data:")
        print("  fetcher = DataFetcher(use_mock=True)")
        print("="*70)
        raise ValueError("Could not fetch real market data. See message above for details.")
    
    # If we got daily data, note it
    if used_interval == "1d":
        print("   ⚠ Note: Using daily data instead of 5m (intraday data unavailable)")
        print("   The model will still work, but results may differ from intraday trading")
    
    # Prepare data
    df_5m = prep_ohlcv(raw_5m)
    df_5m_for_indicators = df_5m.reset_index()
    if 'date' not in df_5m_for_indicators.columns:
        df_5m_for_indicators['date'] = df_5m.index
    
    # ========================================================================
    # 2. Calculate indicators
    # ========================================================================
    print("\n2. Calculating mean reversion indicators...")
    mr = MeanReversionIndicators(df_5m_for_indicators, lookback_period=20)
    full = mr.get_full_analysis()
    
    # Put back on datetime index
    if 'date' in full.columns:
        full = full.set_index('date').sort_index()
    elif not isinstance(full.index, pd.DatetimeIndex):
        if len(full) == len(df_5m):
            full.index = df_5m.index
    
    print(f"   ✓ Calculated indicators for {len(full)} bars")
    print(f"   ✓ Generated {full['signal'].sum()} BUY signals")
    
    # ========================================================================
    # 3. Create labels
    # ========================================================================
    print("\n3. Creating engine-matching labels...")
    y = label_engine_tp_sl_first_long_only(
        df=full,
        delay_bars=1,
        horizon_bars=48,
        stop_loss_pct=0.05,
        entry_price_col="open"
    )
    
    labeled_count = y.notna().sum()
    positive_labels = (y == 1).sum()
    negative_labels = (y == 0).sum()
    
    print(f"   ✓ Labeled {labeled_count} signals")
    print(f"   ✓ Positive (TP hit): {positive_labels}")
    print(f"   ✓ Negative (SL hit): {negative_labels}")
    
    if labeled_count == 0:
        print("\n   ⚠ WARNING: No labeled signals found!")
        print("   This may be due to:")
        print("   - No BUY signals generated")
        print("   - Signals too close to end of data (need horizon_bars)")
        print("   - Using mock data with insufficient price movement")
        return
    
    # ========================================================================
    # 4. Split data for training/validation
    # ========================================================================
    print("\n4. Splitting data for training/validation...")
    
    # Get indices with labels
    labeled_indices = y[y.notna()].index
    X_labeled = full.loc[labeled_indices]
    y_labeled = y.loc[labeled_indices]
    
    # Split by time (earlier data for training, later for validation)
    split_idx = int(len(labeled_indices) * 0.7)
    train_indices = labeled_indices[:split_idx]
    val_indices = labeled_indices[split_idx:]
    
    X_train = full.loc[train_indices]
    y_train = y.loc[train_indices]
    X_val = full.loc[val_indices]
    y_val = y.loc[val_indices]
    
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Validation set: {len(X_val)} samples")
    
    # ========================================================================
    # 5. Train model
    # ========================================================================
    print("\n5. Training ML model...")
    model = MRSignalQualityModel(
        max_depth=4,
        learning_rate=0.05,
        max_iter=600,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    
    # Show feature importance
    print("\n   Top 10 most important features:")
    feature_importance = model.get_feature_importance()
    for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
        print(f"   {i:2d}. {feature:30s} {importance:.4f}")
    
    # ========================================================================
    # 6. Validate model
    # ========================================================================
    print("\n6. Validating model on validation set...")
    
    # Predict on validation set
    predictions = model.predict(X_val)
    
    # Get actual labels for validation set
    y_val_actual = y_val.astype(int)
    
    # Get predicted labels (from signal_ml, but we need probability-based predictions)
    # Use p_smooth > 0.5 as threshold for comparison
    y_val_pred = (predictions['p_smooth'] > 0.5).astype(int)
    
    # Calculate metrics
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
    
    # Classification report
    print("\n   Classification Report:")
    print(classification_report(y_val_actual, y_val_pred, target_names=['SL Hit', 'TP Hit']))
    
    # ========================================================================
    # 7. Backtest comparison
    # ========================================================================
    print("\n7. Running backtest comparison...")
    
    # Prepare data for backtest
    data_for_bt = full.reset_index()[['date', 'close']]
    
    # Original signals
    signals_original = full.reset_index()[['date', 'signal']].copy()
    
    # ML-gated signals
    signals_ml = predictions.reset_index()
    if 'date' not in signals_ml.columns:
        signals_ml['date'] = X_val.index
    signals_ml = signals_ml[['date', 'signal_ml']].rename(columns={'signal_ml': 'signal'})
    
    # Backtest original (on validation period only)
    val_data = data_for_bt[data_for_bt['date'].isin(X_val.index)].copy()
    val_signals_orig = signals_original[signals_original['date'].isin(X_val.index)].copy()
    
    if len(val_data) > 0 and val_signals_orig['signal'].sum() > 0:
        print("\n   Original signals (validation period):")
        bt_original = BacktestEngine(
            data=val_data.copy(),
            signals=val_signals_orig.copy(),
            stop_loss_pct=0.05,
            commission=0.001,
            position_size=0.1
        )
        bt_original.run()
        print(f"   Total Return: {bt_original.metrics['total_return_pct']:.2f}%")
        print(f"   Win Rate: {bt_original.metrics['win_rate_pct']:.2f}%")
        print(f"   Sharpe Ratio: {bt_original.metrics['sharpe_ratio']:.2f}")
    
    # Backtest ML-gated (on validation period only)
    val_signals_ml = signals_ml[signals_ml['date'].isin(X_val.index)].copy()
    
    if len(val_data) > 0 and val_signals_ml['signal'].sum() > 0:
        print("\n   ML-gated signals (validation period):")
        bt_ml = BacktestEngine(
            data=val_data.copy(),
            signals=val_signals_ml.copy(),
            stop_loss_pct=0.05,
            commission=0.001,
            position_size=0.1
        )
        bt_ml.run()
        print(f"   Total Return: {bt_ml.metrics['total_return_pct']:.2f}%")
        print(f"   Win Rate: {bt_ml.metrics['win_rate_pct']:.2f}%")
        print(f"   Sharpe Ratio: {bt_ml.metrics['sharpe_ratio']:.2f}")
    
    # ========================================================================
    # 8. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training and Validation Complete!")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    print(f"  - Validation Accuracy: {accuracy:.2%}")
    print(f"  - Validation F1 Score: {f1:.4f}")
    print("\nModel is ready for use!")


if __name__ == "__main__":
    main()

