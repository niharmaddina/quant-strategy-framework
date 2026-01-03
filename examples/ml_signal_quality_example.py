"""
Example: ML Signal Quality Gating for Mean Reversion Strategy

This example demonstrates:
1. Fetching intraday data (5m/15m/1h)
2. Running MeanReversionIndicators
3. Creating engine-matching labels
4. Training ML model to predict signal quality
5. Generating ML-gated signals for BacktestEngine
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from strategy.mean_reversion import MeanReversionIndicators
from backtest.engine import BacktestEngine
from ml.preprocessing import prep_ohlcv
from ml.labeling import label_engine_tp_sl_first_long_only
from ml.model import MRSignalQualityModel

import pandas as pd


def main():
    """
    End-to-end example: ML signal quality gating for mean reversion.
    """
    print("=" * 60)
    print("ML Signal Quality Example")
    print("=" * 60)
    
    # ========================================================================
    # 1. Fetch intraday data (5m execution timeframe)
    # ========================================================================
    print("\n1. Fetching 5m intraday data...")
    fetcher = DataFetcher()
    
    # Fetch 5m data (limited to 60 days for 5m interval)
    raw_5m = fetcher.fetch_stock_data(
        symbol="SPY",
        start_date="2024-10-01",
        end_date="2024-11-30",
        interval="5m"
    )
    
    print(f"   Fetched {len(raw_5m)} bars of 5m data")
    
    # Prepare data (standardize format, set datetime index)
    df_5m = prep_ohlcv(raw_5m)
    
    # Reset index to get 'date' column for MeanReversionIndicators
    # (MeanReversionIndicators expects 'date' column or uses index)
    df_5m_for_indicators = df_5m.reset_index()
    # Ensure 'date' column exists (reset_index should create it, but be safe)
    if 'date' not in df_5m_for_indicators.columns and len(df_5m_for_indicators) > 0:
        df_5m_for_indicators['date'] = df_5m.index if isinstance(df_5m.index, pd.DatetimeIndex) else pd.to_datetime(df_5m.index)
    
    # ========================================================================
    # 2. Run MeanReversionIndicators
    # ========================================================================
    print("\n2. Calculating mean reversion indicators...")
    mr = MeanReversionIndicators(
        df_5m_for_indicators,
        lookback_period=20,
        z_score_threshold=2.0
    )
    
    # Get full analysis (data + indicators + signals)
    full = mr.get_full_analysis()
    
    # Put back on datetime index for labels/model
    if 'date' in full.columns:
        full = full.set_index('date').sort_index()
    elif not isinstance(full.index, pd.DatetimeIndex):
        # Align with original datetime index
        if len(full) == len(df_5m):
            full.index = df_5m.index
        else:
            # If lengths don't match, try to align by date if available
            if 'date' in full.columns:
                full = full.set_index('date').sort_index()
            else:
                # Fallback: use numeric index (not ideal, but won't break)
                pass
    
    print(f"   Generated {full['signal'].sum()} BUY signals")
    
    # ========================================================================
    # 3. Create engine-matching labels
    # ========================================================================
    print("\n3. Creating engine-matching labels...")
    y = label_engine_tp_sl_first_long_only(
        df=full,
        signal_col="signal",
        delay_bars=1,  # 5m manual reaction => 1 bar delay
        horizon_bars=48,  # 4 hours on 5m (48 bars)
        stop_loss_pct=0.05,  # Must match BacktestEngine.stop_loss_pct
        entry_price_col="open"
    )
    
    labeled_count = y.notna().sum()
    positive_labels = (y == 1).sum()
    print(f"   Labeled {labeled_count} signals ({positive_labels} positive, {labeled_count - positive_labels} negative)")
    
    # ========================================================================
    # 4. Train ML model
    # ========================================================================
    print("\n4. Training ML signal quality model...")
    qm = MRSignalQualityModel(
        max_depth=4,
        learning_rate=0.05,
        max_iter=600,
        random_state=42
    )
    
    qm.fit(full, y)
    
    # Show feature importance
    print("\n   Top 10 most important features:")
    feature_importance = qm.get_feature_importance()
    for i, (feature, importance) in enumerate(feature_importance.head(10).items(), 1):
        print(f"   {i:2d}. {feature:30s} {importance:.4f}")
    
    # ========================================================================
    # 5. Generate ML-gated signals
    # ========================================================================
    print("\n5. Generating ML-gated signals...")
    ml_output = qm.predict(
        full,
        ema_span=20,
        p_on=0.65,  # Turn on gate at 65% probability
        p_off=0.55  # Turn off gate at 55% probability
    )
    
    original_signals = (full['signal'] == 1).sum()
    ml_gated_signals = ml_output['signal_ml'].sum()
    
    print(f"   Original signals: {original_signals}")
    print(f"   ML-gated signals: {ml_gated_signals}")
    print(f"   Filter rate: {(1 - ml_gated_signals / original_signals) * 100:.1f}%")
    
    # ========================================================================
    # 6. Backtest comparison: Original vs ML-gated
    # ========================================================================
    print("\n6. Running backtests...")
    
    # Prepare data for BacktestEngine
    data_for_bt = full.reset_index()[['date', 'close']]
    
    # Original signals
    signals_original = full.reset_index()[['date', 'signal']].copy()
    signals_original = signals_original.rename(columns={'signal': 'signal'})
    
    # ML-gated signals
    signals_ml = ml_output.reset_index()
    # Ensure 'date' column exists (reset_index should create it from datetime index)
    if 'date' not in signals_ml.columns and len(signals_ml) > 0:
        if isinstance(full.index, pd.DatetimeIndex):
            signals_ml['date'] = full.index
        else:
            signals_ml['date'] = pd.to_datetime(signals_ml.index)
    signals_ml = signals_ml[['date', 'signal_ml']].rename(columns={'signal_ml': 'signal'})
    
    # Backtest original signals
    print("\n   Original signals backtest:")
    bt_original = BacktestEngine(
        data=data_for_bt.copy(),
        signals=signals_original.copy(),
        stop_loss_pct=0.05,
        commission=0.001,
        position_size=0.1
    )
    bt_original.run()
    bt_original.print_metrics()
    
    # Backtest ML-gated signals
    print("\n   ML-gated signals backtest:")
    bt_ml = BacktestEngine(
        data=data_for_bt.copy(),
        signals=signals_ml.copy(),
        stop_loss_pct=0.05,
        commission=0.001,
        position_size=0.1
    )
    bt_ml.run()
    bt_ml.print_metrics()
    
    # ========================================================================
    # 7. Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Original strategy:")
    print(f"  - Total return: {bt_original.metrics['total_return_pct']:.2f}%")
    print(f"  - Win rate: {bt_original.metrics['win_rate_pct']:.2f}%")
    print(f"  - Sharpe ratio: {bt_original.metrics['sharpe_ratio']:.2f}")
    
    print(f"\nML-gated strategy:")
    print(f"  - Total return: {bt_ml.metrics['total_return_pct']:.2f}%")
    print(f"  - Win rate: {bt_ml.metrics['win_rate_pct']:.2f}%")
    print(f"  - Sharpe ratio: {bt_ml.metrics['sharpe_ratio']:.2f}")
    
    improvement = bt_ml.metrics['total_return_pct'] - bt_original.metrics['total_return_pct']
    print(f"\nImprovement: {improvement:+.2f}% return")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

