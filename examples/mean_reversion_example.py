"""
Example: Mean Reversion Strategy with Granular Intraday Data

Demonstrates how to use the MeanReversionIndicators class with
intraday data (1h, 15m, 5m intervals) for mean reversion trading.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.fetcher import DataFetcher
from strategy.mean_reversion import MeanReversionIndicators
from backtest.engine import BacktestEngine


def example_mean_reversion_intraday():
    """Example: Mean reversion strategy with 1-hour intraday data."""
    print("=" * 70)
    print("MEAN REVERSION STRATEGY - INTRADAY DATA")
    print("=" * 70)
    print()
    
    # Initialize data fetcher
    # Default to mock data to keep this example runnable without network / rate limit issues.
    # Set use_mock=False if you want real yfinance data.
    fetcher = DataFetcher(use_mock=True)
    
    # Fetch intraday data (1-hour intervals)
    # Note: yfinance limits 1h data to ~730 days
    # Use past dates to ensure data availability
    end_date = datetime(2024, 12, 1)  # Use past date
    start_date = end_date - timedelta(days=7)  # Last 7 days
    
    symbol = "SPY"
    print(f"Fetching {symbol} intraday data (1h interval)...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()
    
    try:
        data = fetcher.fetch_stock_data(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1h"  # 1-hour intervals
        )
        
        print(f"✓ Fetched {len(data)} data points")
        print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print()
        
        # Initialize mean reversion indicators
        print("Calculating mean reversion indicators...")
        mr_indicators = MeanReversionIndicators(
            data=data,
            lookback_period=20,  # 20 periods (20 hours for 1h data)
            z_score_threshold=2.0,  # 2 standard deviations
            bb_std_dev=2.0
        )
        
        print("✓ Indicators calculated")
        print()
        
        # Get current signals
        current_signals = mr_indicators.get_current_signals()
        print("Current Mean Reversion Signals:")
        print("-" * 70)
        print(f"  Signal: {current_signals['signal']}")
        print(f"  Confidence: {current_signals['confidence']:.2%}")
        print(f"  Z-Score: {current_signals['z_score']:.2f}")
        print(f"  Distance from Mean: {current_signals['distance_from_mean_pct']:.2f}%")
        print(f"  BB Position: {current_signals['bb_position']:.2%}")
        print(f"  Mean Reversion Strength: {current_signals['mean_reversion_strength']:.2f}/10")
        print(f"  Expected Reversion: {current_signals['expected_reversion_pct']:.2f}%")
        print(f"  Risk/Reward Ratio: {current_signals['risk_reward_ratio']:.2f}")
        print()
        
        # Get opportunities
        opportunities = mr_indicators.get_mean_reversion_opportunities(
            min_strength=5.0,
            min_confidence=0.5
        )
        
        print(f"Mean Reversion Opportunities (min strength: 5.0, min confidence: 0.5):")
        print(f"  Found {len(opportunities)} opportunities")
        print()
        
        if len(opportunities) > 0:
            print("Top 5 Opportunities:")
            print("-" * 70)
            top_opps = opportunities.nlargest(5, 'mean_reversion_strength')
            for idx, row in top_opps.iterrows():
                signal_type = "BUY" if row['signal'] == 1 else "SELL"
                print(f"  {row.get('date', idx)}: {signal_type}")
                print(f"    Strength: {row['mean_reversion_strength']:.2f}")
                print(f"    Confidence: {row['confidence']:.2%}")
                print(f"    Z-Score: {row['z_score']:.2f}")
                print(f"    Price: ${row['close']:.2f} (Mean: ${row['mean']:.2f})")
                print(f"    Expected Reversion: {row['expected_reversion_pct']:.2f}%")
                print()
        
        # Run backtest
        print("Running backtest...")
        signals_df = mr_indicators.get_signals_df()
        
        # Prepare data for backtest (ensure date column)
        backtest_data = data.copy()
        if 'date' not in backtest_data.columns:
            backtest_data = backtest_data.reset_index()
            if backtest_data.index.name == 'date':
                backtest_data['date'] = backtest_data.index
        
        # Prepare signals for backtest
        backtest_signals = signals_df[['signal']].copy()
        if 'date' not in backtest_signals.columns:
            backtest_signals = backtest_signals.reset_index()
            if backtest_signals.index.name == 'date':
                backtest_signals['date'] = backtest_signals.index
        
        engine = BacktestEngine(
            data=backtest_data,
            signals=backtest_signals,
            initial_capital=100000,
            position_size=0.1,  # 10% per trade
            commission=0.001,  # 0.1% commission
            stop_loss_pct=0.02,  # 2% stop loss (tighter for intraday)
            max_drawdown_pct=0.15  # 15% max drawdown
        )
        
        results = engine.run()
        metrics = engine.get_metrics()
        
        print("✓ Backtest complete")
        print()
        print("Backtest Results:")
        print("-" * 70)
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print()
        
        return mr_indicators, results, metrics
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def example_mean_reversion_15min():
    """Example: Mean reversion strategy with 15-minute intervals."""
    print("=" * 70)
    print("MEAN REVERSION STRATEGY - 15-MINUTE DATA")
    print("=" * 70)
    print()
    
    fetcher = DataFetcher(use_mock=True)
    
    # Fetch 15-minute data (limited to ~60 days)
    # Use past dates to ensure data availability
    end_date = datetime(2024, 12, 1)
    start_date = end_date - timedelta(days=2)  # Last 2 days
    
    symbol = "SPY"
    print(f"Fetching {symbol} 15-minute data...")
    
    try:
        data = fetcher.fetch_stock_data(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="15m"  # 15-minute intervals
        )
        
        print(f"✓ Fetched {len(data)} data points")
        print()
        
        # Use shorter lookback for 15m data
        mr_indicators = MeanReversionIndicators(
            data=data,
            lookback_period=40,  # 40 periods = 10 hours
            z_score_threshold=1.5,  # Slightly tighter for shorter timeframe
            bb_std_dev=2.0
        )
        
        current_signals = mr_indicators.get_current_signals()
        print("Current Signals (15m):")
        print(f"  Signal: {current_signals['signal']}")
        print(f"  Z-Score: {current_signals['z_score']:.2f}")
        print(f"  Strength: {current_signals['mean_reversion_strength']:.2f}/10")
        print()
        
        return mr_indicators
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_granular_price_analysis():
    """Example: Analyzing granular price movements for mean reversion."""
    print("=" * 70)
    print("GRANULAR PRICE ANALYSIS FOR MEAN REVERSION")
    print("=" * 70)
    print()
    
    fetcher = DataFetcher(use_mock=True)
    
    # Fetch 5-minute data for detailed analysis
    # Use past dates to ensure data availability
    end_date = datetime(2024, 12, 1)
    start_date = end_date - timedelta(days=1)  # Last 1 day
    
    symbol = "SPY"
    print(f"Fetching {symbol} 5-minute data for granular analysis...")
    
    try:
        data = fetcher.fetch_stock_data(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="5m"
        )
        
        print(f"✓ Fetched {len(data)} data points")
        print(f"  Average price: ${data['close'].mean():.2f}")
        print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"  Volatility (std): ${data['close'].std():.2f}")
        print()
        
        # Analyze mean reversion opportunities
        mr_indicators = MeanReversionIndicators(
            data=data,
            lookback_period=60,  # 60 periods = 5 hours
            z_score_threshold=1.5,
            bb_std_dev=2.0
        )
        
        # Get full analysis
        analysis = mr_indicators.get_full_analysis()
        
        # Find extreme Z-scores
        extreme_oversold = analysis[analysis['z_score'] < -2.0]
        extreme_overbought = analysis[analysis['z_score'] > 2.0]
        
        print("Extreme Mean Reversion Opportunities:")
        print(f"  Extreme Oversold (Z < -2.0): {len(extreme_oversold)} periods")
        print(f"  Extreme Overbought (Z > 2.0): {len(extreme_overbought)} periods")
        print()
        
        if len(extreme_oversold) > 0:
            print("Sample Oversold Periods:")
            for idx, row in extreme_oversold.head(3).iterrows():
                print(f"  {row.get('date', idx)}: Z={row['z_score']:.2f}, "
                      f"Price=${row['close']:.2f}, Mean=${row['mean']:.2f}")
        
        if len(extreme_overbought) > 0:
            print("Sample Overbought Periods:")
            for idx, row in extreme_overbought.head(3).iterrows():
                print(f"  {row.get('date', idx)}: Z={row['z_score']:.2f}, "
                      f"Price=${row['close']:.2f}, Mean=${row['mean']:.2f}")
        
        print()
        return mr_indicators
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MEAN REVERSION STRATEGY EXAMPLES")
    print("=" * 70)
    print()
    
    # Example 1: 1-hour intraday data
    print("Example 1: 1-Hour Intraday Mean Reversion")
    print("-" * 70)
    example_mean_reversion_intraday()
    print()
    
    # Example 2: 15-minute data
    print("\nExample 2: 15-Minute Mean Reversion")
    print("-" * 70)
    example_mean_reversion_15min()
    print()
    
    # Example 3: Granular price analysis
    print("\nExample 3: Granular Price Analysis (5-minute)")
    print("-" * 70)
    example_granular_price_analysis()
    print()
    
    print("=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)

