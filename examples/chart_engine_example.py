"""
Example: Using the Advanced Chart Engine for comprehensive visualizations.

Demonstrates the new ChartEngine with:
- Comprehensive dashboards
- Strategy comparisons
- Mean reversion analysis
- Multi-symbol comparisons
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtest.engine import BacktestEngine
from backtest.chart_engine import ChartEngine
from data.fetcher import DataFetcher
from strategy.indicators import SwingTradingIndicators
from strategy.mean_reversion import MeanReversionIndicators


def example_comprehensive_dashboard():
    """Example: Create comprehensive backtest dashboard."""
    print("=" * 70)
    print("COMPREHENSIVE BACKTEST DASHBOARD")
    print("=" * 70)
    print()
    
    # Create sample data
    print("Creating sample backtest data...")
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(252) * 2)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices + np.random.randn(252) * 0.5,
        'high': prices + abs(np.random.randn(252) * 1.5),
        'low': prices - abs(np.random.randn(252) * 1.5),
        'volume': np.random.randint(1000000, 5000000, 252)
    })
    
    # Generate signals
    indicators = SwingTradingIndicators(data)
    signals_df = indicators.get_signals_df()
    
    # Run backtest
    engine = BacktestEngine(data, signals_df[['signal']], initial_capital=100000)
    results = engine.run()
    print(f"✓ Backtest complete: {engine.get_metrics()['total_return_pct']:.2f}% return")
    print()
    
    # Create comprehensive dashboard
    print("Generating comprehensive dashboard...")
    chart_engine = ChartEngine()
    fig = chart_engine.plot_comprehensive_dashboard(
        engine,
        title="Swing Trading Strategy - Comprehensive Analysis"
    )
    
    # Save
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/comprehensive_dashboard.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved to {save_path}")
    print()


def example_strategy_comparison():
    """Example: Compare multiple strategies."""
    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(252) * 2)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices + np.random.randn(252) * 0.5,
        'high': prices + abs(np.random.randn(252) * 1.5),
        'low': prices - abs(np.random.randn(252) * 1.5),
        'volume': np.random.randint(1000000, 5000000, 252)
    })
    
    # Strategy 1: Swing Trading
    print("Running Strategy 1: Swing Trading...")
    indicators1 = SwingTradingIndicators(data)
    signals1 = indicators1.get_signals_df()
    engine1 = BacktestEngine(data, signals1[['signal']], initial_capital=100000, position_size=0.1)
    engine1.run()
    
    # Strategy 2: Mean Reversion (with different parameters)
    print("Running Strategy 2: Mean Reversion...")
    mr = MeanReversionIndicators(data, lookback_period=20, z_score_threshold=1.5)
    signals2 = mr.get_signals_df()
    engine2 = BacktestEngine(data, signals2[['signal']], initial_capital=100000, position_size=0.1)
    engine2.run()
    
    # Benchmark: Buy and Hold
    print("Running Benchmark: Buy and Hold...")
    buy_hold_signals = pd.DataFrame({'signal': [1] + [0] * 251}, index=data.index)
    benchmark = BacktestEngine(data, buy_hold_signals, initial_capital=100000, position_size=1.0)
    benchmark.run()
    
    print()
    
    # Create comparison chart
    print("Generating strategy comparison chart...")
    chart_engine = ChartEngine()
    strategies = {
        'Swing Trading': engine1,
        'Mean Reversion': engine2
    }
    
    fig = chart_engine.plot_strategy_comparison(
        strategies,
        benchmark=benchmark,
        title="Strategy Performance Comparison"
    )
    
    # Save
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/strategy_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved to {save_path}")
    print()


def example_mean_reversion_visualization():
    """Example: Mean reversion analysis visualization."""
    print("=" * 70)
    print("MEAN REVERSION ANALYSIS VISUALIZATION")
    print("=" * 70)
    print()
    
    # Create intraday sample data
    print("Creating intraday sample data...")
    dates = pd.date_range(start='2024-01-01 09:30', end='2024-01-05 16:00', freq='1h')
    np.random.seed(42)
    
    # Mean-reverting price series
    base_price = 100
    prices = []
    for i in range(len(dates)):
        if i == 0:
            prices.append(base_price)
        else:
            mean = base_price
            reversion = 0.1 * (mean - prices[-1]) + np.random.randn() * 2
            prices.append(prices[-1] + reversion)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices + np.random.randn(len(dates)) * 0.2,
        'high': prices + abs(np.random.randn(len(dates)) * 0.5),
        'low': prices - abs(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Calculate mean reversion indicators
    print("Calculating mean reversion indicators...")
    mr = MeanReversionIndicators(data, lookback_period=10, z_score_threshold=2.0)
    indicators = mr.get_indicators_df()
    signals = mr.get_signals_df()
    
    print(f"✓ Generated {len(signals[signals['signal'] != 0])} signals")
    print()
    
    # Create visualization
    print("Generating mean reversion analysis chart...")
    chart_engine = ChartEngine()
    fig = chart_engine.plot_mean_reversion_analysis(
        data,
        indicators,
        signals,
        title="Mean Reversion Strategy Analysis"
    )
    
    # Save
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/mean_reversion_analysis.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved to {save_path}")
    print()


def example_multi_symbol_comparison():
    """Example: Compare performance across multiple symbols."""
    print("=" * 70)
    print("MULTI-SYMBOL PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    symbol_results = {}
    
    # Create sample data for each symbol
    for symbol in symbols:
        print(f"Processing {symbol}...")
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        np.random.seed(hash(symbol) % 1000)
        prices = 100 + np.cumsum(np.random.randn(252) * 2)
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices + np.random.randn(252) * 0.5,
            'high': prices + abs(np.random.randn(252) * 1.5),
            'low': prices - abs(np.random.randn(252) * 1.5),
            'volume': np.random.randint(1000000, 5000000, 252)
        })
        
        # Run backtest
        indicators = SwingTradingIndicators(data)
        signals = indicators.get_signals_df()
        engine = BacktestEngine(data, signals[['signal']], initial_capital=100000)
        engine.run()
        
        symbol_results[symbol] = {
            'metrics': engine.get_metrics(),
            'engine': engine
        }
    
    print()
    
    # Create comparison chart
    print("Generating multi-symbol comparison chart...")
    chart_engine = ChartEngine()
    fig = chart_engine.plot_multi_symbol_comparison(
        symbol_results,
        title="Multi-Symbol Performance Comparison"
    )
    
    # Save
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/multi_symbol_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved to {save_path}")
    print()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 70)
    print("ADVANCED CHART ENGINE EXAMPLES")
    print("=" * 70)
    print()
    
    try:
        # Example 1: Comprehensive Dashboard
        example_comprehensive_dashboard()
        
        # Example 2: Strategy Comparison
        example_strategy_comparison()
        
        # Example 3: Mean Reversion Visualization
        example_mean_reversion_visualization()
        
        # Example 4: Multi-Symbol Comparison
        example_multi_symbol_comparison()
        
        print("=" * 70)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 70)
        print("\nCharts saved to output/ directory:")
        print("  - comprehensive_dashboard.png")
        print("  - strategy_comparison.png")
        print("  - mean_reversion_analysis.png")
        print("  - multi_symbol_comparison.png")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

