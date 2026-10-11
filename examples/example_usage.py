"""
Example usage of the quant strategy framework.
Demonstrates how to use the components programmatically.
"""

import os
import matplotlib.pyplot as plt

from backtest.engine import BacktestEngine
from backtest.factory import create_backtest_engine
from backtest.visualizer import BacktestVisualizer
from data.fetcher import DataFetcher
from strategy.indicators import SwingTradingIndicators
from universe.constructor import UniverseConstructor


def example_universe_construction():
    """Example: Using a frozen/manual trading universe (recommended for backtests)."""
    print("="*60)
    print("EXAMPLE: Universe Construction")
    print("="*60)

    # Load the built-in frozen universe file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    universe_file = os.path.join(project_root, "universe", "universe_mean_reversion_manual_50.txt")
    symbols = UniverseConstructor.load_universe(universe_file)

    universe = UniverseConstructor(symbols=symbols)
    print(f"\nLoaded frozen universe: {len(symbols)} symbols")
    print(f"First 15: {', '.join(symbols[:15])}")

    # Sector tagging is local (no API calls)
    print("\nSector tags (sample):")
    for s in symbols[:12]:
        print(f"  {s:6s} -> {universe.get_symbol_sector(s) or 'N/A'}")

    return symbols


def example_signal_generation():
    """Example: Generating trading signals."""
    print("\n" + "="*60)
    print("EXAMPLE: Signal Generation")
    print("="*60)
    
    # Fetch data (mock by default to avoid API rate limits in examples)
    fetcher = DataFetcher(use_mock=True)
    data = fetcher.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    print(f"\nFetched {len(data)} days of data for AAPL")
    
    # Calculate indicators
    indicators = SwingTradingIndicators(data)
    
    # Get current signal
    signal = indicators.get_current_signals()
    print("\nCurrent Trading Signal:")
    print(f"  Type: {signal['signal']}")
    print(f"  Confidence: {signal['confidence']:.2%}")
    print(f"  Price: ${signal['price']:.2f}")
    print(f"  RSI: {signal.get('rsi', 'N/A')}")
    print(f"  MACD: {signal.get('macd', 'N/A')}")
    
    # Get all signals
    signals_df = indicators.get_signals_df()
    print(f"\nTotal signals generated: {len(signals_df[signals_df['signal'] != 0])}")
    print(f"  Bullish: {len(signals_df[signals_df['signal'] == 1])}")
    print(f"  Bearish: {len(signals_df[signals_df['signal'] == -1])}")
    
    return indicators


def example_backtest():
    """Example: Running a backtest."""
    print("\n" + "="*60)
    print("EXAMPLE: Backtesting")
    print("="*60)
    
    # Fetch data and generate signals (mock by default to avoid API rate limits)
    fetcher = DataFetcher(use_mock=True)
    data = fetcher.fetch_stock_data('MSFT', '2023-01-01', '2024-01-01')
    
    indicators = SwingTradingIndicators(data)
    signals = indicators.get_signals_df()
    
    # Run backtest (prefer VectorBT, fall back to legacy if not installed)
    try:
        backtest = create_backtest_engine(
            engine="vectorbt",
            data=data,
            signals=signals,
            initial_capital=100000,
            position_size=0.1,
            commission=0.001,
        )
    except ImportError as e:
        print(f"VectorBT not available in this environment ({e}); using legacy engine.")
        backtest = BacktestEngine(
            data=data,
            signals=signals,
            initial_capital=100000,
            position_size=0.1,
            commission=0.001,
        )
    
    results = backtest.run()
    backtest.print_metrics()
    
    # Get trades
    trades = backtest.get_trades()
    if not trades.empty:
        print("\nFirst 5 trades:")
        print(trades.head().to_string())
    
    return backtest


def example_visualization():
    """Example: Creating visualizations."""
    print("\n" + "="*60)
    print("EXAMPLE: Visualization")
    print("="*60)
    
    # Run backtest first (mock by default to avoid API rate limits in examples)
    fetcher = DataFetcher(use_mock=True)
    data = fetcher.fetch_stock_data('TSLA', '2023-01-01', '2024-01-01')
    
    indicators = SwingTradingIndicators(data)
    signals = indicators.get_signals_df()
    
    try:
        backtest = create_backtest_engine(
            engine="vectorbt",
            data=data,
            signals=signals,
            initial_capital=100000,
            position_size=0.1,
        )
    except ImportError as e:
        print(f"VectorBT not available in this environment ({e}); using legacy engine.")
        backtest = BacktestEngine(
            data=data,
            signals=signals,
            initial_capital=100000,
            position_size=0.1,
        )
    
    backtest.run()
    
    # Create visualizations
    visualizer = BacktestVisualizer(backtest)
    
    # Individual plots
    print("\nGenerating plots...")
    output_dir = "output/examples"
    os.makedirs(output_dir, exist_ok=True)

    fig1 = visualizer.plot_equity_curve()
    fig1.savefig(os.path.join(output_dir, 'example_equity_curve.png'), dpi=150, bbox_inches='tight')
    print("  ✓ Saved equity curve")
    
    fig2 = visualizer.plot_trades()
    fig2.savefig(os.path.join(output_dir, 'example_trades.png'), dpi=150, bbox_inches='tight')
    print("  ✓ Saved trades chart")
    
    fig3 = visualizer.plot_all()
    fig3.savefig(os.path.join(output_dir, 'example_dashboard.png'), dpi=150, bbox_inches='tight')
    print("  ✓ Saved dashboard")
    
    plt.close('all')
    print("\nAll plots saved!")


def example_options_analysis():
    """Example: Fetching options data (for reference)."""
    print("\n" + "="*60)
    print("EXAMPLE: Options Chain Analysis")
    print("="*60)
    
    print("\nNote: options chain fetching requires network access and is rate-limit sensitive.")
    print("Skipping by default in this example script.\n")
    return
    
    # Fetch options chain
    fetcher = DataFetcher()
    options = fetcher.fetch_options_chain('AAPL')
    
    if not options['calls'].empty:
        print(f"\nOptions expiration: {options.get('expiration', 'N/A')}")
        print("\nCalls (first 5):")
        print(options['calls'][['strike', 'lastPrice', 'volume', 'openInterest']].head().to_string())
        
        print("\nPuts (first 5):")
        print(options['puts'][['strike', 'lastPrice', 'volume', 'openInterest']].head().to_string())
    else:
        print("No options data available")


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*60)
    print("QUANT STRATEGY FRAMEWORK - EXAMPLES")
    print("="*60)
    
    # Example 1: Universe construction
    universe = example_universe_construction()
    
    # Example 2: Signal generation
    indicators = example_signal_generation()
    
    # Example 3: Backtesting
    backtest = example_backtest()
    
    # Example 4: Visualization
    example_visualization()
    
    # Example 5: Options data (network-dependent, skipped by default)
    # example_options_analysis()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")

