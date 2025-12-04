"""
Example usage of the quant strategy framework.
Demonstrates how to use the components programmatically.
"""

import matplotlib.pyplot as plt

from backtest.engine import BacktestEngine
from backtest.visualizer import BacktestVisualizer
from data.fetcher import DataFetcher
from strategy.indicators import SwingTradingIndicators
from universe.constructor import UniverseConstructor


def example_universe_construction():
    """Example: Building a trading universe."""
    print("="*60)
    print("EXAMPLE: Universe Construction")
    print("="*60)
    
    # Create universe
    universe = UniverseConstructor()
    
    # Get universe info
    info = universe.get_universe_info()
    print("\nUniverse Information:")
    print(info[['symbol', 'sector', 'market_cap', 'avg_volume']].to_string())
    
    # Create balanced universe
    balanced = universe.create_sector_balanced_universe(stocks_per_sector=2)
    print(f"\nBalanced Universe ({len(balanced)} symbols):")
    print(balanced)
    
    return balanced


def example_signal_generation():
    """Example: Generating trading signals."""
    print("\n" + "="*60)
    print("EXAMPLE: Signal Generation")
    print("="*60)
    
    # Fetch data
    fetcher = DataFetcher()
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
    
    # Fetch data and generate signals
    fetcher = DataFetcher()
    data = fetcher.fetch_stock_data('MSFT', '2023-01-01', '2024-01-01')
    
    indicators = SwingTradingIndicators(data)
    signals = indicators.get_signals_df()
    
    # Run backtest
    backtest = BacktestEngine(
        data=data,
        signals=signals,
        initial_capital=100000,
        position_size=0.1,
        commission=0.001
    )
    
    results = backtest.run()
    backtest.print_metrics()
    
    # Get trades
    trades = backtest.get_trades()
    if not trades.empty:
        print(f"\nFirst 5 trades:")
        print(trades.head().to_string())
    
    return backtest


def example_visualization():
    """Example: Creating visualizations."""
    print("\n" + "="*60)
    print("EXAMPLE: Visualization")
    print("="*60)
    
    # Run backtest first
    fetcher = DataFetcher()
    data = fetcher.fetch_stock_data('TSLA', '2023-01-01', '2024-01-01')
    
    indicators = SwingTradingIndicators(data)
    signals = indicators.get_signals_df()
    
    backtest = BacktestEngine(
        data=data,
        signals=signals,
        initial_capital=100000,
        position_size=0.1
    )
    
    backtest.run()
    
    # Create visualizations
    visualizer = BacktestVisualizer(backtest)
    
    # Individual plots
    print("\nGenerating plots...")
    fig1 = visualizer.plot_equity_curve()
    fig1.savefig('example_equity_curve.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved equity curve")
    
    fig2 = visualizer.plot_trades()
    fig2.savefig('example_trades.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved trades chart")
    
    fig3 = visualizer.plot_all()
    fig3.savefig('example_dashboard.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved dashboard")
    
    plt.close('all')
    print("\nAll plots saved!")


def example_options_analysis():
    """Example: Fetching options data (for reference)."""
    print("\n" + "="*60)
    print("EXAMPLE: Options Chain Analysis")
    print("="*60)
    
    fetcher = DataFetcher()
    
    # Fetch options chain
    options = fetcher.fetch_options_chain('AAPL')
    
    if not options['calls'].empty:
        print(f"\nOptions expiration: {options.get('expiration', 'N/A')}")
        print(f"\nCalls (first 5):")
        print(options['calls'][['strike', 'lastPrice', 'volume', 'openInterest']].head().to_string())
        
        print(f"\nPuts (first 5):")
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
    
    # Example 5: Options data
    example_options_analysis()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")

