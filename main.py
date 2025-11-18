"""
Main execution script for quant strategy and backtesting.
"""

import pandas as pd
from datetime import datetime, timedelta
import argparse
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import DataFetcher
from universe.constructor import UniverseConstructor
from strategy.indicators import SwingTradingIndicators
from backtest.engine import BacktestEngine
from backtest.visualizer import BacktestVisualizer


def run_strategy(
    symbols: list,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    position_size: float = 0.1,
    plot: bool = True
):
    """
    Run strategy analysis and backtesting.
    
    Args:
        symbols: List of stock symbols to analyze
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        plot: Whether to generate plots
    """
    print(f"\n{'='*60}")
    print(f"QUANT STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"{'='*60}\n")
    
    # Fetch data
    print("Fetching market data...")
    fetcher = DataFetcher()
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        try:
            # Fetch stock data
            data = fetcher.fetch_stock_data(symbol, start_date, end_date)
            print(f"  ✓ Fetched {len(data)} days of data")
            
            # Calculate indicators and generate signals
            print("  Calculating indicators...")
            indicators = SwingTradingIndicators(data)
            signals = indicators.get_signals_df()
            
            # Get current signals for manual trading
            current_signal = indicators.get_current_signals()
            print(f"\n  Current Signal for {symbol}:")
            print(f"    Type: {current_signal.get('signal', 'N/A')}")
            print(f"    Confidence: {current_signal.get('confidence', 0):.2%}")
            print(f"    Price: ${current_signal.get('price', 0):.2f}")
            if current_signal.get('rsi'):
                print(f"    RSI: {current_signal['rsi']:.2f}")
            
            # Run backtest
            print("  Running backtest...")
            backtest = BacktestEngine(
                data=data,
                signals=signals,
                initial_capital=initial_capital,
                position_size=position_size
            )
            
            results = backtest.run()
            metrics = backtest.get_metrics()
            
            print(f"\n  Backtest Results for {symbol}:")
            backtest.print_metrics()
            
            # Store results
            all_results[symbol] = {
                'backtest': backtest,
                'indicators': indicators,
                'metrics': metrics,
                'current_signal': current_signal
            }
            
            # Generate plots
            if plot:
                print(f"  Generating visualizations...")
                visualizer = BacktestVisualizer(backtest)
                
                # Save individual plots
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                
                fig = visualizer.plot_all()
                fig.savefig(f"{output_dir}/{symbol}_backtest.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Saved plot to {output_dir}/{symbol}_backtest.png")
        
        except Exception as e:
            print(f"  ✗ Error processing {symbol}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        summary_df = pd.DataFrame({
            symbol: result['metrics'] 
            for symbol, result in all_results.items()
        }).T
        
        print("\nPerformance by Symbol:")
        print(summary_df[['total_return_pct', 'win_rate_pct', 'sharpe_ratio', 'max_drawdown_pct']].to_string())
        
        # Average metrics
        avg_return = summary_df['total_return_pct'].mean()
        avg_sharpe = summary_df['sharpe_ratio'].mean()
        avg_win_rate = summary_df['win_rate_pct'].mean()
        
        print(f"\nAverage Metrics:")
        print(f"  Avg Return: {avg_return:.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Avg Win Rate: {avg_win_rate:.2f}%")
    
    print(f"\n{'='*60}\n")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Quant Strategy Backtesting')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT'],
                        help='Stock symbols to analyze (default: AAPL MSFT)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD). Default: 1 year ago')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD). Default: today')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as fraction of capital (default: 0.1)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--universe', action='store_true',
                        help='Use predefined balanced universe')
    
    args = parser.parse_args()
    
    # Set default dates
    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')
    if args.start is None:
        start_date = datetime.now() - timedelta(days=365)
        args.start = start_date.strftime('%Y-%m-%d')
    
    # Use universe if requested
    if args.universe:
        universe = UniverseConstructor()
        args.symbols = universe.create_sector_balanced_universe(stocks_per_sector=2)
        print(f"Using balanced universe: {args.symbols}")
    
    # Run strategy
    run_strategy(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        position_size=args.position_size,
        plot=not args.no_plot
    )


if __name__ == "__main__":
    main()

