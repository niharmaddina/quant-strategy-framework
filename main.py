"""
Main execution script for quant strategy and backtesting.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest.engine import BacktestEngine
from backtest.factory import create_backtest_engine
from backtest.visualizer import BacktestVisualizer
from data.fetcher import DataFetcher
from strategy.indicators import SwingTradingIndicators
from universe.constructor import UniverseConstructor

# Default frozen universe (manual / survivorship-bias minimizing)
DEFAULT_FROZEN_UNIVERSE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "universe",
    "universe_mean_reversion_manual_50.txt",
)


def run_strategy(
    symbols: list,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    position_size: float = 0.1,
    plot: bool = True,
    engine_name: str = "vectorbt",
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
    print("QUANT STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"{'='*60}\n")
    
    # Fetch data
    print("Fetching market data...")
    fetcher = DataFetcher()
    
    all_results = {}
    total_symbols = len(symbols)
    
    # Batch fetch all data first to minimize API calls
    print(f"\nFetching data for {total_symbols} symbols in batches...")
    all_data = fetcher.fetch_multiple_symbols(
        symbols, start_date, end_date, interval="1d", batch_size=5
    )
    
    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{total_symbols}] Processing {symbol}...")
        
        try:
            # Use pre-fetched data
            data = all_data.get(symbol)
            if data is None or data.empty:
                print(f"  ⚠ No data available for {symbol}, skipping...")
                continue
            print(f"  ✓ Using {len(data)} days of data")
            
            # Show data summary
            summary = fetcher.get_data_summary(data)
            if summary['missing_data']:
                print(f"  ⚠ Missing data: {summary['missing_data']}")
            
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
            try:
                backtest = create_backtest_engine(
                    engine=engine_name,  # type: ignore[arg-type]
                    data=data,
                    signals=signals,
                    initial_capital=initial_capital,
                    position_size=position_size,
                )
            except ImportError as e:
                print(f"  ⚠ {e}")
                print("  Falling back to legacy backtester.")
                backtest = BacktestEngine(
                    data=data,
                    signals=signals,
                    initial_capital=initial_capital,
                    position_size=position_size,
                )
            
            backtest.run()
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
                print("  Generating visualizations...")
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
        
        print("\nAverage Metrics:")
        print(f"  Avg Return: {avg_return:.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Avg Win Rate: {avg_win_rate:.2f}%")
    
    print(f"\n{'='*60}\n")
    
    return all_results


def main():
    """Main entry point."""
    try:
        from utils.validation import (
            validate_symbol, validate_date_range, validate_capital, 
            validate_position_size, normalize_symbol
        )
    except ImportError:
        # Fallback if utils not available
        def validate_symbol(s): return bool(s)
        def validate_date_range(s, e): return (True, None)
        def validate_capital(a): return a > 0
        def validate_position_size(s): return 0 < s <= 1
        def normalize_symbol(s): return s.upper().strip() if s else ""
    
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
    parser.add_argument(
        '--engine',
        type=str,
        default="vectorbt",
        choices=["vectorbt", "legacy"],
        help="Backtest engine to use (default: vectorbt; falls back to legacy if vectorbt isn't installed)",
    )
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--universe', action='store_true',
                        help='Use frozen manual universe (recommended; avoids survivorship/selection bias)')
    parser.add_argument('--universe-file', type=str, default=None,
                        help='Load a fixed universe from a .txt or .csv file (avoids optimized selection bias)')
    parser.add_argument('--universe-manual', action='store_true',
                        help='Build a manual sector-balanced universe (no metadata filters; avoids look-ahead from current fundamentals)')
    parser.add_argument('--target-size', type=int, default=20,
                        help='Target number of symbols when building a universe (default: 20)')
    parser.add_argument('--stocks-per-sector', type=int, default=None,
                        help='If set with --universe-manual, pick N tickers per sector (overrides --target-size)')
    parser.add_argument('--export', type=str, default=None,
                        help='Export results to directory (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    args.symbols = [normalize_symbol(s) for s in args.symbols]
    for symbol in args.symbols:
        if not validate_symbol(symbol):
            print(f"Warning: Invalid symbol format: {symbol}")
    
    if not validate_capital(args.capital):
        raise ValueError(f"Invalid capital: {args.capital}. Must be positive.")
    
    if not validate_position_size(args.position_size):
        raise ValueError(f"Invalid position size: {args.position_size}. Must be between 0 and 1.")
    
    # Set default dates
    if args.end is None:
        args.end = datetime.now().strftime('%Y-%m-%d')
    if args.start is None:
        start_date = datetime.now() - timedelta(days=365)
        args.start = start_date.strftime('%Y-%m-%d')
    
    # Validate date range
    is_valid, error_msg = validate_date_range(args.start, args.end)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Use universe if requested
    if args.universe:
        # Default: frozen universe to minimize survivorship/selection bias and avoid yfinance metadata calls
        if args.universe_file is None and not args.universe_manual:
            args.universe_file = DEFAULT_FROZEN_UNIVERSE_FILE

    if args.universe_file:
        # Frozen universe from file (best for avoiding selection/survivorship bias)
        loaded = UniverseConstructor.load_universe(args.universe_file)
        args.symbols = [normalize_symbol(s) for s in loaded]
        print(f"✓ Loaded universe from file: {len(args.symbols)} symbols")
        print(f"Symbols: {', '.join(args.symbols)}")
    elif args.universe_manual:
        # Manual sector-balanced universe (no yfinance metadata filters)
        print("Building manual sector-balanced universe...")
        universe = UniverseConstructor()
        args.symbols = universe.create_manual_sector_balanced_universe(
            target_size=args.target_size,
            stocks_per_sector=args.stocks_per_sector,
            include_index_etfs=True
        )
        print(f"✓ Built manual universe: {len(args.symbols)} symbols")
        print(f"Symbols: {', '.join(args.symbols)}")
    # Note: we intentionally do NOT auto-build an "optimized" universe here anymore,
    # because it relies on current yfinance metadata and can introduce selection bias.
    
    # Run strategy
    results = run_strategy(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        position_size=args.position_size,
        plot=not args.no_plot,
        engine_name=args.engine,
    )
    
    # Export results if requested
    if args.export:
        export_dir = args.export
        os.makedirs(export_dir, exist_ok=True)
        
        print(f"\nExporting results to {export_dir}...")
        for symbol, result_data in results.items():
            backtest = result_data['backtest']
            
            # Export results
            backtest.export_results_to_csv(f"{export_dir}/{symbol}_results.csv")
            backtest.export_metrics_to_json(f"{export_dir}/{symbol}_metrics.json")
            
            # Export trades if available
            trades = backtest.get_trades()
            if not trades.empty:
                backtest.export_trades_to_csv(f"{export_dir}/{symbol}_trades.csv")
        
        print(f"✓ Exported results for {len(results)} symbols")


if __name__ == "__main__":
    main()

