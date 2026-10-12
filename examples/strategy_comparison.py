"""
Strategy performance comparison: Swing Trading Strategy vs Buy-and-Hold S&P 500
"""

import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator

from backtest.engine import BacktestEngine
from data.fetcher import DataFetcher
from strategy.indicators import SwingTradingIndicators


def run_strategy_comparison(
    symbol: str = "SPY",
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100000,
    position_size: float = 0.1
):
    """
    Run strategy backtest and compare to buy-and-hold S&P 500.
    
    Args:
        symbol: Stock symbol to test (default: SPY for S&P 500)
        start_date: Start date (YYYY-MM-DD). Default: 2 years ago
        end_date: End date (YYYY-MM-DD). Default: today
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
    """
    # Set default dates (2 years of data)
    if end_date is None:
        # Use yesterday to avoid issues with incomplete current day data
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print("="*70)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print("="*70)
    print()
    
    # Fetch data with improved retry logic and alternative methods
    print("Fetching market data...")
    fetcher = DataFetcher()
    data = None
    
    # Try multiple methods to fetch data
    
    # Method 1: Try using DataFetcher
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = fetcher.fetch_stock_data(symbol, start_date, end_date)
            if not data.empty and len(data) > 50:  # Ensure we have enough data
                print(f"✓ Fetched {len(data)} days of data for {symbol}")
                break
            else:
                raise ValueError("Empty or insufficient data returned")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠ Attempt {attempt + 1} failed: {e}")
                print("  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("⚠ DataFetcher failed. Trying direct yfinance...")
    
    # Method 2: Try direct yfinance with period parameter (more reliable)
    if data is None or data.empty or len(data) < 50:
        try:
            print("  Trying alternative fetch method...")
            ticker = yf.Ticker(symbol)
            
            # Try multiple approaches
            data_direct = None
            
            # Approach 1: Try with start/end dates
            try:
                data_direct = ticker.history(start=start_date, end=end_date)
                if data_direct.empty:
                    raise ValueError("Empty result")
            except Exception:
                pass
            
            # Approach 2: Try with period="max" and filter
            if data_direct is None or data_direct.empty:
                try:
                    data_direct = ticker.history(period="max")
                    if not data_direct.empty:
                        # Filter to requested date range
                        data_direct.index = pd.to_datetime(data_direct.index)
                        if start_date:
                            data_direct = data_direct[data_direct.index >= pd.to_datetime(start_date)]
                        if end_date:
                            end_dt_plus = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                            data_direct = data_direct[data_direct.index < end_dt_plus]
                except Exception:
                    pass
            
            # Approach 3: Try 5y period
            if data_direct is None or data_direct.empty:
                try:
                    data_direct = ticker.history(period="5y")
                    if not data_direct.empty:
                        data_direct.index = pd.to_datetime(data_direct.index)
                        if start_date:
                            data_direct = data_direct[data_direct.index >= pd.to_datetime(start_date)]
                        if end_date:
                            end_dt_plus = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                            data_direct = data_direct[data_direct.index < end_dt_plus]
                except Exception:
                    pass
            
            if data_direct is not None and not data_direct.empty:
                # Standardize columns
                data_direct.columns = [col.lower().replace(' ', '_') for col in data_direct.columns]
                data_direct = data_direct.reset_index()
                data_direct.columns = [col.lower().replace(' ', '_') for col in data_direct.columns]
                
                if 'date' not in data_direct.columns and len(data_direct.columns) > 0:
                    # Rename index column to date
                    data_direct = data_direct.rename(columns={data_direct.columns[0]: 'date'})
                
                if not data_direct.empty:
                    data = data_direct
                    print(f"✓ Fetched {len(data)} days of data using direct method")
                    if 'date' in data.columns:
                        print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
        except Exception as e:
            print(f"⚠ Direct yfinance also failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final fallback: mock data (but warn user)
    if data is None or data.empty or len(data) < 50:
        print("⚠ All real data methods failed. Using mock data for demonstration...")
        print("⚠ WARNING: Mock data may not reflect real market conditions!")
        fetcher_mock = DataFetcher(use_mock=True)
        try:
            data = fetcher_mock.fetch_stock_data(symbol, start_date, end_date)
            print(f"✓ Using mock data: {len(data)} days")
            print("⚠ CRITICAL: Results are NOT real - using synthetic data!")
            print("⚠ For real analysis, ensure yfinance API is working and internet connection is active.")
        except Exception as e2:
            print(f"✗ Error with mock data: {e2}")
            return None
    
    if data is None or data.empty:
        print("✗ Could not fetch data")
        return None
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"✗ Missing required columns: {missing}")
        return None
    
    # Verify data date range
    if 'date' in data.columns:
        data_dates = pd.to_datetime(data['date'])
        # Convert to timezone-naive for comparison if needed
        data_min = data_dates.min()
        data_max = data_dates.max()
        if data_max.tz is not None:
            data_min = data_min.tz_localize(None)
            data_max = data_max.tz_localize(None)
        
        print(f"\nData date range: {data_min.strftime('%Y-%m-%d')} to {data_max.strftime('%Y-%m-%d')}")
        print(f"Requested range: {start_date} to {end_date}")
        
        # Compare timezone-naive dates
        end_date_dt = pd.to_datetime(end_date)
        if data_max < end_date_dt - pd.Timedelta(days=30):
            print("⚠ WARNING: Data range is shorter than requested! May be using mock/limited data.")
    
    # Run strategy backtest
    print("\nRunning strategy backtest...")
    indicators = SwingTradingIndicators(data)
    signals = indicators.get_signals_df()
    
    # Debug: Check signal counts
    signal_counts = signals['signal'].value_counts()
    print(f"  Signal distribution: {dict(signal_counts)}")
    print(f"  Total buy signals (1): {len(signals[signals['signal'] == 1])}")
    print(f"  Total sell signals (-1): {len(signals[signals['signal'] == -1])}")
    print(f"  Total neutral signals (0): {len(signals[signals['signal'] == 0])}")
    
    # Show signal strength distribution
    if 'bullish_strength' in signals.columns:
        print(f"  Max bullish strength: {signals['bullish_strength'].max()}")
        print(f"  Max bearish strength: {signals['bearish_strength'].max()}")
        strong = (signals['bullish_strength'] >= 2) | (signals['bearish_strength'] >= 2)
        print(f"  Signals with strength >= 2: {int(strong.sum())}")
    
    backtest = BacktestEngine(
        data=data,
        signals=signals,
        initial_capital=initial_capital,
        position_size=position_size,
        commission=0.001,
        stop_loss_pct=0.05,  # 5% stop loss
        trailing_stop=False,  # Set to True to enable trailing stop
        trailing_stop_pct=0.03,  # 3% trailing stop
        max_drawdown_pct=0.70  # 70% max portfolio drawdown before halting (allows recovery)
    )
    
    backtest.run()
    strategy_results = backtest.results
    strategy_metrics = backtest.get_metrics()
    
    # Debug: Check why trades might not be executing
    print("\n  Backtest Results:")
    print(f"    Total trades executed: {strategy_metrics['total_trades']}")
    print(f"    Final capital: ${strategy_metrics['final_capital']:,.2f}")
    
    # Analyze trade quality
    trades = backtest.get_trades()
    if len(trades) > 0:
        print("\n  Trade Quality Analysis:")
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        if len(wins) > 0 and len(losses) > 0:
            avg_win = wins['pnl'].mean()
            avg_loss = losses['pnl'].mean()
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            print(f"    Average win: \${avg_win:,.2f}")
            print(f"    Average loss: \${avg_loss:,.2f}")
            print(f"    Win/Loss ratio: {win_loss_ratio:.2f}")
            if win_loss_ratio < 1.5:
                print("    ⚠ WARNING: Losses are larger than wins - strategy needs improvement!")
        
        # Check for large losses
        large_losses = trades[trades['pnl'] < -500]
        if len(large_losses) > 0:
            print(f"    ⚠ WARNING: {len(large_losses)} trades with losses > \$500")
            print(f"    Largest loss: \${trades['pnl'].min():,.2f}")
    
    if strategy_metrics['total_trades'] == 0:
        print("\n  ⚠ WARNING: No trades executed!")
        print("    Checking signal alignment...")
        
        # Check if signals are being passed correctly
        if hasattr(backtest, 'results') and 'signal' in backtest.results.columns:
            signal_in_results = backtest.results['signal']
            non_zero_in_results = signal_in_results[signal_in_results != 0]
            print(f"    Non-zero signals in backtest results: {len(non_zero_in_results)}")
            if len(non_zero_in_results) > 0:
                print("    First few non-zero signals:")
                print(f"      {non_zero_in_results.head(5)}")
        
        # Check original signals
        non_zero_original = signals[signals['signal'] != 0]
        print(f"    Non-zero signals in original signals: {len(non_zero_original)}")
        if len(non_zero_original) > 0:
            print("    First few non-zero original signals:")
            print(f"      {non_zero_original[['signal', 'bullish_strength', 'bearish_strength']].head(5)}")
    
    # Calculate buy-and-hold S&P 500 returns
    print("Calculating buy-and-hold S&P 500 returns...")
    
    # Ensure we have a proper date index
    data_for_buyhold = data.copy()
    if 'date' in data_for_buyhold.columns:
        data_for_buyhold = data_for_buyhold.set_index('date')
    elif data_for_buyhold.index.name != 'date':
        # If no date column, use the index
        data_for_buyhold.index.name = 'date'
    
    # Calculate buy-and-hold equity curve
    initial_price = data_for_buyhold['close'].iloc[0]
    final_price = data_for_buyhold['close'].iloc[-1]
    
    # Calculate daily returns
    data_for_buyhold['returns'] = data_for_buyhold['close'].pct_change()
    data_for_buyhold['cumulative_returns'] = (1 + data_for_buyhold['returns']).cumprod()
    
    # Buy-and-hold equity curve
    buyhold_equity = initial_capital * data_for_buyhold['cumulative_returns']
    buyhold_equity.iloc[0] = initial_capital  # Set first value
    
    # Calculate buy-and-hold metrics
    buyhold_total_return = ((final_price / initial_price) - 1) * 100
    buyhold_final_capital = buyhold_equity.iloc[-1]
    
    # Calculate buy-and-hold returns for risk metrics
    buyhold_returns = data_for_buyhold['returns'].dropna()
    buyhold_sharpe = backtest._calculate_sharpe(buyhold_returns)
    buyhold_max_drawdown = backtest._calculate_max_drawdown(buyhold_equity)
    buyhold_volatility = buyhold_returns.std() * (252 ** 0.5) * 100
    
    # Align indices for plotting - use the same index as strategy results
    strategy_equity = strategy_results['equity']
    
    # Ensure both use the same index
    # Strategy results should already have the correct index from backtest
    # Buy-and-hold should match
    if strategy_equity.index.name != buyhold_equity.index.name:
        # Try to align by converting to same index type
        if len(strategy_equity) == len(buyhold_equity):
            buyhold_equity.index = strategy_equity.index
    
    # Get common index
    common_index = strategy_equity.index.intersection(buyhold_equity.index)
    if len(common_index) == 0:
        # If no intersection, use strategy index and reindex buyhold
        common_index = strategy_equity.index
        buyhold_equity = buyhold_equity.reindex(common_index, method='ffill')
    
    strategy_equity_aligned = strategy_equity.loc[common_index]
    buyhold_equity_aligned = buyhold_equity.loc[common_index]
    
    # Print performance numbers
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Strategy':>20} {'Buy & Hold':>20}")
    print("-"*70)
    print(f"{'Total Return':<30} {strategy_metrics['total_return_pct']:>19.2f}% {buyhold_total_return:>19.2f}%")
    print(f"{'Final Capital':<30} ${strategy_metrics['final_capital']:>18,.2f} ${buyhold_final_capital:>18,.2f}")
    print(f"{'Sharpe Ratio':<30} {strategy_metrics['sharpe_ratio']:>20.2f} {buyhold_sharpe:>20.2f}")
    print(f"{'Max Drawdown':<30} {strategy_metrics['max_drawdown_pct']:>19.2f}% {buyhold_max_drawdown:>19.2f}%")
    print(f"{'Volatility (Ann)':<30} {strategy_metrics['volatility_pct']:>19.2f}% {buyhold_volatility:>19.2f}%")
    print(f"{'Total Trades':<30} {strategy_metrics['total_trades']:>20} {'N/A':>20}")
    print(f"{'Win Rate':<30} {strategy_metrics['win_rate_pct']:>19.2f}% {'N/A':>20}")
    print(f"{'Profit Factor':<30} {strategy_metrics['profit_factor']:>20.2f} {'N/A':>20}")
    print("="*70)
    
    # Calculate outperformance
    outperformance = strategy_metrics['total_return_pct'] - buyhold_total_return
    print(f"\nStrategy Outperformance: {outperformance:+.2f}%")
    
    if outperformance > 0:
        print("✓ Strategy outperformed buy-and-hold")
    else:
        print("✗ Strategy underperformed buy-and-hold")
    
    # Create enhanced comparison chart
    print("\nGenerating comparison chart...")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Main equity curve plot
    ax1 = fig.add_subplot(gs[0])
    
    # Plot both equity curves
    ax1.plot(
        common_index,
        strategy_equity_aligned,
        label='Swing Trading Strategy',
        color='#2E86AB',
        linewidth=2.8,
        alpha=0.95,
        zorder=3
    )
    
    ax1.plot(
        common_index,
        buyhold_equity_aligned,
        label='Buy & Hold S&P 500',
        color='#A23B72',
        linewidth=2.8,
        linestyle='--',
        alpha=0.95,
        zorder=2
    )
    
    # Add initial capital reference line
    ax1.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial Capital')
    
    # Formatting main plot
    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=14, fontweight='bold')
    ax1.set_title(
        f'Strategy Performance Comparison: {symbol}',
        fontsize=17,
        fontweight='bold',
        pad=20
    )
    
    # Add subtitle with date range - format properly
    if len(common_index) > 0:
        start_str = pd.to_datetime(common_index[0]).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(common_index[-1]).strftime("%Y-%m-%d")
        date_range_str = f'{start_str} to {end_str} ({len(common_index)} trading days)'
        ax1.text(0.5, 0.98, date_range_str, transform=ax1.transAxes, 
                fontsize=12, ha='center', style='italic', color='gray')
    
    # Legend
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Better x-axis date formatting for longer periods
    if len(common_index) > 0:
        # Use year labels for long periods
        if len(common_index) > 500:
            ax1.xaxis.set_major_locator(YearLocator())
            ax1.xaxis.set_minor_locator(MonthLocator((1, 7)))  # Jan and Jul
            ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
        else:
            ax1.xaxis.set_major_locator(MonthLocator(interval=max(1, len(common_index) // 12)))
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add text box with key metrics
    textstr = (
        f'Strategy Return: {strategy_metrics["total_return_pct"]:.2f}%\n'
        f'Buy & Hold Return: {buyhold_total_return:.2f}%\n'
        f'Outperformance: {outperformance:+.2f}%\n'
        f'Trades: {strategy_metrics["total_trades"]}\n'
        f'Win Rate: {strategy_metrics["win_rate_pct"]:.1f}%\n'
        f'Sharpe Ratio: {strategy_metrics["sharpe_ratio"]:.2f}'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                 edgecolor='black', linewidth=1.5)
    ax1.text(
        0.98, 0.02, textstr,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props,
        family='monospace'
    )
    
    # Add drawdown subplot
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate drawdowns
    strategy_peak = strategy_equity_aligned.expanding().max()
    strategy_dd = (strategy_equity_aligned - strategy_peak) / strategy_peak * 100
    
    buyhold_peak = buyhold_equity_aligned.expanding().max()
    buyhold_dd = (buyhold_equity_aligned - buyhold_peak) / buyhold_peak * 100
    
    ax2.fill_between(common_index, strategy_dd, 0, alpha=0.3, color='#2E86AB', label='Strategy Drawdown')
    ax2.fill_between(common_index, buyhold_dd, 0, alpha=0.3, color='#A23B72', label='Buy & Hold Drawdown')
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Format x-axis for drawdown plot
    if len(common_index) > 500:
        ax2.xaxis.set_major_locator(YearLocator())
        ax2.xaxis.set_major_formatter(DateFormatter('%Y'))
    else:
        ax2.xaxis.set_major_locator(MonthLocator(interval=max(1, len(common_index) // 12)))
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save chart
    output_file = 'strategy_vs_buyhold_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved chart to {output_file}")
    
    plt.show()
    
    return {
        'strategy_metrics': strategy_metrics,
        'buyhold_metrics': {
            'total_return_pct': buyhold_total_return,
            'final_capital': buyhold_final_capital,
            'sharpe_ratio': buyhold_sharpe,
            'max_drawdown_pct': buyhold_max_drawdown,
            'volatility_pct': buyhold_volatility
        },
        'outperformance': outperformance,
        'strategy_equity': strategy_equity_aligned,
        'buyhold_equity': buyhold_equity_aligned
    }


if __name__ == "__main__":
    # Run comparison from 2021 to 2026 (or current date if before 2026)
    start_date = "2021-01-01"
    # Use current date or end of 2025, whichever is earlier
    today = datetime.now()
    if today.year >= 2026:
        end_date = "2026-01-01"
    else:
        end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday to avoid incomplete data
    
    print("Testing strategy from 2021 to 2026")
    print(f"Date range: {start_date} to {end_date}")
    print("="*70)
    
    results = run_strategy_comparison(
        symbol="SPY",
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        position_size=0.1
    )

