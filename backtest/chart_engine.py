"""
Advanced charting and visualization engine for backtest results.
Provides comprehensive charts, comparisons, and analysis visualizations.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:  # type hints only; a real import would be circular
    from backtest.engine import BacktestEngine


class ChartEngine:
    """
    Advanced charting engine for backtest visualization.
    
    Features:
    - Comprehensive performance charts
    - Strategy comparisons
    - Multi-symbol analysis
    - Mean reversion visualizations
    - Trade analysis charts
    - Rolling metrics
    - Professional styling
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (16, 10)):
        """
        Initialize chart engine.
        
        Args:
            style: Matplotlib style (default: 'seaborn-v0_8')
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.default_figsize = figsize
        plt.style.use(style)
        
        # Color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'danger': '#D00000',
            'warning': '#FFB627',
            'info': '#06AED5',
            'light': '#F0F0F0',
            'dark': '#1A1A1A'
        }
    
    def plot_comprehensive_dashboard(
        self,
        backtest_engine,
        title: str = "Backtest Analysis Dashboard",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with all key metrics and charts.
        
        Args:
            backtest_engine: BacktestEngine instance with results
            title: Dashboard title
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if backtest_engine.results is None or backtest_engine.results.empty:
            raise ValueError("No backtest results available. Run backtest first.")
        
        fig = plt.figure(figsize=self.default_figsize)
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        results = backtest_engine.results
        metrics = backtest_engine.get_metrics()
        
        # 1. Equity Curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve_enhanced(ax1, backtest_engine)
        
        # 2. Drawdown Chart
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(ax2, results)
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_returns_distribution(ax3, results)
        
        # 4. Monthly Returns Heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_monthly_returns(ax4, results)
        
        # 5. Rolling Sharpe Ratio
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_rolling_sharpe(ax5, results, risk_free_rate=backtest_engine.risk_free_rate)
        
        # 6. Rolling Volatility
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_rolling_volatility(ax6, results)
        
        # 7. Trade Analysis
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_trade_analysis(ax7, backtest_engine)
        
        # 8. Price Chart with Signals (bottom, full width)
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_price_with_signals(ax8, backtest_engine)
        
        # Add title
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        
        # Add metrics text box
        metrics_text = self._format_metrics_text(metrics)
        fig.text(0.02, 0.02, metrics_text, fontsize=9, 
                verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_strategy_comparison(
        self,
        strategies: Dict[str, 'BacktestEngine'],
        benchmark: Optional['BacktestEngine'] = None,
        title: str = "Strategy Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare multiple strategies side-by-side.
        
        Args:
            strategies: Dictionary of {strategy_name: BacktestEngine}
            benchmark: Optional benchmark strategy (e.g., buy-and-hold)
            title: Chart title
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curves Comparison (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_comparison(ax1, strategies, benchmark)
        
        # 2. Returns Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_returns_comparison(ax2, strategies, benchmark)
        
        # 3. Drawdown Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown_comparison(ax3, strategies, benchmark)
        
        # 4. Metrics Comparison Table
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_metrics_table(ax4, strategies, benchmark)
        
        # 5. Risk-Return Scatter
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_risk_return_scatter(ax5, strategies, benchmark)
        
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_mean_reversion_analysis(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        signals: pd.DataFrame,
        title: str = "Mean Reversion Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot mean reversion analysis with indicators and signals.
        
        Args:
            data: Price data DataFrame
            indicators: Mean reversion indicators DataFrame
            signals: Signals DataFrame
            title: Chart title
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Price with Bollinger Bands and Mean
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_with_bands(ax1, data, indicators, signals)
        
        # 2. Z-Score
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_z_score(ax2, indicators, signals)
        
        # 3. Mean Reversion Strength
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_mean_reversion_strength(ax3, indicators, signals)
        
        # 4. Distance from Mean
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_distance_from_mean(ax4, indicators)
        
        # 5. Volume Profile
        ax5 = fig.add_subplot(gs[2, 1])
        if 'volume' in data.columns:
            self._plot_volume_profile(ax5, data, indicators)
        
        # 6. Signal Timeline
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_signal_timeline(ax6, data, signals)
        
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_multi_symbol_comparison(
        self,
        symbol_results: Dict[str, Dict],
        title: str = "Multi-Symbol Performance Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare performance across multiple symbols.
        
        Args:
            symbol_results: Dict of {symbol: {metrics_dict, engine}}
            title: Chart title
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract metrics
        metrics_df = pd.DataFrame({
            symbol: result['metrics'] 
            for symbol, result in symbol_results.items()
        }).T
        
        # 1. Returns Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_returns_bar(ax1, metrics_df)
        
        # 2. Sharpe Ratio Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sharpe_comparison(ax2, metrics_df)
        
        # 3. Win Rate Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_win_rate_comparison(ax3, metrics_df)
        
        # 4. Max Drawdown Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown_comparison_bar(ax4, metrics_df)
        
        # 5. Risk-Return Scatter
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_risk_return_multi(ax5, metrics_df)
        
        # 6. Metrics Table
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_metrics_table_multi(ax6, metrics_df)
        
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _plot_equity_curve_enhanced(self, ax, engine):
        """Plot enhanced equity curve with annotations."""
        results = engine.results
        equity = results['equity']
        
        # Main equity curve
        ax.plot(equity.index, equity.values, linewidth=2.5, 
               color=self.colors['primary'], label='Equity Curve', zorder=3)
        
        # Initial capital line
        ax.axhline(y=engine.initial_capital, color=self.colors['danger'], 
                  linestyle='--', linewidth=1.5, alpha=0.7, 
                  label=f'Initial Capital (${engine.initial_capital:,.0f})', zorder=1)
        
        # Fill area
        ax.fill_between(equity.index, equity.values, engine.initial_capital,
                       where=(equity.values >= engine.initial_capital),
                       alpha=0.2, color=self.colors['success'], label='Profit')
        ax.fill_between(equity.index, equity.values, engine.initial_capital,
                       where=(equity.values < engine.initial_capital),
                       alpha=0.2, color=self.colors['danger'], label='Loss')
        
        # Final value annotation
        final_value = equity.iloc[-1]
        ax.annotate(f'Final: ${final_value:,.0f}',
                   xy=(equity.index[-1], final_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_drawdown(self, ax, results):
        """Plot drawdown chart."""
        equity = results['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       alpha=0.4, color=self.colors['danger'], label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, linewidth=2, 
               color=self.colors['danger'], zorder=2)
        
        # Max drawdown annotation
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_val:.2f}%',
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(10, -10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_returns_distribution(self, ax, results):
        """Plot returns distribution."""
        returns = results['returns'].dropna() * 100
        
        ax.hist(returns, bins=50, edgecolor='black', alpha=0.7, 
               color=self.colors['info'], label='Returns')
        ax.axvline(x=0, color=self.colors['danger'], linestyle='--', 
                  linewidth=2, label='Zero Return')
        ax.axvline(x=returns.mean(), color=self.colors['success'], 
                  linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        
        ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Returns (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_monthly_returns(self, ax, results):
        """Plot monthly returns heatmap."""
        if 'returns' not in results.columns:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center')
            return
        
        returns = results['returns'].dropna()
        if len(returns) == 0:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center')
            return
        
        # Convert to monthly returns
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        else:
            # If index is not datetime, try to convert
            try:
                returns.index = pd.to_datetime(returns.index)
                monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            except Exception:
                ax.text(0.5, 0.5, 'Cannot calculate monthly returns', ha='center', va='center')
                return
        
        if len(monthly) == 0:
            ax.text(0.5, 0.5, 'No monthly data', ha='center', va='center')
            return
        
        # Create heatmap data
        monthly_df = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })
        
        if len(monthly_df) > 0:
            pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', 
                          vmin=-10, vmax=10, interpolation='nearest')
            
            ax.set_xticks(range(12))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            
            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(12):
                    if j < len(pivot.columns):
                        val = pivot.iloc[i, j]
                        if not pd.isna(val):
                            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                                   color='white' if abs(val) > 5 else 'black', fontsize=7)
        
        ax.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('Year', fontsize=10)
        plt.colorbar(im, ax=ax, label='Return %')
    
    def _plot_rolling_sharpe(self, ax, results, window: int = 30, risk_free_rate: float = 0.02):
        """Plot rolling Sharpe ratio."""
        if 'returns' not in results.columns:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center')
            return
        
        returns = results['returns'].dropna()
        if len(returns) < window:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return
        
        # Calculate rolling Sharpe
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std.replace(0, np.nan)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
               linewidth=2, color=self.colors['primary'], label='Rolling Sharpe')
        ax.axhline(y=0, color=self.colors['danger'], linestyle='--', alpha=0.7)
        ax.axhline(y=1, color=self.colors['success'], linestyle='--', alpha=0.5, label='Sharpe = 1')
        
        ax.set_title(f'Rolling Sharpe Ratio ({window}-period)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_rolling_volatility(self, ax, results, window: int = 30):
        """Plot rolling volatility."""
        if 'returns' not in results.columns:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center')
            return
        
        returns = results['returns'].dropna()
        if len(returns) < window:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return
        
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
        
        ax.plot(rolling_vol.index, rolling_vol.values, 
               linewidth=2, color=self.colors['warning'], label='Rolling Volatility')
        ax.fill_between(rolling_vol.index, rolling_vol.values, alpha=0.3, 
                       color=self.colors['warning'])
        
        ax.set_title(f'Rolling Volatility ({window}-period)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annualized Volatility (%)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_trade_analysis(self, ax, engine):
        """Plot trade analysis."""
        trades = engine.get_trades()
        
        if trades.empty:
            ax.text(0.5, 0.5, 'No trades to analyze', ha='center', va='center')
            return
        
        # Win/Loss distribution
        winning = trades[trades['pnl'] > 0]['pnl']
        losing = trades[trades['pnl'] <= 0]['pnl']
        
        if len(winning) > 0:
            ax.hist(winning, bins=20, alpha=0.6, color=self.colors['success'], 
                   label=f'Wins ({len(winning)})', edgecolor='black')
        if len(losing) > 0:
            ax.hist(losing, bins=20, alpha=0.6, color=self.colors['danger'], 
                   label=f'Losses ({len(losing)})', edgecolor='black')
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('P&L ($)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_price_with_signals(self, ax, engine):
        """Plot price chart with trading signals."""
        results = engine.results
        
        if 'price' not in results.columns:
            ax.text(0.5, 0.5, 'No price data', ha='center', va='center')
            return
        
        price = results['price']
        ax.plot(price.index, price.values, linewidth=1.5, 
               color=self.colors['dark'], label='Price', zorder=1)
        
        # Add signals if available
        if 'signal' in results.columns:
            signals = results['signal']
            buy_signals = signals[signals == 1]
            sell_signals = signals[signals == -1]
            
            if len(buy_signals) > 0:
                buy_prices = price.reindex(buy_signals.index)
                ax.scatter(buy_signals.index, buy_prices.values, 
                          color=self.colors['success'], marker='^', s=150, 
                          label='Buy Signal', zorder=5, edgecolors='black', linewidths=1)
            
            if len(sell_signals) > 0:
                sell_prices = price.reindex(sell_signals.index)
                ax.scatter(sell_signals.index, sell_prices.values, 
                          color=self.colors['danger'], marker='v', s=150, 
                          label='Sell Signal', zorder=5, edgecolors='black', linewidths=1)
        
        ax.set_title('Price Chart with Trading Signals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_equity_comparison(self, ax, strategies, benchmark=None):
        """Plot equity curves for multiple strategies."""
        colors_list = [self.colors['primary'], self.colors['secondary'], 
                      self.colors['success'], self.colors['info'], self.colors['warning']]
        
        for i, (name, engine) in enumerate(strategies.items()):
            if engine.results is not None and 'equity' in engine.results.columns:
                equity = engine.results['equity']
                color = colors_list[i % len(colors_list)]
                ax.plot(equity.index, equity.values, linewidth=2.5, 
                       label=name, color=color, zorder=3)
        
        if benchmark and benchmark.results is not None:
            if 'equity' in benchmark.results.columns:
                equity = benchmark.results['equity']
                ax.plot(equity.index, equity.values, linewidth=2, 
                       linestyle='--', label='Benchmark', color=self.colors['dark'], 
                       alpha=0.7, zorder=2)
        
        ax.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_returns_comparison(self, ax, strategies, benchmark=None):
        """Plot returns comparison bar chart."""
        returns_data = {}
        
        for name, engine in strategies.items():
            metrics = engine.get_metrics()
            returns_data[name] = metrics.get('total_return_pct', 0)
        
        if benchmark:
            metrics = benchmark.get_metrics()
            returns_data['Benchmark'] = metrics.get('total_return_pct', 0)
        
        names = list(returns_data.keys())
        values = list(returns_data.values())
        colors_list = [self.colors['primary'] if 'Benchmark' not in name 
                      else self.colors['dark'] for name in names]
        
        bars = ax.bar(names, values, color=colors_list, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
                   fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Total Returns Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_drawdown_comparison(self, ax, strategies, benchmark=None):
        """Plot drawdown comparison."""
        for name, engine in strategies.items():
            if engine.results is not None and 'equity' in engine.results.columns:
                equity = engine.results['equity']
                peak = equity.expanding().max()
                drawdown = (equity - peak) / peak * 100
                ax.plot(drawdown.index, drawdown.values, linewidth=2, label=name)
        
        if benchmark and benchmark.results is not None:
            if 'equity' in benchmark.results.columns:
                equity = benchmark.results['equity']
                peak = equity.expanding().max()
                drawdown = (equity - peak) / peak * 100
                ax.plot(drawdown.index, drawdown.values, linewidth=2, 
                       linestyle='--', label='Benchmark', alpha=0.7)
        
        ax.fill_between(ax.get_xlim()[0], ax.get_xlim()[1], 0, alpha=0.1, color='red')
        ax.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_metrics_table(self, ax, strategies, benchmark=None):
        """Plot metrics comparison table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        data = []
        for name, engine in strategies.items():
            metrics = engine.get_metrics()
            data.append({
                'Strategy': name,
                'Return %': f"{metrics.get('total_return_pct', 0):.2f}",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max DD %': f"{metrics.get('max_drawdown_pct', 0):.2f}",
                'Win Rate %': f"{metrics.get('win_rate_pct', 0):.2f}"
            })
        
        if benchmark:
            metrics = benchmark.get_metrics()
            data.append({
                'Strategy': 'Benchmark',
                'Return %': f"{metrics.get('total_return_pct', 0):.2f}",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max DD %': f"{metrics.get('max_drawdown_pct', 0):.2f}",
                'Win Rate %': f"{metrics.get('win_rate_pct', 0):.2f}"
            })
        
        df = pd.DataFrame(data)
        table = ax.table(cellText=df.values, colLabels=df.columns,
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    def _plot_risk_return_scatter(self, ax, strategies, benchmark=None):
        """Plot risk-return scatter."""
        returns = []
        risks = []
        labels = []
        
        for name, engine in strategies.items():
            metrics = engine.get_metrics()
            returns.append(metrics.get('total_return_pct', 0))
            risks.append(metrics.get('volatility_pct', 0))
            labels.append(name)
        
        if benchmark:
            metrics = benchmark.get_metrics()
            returns.append(metrics.get('total_return_pct', 0))
            risks.append(metrics.get('volatility_pct', 0))
            labels.append('Benchmark')
        
        ax.scatter(risks, returns, s=200, alpha=0.6, 
                           c=range(len(labels)), cmap='viridis', edgecolors='black')
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (risks[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax.set_xlabel('Volatility (%)', fontsize=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    # Mean reversion specific plots
    def _plot_price_with_bands(self, ax, data, indicators, signals):
        """Plot price with Bollinger Bands."""
        if 'close' not in data.columns:
            return
        
        price = data['close']
        ax.plot(price.index, price.values, linewidth=2, color=self.colors['dark'], label='Price')
        
        if 'bb_upper' in indicators.columns:
            ax.plot(indicators.index, indicators['bb_upper'], 
                   linewidth=1, color=self.colors['danger'], linestyle='--', alpha=0.7, label='Upper Band')
            ax.plot(indicators.index, indicators['bb_lower'], 
                   linewidth=1, color=self.colors['success'], linestyle='--', alpha=0.7, label='Lower Band')
            ax.fill_between(indicators.index, indicators['bb_upper'], indicators['bb_lower'],
                          alpha=0.1, color=self.colors['info'])
        
        if 'mean' in indicators.columns:
            ax.plot(indicators.index, indicators['mean'], 
                   linewidth=1.5, color=self.colors['warning'], linestyle=':', label='Mean')
        
        # Add signals
        if 'signal' in signals.columns:
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]
            
            if len(buy_signals) > 0:
                buy_prices = price.reindex(buy_signals.index)
                ax.scatter(buy_signals.index, buy_prices.values, 
                          color=self.colors['success'], marker='^', s=150, 
                          label='Buy', zorder=5, edgecolors='black')
            
            if len(sell_signals) > 0:
                sell_prices = price.reindex(sell_signals.index)
                ax.scatter(sell_signals.index, sell_prices.values, 
                          color=self.colors['danger'], marker='v', s=150, 
                          label='Sell', zorder=5, edgecolors='black')
        
        ax.set_title('Price with Bollinger Bands', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_z_score(self, ax, indicators, signals):
        """Plot Z-score with thresholds."""
        if 'z_score' not in indicators.columns:
            return
        
        z_score = indicators['z_score']
        ax.plot(z_score.index, z_score.values, linewidth=2, color=self.colors['primary'], label='Z-Score')
        ax.axhline(y=2.0, color=self.colors['danger'], linestyle='--', label='Overbought (+2)')
        ax.axhline(y=-2.0, color=self.colors['success'], linestyle='--', label='Oversold (-2)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.fill_between(z_score.index, 2.0, z_score.values, where=(z_score.values >= 2.0),
                       alpha=0.2, color=self.colors['danger'])
        ax.fill_between(z_score.index, -2.0, z_score.values, where=(z_score.values <= -2.0),
                       alpha=0.2, color=self.colors['success'])
        
        ax.set_title('Z-Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z-Score', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_mean_reversion_strength(self, ax, indicators, signals):
        """Plot mean reversion strength."""
        if 'mean_reversion_strength' not in indicators.columns:
            return
        
        strength = indicators['mean_reversion_strength']
        ax.plot(strength.index, strength.values, linewidth=2, color=self.colors['secondary'], label='Strength')
        ax.axhline(y=5.0, color=self.colors['warning'], linestyle='--', alpha=0.7, label='Strong (5.0)')
        ax.fill_between(strength.index, strength.values, alpha=0.3, color=self.colors['secondary'])
        
        ax.set_title('Mean Reversion Strength', fontsize=12, fontweight='bold')
        ax.set_ylabel('Strength (0-10)', fontsize=10)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_distance_from_mean(self, ax, indicators):
        """Plot distance from mean percentage."""
        if 'distance_from_mean_pct' not in indicators.columns:
            return
        
        distance = indicators['distance_from_mean_pct']
        ax.plot(distance.index, distance.values, linewidth=2, color=self.colors['info'], label='Distance')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=2.0, color=self.colors['danger'], linestyle='--', alpha=0.5, label='+2%')
        ax.axhline(y=-2.0, color=self.colors['success'], linestyle='--', alpha=0.5, label='-2%')
        ax.fill_between(distance.index, 0, distance.values, 
                       where=(distance.values >= 0), alpha=0.2, color=self.colors['danger'])
        ax.fill_between(distance.index, 0, distance.values, 
                       where=(distance.values < 0), alpha=0.2, color=self.colors['success'])
        
        ax.set_title('Distance from Mean', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance (%)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_volume_profile(self, ax, data, indicators):
        """Plot volume profile."""
        if 'volume' not in data.columns:
            ax.text(0.5, 0.5, 'No volume data', ha='center', va='center')
            return
        
        volume = data['volume']
        if 'volume_ma' in indicators.columns:
            volume_ma = indicators['volume_ma']
            ax.plot(volume.index, volume.values, linewidth=1, color=self.colors['info'], 
                   alpha=0.6, label='Volume')
            ax.plot(volume_ma.index, volume_ma.values, linewidth=2, 
                   color=self.colors['warning'], label='Volume MA')
        else:
            ax.plot(volume.index, volume.values, linewidth=1, color=self.colors['info'], label='Volume')
        
        ax.set_title('Volume Profile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_signal_timeline(self, ax, data, signals):
        """Plot signal timeline."""
        if 'signal' not in signals.columns:
            return
        
        signal_values = signals['signal']
        buy_times = signal_values[signal_values == 1].index
        sell_times = signal_values[signal_values == -1].index
        
        if len(buy_times) > 0:
            ax.scatter(buy_times, [1] * len(buy_times), color=self.colors['success'], 
                      marker='^', s=200, label='Buy Signals', zorder=5)
        if len(sell_times) > 0:
            ax.scatter(sell_times, [-1] * len(sell_times), color=self.colors['danger'], 
                      marker='v', s=200, label='Sell Signals', zorder=5)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Signal Timeline', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal', fontsize=10)
        ax.set_ylim(-1.5, 1.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_returns_bar(self, ax, metrics_df):
        """Plot returns bar chart."""
        if 'total_return_pct' not in metrics_df.columns:
            return
        
        returns = metrics_df['total_return_pct'].sort_values()
        colors_list = [self.colors['success'] if x >= 0 else self.colors['danger'] for x in returns.values]
        
        ax.barh(returns.index, returns.values, color=colors_list, alpha=0.7, edgecolor='black')
        
        for i, (idx, val) in enumerate(returns.items()):
            ax.text(val, i, f'{val:.1f}%', va='center', 
                   ha='left' if val >= 0 else 'right', fontweight='bold')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Total Returns by Symbol', fontsize=12, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_sharpe_comparison(self, ax, metrics_df):
        """Plot Sharpe ratio comparison."""
        if 'sharpe_ratio' not in metrics_df.columns:
            return
        
        sharpe = metrics_df['sharpe_ratio'].sort_values()
        colors_list = [self.colors['success'] if x >= 1 else 
                      (self.colors['warning'] if x >= 0 else self.colors['danger']) 
                      for x in sharpe.values]
        
        ax.barh(sharpe.index, sharpe.values, color=colors_list, alpha=0.7, edgecolor='black')
        
        ax.axvline(x=1, color=self.colors['success'], linestyle='--', alpha=0.7, label='Sharpe = 1')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sharpe Ratio', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_win_rate_comparison(self, ax, metrics_df):
        """Plot win rate comparison."""
        if 'win_rate_pct' not in metrics_df.columns:
            return
        
        win_rate = metrics_df['win_rate_pct'].sort_values()
        ax.barh(win_rate.index, win_rate.values, 
                      color=self.colors['info'], alpha=0.7, edgecolor='black')
        
        ax.axvline(x=50, color=self.colors['warning'], linestyle='--', alpha=0.7, label='50%')
        
        ax.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Win Rate (%)', fontsize=10)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_drawdown_comparison_bar(self, ax, metrics_df):
        """Plot max drawdown comparison bar chart."""
        if 'max_drawdown_pct' not in metrics_df.columns:
            return
        
        dd = metrics_df['max_drawdown_pct'].sort_values()
        ax.barh(dd.index, dd.values, color=self.colors['danger'], alpha=0.7, edgecolor='black')
        
        ax.set_title('Max Drawdown Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Max Drawdown (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_risk_return_multi(self, ax, metrics_df):
        """Plot risk-return scatter for multiple symbols."""
        if 'total_return_pct' not in metrics_df.columns or 'volatility_pct' not in metrics_df.columns:
            return
        
        returns = metrics_df['total_return_pct']
        risks = metrics_df['volatility_pct']
        
        ax.scatter(risks, returns, s=300, alpha=0.6, 
                           c=range(len(returns)), cmap='viridis', edgecolors='black', linewidths=2)
        
        for symbol in returns.index:
            ax.annotate(symbol, (risks[symbol], returns[symbol]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax.set_xlabel('Volatility (%)', fontsize=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    def _plot_metrics_table_multi(self, ax, metrics_df):
        """Plot metrics table for multiple symbols."""
        ax.axis('tight')
        ax.axis('off')
        
        # Select key metrics
        key_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
                      'win_rate_pct', 'profit_factor']
        available_metrics = [m for m in key_metrics if m in metrics_df.columns]
        
        if len(available_metrics) == 0:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center')
            return
        
        display_df = metrics_df[available_metrics].copy()
        display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
        
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                         rowLabels=display_df.index, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Style header and row labels
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    def _format_metrics_text(self, metrics: Dict) -> str:
        """Format metrics as text for display."""
        lines = [
            "KEY METRICS:",
            f"Total Return: {metrics.get('total_return_pct', 0):.2f}%",
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%",
            f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Total Trades: {metrics.get('total_trades', 0)}"
        ]
        return "\n".join(lines)

