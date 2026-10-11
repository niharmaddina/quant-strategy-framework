"""
Visualization utilities for backtest results.
"""

import matplotlib.pyplot as plt


class BacktestVisualizer:
    """Create visualizations for backtest results."""
    
    def __init__(self, backtest_engine):
        """
        Initialize visualizer.
        
        Args:
            backtest_engine: BacktestEngine instance with results
        """
        self.engine = backtest_engine
    
    def plot_equity_curve(self, figsize=(12, 6)):
        """Plot equity curve."""
        if self.engine.results is None or self.engine.results.empty:
            raise ValueError("No backtest results available. Run backtest first.")
        
        if 'equity' not in self.engine.results.columns:
            raise ValueError("Results missing 'equity' column")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        equity = self.engine.results['equity']
        ax.plot(equity.index, equity.values, linewidth=2, label='Equity Curve')
        ax.axhline(y=self.engine.initial_capital, color='r', linestyle='--', 
                   label=f'Initial Capital (${self.engine.initial_capital:,.0f})')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(self, figsize=(10, 6)):
        """Plot distribution of returns."""
        if self.engine.results is None or self.engine.results.empty:
            raise ValueError("No backtest results available. Run backtest first.")
        
        if 'returns' not in self.engine.results.columns:
            raise ValueError("Results missing 'returns' column")
        
        returns = self.engine.results['returns'].dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data to plot")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(returns * 100, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Daily Returns (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_trades(self, figsize=(14, 8)):
        """Plot price chart with trade markers."""
        if self.engine.results is None or self.engine.results.empty:
            raise ValueError("No backtest results available. Run backtest first.")
        
        if self.engine.trades.empty:
            raise ValueError("No trades to plot.")
        
        if 'price' not in self.engine.results.columns:
            raise ValueError("Results missing 'price' column")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Price chart with trades
        price = self.engine.results['price']
        ax1.plot(price.index, price.values, linewidth=1.5, label='Price', color='black')
        
        # Mark buy and sell signals
        signals = self.engine.results['signal']
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        if len(buy_signals) > 0:
            buy_prices = price.reindex(buy_signals.index)
            ax1.scatter(buy_signals.index, buy_prices.values, 
                       color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        if len(sell_signals) > 0:
            sell_prices = price.reindex(sell_signals.index)
            ax1.scatter(sell_signals.index, sell_prices.values, 
                       color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Chart with Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Equity curve
        equity = self.engine.results['equity']
        ax2.plot(equity.index, equity.values, linewidth=2, label='Equity', color='blue')
        ax2.axhline(y=self.engine.initial_capital, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Equity Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, figsize=(12, 6)):
        """Plot drawdown chart."""
        if self.engine.results is None or self.engine.results.empty:
            raise ValueError("No backtest results available. Run backtest first.")
        
        if 'equity' not in self.engine.results.columns:
            raise ValueError("Results missing 'equity' column")
        
        equity = self.engine.results['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, linewidth=2, color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Chart')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_all(self, figsize=(14, 10)):
        """Create a comprehensive dashboard."""
        if self.engine.results is None or self.engine.results.empty:
            raise ValueError("No backtest results available. Run backtest first.")
        
        required_cols = ['equity', 'price', 'returns']
        missing_cols = [col for col in required_cols if col not in self.engine.results.columns]
        if missing_cols:
            raise ValueError(f"Results missing required columns: {missing_cols}")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        equity = self.engine.results['equity']
        ax1.plot(equity.index, equity.values, linewidth=2, label='Equity')
        ax1.axhline(y=self.engine.initial_capital, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, linewidth=2, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        ax3 = fig.add_subplot(gs[1, 1])
        returns = self.engine.results['returns'].dropna()
        ax3.hist(returns * 100, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Returns (%)')
        ax3.grid(True, alpha=0.3)
        
        # Price with signals
        ax4 = fig.add_subplot(gs[2, :])
        price = self.engine.results['price']
        ax4.plot(price.index, price.values, linewidth=1.5, color='black', label='Price')
        
        signals = self.engine.results['signal']
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        if len(buy_signals) > 0:
            buy_prices = price.reindex(buy_signals.index)
            ax4.scatter(buy_signals.index, buy_prices.values, 
                       color='green', marker='^', s=80, label='Buy', zorder=5)
        
        if len(sell_signals) > 0:
            sell_prices = price.reindex(sell_signals.index)
            ax4.scatter(sell_signals.index, sell_prices.values, 
                       color='red', marker='v', s=80, label='Sell', zorder=5)
        
        ax4.set_title('Price Chart with Signals')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Backtest Dashboard', fontsize=16, y=0.995)
        return fig

