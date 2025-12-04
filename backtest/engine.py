"""
Backtesting engine for quantitative strategies.
Provides basic performance metrics.
"""

import json
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Simple backtesting engine for strategy evaluation.
    Assumes signals are generated and we follow them with fixed position sizes.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        position_size: float = 0.1,  # 10% of capital per trade
        commission: float = 0.001  # 0.1% commission
    ):
        """
        Initialize backtest engine.
        
        Args:
            data: Price data with 'close' column
            signals: DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
            initial_capital: Starting capital
            position_size: Fraction of capital to use per trade
            commission: Commission rate per trade
        
        Raises:
            ValueError: If data is invalid or missing required columns
        """
        # Validate inputs
        if data.empty:
            raise ValueError("Data DataFrame is empty")
        if signals.empty:
            raise ValueError("Signals DataFrame is empty")
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        if 'signal' not in signals.columns:
            raise ValueError("Signals must contain 'signal' column")
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")
        if not 0 < position_size <= 1:
            raise ValueError(f"Position size must be between 0 and 1, got {position_size}")
        if commission < 0:
            raise ValueError(f"Commission must be non-negative, got {commission}")
        
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        
        self.results = None
        self.metrics = {}
    
    def run(self) -> pd.DataFrame:
        """
        Run backtest.
        
        Returns:
            DataFrame with backtest results
        
        Raises:
            ValueError: If data and signals cannot be aligned
        """
        df = self.data.copy()
        signals = self.signals.copy()
        
        # Align indices
        if 'date' in df.columns:
            df = df.set_index('date')
        if 'date' in signals.columns:
            signals = signals.set_index('date')
        
        # Validate index alignment
        if df.index.empty:
            raise ValueError("Data index is empty after processing")
        
        # Merge data and signals
        backtest_df = pd.DataFrame(index=df.index)
        backtest_df['price'] = df['close']
        backtest_df['signal'] = signals['signal'].reindex(df.index, fill_value=0)
        
        # Validate we have price data
        if backtest_df['price'].isna().all():
            raise ValueError("All price values are NaN")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares
        entry_price = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(backtest_df)):
            current_price = backtest_df['price'].iloc[i]
            signal = backtest_df['signal'].iloc[i]
            prev_signal = backtest_df['signal'].iloc[i-1]
            
            # Close position if signal changes
            if position != 0 and signal != prev_signal and signal == 0:
                # Close position
                pnl = (current_price - entry_price) * position
                commission_cost = abs(position * current_price * self.commission)
                capital += pnl - commission_cost
                
                trades.append({
                    'entry_date': backtest_df.index[i-1],
                    'exit_date': backtest_df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': pnl - commission_cost,
                    'return_pct': ((current_price - entry_price) / entry_price) * 100
                })
                
                position = 0
                entry_price = 0
            
            # Open new position
            if signal != 0 and position == 0:
                trade_capital = capital * self.position_size
                shares = int(trade_capital / current_price)
                
                if shares > 0:
                    commission_cost = shares * current_price * self.commission
                    if trade_capital >= commission_cost:
                        position = shares
                        entry_price = current_price
                        capital -= (shares * current_price + commission_cost)
            
            # Update equity curve
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        # Close final position if open
        if position > 0:
            final_price = backtest_df['price'].iloc[-1]
            pnl = (final_price - entry_price) * position
            commission_cost = abs(position * final_price * self.commission)
            capital += pnl - commission_cost
            
            trades.append({
                'entry_date': backtest_df.index[-1],
                'exit_date': backtest_df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'pnl': pnl - commission_cost,
                'return_pct': ((final_price - entry_price) / entry_price) * 100
            })
        
        # Store results
        backtest_df['equity'] = equity_curve
        backtest_df['returns'] = backtest_df['equity'].pct_change()
        backtest_df['cumulative_returns'] = (1 + backtest_df['returns']).cumprod() - 1
        
        self.results = backtest_df
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate metrics
        self._calculate_metrics()
        
        return backtest_df
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if self.results is None or self.results.empty:
            return
        
        equity = self.results['equity']
        returns = self.results['returns'].dropna()
        
        # Basic metrics
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        total_trades = len(self.trades) if not self.trades.empty else 0
        
        if total_trades > 0:
            winning_trades = self.trades[self.trades['pnl'] > 0]
            losing_trades = self.trades[self.trades['pnl'] <= 0]
            
            win_rate = len(winning_trades) / total_trades * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
            avg_return_per_trade = self.trades['return_pct'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_return_per_trade = 0
        
        # Risk metrics
        if len(returns) > 0:
            sharpe_ratio = self._calculate_sharpe(returns)
            sortino_ratio = self._calculate_sortino(returns)
            max_drawdown = self._calculate_max_drawdown(equity)
            calmar_ratio = self._calculate_calmar(total_return, max_drawdown)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
            volatility = 0
        
        self.metrics = {
            'total_return_pct': total_return,
            'final_capital': equity.iloc[-1],
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_return_per_trade_pct': avg_return_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_pct': max_drawdown,
            'volatility_pct': volatility
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) == 0:
            return 0.0
        
        std_dev = returns.std()
        if std_dev == 0 or np.isnan(std_dev):
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe = excess_returns / (std_dev * np.sqrt(252))
        
        # Handle NaN or infinite values
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        
        return sharpe
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity) == 0:
            return 0.0
        
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min()) * 100
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (downside deviation only).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default: 0.02)
        
        Returns:
            Sortino ratio (annualized)
        """
        if len(returns) == 0:
            return 0.0
        
        # Only consider negative returns (downside)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            # No downside, return high value or 0
            return 0.0
        
        downside_std = downside_returns.std() * np.sqrt(252)
        
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        
        excess_return = returns.mean() * 252 - risk_free_rate
        sortino = excess_return / downside_std
        
        # Handle NaN or infinite values
        if np.isnan(sortino) or np.isinf(sortino):
            return 0.0
        
        return sortino
    
    def _calculate_calmar(self, total_return_pct: float, max_drawdown_pct: float) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            total_return_pct: Total return percentage
            max_drawdown_pct: Maximum drawdown percentage
        
        Returns:
            Calmar ratio
        """
        if max_drawdown_pct == 0:
            return 0.0
        
        # Convert percentage to decimal for calculation
        total_return = total_return_pct / 100
        max_drawdown = max_drawdown_pct / 100
        
        calmar = total_return / max_drawdown
        
        if np.isnan(calmar) or np.isinf(calmar):
            return 0.0
        
        return calmar
    
    def get_metrics(self) -> Dict:
        """Get calculated performance metrics."""
        return self.metrics.copy()
    
    def print_metrics(self):
        """Print formatted metrics."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Return:        {self.metrics['total_return_pct']:.2f}%")
        print(f"Final Capital:       ${self.metrics['final_capital']:,.2f}")
        print(f"Total Trades:        {self.metrics['total_trades']}")
        print(f"Win Rate:            {self.metrics['win_rate_pct']:.2f}%")
        print(f"Avg Win:             ${self.metrics['avg_win']:,.2f}")
        print(f"Avg Loss:            ${self.metrics['avg_loss']:,.2f}")
        print(f"Profit Factor:       {self.metrics['profit_factor']:.2f}")
        print(f"Avg Return/Trade:    {self.metrics['avg_return_per_trade_pct']:.2f}%")
        print(f"Sharpe Ratio:        {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {self.metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:        {self.metrics['calmar_ratio']:.2f}")
        print(f"Max Drawdown:        {self.metrics['max_drawdown_pct']:.2f}%")
        print(f"Volatility (Ann):    {self.metrics['volatility_pct']:.2f}%")
        print("="*50 + "\n")
    
    def get_trades(self) -> pd.DataFrame:
        """Get trade log."""
        return self.trades.copy() if not self.trades.empty else pd.DataFrame()
    
    def export_results_to_csv(self, filepath: str):
        """
        Export backtest results to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        self.results.to_csv(filepath, index=True)
    
    def export_trades_to_csv(self, filepath: str):
        """
        Export trade log to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        if self.trades.empty:
            raise ValueError("No trades to export.")
        
        self.trades.to_csv(filepath, index=False)
    
    def export_metrics_to_json(self, filepath: str):
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            elif isinstance(value, (int, float)):
                metrics_serializable[key] = value
            else:
                metrics_serializable[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

