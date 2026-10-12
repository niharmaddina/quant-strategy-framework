"""
Backtesting engine for quantitative strategies.
Provides basic performance metrics.
"""

from typing import Dict

import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Simple backtesting engine for strategy evaluation.
    Assumes signals are generated and we follow them with fixed position sizes.

    Risk metrics come in two families, because a single-position account that is
    in cash most of the time makes the usual account-level ratios misleading:

    * ``sharpe_ratio``, ``sortino_ratio``, ``volatility_pct``, ``max_drawdown_pct``
      are computed on the daily returns of total account equity over every bar.
      Cash earns nothing in this engine, so an account that is rarely invested has
      near-zero mean return and near-zero volatility; subtracting the risk-free rate
      then gives a large negative Sharpe (roughly -rf / tiny volatility). That value
      is finite and correctly says "T-bills would have done better", but it says
      nothing about the trades themselves.
    * ``in_market_sharpe_ratio`` and ``in_market_sortino_ratio`` are computed only
      on the days a position was held, from the return on the capital at risk (see
      ``_in_market_returns``). They describe the quality of the trades independently
      of how often and how large they were; ``exposure_pct``, ``avg_holding_days``
      and ``in_market_days`` report that separately. They are NaN, not 0, when
      undefined (fewer than two in-market days, or zero variance).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        position_size: float = 0.1,  # 10% of capital per trade
        commission: float = 0.001,  # 0.1% commission
        stop_loss_pct: float = 0.05,  # 5% stop loss
        trailing_stop: bool = False,  # Use trailing stop loss
        trailing_stop_pct: float = 0.03,  # 3% trailing stop
        max_drawdown_pct: float = 0.20,  # 20% max portfolio drawdown before stopping
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        max_holding_days: int = 20,  # Force exit after this many bars in a trade
    ):
        """
        Initialize backtest engine with risk management features.

        Args:
            data: Price data with 'close' column
            signals: DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
            initial_capital: Starting capital
            position_size: Fraction of capital to use per trade (0.1 = 10%)
            commission: Commission rate per trade (0.001 = 0.1%)
            stop_loss_pct: Stop loss percentage (0.05 = 5% stop loss)
            trailing_stop: If True, use trailing stop loss instead of fixed
            trailing_stop_pct: Trailing stop percentage (0.03 = 3% trailing stop)
            max_drawdown_pct: Max portfolio drawdown before halting (0.20 = 20%)
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (0.02 = 2%)
            max_holding_days: Maximum number of bars a position may be held before it
                is closed at that bar's close (default 20, the swing-trade horizon)

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
        if not 0 < stop_loss_pct < 1:
            raise ValueError(f"Stop loss must be between 0 and 1, got {stop_loss_pct}")
        if not 0 < trailing_stop_pct < 1:
            raise ValueError(f"Trailing stop must be between 0 and 1, got {trailing_stop_pct}")
        if not 0 < max_drawdown_pct <= 1:
            raise ValueError(f"Max drawdown must be between 0 and 1, got {max_drawdown_pct}")
        # Allow 1.0 to effectively disable max drawdown protection
        if int(max_holding_days) < 1 or int(max_holding_days) != max_holding_days:
            raise ValueError(f"max_holding_days must be a positive integer, got {max_holding_days}")

        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop = trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_free_rate = risk_free_rate
        self.max_holding_days = int(max_holding_days)

        self.results = None
        self.metrics = {}
        self.trades = pd.DataFrame()
    
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
        
        # Align indices - ensure both use the same index
        # Store original indices to preserve alignment
        data_has_date_col = 'date' in df.columns
        signals_has_date_col = 'date' in signals.columns
        
        # If data has date column, convert to date index
        if data_has_date_col:
            df = df.set_index('date')
        
        # Align signals index with data index
        # If signals has date column, use it; otherwise use data's index
        if signals_has_date_col:
            signals = signals.set_index('date')
        else:
            # Signals has RangeIndex - align it with data's index
            # This preserves the positional alignment
            if len(signals) == len(df):
                signals.index = df.index
            else:
                # Length mismatch - try to reindex
                signals = signals.reindex(df.index, fill_value=0)
        
        # Validate index alignment
        if df.index.empty:
            raise ValueError("Data index is empty after processing")
        
        # Merge data and signals - ensure indices match
        backtest_df = pd.DataFrame(index=df.index)
        backtest_df['price'] = df['close']
        
        # Align signals to data index
        if 'signal' in signals.columns:
            # Use the signal column, ensuring proper alignment
            backtest_df['signal'] = signals['signal']
        else:
            # If signal column doesn't exist, create zeros
            backtest_df['signal'] = 0
        
        # Ensure signal values are integers and handle any NaN
        backtest_df['signal'] = backtest_df['signal'].fillna(0).astype(int)
        
        # Debug: Check signal alignment
        non_zero_signals = backtest_df[backtest_df['signal'] != 0]
        if len(non_zero_signals) == 0:
            # Log warning but continue - might be legitimate (no signals generated)
            pass
        else:
            # Verify signals are properly aligned
            pass
        
        # Validate we have price data
        if backtest_df['price'].isna().all():
            raise ValueError("All price values are NaN")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of shares
        entry_price = 0
        entry_index = 0  # Track when position was opened
        entry_commission_cost = 0.0  # Track entry commission for accurate PnL logging
        stop_loss_price = 0  # Stop loss price level
        highest_price = 0  # For trailing stop
        trades = []
        equity_curve = []  # Will append one value per day
        position_curve = []  # Shares held at the close of each day (0 when flat)
        peak_equity = capital  # For max drawdown protection
        trading_halted = False  # Stop trading if max drawdown exceeded
        max_holding_days = self.max_holding_days  # Maximum holding period for swing trades

        # Process from day 0 to handle signals on first day
        # Start from i=0 to catch buy signals on the first day
        for i in range(len(backtest_df)):
            if i == 0:
                # First day: check for initial buy signal
                current_price = backtest_df['price'].iloc[i]
                signal = int(backtest_df['signal'].iloc[i])

                # Skip if price is invalid
                if pd.isna(current_price) or current_price <= 0:
                    current_equity = capital + (position * current_price if position > 0 else 0)
                    equity_curve.append(current_equity)
                    position_curve.append(position)
                    continue
                
                # Calculate current equity
                current_equity = capital + (position * current_price if position > 0 else 0)
                
                # Update peak equity
                if current_equity > peak_equity:
                    peak_equity = current_equity
                
                # Check for opening position on first day
                if signal == 1 and position == 0 and not trading_halted:
                    trade_capital = capital * self.position_size
                    shares = int(trade_capital / current_price)
                    
                    if shares > 0:
                        commission_cost = shares * current_price * self.commission
                        if trade_capital >= commission_cost:
                            position = shares
                            entry_price = current_price
                            entry_index = i  # Track when position was opened
                            entry_commission_cost = commission_cost
                            capital -= (shares * current_price + commission_cost)
                            
                            # Set stop loss price
                            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                            
                            # Initialize trailing stop
                            if self.trailing_stop:
                                highest_price = entry_price
                                stop_loss_price = highest_price * (1 - self.trailing_stop_pct)
                
                # Update equity curve
                current_equity = capital + (position * current_price if position > 0 else 0)
                equity_curve.append(current_equity)
                position_curve.append(position)
                continue

            # Days 1 onwards: normal processing
            current_price = backtest_df['price'].iloc[i]
            signal = int(backtest_df['signal'].iloc[i])  # Ensure integer
            prev_signal = int(backtest_df['signal'].iloc[i-1])  # Ensure integer

            # Skip if price is invalid
            if pd.isna(current_price) or current_price <= 0:
                current_equity = capital + (position * backtest_df['price'].iloc[i-1] if position > 0 else 0)
                equity_curve.append(current_equity)
                position_curve.append(position)
                continue
            
            # Calculate current equity
            current_equity = capital + (position * current_price if position > 0 else 0)
            
            # Update peak equity
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            # Check stop loss FIRST (before max drawdown) if we have a position
            if position > 0:
                stop_loss_triggered = False
                exit_reason = 'signal'
                
                # Regular stop loss check
                if current_price <= stop_loss_price:
                    stop_loss_triggered = True
                    exit_reason = 'stop_loss'
                
                # Trailing stop loss check
                if self.trailing_stop and current_price > highest_price:
                    highest_price = current_price
                    # Update trailing stop to be trailing_stop_pct below highest price
                    stop_loss_price = highest_price * (1 - self.trailing_stop_pct)
                elif self.trailing_stop and current_price <= stop_loss_price:
                    stop_loss_triggered = True
                    exit_reason = 'trailing_stop'
                
                # Close position if stop loss triggered
                if stop_loss_triggered:
                    # IMPORTANT: add back full sale proceeds (principal + PnL), minus exit commission
                    # (Entry cost + entry commission were already deducted at entry)
                    exit_commission_cost = abs(position * current_price * self.commission)
                    proceeds = position * current_price
                    cost_basis = position * entry_price
                    pnl_net = proceeds - exit_commission_cost - cost_basis - entry_commission_cost
                    capital += proceeds - exit_commission_cost
                    
                    trades.append({
                        'entry_date': backtest_df.index[entry_index],
                        'exit_date': backtest_df.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': position,
                        'pnl': pnl_net,
                        'return_pct': ((current_price - entry_price) / entry_price) * 100,
                        'bars_held': i - entry_index,
                        'exit_reason': exit_reason
                    })

                    position = 0
                    entry_price = 0
                    entry_index = 0
                    entry_commission_cost = 0.0
                    stop_loss_price = 0
                    highest_price = 0
                    # Recalculate equity after closing position
                    current_equity = capital
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    # Update equity curve before continuing
                    equity_curve.append(current_equity)
                    position_curve.append(position)
                    continue  # Skip signal-based exit logic
            
            # Check max drawdown protection AFTER processing stop loss
            # Only halt if we're not in a position (to avoid halting mid-trade)
            # Allow trading to resume if equity recovers above the halt threshold
            if position == 0 and self.max_drawdown_pct < 1.0:
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
                if drawdown >= self.max_drawdown_pct:
                    trading_halted = True
                    # Don't open new positions, but let existing logic handle it
                elif trading_halted:
                    # If we were halted but drawdown has improved, allow trading to resume
                    # Resume if drawdown is below 80% of the threshold (some recovery)
                    if drawdown < (self.max_drawdown_pct * 0.8):
                        trading_halted = False
            
            # Close position if we have one and signal indicates exit
            # Improved exit logic for SWING TRADES: close on sell signal, trend reversal,
            # profit target, or max holding period
            if position != 0 and not trading_halted:
                # Calculate current return
                current_return_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Calculate days held
                days_held = i - entry_index
                
                # Exit conditions for swing trades:
                # 1. Explicit sell signal
                # 2. Trend reversal (price breaks below key MA)
                # 3. Take profit (2:1 risk/reward or 3% gain)
                # 4. Maximum holding period reached (force exit for swing trades)
                # 5. Neutral signal after holding (exit on neutral to avoid holding too long)
                
                trend_reversal = False
                if 'sma_20' in backtest_df.columns:
                    # Exit if price breaks below 20-day SMA (trend weakening)
                    trend_reversal = current_price < backtest_df['sma_20'].iloc[i]
                
                # Take profit: 2:1 risk/reward or 3% gain (whichever comes first)
                profit_target = max(entry_price * 1.03, entry_price + (entry_price - stop_loss_price) * 2)
                take_profit = current_price >= profit_target
                
                # Maximum holding period: force exit after max_holding_days (swing trades should be days to weeks)
                max_holding_reached = days_held >= max_holding_days
                
                # Exit on neutral signal if we've held for a while (prevents holding indefinitely)
                # Exit on neutral after 5 days to avoid dead positions
                neutral_exit = (signal == 0) and (days_held >= 5)
                
                should_close = (
                    signal == -1 or  # Explicit sell signal
                    (prev_signal == 1 and signal == -1) or  # Buy to sell transition
                    trend_reversal or  # Trend reversal
                    take_profit or  # Profit target hit
                    max_holding_reached or  # Maximum holding period
                    neutral_exit  # Neutral signal after holding period
                )
                
                if should_close:
                    # Determine exit reason
                    exit_reason = 'signal'
                    if max_holding_reached:
                        exit_reason = 'max_holding_period'
                    elif take_profit:
                        exit_reason = 'take_profit'
                    elif trend_reversal:
                        exit_reason = 'trend_reversal'
                    elif neutral_exit:
                        exit_reason = 'neutral_exit'
                    
                    # Close position
                    exit_commission_cost = abs(position * current_price * self.commission)
                    proceeds = position * current_price
                    cost_basis = position * entry_price
                    pnl_net = proceeds - exit_commission_cost - cost_basis - entry_commission_cost
                    capital += proceeds - exit_commission_cost
                    
                    trades.append({
                        'entry_date': backtest_df.index[entry_index],
                        'exit_date': backtest_df.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': position,
                        'pnl': pnl_net,
                        'return_pct': current_return_pct,
                        'bars_held': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_index = 0
                    entry_commission_cost = 0.0
                    stop_loss_price = 0
                    highest_price = 0
            
            # Open new long position if signal is 1 (buy) and we don't have a position
            # Only open if trading is not halted
            if signal == 1 and position == 0 and not trading_halted:
                trade_capital = capital * self.position_size
                shares = int(trade_capital / current_price)
                
                if shares > 0:
                    commission_cost = shares * current_price * self.commission
                    if trade_capital >= commission_cost:
                        position = shares
                        entry_price = current_price
                        entry_index = i  # Track when position was opened
                        entry_commission_cost = commission_cost
                        capital -= (shares * current_price + commission_cost)
                        
                        # Set stop loss price
                        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
                        
                        # Initialize trailing stop
                        if self.trailing_stop:
                            highest_price = entry_price
                            stop_loss_price = highest_price * (1 - self.trailing_stop_pct)
            
            # Update equity curve
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
            position_curve.append(position)

        # Close final position if open
        if position > 0:
            final_price = backtest_df['price'].iloc[-1]
            exit_commission_cost = abs(position * final_price * self.commission)
            proceeds = position * final_price
            cost_basis = position * entry_price
            pnl_net = proceeds - exit_commission_cost - cost_basis - entry_commission_cost
            capital += proceeds - exit_commission_cost

            trades.append({
                'entry_date': backtest_df.index[entry_index],
                'exit_date': backtest_df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'pnl': pnl_net,
                'return_pct': ((final_price - entry_price) / entry_price) * 100,
                'bars_held': len(backtest_df) - 1 - entry_index,
                'exit_reason': 'end_of_period'
            })

        # Store results
        backtest_df['equity'] = equity_curve
        backtest_df['position'] = position_curve
        backtest_df['returns'] = backtest_df['equity'].pct_change()
        backtest_df['cumulative_returns'] = (1 + backtest_df['returns']).cumprod() - 1
        # A day's close-to-close return was earned with a position open iff shares were held at the previous close.
        backtest_df['in_market'] = backtest_df['position'].shift(1).fillna(0) > 0

        self.results = backtest_df
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate metrics
        self._calculate_metrics()
        
        return backtest_df
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics from ``self.results`` and ``self.trades``.

        Account-level ratios (``sharpe_ratio``, ``sortino_ratio``, ``calmar_ratio``,
        ``volatility_pct``, ``max_drawdown_pct``) use every bar of account equity and
        keep their long-standing conventions (0.0 when undefined). The in-market
        family added for mostly-idle accounts is:

        * ``exposure_pct``: percentage of bars on which shares were held at the close.
        * ``in_market_days``: number of bars whose close-to-close return was earned with
          a position open (shares held at the previous close).
        * ``avg_holding_days``: mean ``bars_held`` over closed trades (NaN if no trades).
          For a trade entered at bar e and exited at bar x this is x - e, which is also
          the number of bars it contributes to ``exposure_pct`` and ``in_market_days``
          (the position opened at the end of bar e is held at closes e..x-1 and earns
          the returns of bars e+1..x). The forced end-of-period exit counts one bar
          less than the exposure it created, since the last bar's close is both the
          mark and the liquidation.
        * ``in_market_sharpe_ratio``, ``in_market_sortino_ratio``: same formulas and
          risk-free convention as the account ratios, applied to the return on capital
          at risk over in-market days only (see ``_in_market_returns``). NaN when fewer
          than two in-market days exist or their variance is zero, because "undefined"
          and "zero" mean different things for a ratio and the account-level 0.0
          convention already hides that distinction.
        """
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
            losing_pnl_sum = losing_trades['pnl'].sum()
            if len(losing_trades) > 0 and losing_pnl_sum != 0:
                profit_factor = abs(winning_trades['pnl'].sum() / losing_pnl_sum)
            else:
                profit_factor = 0
            avg_return_per_trade = self.trades['return_pct'].mean()
            
            # Count exit reasons
            if 'exit_reason' in self.trades.columns:
                stop_loss_trades = len(self.trades[self.trades['exit_reason'] == 'stop_loss'])
                trailing_stop_trades = len(self.trades[self.trades['exit_reason'] == 'trailing_stop'])
                signal_trades = len(self.trades[self.trades['exit_reason'] == 'signal'])
                max_dd_trades = len(self.trades[self.trades['exit_reason'] == 'max_drawdown'])
            else:
                stop_loss_trades = 0
                trailing_stop_trades = 0
                signal_trades = total_trades
                max_dd_trades = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_return_per_trade = 0
            stop_loss_trades = 0
            trailing_stop_trades = 0
            signal_trades = 0
            max_dd_trades = 0
        
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

        # Exposure and in-market metrics (see docstring)
        position = self.results['position']
        exposure_pct = float((position > 0).mean() * 100.0) if len(position) else 0.0
        in_market_returns = self._in_market_returns()
        in_market_days = int(len(in_market_returns))
        in_market_sharpe = self._calculate_sharpe(in_market_returns, undefined=np.nan)
        in_market_sortino = self._calculate_sortino(in_market_returns, undefined=np.nan)
        if total_trades > 0 and 'bars_held' in self.trades.columns:
            avg_holding_days = float(self.trades['bars_held'].mean())
        else:
            avg_holding_days = np.nan

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
            'volatility_pct': volatility,
            'exposure_pct': exposure_pct,
            'avg_holding_days': avg_holding_days,
            'in_market_days': in_market_days,
            'in_market_sharpe_ratio': in_market_sharpe,
            'in_market_sortino_ratio': in_market_sortino,
            'stop_loss_trades': stop_loss_trades,
            'trailing_stop_trades': trailing_stop_trades,
            'signal_trades': signal_trades,
            'max_drawdown_trades': max_dd_trades
        }

    def _in_market_returns(self) -> pd.Series:
        """
        Daily return on the capital at risk, on in-market days only.

        For every bar i whose previous close had shares held (``position[i-1] > 0``)
        the return is ``(equity[i] - equity[i-1]) / (position[i-1] * price[i-1])``:
        the day's change in account equity divided by the market value of the position
        at the previous close. Bars with no position are dropped rather than counted
        as zero, so the series is the stitched-together experience of the money that
        was actually invested.

        Why this basis: the account-level Sharpe is diluted by the cash fraction
        (1 - position_size, 90% at the default sizing) on invested days and by the
        idle days in between, so it mixes trade quality with sizing and frequency.
        The return on capital at risk is independent of position_size to first order
        and is directly comparable with the Sharpe of a fully invested buy-and-hold
        series, which is what the worked example reports next to it.

        Bookkeeping: the exit commission falls on an in-market day and is included.
        The entry commission is paid at the close of the entry bar, which is not an
        in-market day (nothing was held at the previous close), so it is excluded;
        at 0.1% per side that omits 0.1% of the position value once per trade.
        Returns an empty Series when the account never held a position.
        """
        if self.results is None or self.results.empty or 'position' not in self.results.columns:
            return pd.Series(dtype=float)
        equity = self.results['equity'].astype(float)
        price = self.results['price'].astype(float)
        position = self.results['position'].astype(float)
        prev_position = position.shift(1)
        prev_value = prev_position * price.shift(1)
        mask = (prev_position > 0) & (prev_value > 0)
        if not mask.any():
            return pd.Series(dtype=float)
        pnl = equity - equity.shift(1)
        return (pnl[mask] / prev_value[mask]).astype(float)

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = None, undefined: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (annualized): (mean daily return * 252 - rf) / (daily std * sqrt(252)).

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (default: self.risk_free_rate)
            undefined: Value returned when the ratio is undefined (empty input, fewer
                than two observations, zero or NaN standard deviation, non-finite
                result). The account-level metrics keep the historical 0.0; the
                in-market metrics pass NaN.
        """
        if len(returns) == 0:
            return undefined

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        std_dev = returns.std()
        if std_dev == 0 or np.isnan(std_dev):
            return undefined

        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe = excess_returns / (std_dev * np.sqrt(252))

        # Handle NaN or infinite values
        if np.isnan(sharpe) or np.isinf(sharpe):
            return undefined

        return sharpe
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity) == 0:
            return 0.0
        
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min()) * 100
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = None, undefined: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside deviation only).

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default: from self.risk_free_rate)
            undefined: Value returned when the ratio is undefined (no observations, no
                negative returns, zero or NaN downside deviation, non-finite result);
                0.0 for the account-level metric, NaN for the in-market one

        Returns:
            Sortino ratio (annualized)
        """
        if len(returns) == 0:
            return undefined

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Only consider negative returns (downside)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            # No downside: the ratio is undefined rather than infinite
            return undefined

        downside_std = downside_returns.std() * np.sqrt(252)

        if downside_std == 0 or np.isnan(downside_std):
            return undefined

        excess_return = returns.mean() * 252 - risk_free_rate
        sortino = excess_return / downside_std

        # Handle NaN or infinite values
        if np.isnan(sortino) or np.isinf(sortino):
            return undefined

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
        print(f"Exposure:            {self.metrics['exposure_pct']:.1f}% of bars")
        print(f"Avg Holding:         {self.metrics['avg_holding_days']:.1f} bars")
        print(f"In-market Sharpe:    {self.metrics['in_market_sharpe_ratio']:.2f} "
              f"({self.metrics['in_market_days']} bars)")
        print(f"In-market Sortino:   {self.metrics['in_market_sortino_ratio']:.2f}")
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
        
        # Convert numpy types to native Python types for JSON serialization; NaN (undefined ratio) becomes null
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            if isinstance(value, float) and np.isnan(value):
                metrics_serializable[key] = None
            elif isinstance(value, (int, float)):
                metrics_serializable[key] = value
            else:
                metrics_serializable[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

