"""
Universe construction for quantitative strategy.
Provides guidance and utilities for building a focused trading universe.
"""

import pandas as pd
from typing import List, Dict, Optional
import yfinance as yf


class UniverseConstructor:
    """
    Constructs and manages trading universe.
    
    GUIDANCE FOR UNIVERSE CONSTRUCTION:
    ====================================
    
    For options swing trading, focus on:
    
    1. LIQUIDITY: Choose stocks with high options volume
       - Daily options volume > 100k contracts
       - Tight bid-ask spreads
       - High open interest
    
    2. VOLATILITY: Stocks with sufficient movement
       - IV > 20% (for premium collection)
       - Historical volatility allows for swing opportunities
       - Avoid extremely low volatility stocks
    
    3. MARKET CAP: Mid to large cap preferred
       - Market cap > $2B (reduces manipulation risk)
       - Better options liquidity
       - More predictable price action
    
    4. SECTOR DIVERSIFICATION: Reduce correlation
       - Spread across 3-5 sectors
       - Avoid highly correlated stocks
    
    5. SIZE: Keep universe manageable
       - Start with 10-20 stocks
       - Focus on quality over quantity
       - Easier to monitor and analyze
    
    RECOMMENDED APPROACH:
    - Start with SPY components or sector ETFs
    - Filter by options volume and liquidity
     - Add 2-3 stocks per sector for diversification
    """
    
    # Pre-defined universe examples
    POPULAR_OPTIONS_STOCKS = [
        # Tech
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA',
        # Finance
        'JPM', 'BAC', 'GS',
        # Consumer
        'AMZN', 'META', 'GOOGL', 'NFLX',
        # Healthcare
        'UNH', 'JNJ', 'PFE',
        # Energy
        'XOM', 'CVX',
        # ETFs (for sector exposure)
        'SPY', 'QQQ', 'IWM', 'DIA'
    ]
    
    SECTOR_ETFS = {
        'Technology': ['XLK', 'QQQ'],
        'Finance': ['XLF'],
        'Healthcare': ['XLV'],
        'Energy': ['XLE'],
        'Consumer': ['XLY'],
        'Industrial': ['XLI']
    }
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize universe constructor.
        
        Args:
            symbols: Optional list of symbols to use. If None, uses default.
        """
        self.symbols = symbols or self.POPULAR_OPTIONS_STOCKS.copy()
        self.universe_metadata = {}
    
    def add_symbol(self, symbol: str):
        """Add a symbol to the universe."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from the universe."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
    
    def get_universe(self) -> List[str]:
        """Get current universe symbols."""
        return self.symbols.copy()
    
    def filter_by_liquidity(
        self,
        min_volume: float = 1_000_000,
        min_options_volume: Optional[int] = None
    ) -> List[str]:
        """
        Filter universe by liquidity metrics.
        
        Note: This is a simplified filter. For production, you'd want
        to check actual options volume from your data provider.
        
        Args:
            min_volume: Minimum average daily stock volume
            min_options_volume: Minimum options volume (if available)
        
        Returns:
            Filtered list of symbols
        """
        filtered = []
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                avg_volume = info.get('averageVolume', 0)
                
                if avg_volume >= min_volume:
                    filtered.append(symbol)
            except Exception as e:
                print(f"Error checking {symbol}: {e}")
        
        return filtered
    
    def create_sector_balanced_universe(
        self,
        stocks_per_sector: int = 2
    ) -> List[str]:
        """
        Create a sector-balanced universe.
        
        Args:
            stocks_per_sector: Number of stocks per sector
        
        Returns:
            Balanced universe list
        """
        # This is a simplified example
        # In practice, you'd map stocks to sectors and balance
        balanced = []
        
        # Add some ETFs for broad exposure
        balanced.extend(['SPY', 'QQQ'])
        
        # Add diverse individual stocks
        balanced.extend([
            'AAPL', 'MSFT',  # Tech
            'JPM', 'BAC',    # Finance
            'JNJ', 'UNH',    # Healthcare
            'XOM', 'CVX',    # Energy
            'AMZN', 'META'   # Consumer
        ])
        
        return balanced[:stocks_per_sector * 5]  # Limit size
    
    def get_universe_info(self) -> pd.DataFrame:
        """
        Get basic information about universe symbols.
        
        Returns:
            DataFrame with symbol metadata
        """
        info_list = []
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                info_list.append({
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'beta': info.get('beta', 0)
                })
            except Exception as e:
                print(f"Error fetching info for {symbol}: {e}")
        
        return pd.DataFrame(info_list)
    
    def export_universe(self, filepath: str):
        """Export universe to file."""
        with open(filepath, 'w') as f:
            for symbol in self.symbols:
                f.write(f"{symbol}\n")
    
    @staticmethod
    def load_universe(filepath: str) -> List[str]:
        """Load universe from file."""
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

