"""
================================================================================
UNIVERSE CONSTRUCTOR - TRADING UNIVERSE BUILDER
================================================================================

Universe construction system for quantitative trading strategies.
Designed filtering, risk management, and
sector diversification capabilities.

Key Features:
    - Multi-factor filtering (liquidity, quality, volatility, market cap)
    - Sector diversification with exposure limits
    - Correlation analysis and de-duplication
    - Risk metrics (beta, sector concentration)
    - Universe scoring and ranking
    - Strategy testing integration
    - Performance optimization with caching


Version: 2.0
================================================================================
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import warnings

import pandas as pd
import yfinance as yf


# ============================================================================
# CONSTANTS - SECTOR CLASSIFICATIONS
# ============================================================================

# GICS Sector Classifications
# Stocks organized by market cap tiers and sub-industries for optimal selection
SECTOR_MAPPING = {
    'Technology': [
        # Mega Cap Technology (Market Cap > $100B)
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'CSCO',
        
        # Large Cap Technology ($10B - $100B)
        'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU', 'NXPI', 'MRVL',
        
        # Mid Cap Technology ($2B - $10B)
        'ANET', 'SNPS', 'CDNS', 'FTNT', 'PANW', 'CRWD', 'ZS', 'NET', 'DDOG', 'TEAM',
        
        # Software & Cloud Services
        'NOW', 'WDAY', 'VEEV', 'DOCN', 'ESTC', 'MDB', 'SPLK', 'OKTA', 'ZM', 'DOCU',
        
        # Semiconductors & Equipment
        'TSM', 'ASML', 'AMBA', 'SWKS', 'QRVO', 'ON', 'MPWR', 'ALGM', 'DIOD', 'SLAB'
    ],
    
    'Financial Services': [
        # Mega Cap Banks (Market Cap > $100B)
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
        
        # Regional Banks
        'USB', 'TFC', 'PNC', 'BK', 'STT', 'CFG', 'HBAN', 'KEY', 'MTB', 'ZION',
        
        # Credit Cards & Payment Processing
        'AXP', 'COF', 'V', 'MA', 'FIS', 'FISV', 'GPN', 'FLT', 'WU', 'SQ',
        
        # Insurance Companies
        'BRK.B', 'AIG', 'PRU', 'MET', 'ALL', 'TRV', 'PGR', 'AFL', 'HIG', 'PFG',
        
        # Investment Services & Exchanges
        'IBKR', 'TROW', 'BEN', 'NDAQ', 'CME', 'ICE', 'MCO', 'SPGI', 'MSCI', 'FDS'
    ],
    
    'Healthcare': [
        # Pharmaceutical Giants
        'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'REGN',
        
        # Healthcare Services & Managed Care
        'UNH', 'CVS', 'CI', 'HUM', 'ELV', 'CNC', 'MOH', 'AGL', 'HQY', 'TDOC',
        
        # Medical Devices & Equipment
        'ISRG', 'SYK', 'BSX', 'EW', 'ZBH', 'BAX', 'BDX', 'HOLX', 'ALGN', 'SWAV',
        
        # Biotechnology
        'BIIB', 'VRTX', 'ILMN', 'ALNY', 'IONS', 'FOLD', 'ARWR', 'SGMO', 'BLUE', 'RGNX',
        
        # Healthcare Facilities & Services
        'HCA', 'THC', 'UHS', 'CYH', 'SEM', 'ACHC', 'ENSG', 'AMED', 'LPNT', 'OPCH'
    ],
    
    'Consumer Discretionary': [
        # E-commerce & Retail
        'AMZN', 'HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'BBY', 'BBWI',
        
        # Automotive Manufacturers
        'TSLA', 'GM', 'F', 'STLA', 'HMC', 'TM', 'RIVN', 'LCID', 'FORD', 'WKHS',
        
        # Entertainment & Media (Non-Communication Services)
        'LGF.A', 'AMC',
        
        # Restaurants & Leisure
        'SBUX', 'MCD', 'YUM', 'CMG', 'DPZ', 'DIN', 'WEN', 'JACK', 'CAKE', 'BLMN',
        
        # Apparel & Luxury Goods
        'NKE', 'LULU', 'DKS', 'HIBB', 'ASO', 'BOOT', 'FL', 'CAL', 'DXLG', 'ZUMZ',
        
        # Travel & Booking Services
        'BKNG', 'TCOM', 'ABNB', 'MAR', 'HLT', 'H', 'WH', 'LVS'
    ],
    
    'Consumer Staples': [
        # Food & Beverage Giants
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'TGT', 'KR', 'SFM', 'GO', 'SPLS',
        
        # Household Products
        'CL', 'CHD', 'CLX', 'KMB', 'ENR', 'NWL', 'SON', 'FUL', 'REYN', 'SPB',
        
        # Tobacco & Alcohol
        'PM', 'MO', 'STZ', 'TAP', 'SAM', 'BUD', 'BF.B', 'MGPI', 'VST', 'PRMW',
        
        # Personal Care Products
        'EL', 'REV', 'COTY', 'OLPX', 'USNA', 'NUS', 'HLF', 'NATR', 'HLMN'
    ],
    
    'Energy': [
        # Integrated Oil & Gas Majors
        'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'BKR', 'NOV', 'FTI', 'OII', 'WHD',
        
        # Exploration & Production
        'EOG', 'MPC', 'VLO', 'PSX', 'HES', 'DVN', 'FANG', 'CTRA', 'OVV', 'MRO',
        
        # Midstream & Pipelines
        'KMI', 'EPD', 'ET', 'WMB', 'OKE', 'TRGP', 'PAGP', 'AM', 'USAC', 'DCP',
        
        # Renewable Energy (Solar/Wind)
        'CSIQ', 'FSLR', 'SPWR', 'PLUG',
        
        # Oil Services & Equipment
        'RIG', 'VAL', 'HP', 'NBR', 'PTEN', 'LBRT', 'NEX', 'PUMP', 'WTTR', 'NINE'
    ],
    
    'Industrials': [
        # Aerospace & Defense
        'BA', 'RTX', 'LMT', 'NOC', 'GD', 'TDG', 'HWM', 'TXT', 'HII', 'LDOS',
        
        # Machinery & Heavy Equipment
        'CAT', 'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'FTV', 'AME', 'GGG',
        
        # Transportation & Logistics
        'UPS', 'FDX', 'JBHT', 'CHRW', 'KNX', 'XPO', 'ODFL', 'ARCB', 'RXO', 'HUBG',
        
        # Electrical Equipment
        'GE', 'HON', 'ROK', 'AOS', 'WWD', 'AWI', 'ATKR', 'TTC', 'ENS', 'ESE',
        
        # Industrial Services & Distribution
        'FAST', 'GWW', 'WCC', 'MSM', 'AIRC', 'CSWI', 'FIX', 'GVA', 'DY'
    ],
    
    'Communication Services': [
        # Internet & Social Media Platforms
        'GOOGL', 'META', 'SNAP', 'PINS', 'BMBL', 'MTCH', 'IAC', 'ANGI', 'TRIP', 'EXPE',
        
        # Telecommunications
        'VZ', 'T', 'TMUS', 'LUMN', 'USM', 'CNSL', 'ATUS', 'SHEN', 'VSAT', 'IRDM',
        
        # Media & Entertainment
        'DIS', 'CMCSA', 'NFLX', 'WBD', 'PARA', 'FOXA', 'NWS', 'LSG', 'IMAX', 'CNK',
        
        # Gaming & Interactive Entertainment
        'EA', 'TTWO', 'ATVI', 'U', 'RBLX', 'DKNG', 'PENN', 'LNW', 'GMBL', 'GAN',
        
        # Advertising & Marketing Services
        'OMC', 'IPG', 'PUBM', 'TTD', 'MGNI', 'TRMR', 'ADTH', 'PERI', 'CMPR', 'STGW'
    ],
    
    'Real Estate': [
        # REITs - Data Centers & Technology
        'AMT', 'EQIX', 'DLR', 'CONE', 'QTS', 'COR', 'INVH', 'PLD', 'EXPI', 'CBRE',
        
        # REITs - Residential
        'EQR', 'AVB', 'UDR', 'MAA', 'CPT', 'ESS', 'AIV', 'NLY', 'AGNC', 'TWO',
        
        # REITs - Commercial & Retail
        'SPG', 'O', 'PSA', 'WELL', 'VTR', 'PEAK', 'VICI', 'STOR', 'STAG', 'FR',
        
        # REITs - Industrial & Logistics
        'CUBE', 'LSI', 'TRNO', 'REXR', 'EGP', 'DRE', 'BXP', 'SLG',
        
        # Real Estate Services & Technology
        'Z', 'RDFN', 'OPEN', 'COMP', 'HOUS', 'REAX', 'RMAX', 'REZI', 'RKT', 'UWMC'
    ],
    
    'Materials': [
        # Chemicals
        'LIN', 'APD', 'ECL', 'SHW', 'DD', 'PPG', 'ALB', 'FMC', 'CF', 'NEM',
        
        # Metals & Mining
        'FCX', 'VALE', 'NUE', 'STLD', 'X', 'CLF', 'CMC', 'RS', 'CENX', 'MT',
        
        # Gold & Precious Metals
        'GOLD', 'AEM', 'AGI', 'FNV', 'WPM', 'RGLD', 'SAND', 'OR', 'HL', 'MUX',
        
        # Construction Materials
        'VMC', 'MLM', 'EXP', 'USCR', 'SUM', 'OC', 'USG', 'BCC', 'TREX', 'AZEK',
        
        # Packaging & Containers
        'PKG', 'WRK', 'IP', 'SEE', 'BALL', 'CCK', 'SLGN', 'GPK', 'AVY', 'BERY'
    ],
    
    'Utilities': [
        # Electric Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES',
        
        # Multi-Utilities
        'PEG', 'ETR', 'FE', 'ED', 'AEE', 'CMS', 'LNT', 'ATO', 'CNP', 'NI',
        
        # Water Utilities
        'AWK', 'WTRG', 'SBS', 'CWT', 'YORW', 'CTWS', 'SJW', 'MSEX', 'ARTNA', 'GWR',
        
        # Renewable Energy Utilities
        'BEP', 'BEPC', 'NEP', 'ORA', 'CWEN', 'NEE', 'NOVA', 'ENPH', 'SEDG', 'RUN', 'ARRY',
        
        # Gas Utilities & Pipelines
        'OKE', 'WMB', 'TRGP', 'KMI', 'EPD', 'ET', 'PAGP', 'AM', 'USAC', 'DCP'
    ]
}


# Sector ETFs for broad market exposure
SECTOR_ETFS = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Communication Services': 'XLC',
        'Real Estate': 'XLRE',
        'Materials': 'XLB',
        'Utilities': 'XLU'
    }


# Major Market Indices (ETF Proxies)
INDEX_COMPONENTS = {
    'SP500': ['SPY'],
    'NASDAQ100': ['QQQ'],
    'DOW': ['DIA'],
    'RUSSELL2000': ['IWM']
}


# ============================================================================
# UNIVERSE CONSTRUCTOR CLASS
# ============================================================================

class UniverseConstructor:
    """
    ============================================================================
    INSTITUTIONAL-GRADE UNIVERSE CONSTRUCTOR
    ============================================================================
    
    Professional system for building and managing trading universes with
    sophisticated filtering, risk management, and strategy testing capabilities.
    
    DESIGN PHILOSOPHY:
    ------------------
    This class is designed to make strategy testing extremely easy by providing:
    
    1. Pre-organized sector classifications with 500+ stocks
    2. One-line universe building with intelligent defaults
    3. Seamless integration with backtesting and strategy systems
    4. Comprehensive metadata for all universe members
    5. Risk management built-in (sector limits, correlation controls)
    
    USAGE PATTERNS:
    ---------------
    
    Quick Start - Default Universe:
        universe = UniverseConstructor()
        symbols = universe.get_universe()
    
    Strategy Testing - Optimized Universe:
        universe = UniverseConstructor()
        symbols = universe.build_optimized_universe(target_size=20)
        # Now test your strategy on these symbols
    
    Custom Universe - Sector Balanced:
        universe = UniverseConstructor()
        symbols = universe.create_sector_balanced_universe(
            target_size=30,
            stocks_per_sector=3
        )
    
    ============================================================================
    """
    
    # Class-level sector mapping (shared across instances)
    SECTOR_MAPPING = SECTOR_MAPPING
    SECTOR_ETFS = SECTOR_ETFS
    INDEX_COMPONENTS = INDEX_COMPONENTS
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        min_market_cap: float = 2_000_000_000,      # $2B minimum
        min_avg_volume: float = 1_000_000,          # 1M shares/day
        max_beta: float = 2.5,                      # Maximum volatility
        min_beta: float = 0.3,                      # Minimum volatility
        max_sector_exposure: float = 0.30,          # 30% max per sector
        max_correlation: float = 0.85,               # Correlation threshold
        lookback_days: int = 252,                    # 1 year for calculations
        api_delay: float = 2.0                       # API rate limiting (seconds)
    ):
        """
        Initialize universe constructor
        
        Parameters:
        -----------
        symbols : Optional[List[str]]
            Initial symbol list. If None, uses high-quality default universe.
            
        min_market_cap : float
            Minimum market capitalization in USD (default: $2B)
            Filters out micro-cap and small-cap stocks
            
        min_avg_volume : float
            Minimum average daily volume in shares (default: 1M)
            Ensures sufficient liquidity for trading
            
        max_beta : float
            Maximum beta threshold (default: 2.5)
            Filters out extremely volatile stocks
            
        min_beta : float
            Minimum beta threshold (default: 0.3)
            Filters out low-volatility stocks (not ideal for swing trading)
            
        max_sector_exposure : float
            Maximum allocation to any single sector (default: 0.30 = 30%)
            Prevents over-concentration in one sector
            
        max_correlation : float
            Maximum correlation for de-duplication (default: 0.85)
            Removes highly correlated stocks to reduce redundancy
            
        lookback_days : int
            Days of historical data for correlation/volatility (default: 252 = 1 year)
            
        api_delay : float
            Delay between API calls in seconds (default: 2.0)
            Prevents rate limiting from data providers
            
        Examples:
        ---------
        >>> # Default initialization
        >>> universe = UniverseConstructor()
        >>> 
        >>> # Custom parameters for swing trading
        >>> universe = UniverseConstructor(
        ...     min_market_cap=5_000_000_000,  # $5B minimum
        ...     min_avg_volume=2_000_000,      # 2M shares/day
        ...     max_beta=1.8,                  # Moderate volatility
        ...     min_beta=0.8
        ... )
        """
        # Store parameters
        self.symbols = symbols or self._get_default_universe()
        self.min_market_cap = min_market_cap
        self.min_avg_volume = min_avg_volume
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        self.lookback_days = lookback_days
        self.api_delay = api_delay
        
        # Performance optimization - caching
        self._metadata_cache = {}
        self._correlation_cache = {}
        self._last_api_call_time = 0
        
        # Rate limiting management
        self._rate_limited = False
        self._rate_limit_wait_time = 60  # Wait 60 seconds if rate limited
    
    # ========================================================================
    # DEFAULT UNIVERSE GENERATION
    # ========================================================================
        
    def _get_default_universe(self) -> List[str]:
        """
        Generate default high-quality universe for immediate use.
        
        Returns:
        --------
        List[str]
            List of high-quality, liquid symbols suitable for strategy testing
            
        Notes:
        ------
        This default universe includes:
        - Mega-cap technology stocks (high options volume)
        - Major financial institutions
        - Healthcare leaders
        - Consumer discretionary leaders
        - Energy majors
        - Broad market ETFs for diversification
        """
        default = [
            # Technology (high options volume, liquid)
            'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL', 'AMZN',
            
            # Financial Services
            'JPM', 'BAC', 'GS', 'MS',
            
            # Healthcare
            'UNH', 'JNJ', 'PFE', 'ABBV',
            
            # Consumer Discretionary
            'NFLX', 'HD', 'NKE',
            
            # Energy
            'XOM', 'CVX',
            
            # ETFs for broad market exposure
            'SPY', 'QQQ', 'IWM'
        ]
        
        return default
    
    # ========================================================================
    # SECTOR CLASSIFICATION
    # ========================================================================
    
    def get_symbol_sector(self, symbol: str) -> Optional[str]:
        """
        Get GICS sector classification for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        Optional[str]
            Sector name if found, None otherwise
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> universe.get_symbol_sector('AAPL')
        'Technology'
        >>> universe.get_symbol_sector('JPM')
        'Financial Services'
        """
        for sector, symbols in self.SECTOR_MAPPING.items():
            if symbol in symbols:
                return sector
        return None
    
    def get_sector_symbols(self, sector: str) -> List[str]:
        """
        Get all symbols in a specific sector.
        
        Parameters:
        -----------
        sector : str
            Sector name (e.g., 'Technology', 'Healthcare')
            
        Returns:
        --------
        List[str]
            List of symbols in that sector
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> tech_stocks = universe.get_sector_symbols('Technology')
        >>> len(tech_stocks)
        50
        """
        return self.SECTOR_MAPPING.get(sector, [])
    
    def get_all_sectors(self) -> List[str]:
        """
        Get list of all available sectors.
        
        Returns:
        --------
        List[str]
            List of sector names
        """
        return list(self.SECTOR_MAPPING.keys())
    
    # ========================================================================
    # METADATA FETCHING & CACHING
    # ========================================================================
    
    def fetch_symbol_metadata(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> Dict:
        """
        Fetch comprehensive metadata for a symbol with intelligent caching.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
            
        use_cache : bool
            Whether to use cached data (default: True)
        
        Returns:
        --------
        Dict
            Dictionary containing:
            - Basic info: symbol, name, sector, industry
            - Market data: market_cap, avg_volume, current_price
            - Risk metrics: beta, volatility indicators
            - Financial metrics: pe_ratio, profit_margins, debt_to_equity
            - Performance: revenue_growth, earnings_growth
            - Price metrics: 52-week high/low, price vs 52w high
            
        Notes:
        ------
        - Automatically handles rate limiting
        - Caches results for performance
        - Returns minimal metadata on errors (graceful degradation)
        
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> metadata = universe.fetch_symbol_metadata('AAPL')
        >>> print(f"Market Cap: ${metadata['market_cap']/1e9:.1f}B")
        >>> print(f"Beta: {metadata['beta']:.2f}")
        """
        # Check cache first
        if use_cache and symbol in self._metadata_cache:
            return self._metadata_cache[symbol]
        
        # Check rate limiting
        if self._rate_limited:
            warnings.warn(
                f"Rate limited. Skipping {symbol}. "
                f"Wait {self._rate_limit_wait_time} seconds before retrying."
            )
            return self._get_minimal_metadata(symbol, error='Rate limit active', rate_limited=True)
        
        # Rate limiting: add delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call_time
        if time_since_last_call < self.api_delay:
            time.sleep(self.api_delay - time_since_last_call)
        self._last_api_call_time = time.time()
        
        # Fetch data with error handling
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get info with better error handling
            try:
                info = ticker.info
                if not info or len(info) == 0:
                    raise ValueError("Empty response from API")
            except Exception as e:
                # Check for rate limiting
                error_str = str(e)
                error_lower = error_str.lower()
                
                if ('429' in error_str or 
                    'too many' in error_lower or 
                    'rate limit' in error_lower or
                    'too many requests' in error_lower):
                    self._rate_limited = True
                    warnings.warn(f"Rate limit detected for {symbol}. Setting rate limit flag.")
                    raise ConnectionError("Rate limit exceeded")
                
                if 'expecting value' in error_lower or 'json' in error_lower:
                    self._rate_limited = True
                    warnings.warn(f"Invalid response (likely rate limit) for {symbol}.")
                    raise ConnectionError("Rate limit exceeded - invalid response")
                
                raise
            
            # Build comprehensive metadata dictionary
            metadata = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', self.get_symbol_sector(symbol) or 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 1.0),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'dividend_yield': info.get('dividendYield', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'profit_margins': info.get('profitMargins', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'quote_type': info.get('quoteType', 'EQUITY')
            }
            
            # Calculate additional derived metrics
            if metadata['current_price'] and metadata['fifty_two_week_high']:
                metadata['price_vs_52w_high'] = (
                    metadata['current_price'] / metadata['fifty_two_week_high']
                )
            
            # Cache the result
            if use_cache:
                self._metadata_cache[symbol] = metadata
            
            return metadata
            
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            
            # Handle rate limiting
            if ('429' in error_str or 
                'too many' in error_lower or 
                'rate limit' in error_lower or 
                isinstance(e, ConnectionError)):
                self._rate_limited = True
                warnings.warn(
                    f"Rate limit hit for {symbol}. "
                    f"All subsequent API calls will be skipped for {self._rate_limit_wait_time} seconds."
                )
                return self._get_minimal_metadata(symbol, error='Rate limit exceeded', rate_limited=True)
            
            # Return minimal metadata on other errors
            return self._get_minimal_metadata(symbol, error=error_str)
    
    def _get_minimal_metadata(self, symbol: str, error: str = '', rate_limited: bool = False) -> Dict:
        """
        Generate minimal metadata structure for error cases.
        
        Internal helper method for graceful error handling.
        """
        return {
            'symbol': symbol,
            'name': 'Rate Limited' if rate_limited else f'Error: {error[:50]}',
            'sector': self.get_symbol_sector(symbol) or 'N/A',
            'market_cap': 0,
                'avg_volume': 0,
                'beta': 1.0,
                'current_price': 0,
            'error': error,
            'rate_limited': rate_limited
            }
    
    # ========================================================================
    # FILTERING METHODS
    # ========================================================================
    
    def filter_by_liquidity(
        self,
        symbols: Optional[List[str]] = None,
        min_volume: Optional[float] = None,
        min_market_cap: Optional[float] = None
    ) -> List[str]:
        """
        Filter symbols by liquidity criteria (volume and market cap).
        
        Parameters:
        -----------
        symbols : Optional[List[str]]
            Symbols to filter. If None, uses current universe.
            
        min_volume : Optional[float]
            Minimum average daily volume (overrides instance default)
            
        min_market_cap : Optional[float]
            Minimum market cap (overrides instance default)
        
        Returns:
        --------
        List[str]
            Filtered list of symbols meeting liquidity criteria
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> liquid = universe.filter_by_liquidity(
        ...     min_volume=2_000_000,      # 2M shares/day
        ...     min_market_cap=5_000_000_000  # $5B
        ... )
        """
        symbols = symbols or self.symbols
        min_volume = min_volume or self.min_avg_volume
        min_market_cap = min_market_cap or self.min_market_cap
        
        filtered = []
        for symbol in symbols:
            try:
                metadata = self.fetch_symbol_metadata(symbol)
                
                avg_vol = metadata.get('avg_volume', 0)
                market_cap = metadata.get('market_cap', 0)
                
                if avg_vol >= min_volume and market_cap >= min_market_cap:
                    filtered.append(symbol)
                    
            except Exception as e:
                warnings.warn(f"Error filtering {symbol}: {e}")
                continue
        
        return filtered
    
    def filter_by_volatility(
        self,
        symbols: Optional[List[str]] = None,
        min_beta: Optional[float] = None,
        max_beta: Optional[float] = None
    ) -> List[str]:
        """
        Filter symbols by volatility (beta) range.
        
        Parameters:
        -----------
        symbols : Optional[List[str]]
            Symbols to filter
            
        min_beta : Optional[float]
            Minimum beta threshold (overrides instance default)
            
        max_beta : Optional[float]
            Maximum beta threshold (overrides instance default)
        
        Returns:
        --------
        List[str]
            Filtered list of symbols within beta range
            
        Examples:
        ---------
        >>> # Filter for moderate volatility (ideal for swing trading)
        >>> universe = UniverseConstructor()
        >>> moderate_vol = universe.filter_by_volatility(
        ...     min_beta=0.8,
        ...     max_beta=1.5
        ... )
        """
        symbols = symbols or self.symbols
        min_beta = min_beta or self.min_beta
        max_beta = max_beta or self.max_beta
        
        filtered = []
        for symbol in symbols:
            try:
                metadata = self.fetch_symbol_metadata(symbol)
                beta = metadata.get('beta', 1.0)
                
                if min_beta <= beta <= max_beta:
                    filtered.append(symbol)
                    
            except Exception as e:
                warnings.warn(f"Error filtering {symbol} by volatility: {e}")
                continue
        
        return filtered
    
    def filter_by_quality(
        self,
        symbols: Optional[List[str]] = None,
        min_market_cap: Optional[float] = None,
        max_debt_to_equity: Optional[float] = None,
        min_profit_margin: Optional[float] = None
    ) -> List[str]:
        """
        Filter symbols by quality metrics (profitability, debt levels).
        
        Parameters:
        -----------
        symbols : Optional[List[str]]
            Symbols to filter
            
        min_market_cap : Optional[float]
            Minimum market cap
            
        max_debt_to_equity : Optional[float]
            Maximum debt-to-equity ratio (e.g., 2.0 = 200%)
            
        min_profit_margin : Optional[float]
            Minimum profit margin (e.g., 0.05 = 5%)
        
        Returns:
        --------
        List[str]
            Filtered list of high-quality symbols
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> quality_stocks = universe.filter_by_quality(
        ...     max_debt_to_equity=1.0,    # D/E < 100%
        ...     min_profit_margin=0.10     # Profit margin > 10%
        ... )
        """
        symbols = symbols or self.symbols
        min_market_cap = min_market_cap or self.min_market_cap
        
        filtered = []
        for symbol in symbols:
            try:
                metadata = self.fetch_symbol_metadata(symbol)
                
                market_cap = metadata.get('market_cap', 0)
                if market_cap < min_market_cap:
                    continue
                
                # Optional quality filters
                if max_debt_to_equity is not None:
                    dte = metadata.get('debt_to_equity')
                    if dte is not None and dte > max_debt_to_equity:
                        continue
                
                if min_profit_margin is not None:
                    pm = metadata.get('profit_margins')
                    if pm is not None and pm < min_profit_margin:
                        continue
                
                filtered.append(symbol)
                
            except Exception as e:
                warnings.warn(f"Error filtering {symbol} by quality: {e}")
                continue
        
        return filtered
    
    # ========================================================================
    # CORRELATION ANALYSIS
    # ========================================================================
    
    def calculate_correlation_matrix(
        self,
        symbols: List[str],
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for symbols using historical returns.
        
        Parameters:
        -----------
        symbols : List[str]
            List of symbols to analyze
            
        lookback_days : Optional[int]
            Days of historical data to use (overrides instance default)
        
        Returns:
        --------
        pd.DataFrame
            Correlation matrix with symbols as index and columns
            
        Notes:
        ------
        - Uses daily returns for correlation calculation
        - Automatically handles missing data
        - Caches results for performance
        - Returns empty DataFrame if rate limited
        
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> corr_matrix = universe.calculate_correlation_matrix(
        ...     ['AAPL', 'MSFT', 'NVDA', 'JPM', 'BAC']
        ... )
        >>> print(corr_matrix.loc['AAPL', 'MSFT'])  # Correlation between AAPL and MSFT
        """
        lookback_days = lookback_days or self.lookback_days
        
        # Check cache
        cache_key = tuple(sorted(symbols))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]
        
        # Check rate limiting
        if self._rate_limited:
            warnings.warn("Rate limited. Skipping correlation calculation.")
            return pd.DataFrame()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
        
        # Fetch historical prices
        prices = {}
        for symbol in symbols:
            # Rate limiting for historical data
            current_time = time.time()
            time_since_last_call = current_time - self._last_api_call_time
            if time_since_last_call < self.api_delay:
                time.sleep(self.api_delay - time_since_last_call)
            self._last_api_call_time = time.time()
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty:
                    prices[symbol] = data['Close']
                    
            except Exception as e:
                error_str = str(e)
                error_lower = error_str.lower()
                if '429' in error_str or 'too many' in error_lower or 'rate limit' in error_lower:
                    self._rate_limited = True
                    warnings.warn(f"Rate limit hit for {symbol}. Stopping correlation calculation.")
                    break
                warnings.warn(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not prices:
            return pd.DataFrame()
        
        # Align prices and calculate returns
        price_df = pd.DataFrame(prices)
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation
        corr_matrix = returns_df.corr()
        
        # Cache result
        self._correlation_cache[cache_key] = corr_matrix
        
        return corr_matrix
    
    def remove_highly_correlated(
        self,
        symbols: Optional[List[str]] = None,
        max_correlation: Optional[float] = None,
        prefer_larger_cap: bool = True
    ) -> List[str]:
        """
        Remove highly correlated symbols, keeping the best representative.
        
        Parameters:
        -----------
        symbols : Optional[List[str]]
            Symbols to deduplicate
            
        max_correlation : Optional[float]
            Maximum allowed correlation (overrides instance default)
            
        prefer_larger_cap : bool
            If True, keep larger market cap when removing duplicates
        
        Returns:
        --------
        List[str]
            Deduplicated symbol list with reduced correlation
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> # Remove stocks with correlation > 0.85
        >>> deduplicated = universe.remove_highly_correlated(
        ...     symbols=['JPM', 'BAC', 'WFC', 'C'],  # Banks (highly correlated)
        ...     max_correlation=0.80
        ... )
        """
        symbols = symbols or self.symbols
        max_correlation = max_correlation or self.max_correlation
        
        if len(symbols) <= 1:
            return symbols
        
        # Get metadata for all symbols
        metadata = {s: self.fetch_symbol_metadata(s) for s in symbols}
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return symbols
        
        # Find highly correlated pairs
        to_remove = set()
        for i, sym1 in enumerate(symbols):
            if sym1 in to_remove:
                continue
            
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if sym2 in to_remove:
                    continue
                
                if sym1 in corr_matrix.index and sym2 in corr_matrix.columns:
                    corr = corr_matrix.loc[sym1, sym2]
                    
                    if not pd.isna(corr) and abs(corr) > max_correlation:
                        # Decide which to remove
                        if prefer_larger_cap:
                            cap1 = metadata[sym1].get('market_cap', 0)
                            cap2 = metadata[sym2].get('market_cap', 0)
                            
                            if cap1 < cap2:
                                to_remove.add(sym1)
                            else:
                                to_remove.add(sym2)
                        else:
                            to_remove.add(sym2)
        
        return [s for s in symbols if s not in to_remove]
    
    # ========================================================================
    # UNIVERSE BUILDING METHODS
    # ========================================================================
    
    def create_sector_balanced_universe(
        self,
        target_size: int = 20,
        stocks_per_sector: Optional[int] = None,
        min_per_sector: int = 1,
        max_per_sector: Optional[int] = None
    ) -> List[str]:
        """
        Create a sector-balanced universe with proper diversification.
        
        Parameters:
        -----------
        target_size : int
            Target number of stocks in universe (default: 20)
            
        stocks_per_sector : Optional[int]
            Fixed number per sector (if specified, overrides target_size logic)
            
        min_per_sector : int
            Minimum stocks per sector (default: 1)
            
        max_per_sector : Optional[int]
            Maximum stocks per sector (prevents over-concentration)
        
        Returns:
        --------
        List[str]
            Balanced universe list with sector diversification
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> balanced = universe.create_sector_balanced_universe(
        ...     target_size=30,
        ...     stocks_per_sector=3  # 3 stocks from each sector
        ... )
        """
        # Get all candidate symbols from sector mapping
        all_candidates = []
        for sector, symbols in self.SECTOR_MAPPING.items():
            for symbol in symbols:
                if symbol not in all_candidates:
                    all_candidates.append(symbol)
        
        # Apply filters
        filtered = self.filter_by_liquidity(all_candidates)
        filtered = self.filter_by_volatility(filtered)
        
        # Group by sector
        sector_groups = {}
        for symbol in filtered:
            sector = self.get_symbol_sector(symbol)
            if sector:
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(symbol)
        
        # Select stocks per sector
        selected = []
        if stocks_per_sector:
            # Fixed number per sector
            for sector, symbols in sector_groups.items():
                # Sort by market cap (largest first)
                symbols_with_meta = [
                    (s, self.fetch_symbol_metadata(s).get('market_cap', 0))
                    for s in symbols
                ]
                symbols_with_meta.sort(key=lambda x: x[1], reverse=True)
                
                selected.extend([s for s, _ in symbols_with_meta[:stocks_per_sector]])
        else:
            # Dynamic allocation to reach target size
            sectors = list(sector_groups.keys())
            per_sector = max(min_per_sector, target_size // len(sectors))
            
            if max_per_sector:
                per_sector = min(per_sector, max_per_sector)
            
            for sector, symbols in sector_groups.items():
                # Sort by market cap
                symbols_with_meta = [
                    (s, self.fetch_symbol_metadata(s).get('market_cap', 0))
                    for s in symbols
                ]
                symbols_with_meta.sort(key=lambda x: x[1], reverse=True)
                
                selected.extend([s for s, _ in symbols_with_meta[:per_sector]])
        
        # Remove highly correlated
        selected = self.remove_highly_correlated(selected)
        
        # Trim to target size if needed
        if len(selected) > target_size:
            scored = self.score_universe(selected)
            selected = [s for s, _ in scored[:target_size]]
        
        return selected

    def create_manual_sector_balanced_universe(
        self,
        target_size: int = 20,
        stocks_per_sector: Optional[int] = None,
        sectors: Optional[List[str]] = None,
        include_index_etfs: bool = True
    ) -> List[str]:
        """
        Create a manual sector-balanced universe WITHOUT using metadata filters.

        Why this exists:
        - Avoids look-ahead/selection bias from using "current" fundamentals (market cap, beta, etc.)
        - Produces a stable, reproducible universe suitable for backtests

        How selection works:
        - Starts from the curated `SECTOR_MAPPING` lists (ordered roughly by market-cap tiers)
        - Picks the first N tickers per sector (or an even-ish allocation to reach target_size)
        - Optionally appends index ETF proxies (SPY/QQQ/DIA/IWM)
        """
        chosen_sectors = sectors or list(self.SECTOR_MAPPING.keys())

        # Build per-sector candidate lists in a deterministic order
        sector_candidates: Dict[str, List[str]] = {}
        for sector in chosen_sectors:
            symbols = self.SECTOR_MAPPING.get(sector, [])
            # De-dupe within the sector while preserving order
            seen = set()
            deduped = []
            for s in symbols:
                if s not in seen:
                    deduped.append(s)
                    seen.add(s)
            if deduped:
                sector_candidates[sector] = deduped

        if not sector_candidates:
            return []

        selected: List[str] = []

        if stocks_per_sector is not None:
            for sector in sector_candidates:
                selected.extend(sector_candidates[sector][: max(0, stocks_per_sector)])
        else:
            # Even-ish allocation to reach target_size
            sector_list = list(sector_candidates.keys())
            per_sector = max(1, target_size // len(sector_list))
            for sector in sector_list:
                selected.extend(sector_candidates[sector][:per_sector])

        # Global de-dupe while preserving order
        seen_global = set()
        selected_deduped: List[str] = []
        for s in selected:
            if s not in seen_global:
                selected_deduped.append(s)
                seen_global.add(s)

        if include_index_etfs:
            for _, etfs in self.INDEX_COMPONENTS.items():
                for etf in etfs:
                    if etf not in seen_global:
                        selected_deduped.append(etf)
                        seen_global.add(etf)

        # Trim to target_size if needed (do NOT score/filter; keep deterministic)
        if len(selected_deduped) > target_size:
            selected_deduped = selected_deduped[:target_size]

        return selected_deduped
    
    def build_optimized_universe(
        self,
        target_size: int = 20,
        apply_filters: bool = True,
        remove_correlated: bool = True
    ) -> List[str]:
        """
        Build an optimized universe using all filters and optimizations.
        
        This is the recommended method for strategy testing as it applies
        all quality filters and optimizations automatically.
        
        Parameters:
        -----------
        target_size : int
            Target number of symbols in final universe
            
        apply_filters : bool
            Apply liquidity, volatility, and quality filters
            
        remove_correlated : bool
            Remove highly correlated symbols
            
        Returns:
        --------
        List[str]
            Optimized universe list ready for strategy testing
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> 
        >>> # Build optimized universe for strategy testing
        >>> symbols = universe.build_optimized_universe(
        ...     target_size=25,
        ...     apply_filters=True,
        ...     remove_correlated=True
        ... )
        >>> 
        >>> # Now test your strategy on these symbols
        >>> for symbol in symbols:
        ...     # Run your strategy logic here
        ...     pass
        """
        candidates = self.symbols.copy()
        
        if apply_filters:
            candidates = self.filter_by_liquidity(candidates)
            candidates = self.filter_by_volatility(candidates)
            candidates = self.filter_by_quality(candidates)
        
        if remove_correlated:
            candidates = self.remove_highly_correlated(candidates)
        
        # Score and select top N
        scored = self.score_universe(candidates)
        optimized = [symbol for symbol, _ in scored[:target_size]]
        
        self.symbols = optimized
        return optimized
    
    # ========================================================================
    # SCORING & RANKING
    # ========================================================================
    
    def score_universe(
        self,
        symbols: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Score and rank universe symbols using multi-factor scoring system.
        
        Scoring Methodology:
        --------------------
        Total Score: 0-100 points
        
        Liquidity Score (0-30 points):
            - Market Cap: $10B+ = 15pts, $2B+ = 10pts
            - Volume: 5M+ = 15pts, 1M+ = 10pts
        
        Quality Score (0-25 points):
            - Profit Margin: >20% = 15pts, >10% = 10pts
            - Debt/Equity: <0.5 = 10pts, <1.0 = 5pts
        
        Volatility Score (0-25 points):
            - Beta 0.8-1.5 (sweet spot) = 25pts
            - Beta 0.5-2.0 = 15pts
            - Beta 0.3-2.5 = 10pts
        
        Momentum Score (0-20 points):
            - Price vs 52w High: 70-95% = 20pts
            - Price vs 52w High: 50-98% = 10pts
        
        Parameters:
        -----------
        symbols : Optional[List[str]]
            Symbols to score. If None, uses current universe.
        
        Returns:
        --------
        List[Tuple[str, float]]
            List of (symbol, score) tuples, sorted by score descending
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> scored = universe.score_universe(['AAPL', 'MSFT', 'NVDA'])
        >>> for symbol, score in scored:
        ...     print(f"{symbol}: {score:.1f} points")
        """
        symbols = symbols or self.symbols
        
        scores = []
        for symbol in symbols:
            try:
                metadata = self.fetch_symbol_metadata(symbol)
                score = 0.0
                
                # Liquidity score (0-30 points)
                market_cap = metadata.get('market_cap', 0)
                avg_volume = metadata.get('avg_volume', 0)
                
                if market_cap >= 10_000_000_000:
                    score += 15
                elif market_cap >= 2_000_000_000:
                    score += 10
                
                if avg_volume >= 5_000_000:
                    score += 15
                elif avg_volume >= 1_000_000:
                    score += 10
                
                # Quality score (0-25 points)
                profit_margin = metadata.get('profit_margins')
                if profit_margin is not None:
                    if profit_margin > 0.20:
                        score += 15
                    elif profit_margin > 0.10:
                        score += 10
                
                debt_to_equity = metadata.get('debt_to_equity')
                if debt_to_equity is not None:
                    if debt_to_equity < 0.5:
                        score += 10
                    elif debt_to_equity < 1.0:
                        score += 5
                
                # Volatility score (0-25 points)
                beta = metadata.get('beta', 1.0)
                if 0.8 <= beta <= 1.5:
                    score += 25
                elif 0.5 <= beta <= 2.0:
                    score += 15
                elif 0.3 <= beta <= 2.5:
                    score += 10
                
                # Momentum score (0-20 points)
                price_vs_52w = metadata.get('price_vs_52w_high')
                if price_vs_52w is not None:
                    if 0.7 <= price_vs_52w <= 0.95:
                        score += 20
                    elif 0.5 <= price_vs_52w <= 0.98:
                        score += 10
                
                scores.append((symbol, score))
                
            except Exception as e:
                warnings.warn(f"Error scoring {symbol}: {e}")
                scores.append((symbol, 0.0))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    # ========================================================================
    # ANALYSIS & REPORTING
    # ========================================================================
    
    def get_universe_info(self) -> pd.DataFrame:
        """
        Get comprehensive information about all universe symbols.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with symbol metadata including:
            - Basic info: symbol, name, sector, industry
            - Market data: market_cap, avg_volume, current_price
            - Risk metrics: beta
            - Financial metrics: pe_ratio, profit_margins, debt_to_equity
            - Score: calculated universe score
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> info = universe.get_universe_info()
        >>> print(info[['symbol', 'sector', 'market_cap', 'score']].head())
        """
        info_list = []
        
        for symbol in self.symbols:
            metadata = self.fetch_symbol_metadata(symbol)
            info_list.append(metadata)
        
        df = pd.DataFrame(info_list)
        
        # Add scoring if not already present
        if 'score' not in df.columns:
            scored = self.score_universe(self.symbols)
            score_dict = {symbol: score for symbol, score in scored}
            df['score'] = df['symbol'].map(score_dict).fillna(0)
        
        return df
    
    def analyze_sector_exposure(self) -> pd.DataFrame:
        """
        Analyze sector exposure of current universe.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns:
            - sector: Sector name
            - count: Number of stocks in sector
            - percentage: Percentage of universe
            - within_limit: Boolean indicating if within max_sector_exposure limit
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> exposure = universe.analyze_sector_exposure()
        >>> print(exposure)
        """
        metadata = self.get_universe_info()
        
        sector_counts = metadata['sector'].value_counts()
        sector_pct = (sector_counts / len(metadata) * 100).round(2)
        
        analysis = pd.DataFrame({
            'sector': sector_counts.index,
            'count': sector_counts.values,
            'percentage': sector_pct.values
        })
        
        # Check for concentration
        analysis['within_limit'] = analysis['percentage'] <= (self.max_sector_exposure * 100)
        
        return analysis.sort_values('percentage', ascending=False)
    
    # ========================================================================
    # STRATEGY TESTING HELPERS
    # ========================================================================
    
    def get_universe_for_strategy_testing(
        self,
        strategy_type: str = 'swing_trading',
        target_size: int = 20
    ) -> List[str]:
        """
        Get optimized universe specifically configured for strategy testing.
        
        This method provides pre-configured universes optimized for different
        strategy types, making it extremely easy to test strategies.
        
        Parameters:
        -----------
        strategy_type : str
            Type of strategy:
            - 'swing_trading': Moderate volatility, high liquidity (default)
            - 'day_trading': Higher volatility, very high liquidity
            - 'position_trading': Lower volatility, quality focus
            - 'options_trading': High options volume, moderate volatility
            
        target_size : int
            Target number of symbols
            
        Returns:
        --------
        List[str]
            Optimized universe for the specified strategy type
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> 
        >>> # Get universe optimized for swing trading
        >>> symbols = universe.get_universe_for_strategy_testing(
        ...     strategy_type='swing_trading',
        ...     target_size=25
        ... )
        >>> 
        >>> # Now test your swing trading strategy
        >>> for symbol in symbols:
        ...     # Your strategy logic here
        ...     pass
        """
        if strategy_type == 'swing_trading':
            # Moderate volatility, high liquidity
            self.min_beta = 0.8
            self.max_beta = 1.5
            self.min_avg_volume = 1_000_000
            self.min_market_cap = 2_000_000_000
            
        elif strategy_type == 'day_trading':
            # Higher volatility, very high liquidity
            self.min_beta = 1.0
            self.max_beta = 2.0
            self.min_avg_volume = 5_000_000
            self.min_market_cap = 5_000_000_000
            
        elif strategy_type == 'position_trading':
            # Lower volatility, quality focus
            self.min_beta = 0.5
            self.max_beta = 1.2
            self.min_avg_volume = 500_000
            self.min_market_cap = 10_000_000_000
            
        elif strategy_type == 'options_trading':
            # High options volume, moderate volatility
            self.min_beta = 0.8
            self.max_beta = 1.8
            self.min_avg_volume = 2_000_000
            self.min_market_cap = 3_000_000_000
        
        return self.build_optimized_universe(
            target_size=target_size,
            apply_filters=True,
            remove_correlated=True
        )
    
    def get_universe_summary(self) -> Dict:
        """
        Get comprehensive summary of current universe.
        
        Returns:
        --------
        Dict
            Dictionary with universe statistics including:
            - total_symbols: Number of symbols
            - sector_distribution: Count by sector
            - avg_market_cap: Average market capitalization
            - avg_volume: Average daily volume
            - avg_beta: Average beta
            - sector_exposure: Sector exposure analysis
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> summary = universe.get_universe_summary()
        >>> print(f"Total symbols: {summary['total_symbols']}")
        >>> print(f"Average market cap: ${summary['avg_market_cap']/1e9:.1f}B")
        """
        info = self.get_universe_info()
        exposure = self.analyze_sector_exposure()
        
        return {
            'total_symbols': len(self.symbols),
            'sector_distribution': exposure.to_dict('records'),
            'avg_market_cap': float(info['market_cap'].mean()),
            'avg_volume': float(info['avg_volume'].mean()),
            'avg_beta': float(info['beta'].mean()),
            'avg_score': float(info['score'].mean()) if 'score' in info.columns else 0.0,
            'sector_exposure': exposure
        }
    
    # ========================================================================
    # UNIVERSE MANAGEMENT
    # ========================================================================
    
    def get_universe(self) -> List[str]:
        """
        Get current universe symbols.
        
        Returns:
        --------
        List[str]
            Copy of current universe symbol list
        """
        return self.symbols.copy()
    
    def add_symbol(self, symbol: str):
        """
        Add a symbol to the universe.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol to add
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            # Clear cache for this symbol
            if symbol in self._metadata_cache:
                del self._metadata_cache[symbol]
    
    def remove_symbol(self, symbol: str):
        """
        Remove a symbol from the universe.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol to remove
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
    
    def reset_rate_limit(self):
        """
        Reset the rate limit flag.
        
        Call this after waiting for rate limits to clear.
        """
        self._rate_limited = False
        warnings.warn("Rate limit flag reset. You can now make API calls again.")
    
    # ========================================================================
    # EXPORT & IMPORT
    # ========================================================================
    
    def export_universe(self, filepath: str, include_metadata: bool = False):
        """
        Export universe to file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
            
        include_metadata : bool
            If True, export as CSV with full metadata
            If False, export as simple text file with symbols only
            
        Examples:
        ---------
        >>> universe = UniverseConstructor()
        >>> universe.export_universe('my_universe.txt')
        >>> universe.export_universe('my_universe.csv', include_metadata=True)
        """
        if include_metadata:
            info = self.get_universe_info()
            info.to_csv(filepath, index=False)
        else:
            with open(filepath, 'w') as f:
                for symbol in self.symbols:
                    f.write(f"{symbol}\n")
    
    @staticmethod
    def load_universe(filepath: str) -> List[str]:
        """
        Load universe from file.
        
        Parameters:
        -----------
        filepath : str
            Input file path (CSV or text file)
        
        Returns:
        --------
        List[str]
            List of symbols from file
            
        Examples:
        ---------
        >>> symbols = UniverseConstructor.load_universe('my_universe.txt')
        >>> universe = UniverseConstructor(symbols=symbols)
        """
        def _load_text_symbols(path: str) -> List[str]:
            symbols: List[str] = []
            with open(path, 'r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    # Allow comment lines in universe text files
                    if line.startswith('#') or line.startswith(';'):
                        continue
                    # Allow inline comments: "AAPL  # apple"
                    if '#' in line:
                        line = line.split('#', 1)[0].strip()
                    if ';' in line:
                        # common alt comment delimiter
                        line = line.split(';', 1)[0].strip()
                    if line:
                        symbols.append(line)
            return symbols

        try:
            # Try CSV first
            df = pd.read_csv(filepath)
            if 'symbol' in df.columns:
                return df['symbol'].tolist()
            else:
                # Plain text file
                return _load_text_symbols(filepath)
        except Exception:
            # Fallback to plain text
            return _load_text_symbols(filepath)
