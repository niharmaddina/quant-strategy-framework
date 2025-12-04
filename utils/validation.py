"""
Validation utilities for the trading strategy framework.
"""

import re
from datetime import datetime
from typing import Optional, Tuple


def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: 1-5 uppercase letters/numbers
    pattern = r'^[A-Z0-9]{1,5}$'
    return bool(re.match(pattern, symbol.upper()))


def validate_date(date_str: str, date_format: str = "%Y-%m-%d") -> bool:
    """
    Validate date string format.
    
    Args:
        date_str: Date string to validate
        date_format: Expected date format (default: YYYY-MM-DD)
    
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, date_format)
        return True
    except (ValueError, TypeError):
        return False


def validate_capital(amount: float) -> bool:
    """
    Validate capital amount.
    
    Args:
        amount: Capital amount
    
    Returns:
        True if valid (positive), False otherwise
    """
    return isinstance(amount, (int, float)) and amount > 0


def validate_position_size(size: float) -> bool:
    """
    Validate position size.
    
    Args:
        size: Position size as fraction of capital
    
    Returns:
        True if valid (0 < size <= 1), False otherwise
    """
    return isinstance(size, (int, float)) and 0 < size <= 1


def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:
    """
    Validate date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not validate_date(start_date):
        return False, f"Invalid start date format: {start_date}. Use YYYY-MM-DD"
    
    if not validate_date(end_date):
        return False, f"Invalid end date format: {end_date}. Use YYYY-MM-DD"
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start >= end:
            return False, "Start date must be before end date"
        
        return True, None
    except Exception as e:
        return False, f"Error validating date range: {e}"


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol format (uppercase, strip whitespace).
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Normalized symbol
    """
    if not symbol:
        return ""
    return symbol.upper().strip()

