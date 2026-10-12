"""
Fast unit tests for utility functions.
"""

import pytest

from utils.validation import (
    normalize_symbol,
    validate_capital,
    validate_date,
    validate_date_range,
    validate_position_size,
    validate_symbol,
)


class TestValidationUtils:
    """Test suite for validation utilities."""
    
    @pytest.mark.fast
    def test_validate_symbol(self):
        """Test symbol validation."""
        assert validate_symbol("AAPL") is True
        assert validate_symbol("MSFT") is True
        assert validate_symbol("TSLA") is True
        assert validate_symbol("INVALID123") is False
        assert validate_symbol("") is False
        assert validate_symbol("TOOLONG123") is False
    
    @pytest.mark.fast
    def test_validate_date(self):
        """Test date validation."""
        assert validate_date("2023-01-01") is True
        assert validate_date("2024-12-31") is True
        assert validate_date("2023-13-01") is False
        assert validate_date("invalid") is False
        assert validate_date("2023/01/01") is False
    
    @pytest.mark.fast
    def test_validate_capital(self):
        """Test capital validation."""
        assert validate_capital(100000) is True
        assert validate_capital(50000.5) is True
        assert validate_capital(0) is False
        assert validate_capital(-1000) is False
    
    @pytest.mark.fast
    def test_validate_position_size(self):
        """Test position size validation."""
        assert validate_position_size(0.1) is True
        assert validate_position_size(0.5) is True
        assert validate_position_size(1.0) is True
        assert validate_position_size(0) is False
        assert validate_position_size(1.5) is False
        assert validate_position_size(-0.1) is False
    
    @pytest.mark.fast
    def test_validate_date_range(self):
        """Test date range validation."""
        is_valid, error = validate_date_range("2023-01-01", "2024-01-01")
        assert is_valid is True
        assert error is None
        
        is_valid, error = validate_date_range("2024-01-01", "2023-01-01")
        assert is_valid is False
        assert "before" in error.lower()
        
        is_valid, error = validate_date_range("invalid", "2024-01-01")
        assert is_valid is False
    
    @pytest.mark.fast
    def test_normalize_symbol(self):
        """Test symbol normalization."""
        assert normalize_symbol("aapl") == "AAPL"
        assert normalize_symbol("  MSFT  ") == "MSFT"
        assert normalize_symbol("tsla") == "TSLA"
        assert normalize_symbol("") == ""

