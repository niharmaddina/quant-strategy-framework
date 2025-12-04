"""
Fast unit tests for DataFetcher with mock data support.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from data.fetcher import DataFetcher
from utils.validation import validate_symbol, validate_date


class TestDataFetcher:
    """Test suite for DataFetcher with fast mock-based tests."""
    
    def test_init(self):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher()
        assert fetcher is not None
    
    def test_validate_symbol(self):
        """Test symbol validation."""
        assert validate_symbol("AAPL") is True
        assert validate_symbol("MSFT") is True
        assert validate_symbol("INVALID123") is False
        assert validate_symbol("") is False
    
    def test_validate_date(self):
        """Test date validation."""
        assert validate_date("2023-01-01") is True
        assert validate_date("2023-13-01") is False
        assert validate_date("invalid") is False
    
    @pytest.mark.fast
    def test_fetch_stock_data_with_mock(self, mock_data_fetcher, sample_price_data):
        """Test fetching data with mock (fast, no network)."""
        mock_data_fetcher.set_mock_data(sample_price_data)
        
        result = mock_data_fetcher.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'close' in result.columns
        assert 'date' in result.columns
    
    @pytest.mark.fast
    def test_data_quality_validation(self, mock_data_fetcher, sample_price_data):
        """Test data quality checks."""
        mock_data_fetcher.set_mock_data(sample_price_data)
        data = mock_data_fetcher.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
        
        summary = mock_data_fetcher.get_data_summary(data)
        
        assert summary['total_days'] > 0
        assert 'date_range' in summary
        assert 'price_range' in summary
    
    def test_invalid_symbol_raises_error(self, mock_data_fetcher):
        """Test that invalid symbol raises ValueError."""
        # Mock fetcher should still validate inputs
        with pytest.raises(ValueError, match="Invalid symbol"):
            mock_data_fetcher.fetch_stock_data('', '2023-01-01', '2023-12-31')
    
    def test_invalid_date_format_raises_error(self, mock_data_fetcher):
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError):
            mock_data_fetcher.fetch_stock_data('AAPL', 'invalid-date', '2023-12-31')
    
    @pytest.mark.fast
    def test_fetch_multiple_symbols_mock(self, mock_data_fetcher):
        """Test fetching multiple symbols with mock data."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = mock_data_fetcher.fetch_multiple_symbols(
            symbols, '2023-01-01', '2023-12-31'
        )
        
        assert len(result) == 3
        assert all(s in result for s in symbols)
        assert all(isinstance(df, pd.DataFrame) for df in result.values())

