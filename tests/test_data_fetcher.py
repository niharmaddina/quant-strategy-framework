"""
Fast unit tests for DataFetcher with mock data support.
"""

import pytest
import pandas as pd

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


class TestDataFetcherBatching:
    """Batching behavior tests (still fast; uses mock data)."""

    @pytest.mark.fast
    def test_fetch_multiple_symbols_batching(self):
        fetcher = DataFetcher(use_mock=True, batch_delay=0.0)

        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
        data_dict = fetcher.fetch_multiple_symbols(
            symbols,
            start_date="2024-01-01",
            end_date="2024-12-31",
            interval="1d",
            batch_size=2
        )

        assert isinstance(data_dict, dict)
        assert len(data_dict) == len(symbols)
        for symbol in symbols:
            assert symbol in data_dict
            assert isinstance(data_dict[symbol], pd.DataFrame)

    @pytest.mark.fast
    def test_fetch_batch_method(self):
        fetcher = DataFetcher(use_mock=True, batch_delay=0.0)

        requests = [
            ("SPY", "2024-01-01", "2024-12-31", "1d"),
            ("QQQ", "2024-01-01", "2024-12-31", "1d"),
            ("AAPL", "2024-01-01", "2024-12-31", "1d"),
        ]

        results = fetcher.fetch_batch(requests, batch_size=2)

        assert isinstance(results, dict)
        assert len(results) == len(requests)

        # Keys are formatted: "{symbol}_{start}_{end}_{interval}"
        for key in results.keys():
            parts = key.split("_")
            assert len(parts) >= 4

        for _, df in results.items():
            assert isinstance(df, pd.DataFrame)

    @pytest.mark.fast
    def test_batch_delay_configuration(self):
        fetcher_fast = DataFetcher(use_mock=True, batch_delay=0.1)
        fetcher_slow = DataFetcher(use_mock=True, batch_delay=1.0)
        assert fetcher_fast.batch_delay == 0.1
        assert fetcher_slow.batch_delay == 1.0

    @pytest.mark.fast
    def test_rate_limit_delay_configuration(self):
        fetcher = DataFetcher(use_mock=True, rate_limit_delay=10.0)
        assert fetcher.rate_limit_delay == 10.0

    @pytest.mark.fast
    def test_fetch_batch_empty_requests(self):
        fetcher = DataFetcher(use_mock=True)
        results = fetcher.fetch_batch([], batch_size=5)
        assert isinstance(results, dict)
        assert len(results) == 0

    @pytest.mark.fast
    def test_fetch_batch_single_request(self):
        fetcher = DataFetcher(use_mock=True)
        requests = [("SPY", "2024-01-01", "2024-12-31", "1d")]
        results = fetcher.fetch_batch(requests, batch_size=5)
        assert len(results) == 1
        assert "SPY_2024-01-01_2024-12-31_1d" in results
        assert isinstance(results["SPY_2024-01-01_2024-12-31_1d"], pd.DataFrame)

