import pandas as pd
from src.indicators import sma, rsi


def test_sma_basic():
    data = pd.Series([1, 2, 3, 4, 5])
    result = sma(data, window=3)
    # last value should be mean of 3,4,5 = 4
    assert result.iloc[-1] == 4


def test_rsi_range():
    data = pd.Series(range(1, 50))
    r = rsi(data, window=14)
    # RSI should be between 0 and 100 (ignoring NaNs)
    r_non_nan = r.dropna()
    assert (r_non_nan >= 0).all()
    assert (r_non_nan <= 100).all()
