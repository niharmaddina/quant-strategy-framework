from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from .utils import RAW_DATA_DIR, get_logger

logger = get_logger(__name__)


@dataclass
class DataLoaderConfig:
    ticker: str
    start: str
    end: str
    timeframe: str = "1d"  # you can extend later


class DataLoader:
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.cache_path = (
            RAW_DATA_DIR
            / f"{config.ticker}_{config.start}_{config.end}_{config.timeframe}.csv"
        )

    def _download_from_yahoo(self) -> pd.DataFrame:
        logger.info(
            f"Downloading {self.config.ticker} from {self.config.start} "
            f"to {self.config.end} ({self.config.timeframe})"
        )
        df = yf.download(
            self.config.ticker,
            start=self.config.start,
            end=self.config.end,
            interval=self.config.timeframe,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise ValueError("Downloaded data is empty. Check ticker or date range.")
        df = df.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Adj Close",
                "Volume": "Volume",
            }
        )
        df.index.name = "Date"
        return df

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        if self.cache_path.exists():
            logger.info(f"Loading cached data from {self.cache_path}")
            df = pd.read_csv(self.cache_path, parse_dates=["Date"], index_col="Date")
            return df
        return None

    def _save_to_cache(self, df: pd.DataFrame):
        logger.info(f"Caching data to {self.cache_path}")
        df.to_csv(self.cache_path)

    def get_price_data(self, force_download: bool = False) -> pd.DataFrame:
        if not force_download:
            cached = self._load_from_cache()
            if cached is not None:
                return cached

        df = self._download_from_yahoo()
        self._save_to_cache(df)
        return df
