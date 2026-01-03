"""
Machine learning module for signal quality prediction.

This module provides:
- Labeling utilities for training data (TP/SL matching BacktestEngine logic)
- Feature engineering for mean reversion signals
- ML model for predicting signal quality
- Preprocessing utilities for intraday data
"""

from ml.labeling import label_engine_tp_sl_first_long_only
from ml.features import build_mr_quality_features
from ml.model import MRSignalQualityModel
from ml.preprocessing import prep_ohlcv, resample_ohlcv

__all__ = [
    'label_engine_tp_sl_first_long_only',
    'build_mr_quality_features',
    'MRSignalQualityModel',
    'prep_ohlcv',
    'resample_ohlcv',
]

