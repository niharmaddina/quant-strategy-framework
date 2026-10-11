"""
Feature engineering for mean reversion signal quality prediction.

Builds features from MeanReversionIndicators outputs and additional
regime/chaos features for ML model training.
"""

import pandas as pd


def build_mr_quality_features(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature set for mean reversion signal quality prediction.
    
    Uses MeanReversionIndicators outputs plus regime/chaos hooks.
    Designed to work with get_full_analysis() output from MeanReversionIndicators.
    
    Args:
        full_df: DataFrame with indicators and signals (from MeanReversionIndicators)
        
    Returns:
        DataFrame with feature columns for ML model
    """
    X = pd.DataFrame(index=full_df.index)
    
    # Core mean reversion indicators (best predictors)
    mr_cols = [
        "z_score", "z_score_normalized", "z_score_percentile",
        "bb_width", "bb_position",
        "distance_from_mean_pct", "abs_distance_from_mean_pct",
        "distance_from_median_pct",
        "mean_reversion_strength",
        "volatility_ratio", "high_volatility", "low_volatility", "normal_volatility",
        "volume_ratio", "volume_confirmation",
        "confidence",
        "risk_reward_ratio",
    ]
    
    for c in mr_cols:
        if c in full_df.columns:
            X[c] = full_df[c]
    
    # Deltas = regime transition signals
    # First differences (1-bar changes)
    delta_cols = [
        "z_score", "bb_width", "volatility_ratio",
        "mean_reversion_strength", "distance_from_mean_pct"
    ]
    
    for c in delta_cols:
        if c in X.columns:
            X[f"{c}_d1"] = X[c].diff()
            X[f"{c}_d3"] = X[c].diff(3)
    
    # Basic return/volatility context
    if 'close' in full_df.columns:
        r1 = full_df["close"].pct_change()
        X["ret_1"] = r1
        X["rv_20"] = r1.rolling(20).std()
        X["vol_of_vol_60"] = X["rv_20"].rolling(60).std()
    
    # Optional: chaos/regime features from higher timeframes
    # These can be merged down from 15m/1h analysis
    chaos_cols = [
        "hurst_15m", "perm_entropy_15m",
        "hurst_1h", "perm_entropy_1h",
        "rqa_det_15m", "rqa_lam_15m", "rqa_tt_15m",
    ]
    
    for c in chaos_cols:
        if c in full_df.columns:
            X[c] = full_df[c]
    
    # Fill NaN values with 0 for delta features (first rows have no history)
    X = X.fillna(0.0)
    
    return X

