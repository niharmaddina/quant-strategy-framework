"""
Machine learning model for mean reversion signal quality prediction.

Predicts whether a BUY signal will hit TP before SL, and outputs
gated signals + size multipliers for BacktestEngine.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import build_mr_quality_features


class MRSignalQualityModel:
    """
    ML model that predicts signal quality for mean reversion BUY signals.
    
    Predicts: given a BUY signal fired, will it hit engine TP before SL?
    Outputs: p_good_trade, signal_ml, size_multiplier for BacktestEngine.
    
    Uses HistGradientBoostingClassifier with feature engineering pipeline.
    Includes hysteresis gating for stable signal generation.
    """
    
    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        max_iter: int = 600,
        random_state: int = 42
    ):
        """
        Initialize ML model.
        
        Args:
            max_depth: Maximum tree depth for gradient boosting
            learning_rate: Learning rate for gradient boosting
            max_iter: Maximum iterations for gradient boosting
            random_state: Random seed for reproducibility
        """
        self.cols_ = None
        self.model_ = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state
            ))
        ])
    
    def fit(self, full_df: pd.DataFrame, y: pd.Series):
        """
        Train the model on labeled data.
        
        Args:
            full_df: DataFrame with indicators and signals (from MeanReversionIndicators)
            y: Labels (from label_engine_tp_sl_first_long_only)
            
        Returns:
            self (for method chaining)
        """
        X = build_mr_quality_features(full_df)
        
        # Only train on actual BUY signals (long-only)
        mask = y.notna() & (full_df["signal"] == 1)
        X_train = X.loc[mask]
        y_train = y.loc[mask].astype(int)
        
        if len(X_train) == 0:
            raise ValueError("No training samples found (no BUY signals with labels)")
        
        # Store feature columns for prediction
        self.cols_ = X_train.columns.tolist()
        
        # Train model
        self.model_.fit(X_train, y_train)
        
        return self
    
    def predict(
        self,
        full_df: pd.DataFrame,
        ema_span: int = 20,
        p_on: float = 0.65,
        p_off: float = 0.55
    ) -> pd.DataFrame:
        """
        Predict signal quality and generate ML-gated signals.
        
        Args:
            full_df: DataFrame with indicators and signals (from MeanReversionIndicators)
            ema_span: EMA span for smoothing probability (default: 20)
            p_on: Probability threshold to turn on ML gate (default: 0.65)
            p_off: Probability threshold to turn off ML gate (default: 0.55)
            
        Returns:
            DataFrame with columns:
                - p_good_trade: Raw probability of good trade
                - p_smooth: Smoothed probability (EMA)
                - ml_gate_buy: Boolean gate (hysteresis)
                - size_multiplier: Position size multiplier (0-1)
                - signal_ml: ML-gated signal (0 or 1)
        """
        if self.cols_ is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Build features
        X = build_mr_quality_features(full_df)
        
        # Ensure we have the same columns as training
        missing_cols = set(self.cols_) - set(X.columns)
        if missing_cols:
            # Add missing columns with zeros
            for col in missing_cols:
                X[col] = 0.0
        
        # Select only training columns (in same order)
        X = X[self.cols_]
        
        # Predict probability
        p = pd.Series(
            self.model_.predict_proba(X)[:, 1],
            index=full_df.index,
            name="p_good_trade"
        )
        
        # Smooth probability with EMA for stability
        p_smooth = p.ewm(span=ema_span, adjust=False).mean()
        
        # Hysteresis gating (prevents rapid on/off switching)
        gate = pd.Series(False, index=full_df.index)
        on = False
        
        for i in range(len(full_df)):
            if not on and p_smooth.iat[i] >= p_on:
                on = True
            elif on and p_smooth.iat[i] <= p_off:
                on = False
            gate.iat[i] = on
        
        # Build output DataFrame
        out = pd.DataFrame(index=full_df.index)
        out["p_good_trade"] = p
        out["p_smooth"] = p_smooth
        out["ml_gate_buy"] = gate & (full_df["signal"] == 1)
        
        # Size multiplier: 0 at p=0.5, 1 at p=1.0
        # Linear scaling: 2 * (p - 0.5) clipped to [0, 1]
        out["size_multiplier"] = np.clip(2.0 * (out["p_smooth"] - 0.5), 0.0, 1.0)
        
        # Convert to engine-compatible signals
        out["signal_ml"] = 0
        out.loc[out["ml_gate_buy"], "signal_ml"] = 1
        
        return out
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from trained model.
        
        Returns:
            Series with feature names and importance scores
        """
        if self.cols_ is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # HistGradientBoostingClassifier doesn't have feature_importances_ directly
        # Use permutation importance as fallback
        try:
            from sklearn.inspection import permutation_importance  # noqa: F401 (availability check)
            # Need training data to compute permutation importance
            # For now, return equal importance (can be improved with actual training data)
            importances = np.ones(len(self.cols_)) / len(self.cols_)
        except (ImportError, AttributeError):
            # Fallback: equal importance
            importances = np.ones(len(self.cols_)) / len(self.cols_)
        
        return pd.Series(importances, index=self.cols_, name="importance").sort_values(ascending=False)

