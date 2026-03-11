"""
research/regime_detection/features.py
======================================
Computes the 4 rolling features used to train the HMM regime model.

Features
--------
  volatility         — realized annualized vol (captures regime risk level)
  autocorrelation    — return autocorrelation lag-1 (trend vs mean-reversion)
  index_correlation  — rolling correlation with benchmark (risk-on/off)
  skewness           — return distribution skew (tail risk)

All features are computed on a rolling window so the HMM sees a
smooth, regime-appropriate signal rather than raw daily noise.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew


def compute_features(
    prices: pd.Series,
    index_prices: pd.Series,
    window: int = 20,
    autocorr_lag: int = 1,
) -> pd.DataFrame:
    """
    Build the 4-feature matrix for HMM training/inference.

    Parameters
    ----------
    prices        : pd.Series  — asset close prices
    index_prices  : pd.Series  — benchmark close prices (e.g. SPY)
    window        : int        — rolling window in trading days (default 20)
    autocorr_lag  : int        — lag for autocorrelation (default 1)

    Returns
    -------
    pd.DataFrame with columns:
        volatility, autocorrelation, index_correlation, skewness
    (rows with NaN dropped — first `window` rows will be absent)
    """
    # Accept both Series and DataFrame (framework uses both)
    if isinstance(prices, pd.DataFrame):
        prices = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    if isinstance(index_prices, pd.DataFrame):
        index_prices = index_prices["Close"] if "Close" in index_prices.columns else index_prices.iloc[:, 0]

    returns     = prices.pct_change()
    idx_returns = index_prices.pct_change()

    # Align on shared dates (handles missing index days cleanly)
    returns, idx_returns = returns.align(idx_returns, join="inner")

    features = pd.DataFrame(index=returns.index)

    # 1. Realized Volatility — annualized
    features["volatility"] = (
        returns.rolling(window).std() * np.sqrt(252)
    )

    # 2. Return Autocorrelation — positive=trending, negative=mean-reverting
    features["autocorrelation"] = (
        returns.rolling(window)
               .apply(lambda x: pd.Series(x).autocorr(lag=autocorr_lag), raw=False)
    )

    # 3. Rolling Correlation with Benchmark Index
    features["index_correlation"] = (
        returns.rolling(window).corr(idx_returns)
    )

    # 4. Rolling Skewness — negative = fat left tail / crash risk
    features["skewness"] = (
        returns.rolling(window)
               .apply(lambda x: scipy_skew(x, bias=False), raw=True)
    )

    return features.dropna()


def normalize_features(
    features: pd.DataFrame,
    mean: pd.Series = None,
    std: pd.Series = None,
) -> tuple:
    """
    Standardize features to zero mean / unit variance.

    Pass in training mean/std when normalizing out-of-sample data
    so inference uses the same distribution as training.

    Returns
    -------
    (normalized_df, mean_series, std_series)
    """
    if mean is None:
        mean = features.mean()
    if std is None:
        std  = features.std()

    std = std.replace(0, 1)  # guard against zero-variance features
    normalized = (features - mean) / std
    return normalized, mean, std

