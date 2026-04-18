"""
features/feature_engine.py
--------------------------
Feature extraction layer for the alpha model pipeline.

This module ONLY extracts continuous numerical signals from price data.
It does NOT generate buy/sell decisions, does NOT output positions,
and does NOT contain trading rules or thresholds.

Pipeline position:
    raw data → feature_engine → alpha_model → position_bridge → backtest

Signals produced:
    returns      : log returns
    rsi_signal   : normalized RSI in [-1, 1]
    momentum     : rolling mean of returns
    volatility   : rolling std of returns (annualized)
    zscore       : price z-score vs rolling mean
    volume_signal: normalized volume deviation (if Volume available)
    trend        : EMA ratio — where price sits relative to trend
    dispersion   : high-low range normalized by close (intraday range signal)

Key design rule:
    Every output is a continuous float.
    No if/else rules. No position logic. No thresholds.
    "Describe the market" — not "trade the market."
"""

import pandas as pd
import numpy as np

# ── RSI helper ─────────────────────────────────────────────────────────────────

def _compute_rsi(series, period=14):
    """
    Wilder's RSI computed from scratch.
    Returns raw RSI values in [0, 100].
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Individual feature functions ───────────────────────────────────────────────

def compute_returns(df, log=True):
    """
    Price returns.

    Args:
        df  : DataFrame with 'Close' column
        log : if True use log returns (default), else pct_change

    Returns:
        Series — continuous, unbounded
    """
    if log:
        return np.log(df["Close"] / df["Close"].shift(1))
    return df["Close"].pct_change()


def compute_rsi_signal(df, period=14):
    """
    Normalized RSI signal.

    Transforms RSI [0, 100] → rsi_signal [-1, 1] by centering on 50:
        rsi_signal = (RSI - 50) / 50

    Interpretation:
        -1.0 = maximally oversold
         0.0 = neutral
        +1.0 = maximally overbought

    This preserves the full gradient — RSI=29 and RSI=31 remain
    distinct values (-0.42 vs -0.38) rather than collapsing to
    the same buy signal.

    Args:
        df     : DataFrame with 'Close' column
        period : RSI lookback period (default 14)

    Returns:
        Series in approximately [-1, 1]
    """
    rsi = _compute_rsi(df["Close"], period)
    return (rsi - 50) / 50


def compute_momentum(df, window=10):
    """
    Price momentum — rolling mean of returns.

    Captures the average direction of recent price movement.
    Positive = trending up, negative = trending down.
    Magnitude reflects strength of trend.

    Args:
        df     : DataFrame with 'Close' column
        window : rolling window for mean (default 10)

    Returns:
        Series — continuous, centered near 0
    """
    returns = np.log(df["Close"] / df["Close"].shift(1))
    return returns.rolling(window).mean()


def compute_volatility(df, window=20, annualize=True):
    """
    Realized volatility — rolling std of returns.

    Measures market uncertainty. High volatility = unstable,
    low volatility = calm. Not a directional signal — always positive.

    Args:
        df        : DataFrame with 'Close' column
        window    : rolling window for std (default 20)
        annualize : if True, annualize by sqrt(252) (default True)

    Returns:
        Series — positive continuous values
    """
    returns = np.log(df["Close"] / df["Close"].shift(1))
    vol     = returns.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol


def compute_zscore(df, window=20):
    """
    Price z-score relative to rolling mean.

    Measures how far price has deviated from its recent average,
    in units of standard deviation. Mean-reversion signal.

    z = (Close - rolling_mean) / rolling_std

    Interpretation:
        +2.0 = price is 2 std above recent mean (stretched high)
        -2.0 = price is 2 std below recent mean (stretched low)

    Args:
        df     : DataFrame with 'Close' column
        window : rolling window for mean and std (default 20)

    Returns:
        Series — continuous, approximately N(0,1) distributed
    """
    mean = df["Close"].rolling(window).mean()
    std  = df["Close"].rolling(window).std().replace(0, np.nan)
    return (df["Close"] - mean) / std


def compute_volume_signal(df, window=20):
    """
    Normalized volume deviation.

    Measures whether current volume is above or below its recent average.
    High relative volume = increased market participation / conviction.

    volume_signal = (Volume - rolling_mean) / rolling_std

    Returns NaN if 'Volume' column is not present.

    Args:
        df     : DataFrame with 'Volume' column (optional)
        window : rolling window (default 20)

    Returns:
        Series — continuous z-score of volume, or NaN series if no Volume
    """
    if "Volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    vol  = df["Volume"].astype(float)
    mean = vol.rolling(window).mean()
    std  = vol.rolling(window).std().replace(0, np.nan)
    return (vol - mean) / std


def compute_trend(df, fast=10, slow=50):
    """
    EMA trend signal — ratio of fast EMA to slow EMA, normalized.

    Captures where price sits relative to its trend structure.
    Positive = fast EMA above slow EMA (uptrend)
    Negative = fast EMA below slow EMA (downtrend)

    trend = (EMA_fast - EMA_slow) / EMA_slow

    Args:
        df   : DataFrame with 'Close' column
        fast : fast EMA period (default 10)
        slow : slow EMA period (default 50)

    Returns:
        Series — continuous, centered near 0
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow) / ema_slow


def compute_dispersion(df):
    """
    Intraday price range normalized by close.

    Measures how much price moved within the bar relative to its level.
    High dispersion = volatile/uncertain bar.
    Low dispersion = tight, directional bar.

    dispersion = (High - Low) / Close

    Returns NaN if High/Low columns are not present.

    Args:
        df : DataFrame with 'High', 'Low', 'Close' columns (optional)

    Returns:
        Series — positive continuous values, or NaN series if no H/L
    """
    if "High" not in df.columns or "Low" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return (df["High"] - df["Low"]) / df["Close"]


# ── Main entry point ───────────────────────────────────────────────────────────

def extract_features(
    df,
    rsi_period     = 14,
    momentum_window = 10,
    vol_window      = 20,
    zscore_window   = 20,
    volume_window   = 20,
    trend_fast      = 10,
    trend_slow      = 50,
    log_returns     = True,
):
    """
    Extract all features from a price DataFrame.

    Adds continuous signal columns to the DataFrame without modifying
    existing columns. All outputs are floating-point numbers — no
    discrete signals, no positions, no trading rules.

    Args:
        df              : DataFrame with at least 'Close' column.
                          'High', 'Low', 'Volume' used if available.
        rsi_period      : RSI lookback (default 14)
        momentum_window : momentum rolling window (default 10)
        vol_window      : volatility rolling window (default 20)
        zscore_window   : z-score rolling window (default 20)
        volume_window   : volume signal rolling window (default 20)
        trend_fast      : trend signal fast EMA (default 10)
        trend_slow      : trend signal slow EMA (default 50)
        log_returns     : use log returns if True (default True)

    Returns:
        DataFrame with added feature columns:
            returns, rsi_signal, momentum, volatility,
            zscore, volume_signal, trend, dispersion

    Example output (not a position — just information):
        returns       = -0.023   (down 2.3% today)
        rsi_signal    = -0.38    (mildly oversold)
        momentum      = -0.008   (weak negative momentum)
        volatility    =  0.62    (62% annualized vol)
        zscore        = -1.4     (1.4 std below rolling mean)
        volume_signal =  1.2     (volume 1.2 std above average)
        trend         = -0.031   (fast EMA 3.1% below slow EMA)
        dispersion    =  0.024   (2.4% intraday range)
    """
    out = df.copy()

    out["returns"]       = compute_returns(out,        log=log_returns)
    out["rsi_signal"]    = compute_rsi_signal(out,     period=rsi_period)
    out["momentum"]      = compute_momentum(out,       window=momentum_window)
    out["volatility"]    = compute_volatility(out,     window=vol_window)
    out["zscore"]        = compute_zscore(out,         window=zscore_window)
    out["volume_signal"] = compute_volume_signal(out,  window=volume_window)
    out["trend"]         = compute_trend(out,          fast=trend_fast, slow=trend_slow)
    out["dispersion"]    = compute_dispersion(out)

    return out


# ── Feature column registry ────────────────────────────────────────────────────

FEATURE_COLS = [
    "returns",
    "rsi_signal",
    "momentum",
    "volatility",
    "zscore",
    "volume_signal",
    "trend",
    "dispersion",
]

CORE_FEATURE_COLS = [
    "returns",
    "rsi_signal",
    "momentum",
    "volatility",
    "zscore",
]


def get_feature_cols(df, include_optional=True):
    """
    Return list of feature columns present in df.
    Skips volume_signal and dispersion if they are all NaN
    (i.e. Volume/High/Low were not in the source data).
    """
    cols = FEATURE_COLS if include_optional else CORE_FEATURE_COLS
    return [c for c in cols if c in df.columns and not df[c].isna().all()]


