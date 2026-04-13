"""
risk/signal_strength.py
-----------------------
Signal Strength Position Sizing.

Sizes positions proportionally to how strong the underlying indicator
signal is, rather than treating every signal as equal. A very oversold
RSI gets a larger position than a mildly oversold one.

Works by extracting a normalized strength score (0 to 1) from whichever
indicator columns are present in the DataFrame.

Standalone usage:
    from risk.signal_strength import apply
    df = apply(df, max_position=1.0)

Interface:
    Input  : df with 'position' (0/1/-1) and indicator columns
             (rsi, zscore, macd_hist, pct_k, deviation_pct, roc, momentum_return)
    Output : df with 'sized_position', 'signal_score' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "SignalStrength"
MODULE_TYPE = "sizing"


def _extract_score(df):
    """
    Detect which indicator is present and extract a normalized
    strength score in [0, 1]. Higher = stronger signal.

    Falls back to binary (0 or 1) if no known indicator is found.
    """
    # RSI — strength = distance from 50 midpoint, normalized to [0,1]
    if "rsi" in df.columns:
        rsi = df["rsi"]
        # RSI < 50 → long signal strength, RSI > 50 → short signal strength
        score = (rsi - 50).abs() / 50.0
        return score.clip(0, 1)

    # Z-score — strength = abs(zscore) normalized by typical range (0-3)
    if "zscore" in df.columns:
        score = df["zscore"].abs() / 3.0
        return score.clip(0, 1)

    # MACD histogram — strength = abs(hist) normalized by rolling max
    if "macd_hist" in df.columns:
        hist  = df["macd_hist"].abs()
        roll_max = hist.rolling(60).max().replace(0, np.nan)
        score = (hist / roll_max).fillna(0)
        return score.clip(0, 1)

    # Stochastic %K — distance from 50 midpoint
    if "pct_k" in df.columns:
        score = (df["pct_k"] - 50).abs() / 50.0
        return score.clip(0, 1)

    # VWAP deviation — abs deviation normalized by typical range
    if "deviation_pct" in df.columns:
        score = df["deviation_pct"].abs() / 0.05   # normalize to 5% max deviation
        return score.clip(0, 1)

    # ROC — abs momentum normalized by rolling max
    if "roc" in df.columns:
        roc      = df["roc"].abs()
        roll_max = roc.rolling(60).max().replace(0, np.nan)
        score    = (roc / roll_max).fillna(0)
        return score.clip(0, 1)

    # Dual momentum return
    if "momentum_return" in df.columns:
        mr       = df["momentum_return"].abs()
        roll_max = mr.rolling(60).max().replace(0, np.nan)
        score    = (mr / roll_max).fillna(0)
        return score.clip(0, 1)

    # No known indicator — fall back to binary
    return df["position"].abs().clip(0, 1)


def apply(df, min_score=0.0, max_position=1.0):
    """
    Signal Strength Position Sizing.

    Extracts a normalized strength score from whichever indicator
    columns are present, then uses that to scale position size.

    sized_position = direction * score * max_position

    Args:
        df           : DataFrame with 'position' and indicator columns
        min_score    : minimum score threshold — signals below this are ignored (default 0.0)
        max_position : maximum allowed position size (default 1.0)

    Returns:
        DataFrame with 'signal_score', 'sized_position' added
    """
    df = df.copy()

    score             = _extract_score(df)
    df["signal_score"] = score

    # Zero out weak signals below threshold
    df.loc[df["signal_score"] < min_score, "signal_score"] = 0

    source_col = "sized_position" if "sized_position" in df.columns else "position"
    direction  = df[source_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    df["sized_position"] = (direction * df["signal_score"] * max_position).clip(
        lower=-max_position, upper=max_position
    )

    return df


def get_params(min_score=0.0, max_position=1.0):
    return {"min_score": min_score, "max_position": max_position}
