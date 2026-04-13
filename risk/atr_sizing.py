"""
risk/atr_sizing.py
------------------
ATR-Based Position Sizing.

Sizes positions so that one ATR move equals a fixed percentage of
capital. Naturally gives larger positions in low-volatility environments
and smaller positions when volatility is high — self-adjusting.

Standalone usage:
    from risk.atr_sizing import apply
    df = apply(df, atr_period=14, atr_risk_pct=0.01)

Interface:
    Input  : df with 'position' (0/1/-1), 'High', 'Low', 'Close' columns
    Output : df with 'sized_position' and 'atr' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "ATR_Sizing"
MODULE_TYPE = "sizing"


def _compute_atr(df, period):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def apply(df, atr_period=14, atr_risk_pct=0.01, max_position=1.0):
    """
    ATR-Based Position Sizing.

    Position size = atr_risk_pct / (ATR / Close)

    Interpretation: if ATR is 2% of price and you want to risk 1%
    of capital per ATR move, position = 1% / 2% = 0.5 (half position).
    High vol → smaller size. Low vol → larger size, capped at max_position.

    Args:
        df           : DataFrame with 'position', 'High', 'Low', 'Close'
        atr_period   : ATR lookback period (default 14)
        atr_risk_pct : capital fraction to risk per 1 ATR move (default 0.01 = 1%)
        max_position : maximum allowed position size (default 1.0)

    Returns:
        DataFrame with 'atr', 'atr_pct', 'sized_position' columns added
    """
    df = df.copy()

    df["atr"]     = _compute_atr(df, atr_period)
    df["atr_pct"] = df["atr"] / df["Close"]   # ATR as % of price

    # Size = desired_risk / atr_pct — bigger ATR = smaller position
    raw_size = atr_risk_pct / df["atr_pct"].replace(0, np.nan)
    raw_size = raw_size.clip(upper=max_position).fillna(0)

    # Apply direction from signal
    df["sized_position"] = (df["position"].abs() * raw_size) * df["position"].apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    df["sized_position"] = df["sized_position"].clip(lower=-max_position, upper=max_position)

    return df


def get_params(atr_period=14, atr_risk_pct=0.01, max_position=1.0):
    return {"atr_period": atr_period, "atr_risk_pct": atr_risk_pct, "max_position": max_position}
