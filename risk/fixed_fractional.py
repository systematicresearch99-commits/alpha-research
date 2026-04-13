"""
risk/fixed_fractional.py
------------------------
Fixed Fractional Position Sizing.

Sizes each trade so that a fixed percentage of current capital
is risked per trade. The simplest and most widely used sizing method.

Standalone usage:
    from risk.fixed_fractional import apply
    df = apply(df, risk_pct=0.01)

Interface:
    Input  : df with 'position' (0/1/-1) and 'Close' columns
    Output : df with 'sized_position' column added (0.0 → 1.0)
"""

import pandas as pd
import numpy as np

MODULE_NAME  = "FixedFractional"
MODULE_TYPE  = "sizing"      # "sizing" | "stop" — used by stacking logic


def apply(df, risk_pct=0.01, max_position=1.0):
    """
    Fixed Fractional Position Sizing.

    Scales the binary signal (0/1/-1) by risk_pct as a fraction of
    full capital allocation. Every trade risks the same fixed percentage.

    Args:
        df           : DataFrame with 'position' column (0/1/-1)
        risk_pct     : fraction of capital to risk per trade (default 0.01 = 1%)
        max_position : maximum allowed position size as fraction (default 1.0)

    Returns:
        DataFrame with 'sized_position' column added
    """
    df = df.copy()

    sized = df["position"] * risk_pct
    df["sized_position"] = sized.clip(lower=-max_position, upper=max_position)

    return df


def get_params(risk_pct=0.01, max_position=1.0):
    return {"risk_pct": risk_pct, "max_position": max_position}
