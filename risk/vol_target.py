"""
risk/vol_target.py
------------------
Volatility Targeting.

Dynamically scales position size so the strategy always targets
the same annualized realized volatility. In high-vol periods the
position shrinks; in low-vol periods it grows (capped at max_position).

Used by many systematic funds including AQR and Winton to make
returns more consistent across market regimes.

Standalone usage:
    from risk.vol_target import apply
    df = apply(df, target_vol=0.10, vol_window=20)

Interface:
    Input  : df with 'position' (0/1/-1) and 'Close' columns
    Output : df with 'sized_position', 'realized_vol', 'vol_scalar' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "VolatilityTarget"
MODULE_TYPE = "sizing"


def apply(df, target_vol=0.10, vol_window=20, max_position=1.0):
    """
    Volatility Targeting.

    vol_scalar = target_vol / realized_vol
    sized_position = position * vol_scalar  (capped at max_position)

    Realized vol is computed as annualized rolling std of daily returns.
    The scalar is computed on the prior window so there's no lookahead.

    Args:
        df           : DataFrame with 'position' and 'Close' columns
        target_vol   : target annualized volatility (default 0.10 = 10%)
        vol_window   : rolling window for realized vol (default 20 days)
        max_position : maximum allowed position size (default 1.0)

    Returns:
        DataFrame with 'realized_vol', 'vol_scalar', 'sized_position' added
    """
    df = df.copy()

    daily_returns      = df["Close"].pct_change()
    df["realized_vol"] = daily_returns.rolling(vol_window).std() * np.sqrt(252)

    # Shift 1 — use yesterday's vol to size today's position
    vol_lagged        = df["realized_vol"].shift(1)
    df["vol_scalar"]  = (target_vol / vol_lagged).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["vol_scalar"]  = df["vol_scalar"].clip(upper=max_position)

    source_col = "sized_position" if "sized_position" in df.columns else "position"
    direction  = df[source_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    df["sized_position"] = (direction * df["vol_scalar"]).clip(
        lower=-max_position, upper=max_position
    )

    return df


def get_params(target_vol=0.10, vol_window=20, max_position=1.0):
    return {"target_vol": target_vol, "vol_window": vol_window, "max_position": max_position}
