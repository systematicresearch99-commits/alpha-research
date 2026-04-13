"""
risk/fixed_stop.py
------------------
Fixed Percentage Stop Loss.

Exits a position if price moves N% against the entry price.
The simplest stop loss — easy to reason about and widely used.

Standalone usage:
    from risk.fixed_stop import apply
    df = apply(df, stop_pct=0.05)

Interface:
    Input  : df with 'position' (0/1/-1) and 'Close' columns
    Output : df with 'sized_position', 'stop_price', 'stopped_out' columns added

Note on stacking:
    When stacked after a sizing module, this reads 'sized_position'
    and zeroes it out on stop triggers. When used standalone it reads
    'position' directly.
"""

import pandas as pd
import numpy as np

MODULE_NAME = "FixedStop"
MODULE_TYPE = "stop"


def apply(df, stop_pct=0.05):
    """
    Fixed Percentage Stop Loss.

    Tracks entry price for each trade. If Close moves more than
    stop_pct against the entry direction, position is forced to 0
    until the strategy re-enters.

    Args:
        df       : DataFrame with 'position' (or 'sized_position') and 'Close'
        stop_pct : maximum adverse move before forced exit (default 0.05 = 5%)

    Returns:
        DataFrame with 'sized_position', 'stop_price', 'stopped_out' added
    """
    df = df.copy()

    # Use sized_position if already set by a sizing module, else use position
    source_col = "sized_position" if "sized_position" in df.columns else "position"

    sized       = df[source_col].copy()
    stop_price  = pd.Series(np.nan, index=df.index)
    stopped_out = pd.Series(0, index=df.index)

    entry_price  = None
    entry_dir    = 0
    is_stopped   = False

    for i in range(len(df)):
        close     = df["Close"].iloc[i]
        raw_sig   = sized.iloc[i]
        direction = 1 if raw_sig > 0 else (-1 if raw_sig < 0 else 0)

        # New entry — reset stop
        if direction != 0 and direction != entry_dir:
            entry_price = close
            entry_dir   = direction
            is_stopped  = False

        # Flat signal — reset
        if direction == 0:
            entry_price = None
            entry_dir   = 0
            is_stopped  = False

        # Check stop condition
        if entry_price and entry_dir != 0 and not is_stopped:
            pct_move = (close - entry_price) / entry_price
            adverse  = pct_move * entry_dir   # negative = moving against us

            if adverse < -stop_pct:
                is_stopped         = True
                stopped_out.iloc[i] = 1
                sized.iloc[i]       = 0
                stop_price.iloc[i]  = close
                entry_price         = None
                entry_dir           = 0
                continue

        if is_stopped:
            sized.iloc[i] = 0

    df["sized_position"] = sized
    df["stop_price"]     = stop_price
    df["stopped_out"]    = stopped_out

    return df


def get_params(stop_pct=0.05):
    return {"stop_pct": stop_pct}
