"""
risk/time_stop.py
-----------------
Time Stop.

Exits any position after N bars regardless of P&L. Prevents capital
being tied up in stagnant trades and forces the strategy to re-evaluate.
Works well combined with mean reversion strategies where the thesis
has a natural time horizon.

Standalone usage:
    from risk.time_stop import apply
    df = apply(df, max_hold_days=10)

Interface:
    Input  : df with 'position' (0/1/-1) and 'Close' columns
    Output : df with 'sized_position', 'hold_days', 'time_stopped' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "TimeStop"
MODULE_TYPE = "stop"


def apply(df, max_hold_days=10):
    """
    Time Stop.

    Counts bars since trade entry. Forces position to 0 when
    max_hold_days is reached, regardless of profit or loss.
    Position stays flat until strategy generates a new signal.

    Args:
        df            : DataFrame with 'position' (or 'sized_position') and 'Close'
        max_hold_days : maximum bars to hold a position (default 10)

    Returns:
        DataFrame with 'sized_position', 'hold_days', 'time_stopped' added
    """
    df = df.copy()

    source_col   = "sized_position" if "sized_position" in df.columns else "position"
    sized        = df[source_col].copy()
    hold_days    = pd.Series(0, index=df.index)
    time_stopped = pd.Series(0, index=df.index)

    bars_held   = 0
    entry_dir   = 0
    time_out    = False

    for i in range(len(df)):
        raw_sig   = sized.iloc[i]
        direction = 1 if raw_sig > 0 else (-1 if raw_sig < 0 else 0)

        # New entry or direction flip
        if direction != 0 and direction != entry_dir:
            bars_held = 1
            entry_dir = direction
            time_out  = False

        # Flat signal — reset
        elif direction == 0:
            bars_held = 0
            entry_dir = 0
            time_out  = False

        # Continuing in same direction
        elif direction == entry_dir:
            bars_held += 1

        # Time stop triggered
        if bars_held >= max_hold_days and entry_dir != 0:
            time_out             = True
            time_stopped.iloc[i] = 1
            sized.iloc[i]        = 0
            bars_held            = 0
            entry_dir            = 0

        if time_out:
            sized.iloc[i] = 0

        hold_days.iloc[i] = bars_held

    df["sized_position"] = sized
    df["hold_days"]      = hold_days
    df["time_stopped"]   = time_stopped

    return df


def get_params(max_hold_days=10):
    return {"max_hold_days": max_hold_days}
