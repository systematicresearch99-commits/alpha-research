"""
risk/atr_trailing_stop.py
-------------------------
ATR Trailing Stop Loss.

Trails a stop behind the price by N * ATR. As price moves in your
favour the stop ratchets up (for longs) or down (for shorts) but
never moves against you. Lets winners run while cutting losers.

Standalone usage:
    from risk.atr_trailing_stop import apply
    df = apply(df, atr_period=14, atr_multiplier=2.0)

Interface:
    Input  : df with 'position' (0/1/-1), 'High', 'Low', 'Close' columns
    Output : df with 'sized_position', 'trail_stop', 'stopped_out' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "ATR_TrailingStop"
MODULE_TYPE = "stop"


def _compute_atr(df, period):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def apply(df, atr_period=14, atr_multiplier=2.0):
    """
    ATR Trailing Stop Loss.

    For longs  : trail_stop = max(trail_stop, Close - multiplier * ATR)
                 stop triggers when Close < trail_stop
    For shorts : trail_stop = min(trail_stop, Close + multiplier * ATR)
                 stop triggers when Close > trail_stop

    Args:
        df             : DataFrame with 'position', 'High', 'Low', 'Close'
        atr_period     : ATR lookback period (default 14)
        atr_multiplier : ATR multiplier for stop distance (default 2.0)

    Returns:
        DataFrame with 'atr', 'trail_stop', 'stopped_out', 'sized_position' added
    """
    df = df.copy()

    df["atr"] = _compute_atr(df, atr_period)

    source_col  = "sized_position" if "sized_position" in df.columns else "position"
    sized       = df[source_col].copy()
    trail_stop  = pd.Series(np.nan, index=df.index)
    stopped_out = pd.Series(0, index=df.index)

    current_stop = None
    entry_dir    = 0
    is_stopped   = False

    for i in range(len(df)):
        close     = df["Close"].iloc[i]
        atr       = df["atr"].iloc[i]
        raw_sig   = sized.iloc[i]
        direction = 1 if raw_sig > 0 else (-1 if raw_sig < 0 else 0)

        if pd.isna(atr):
            continue

        # New entry
        if direction != 0 and direction != entry_dir:
            entry_dir    = direction
            is_stopped   = False
            current_stop = (close - atr_multiplier * atr) if direction == 1 \
                           else (close + atr_multiplier * atr)

        # Flat — reset
        if direction == 0:
            current_stop = None
            entry_dir    = 0
            is_stopped   = False

        # Update trailing stop — only ratchet in favourable direction
        if current_stop is not None and entry_dir != 0 and not is_stopped:
            if entry_dir == 1:
                new_stop     = close - atr_multiplier * atr
                current_stop = max(current_stop, new_stop)   # only move up
                if close < current_stop:
                    is_stopped          = True
                    stopped_out.iloc[i] = 1
                    sized.iloc[i]       = 0
                    current_stop        = None
                    entry_dir           = 0
            elif entry_dir == -1:
                new_stop     = close + atr_multiplier * atr
                current_stop = min(current_stop, new_stop)   # only move down
                if close > current_stop:
                    is_stopped          = True
                    stopped_out.iloc[i] = 1
                    sized.iloc[i]       = 0
                    current_stop        = None
                    entry_dir           = 0

        if is_stopped:
            sized.iloc[i] = 0

        trail_stop.iloc[i] = current_stop if current_stop else np.nan

    df["sized_position"] = sized
    df["trail_stop"]     = trail_stop
    df["stopped_out"]    = stopped_out

    return df


def get_params(atr_period=14, atr_multiplier=2.0):
    return {"atr_period": atr_period, "atr_multiplier": atr_multiplier}
