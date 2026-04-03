import pandas as pd
import numpy as np

STRATEGY_NAME = "ATR_Breakout"


def _compute_atr(df, period=14):
    """
    Average True Range (Wilder's smoothing).
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def generate_signals(data, atr_period=14, multiplier=2.0):
    """
    ATR Breakout strategy.

    Measures volatility via ATR, then signals when price moves
    more than N * ATR from the prior close — indicating a genuine
    volatility expansion breakout.

    Signal logic:
        Long  (position= 1) when Close > PrevClose + multiplier * ATR
        Short (position=-1) when Close < PrevClose - multiplier * ATR
        Flat  (position= 0) otherwise (inside the ATR band)

    Position is held until the opposite signal fires.

    Args:
        data       : DataFrame with 'High', 'Low', 'Close' columns
        atr_period : ATR smoothing period (default 14)
        multiplier : ATR multiplier for breakout threshold (default 2.0)

    Returns:
        DataFrame with atr, upper_break, lower_break, signal, position columns added
    """
    df = data.copy()

    df["atr"]         = _compute_atr(df, atr_period)
    df["upper_break"] = df["Close"].shift(1) + multiplier * df["atr"]
    df["lower_break"] = df["Close"].shift(1) - multiplier * df["atr"]

    # State machine — hold position between signals
    signal   = pd.Series(0, index=df.index)
    position = 0

    for i in range(len(df)):
        close = df["Close"].iloc[i]
        upper = df["upper_break"].iloc[i]
        lower = df["lower_break"].iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            signal.iloc[i] = 0
            continue

        if close > upper:
            position = 1
        elif close < lower:
            position = -1

        signal.iloc[i] = position

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(atr_period=14, multiplier=2.0):
    return {"atr_period": atr_period, "multiplier": multiplier}
