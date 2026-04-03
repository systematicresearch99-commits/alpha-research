import pandas as pd
import numpy as np

STRATEGY_NAME = "Keltner_Breakout"


def _compute_atr(df, period=14):
    """ATR using Wilder's EMA smoothing."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def generate_signals(data, ema_period=20, atr_period=10, multiplier=2.0):
    """
    Keltner Channel Breakout strategy.

    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    This makes the channel smoother and less prone to sudden width spikes
    during gap moves.

    Channel:
        Middle = EMA(Close, ema_period)
        Upper  = Middle + multiplier * ATR(atr_period)
        Lower  = Middle - multiplier * ATR(atr_period)

    Signal logic:
        Long  (position= 1) when Close breaks ABOVE upper channel
        Short (position=-1) when Close breaks BELOW lower channel
        Exit to flat when Close crosses back through middle (EMA)

    Args:
        data       : DataFrame with 'High', 'Low', 'Close' columns
        ema_period : EMA period for the middle line (default 20)
        atr_period : ATR period for channel width (default 10)
        multiplier : ATR multiplier for channel width (default 2.0)

    Returns:
        DataFrame with middle, upper_band, lower_band, signal, position columns added
    """
    df = data.copy()

    df["middle"]     = df["Close"].ewm(span=ema_period, adjust=False).mean()
    atr              = _compute_atr(df, atr_period)
    df["upper_band"] = df["middle"] + multiplier * atr
    df["lower_band"] = df["middle"] - multiplier * atr

    # State machine
    signal   = pd.Series(0, index=df.index)
    in_trade = 0  # 0=flat, 1=long, -1=short

    for i in range(len(df)):
        close  = df["Close"].iloc[i]
        upper  = df["upper_band"].iloc[i]
        lower  = df["lower_band"].iloc[i]
        middle = df["middle"].iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            signal.iloc[i] = 0
            continue

        if in_trade == 0:
            if close > upper:
                in_trade = 1       # breakout above → long
            elif close < lower:
                in_trade = -1      # breakdown below → short
        elif in_trade == 1:
            if close < middle:
                in_trade = 0       # close back below EMA → exit long
        elif in_trade == -1:
            if close > middle:
                in_trade = 0       # close back above EMA → exit short

        signal.iloc[i] = in_trade

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(ema_period=20, atr_period=10, multiplier=2.0):
    return {"ema_period": ema_period, "atr_period": atr_period, "multiplier": multiplier}
