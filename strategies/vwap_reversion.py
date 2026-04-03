import pandas as pd
import numpy as np

STRATEGY_NAME = "VWAP_Reversion"


def _compute_rolling_vwap(df, window):
    """
    Rolling VWAP over a N-day window.

    True intraday VWAP resets each session — on daily bars we approximate
    with a rolling window VWAP, which captures the volume-weighted average
    price over the last N days. This is the standard adaptation for EOD data.

    VWAP = sum(Typical_Price * Volume) / sum(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = df["typical_price"] * df["Volume"]

    vwap = tp_vol.rolling(window).sum() / df["Volume"].rolling(window).sum()
    return vwap


def generate_signals(data, window=20, entry_pct=0.02, exit_pct=0.005):
    """
    VWAP Mean Reversion strategy (adapted for daily bars).

    Price deviating significantly from VWAP is unsustainable — large
    participants use VWAP as a benchmark and will fade extreme deviations.

    Signal logic:
        Long  (position= 1) when Close < VWAP * (1 - entry_pct)
                              (price is entry_pct% BELOW VWAP — oversold)
                              exit when Close > VWAP * (1 - exit_pct)
        Short (position=-1) when Close > VWAP * (1 + entry_pct)
                              (price is entry_pct% ABOVE VWAP — overbought)
                              exit when Close < VWAP * (1 + exit_pct)

    Args:
        data      : DataFrame with 'High', 'Low', 'Close', 'Volume' columns
        window    : rolling VWAP window in days (default 20)
        entry_pct : % deviation from VWAP to enter trade (default 0.02 = 2%)
        exit_pct  : % deviation from VWAP to exit trade (default 0.005 = 0.5%)

    Returns:
        DataFrame with vwap, deviation_pct, signal, position columns added
    """
    df = data.copy()

    df["vwap"]          = _compute_rolling_vwap(df, window)
    df["deviation_pct"] = (df["Close"] - df["vwap"]) / df["vwap"]

    # State machine
    signal   = pd.Series(0, index=df.index)
    in_trade = 0  # 0=flat, 1=long, -1=short

    for i in range(len(df)):
        dev  = df["deviation_pct"].iloc[i]
        vwap = df["vwap"].iloc[i]

        if pd.isna(dev) or pd.isna(vwap):
            signal.iloc[i] = 0
            continue

        if in_trade == 0:
            if dev < -entry_pct:
                in_trade = 1       # too far below VWAP → long
            elif dev > entry_pct:
                in_trade = -1      # too far above VWAP → short
        elif in_trade == 1:
            if dev > -exit_pct:
                in_trade = 0       # reverted close enough → exit long
        elif in_trade == -1:
            if dev < exit_pct:
                in_trade = 0       # reverted close enough → exit short

        signal.iloc[i] = in_trade

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    # Clean up intermediate column
    df.drop(columns=["typical_price"], inplace=True, errors="ignore")

    return df


def get_params(window=20, entry_pct=0.02, exit_pct=0.005):
    return {"window": window, "entry_pct": entry_pct, "exit_pct": exit_pct}
