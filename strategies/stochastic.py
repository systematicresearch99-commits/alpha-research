import pandas as pd

STRATEGY_NAME = "Stochastic"


def _compute_stochastic(df, k_period=14, d_period=3):
    """
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)  — the signal line
    """
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()

    pct_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    pct_d = pct_k.rolling(d_period).mean()

    return pct_k, pct_d


def generate_signals(data, k_period=14, d_period=3, oversold=20, overbought=80):
    """
    Stochastic Oscillator strategy.

    Signal logic:
        Long  (position= 1) when %K crosses ABOVE %D in oversold zone (%K < oversold)
                              exit when %K crosses ABOVE overbought zone
        Short (position=-1) when %K crosses BELOW %D in overbought zone (%K > overbought)
                              exit when %K crosses BELOW oversold zone

    Uses %K/%D crossover inside extreme zones to avoid whipsaws in the middle.

    Args:
        data       : DataFrame with 'High', 'Low', 'Close' columns
        k_period   : %K lookback period (default 14)
        d_period   : %D smoothing period (default 3)
        oversold   : oversold threshold (default 20)
        overbought : overbought threshold (default 80)

    Returns:
        DataFrame with pct_k, pct_d, signal, position columns added
    """
    df = data.copy()

    df["pct_k"], df["pct_d"] = _compute_stochastic(df, k_period, d_period)

    # State machine
    signal   = pd.Series(0, index=df.index)
    in_trade = 0  # 0=flat, 1=long, -1=short

    for i in range(1, len(df)):
        k      = df["pct_k"].iloc[i]
        d      = df["pct_d"].iloc[i]
        k_prev = df["pct_k"].iloc[i - 1]
        d_prev = df["pct_d"].iloc[i - 1]

        if pd.isna(k) or pd.isna(d):
            signal.iloc[i] = 0
            continue

        k_cross_up   = (k_prev < d_prev) and (k > d)   # %K crossed above %D
        k_cross_down = (k_prev > d_prev) and (k < d)   # %K crossed below %D

        if in_trade == 0:
            if k_cross_up and k < oversold:
                in_trade = 1       # oversold crossover → long
            elif k_cross_down and k > overbought:
                in_trade = -1      # overbought crossunder → short
        elif in_trade == 1:
            if k > overbought:
                in_trade = 0       # exit long at overbought
        elif in_trade == -1:
            if k < oversold:
                in_trade = 0       # exit short at oversold

        signal.iloc[i] = in_trade

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(k_period=14, d_period=3, oversold=20, overbought=80):
    return {
        "k_period":   k_period,
        "d_period":   d_period,
        "oversold":   oversold,
        "overbought": overbought,
    }
