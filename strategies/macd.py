import pandas as pd

STRATEGY_NAME = "MACD"


def generate_signals(data, fast=12, slow=26, signal_period=9):
    """
    MACD (Moving Average Convergence Divergence) strategy.

    Signal logic:
        Long  (position= 1) when MACD line crosses ABOVE signal line
                              (histogram goes positive)
        Short (position=-1) when MACD line crosses BELOW signal line
                              (histogram goes negative)

    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
        data          : DataFrame with 'Close' column
        fast          : fast EMA period (default 12)
        slow          : slow EMA period (default 26)
        signal_period : signal line EMA period (default 9)

    Returns:
        DataFrame with macd, macd_signal, macd_hist, signal, position columns added
    """
    df = data.copy()

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    df["signal"] = 0
    df.loc[df["macd_hist"] > 0, "signal"] =  1
    df.loc[df["macd_hist"] < 0, "signal"] = -1

    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(fast=12, slow=26, signal_period=9):
    return {"fast": fast, "slow": slow, "signal_period": signal_period}
