import pandas as pd

STRATEGY_NAME = "EMA_Crossover"


def generate_signals(data, short_window=12, long_window=26):
    """
    Exponential Moving Average crossover strategy.

    Signal logic:
        Long  (position= 1) when short_ema > long_ema
        Short (position=-1) when short_ema < long_ema

    EMA reacts faster to price changes than SMA, reducing lag
    on entries and exits.

    Args:
        data         : DataFrame with 'Close' column
        short_window : fast EMA period (default 12)
        long_window  : slow EMA period (default 26)

    Returns:
        DataFrame with short_ema, long_ema, signal, position columns added
    """
    df = data.copy()

    df["short_ema"] = df["Close"].ewm(span=short_window, adjust=False).mean()
    df["long_ema"]  = df["Close"].ewm(span=long_window,  adjust=False).mean()

    df["signal"] = 0
    df.loc[df["short_ema"] > df["long_ema"], "signal"] =  1
    df.loc[df["short_ema"] < df["long_ema"], "signal"] = -1

    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(short_window=12, long_window=26):
    return {"short_window": short_window, "long_window": long_window}
