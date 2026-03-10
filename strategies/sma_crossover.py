import pandas as pd

STRATEGY_NAME = "SMA_Crossover"

def generate_signals(data, short_window=20, long_window=50):
    """
    Simple Moving Average crossover strategy.

    Signal logic:
        long  (position=1) when short_ma > long_ma
        flat  (position=0) when short_ma <= long_ma

    Position is shifted by 1 bar to prevent lookahead bias
    (signal on close of day N → trade executes open of day N+1).

    Args:
        data         : DataFrame with 'Close' column
        short_window : fast MA period (default 20)
        long_window  : slow MA period (default 50)

    Returns:
        DataFrame with signal and position columns added
    """
    df = data.copy()

    df["short_ma"] = df["Close"].rolling(short_window).mean()
    df["long_ma"]  = df["Close"].rolling(long_window).mean()

    df["signal"]   = 0
    df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1

    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1)

    return df


def get_params(short_window=20, long_window=50):
    return {"short_window": short_window, "long_window": long_window}