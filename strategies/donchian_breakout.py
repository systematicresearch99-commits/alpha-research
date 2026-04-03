import pandas as pd

STRATEGY_NAME = "Donchian_Breakout"


def generate_signals(data, window=20):
    """
    Donchian Channel Breakout strategy (Turtle Trading classic).

    Signal logic:
        Long  (position= 1) when Close breaks ABOVE the N-day high
        Short (position=-1) when Close breaks BELOW the N-day low
        Flat  (position= 0) when price is inside the channel

    Channel is computed on prior N bars (shift 1) to avoid lookahead.
    Once in a trade, position is held until the opposite breakout occurs.

    Args:
        data   : DataFrame with 'High', 'Low', 'Close' columns
        window : lookback period for channel (default 20)

    Returns:
        DataFrame with upper_band, lower_band, signal, position columns added
    """
    df = data.copy()

    # Use shift(1) so today's signal uses yesterday's channel
    df["upper_band"] = df["High"].rolling(window).max().shift(1)
    df["lower_band"] = df["Low"].rolling(window).min().shift(1)

    # State machine to hold position between breakouts
    signal   = pd.Series(0, index=df.index)
    position = 0

    for i in range(len(df)):
        close = df["Close"].iloc[i]
        upper = df["upper_band"].iloc[i]
        lower = df["lower_band"].iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            signal.iloc[i] = 0
            continue

        if close > upper:
            position = 1
        elif close < lower:
            position = -1

        signal.iloc[i] = position

    df["signal"] = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(window=20):
    return {"window": window}
