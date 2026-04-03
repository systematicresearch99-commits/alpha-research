import pandas as pd

STRATEGY_NAME = "Bollinger_Reversion"


def generate_signals(data, window=20, num_std=2.0):
    """
    Bollinger Band Mean Reversion strategy.

    Signal logic:
        Long  (position= 1) when Close crosses BELOW lower band (oversold)
                              exit long when Close crosses ABOVE middle band
        Short (position=-1) when Close crosses ABOVE upper band (overbought)
                              exit short when Close crosses BELOW middle band

    Bands:
        Middle = SMA(window)
        Upper  = Middle + num_std * rolling_std
        Lower  = Middle - num_std * rolling_std

    Args:
        data    : DataFrame with 'Close' column
        window  : rolling window for mean and std (default 20)
        num_std : number of standard deviations for bands (default 2.0)

    Returns:
        DataFrame with middle_band, upper_band, lower_band, signal, position columns added
    """
    df = data.copy()

    df["middle_band"] = df["Close"].rolling(window).mean()
    rolling_std       = df["Close"].rolling(window).std()
    df["upper_band"]  = df["middle_band"] + num_std * rolling_std
    df["lower_band"]  = df["middle_band"] - num_std * rolling_std

    # State machine: track current position
    signal   = pd.Series(0, index=df.index)
    in_trade = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(df)):
        close  = df["Close"].iloc[i]
        upper  = df["upper_band"].iloc[i]
        lower  = df["lower_band"].iloc[i]
        middle = df["middle_band"].iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            signal.iloc[i] = 0
            continue

        if in_trade == 0:
            if close < lower:
                in_trade = 1       # enter long
            elif close > upper:
                in_trade = -1      # enter short
        elif in_trade == 1:
            if close > middle:
                in_trade = 0       # exit long
        elif in_trade == -1:
            if close < middle:
                in_trade = 0       # exit short

        signal.iloc[i] = in_trade

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(window=20, num_std=2.0):
    return {"window": window, "num_std": num_std}
