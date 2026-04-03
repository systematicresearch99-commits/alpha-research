import pandas as pd

STRATEGY_NAME = "ZScore_Reversion"


def generate_signals(data, window=30, entry_z=2.0, exit_z=0.5):
    """
    Z-Score Mean Reversion strategy.

    Z-score measures how many standard deviations price is from its
    rolling mean. Extreme z-scores signal overextension — we fade them.

    Signal logic:
        Long  (position= 1) when z-score < -entry_z  (price far below mean)
                              exit when z-score > -exit_z
        Short (position=-1) when z-score >  entry_z  (price far above mean)
                              exit when z-score <  exit_z

    Z = (Close - rolling_mean) / rolling_std

    Args:
        data    : DataFrame with 'Close' column
        window  : rolling window for mean/std calculation (default 30)
        entry_z : z-score threshold to enter trade (default 2.0)
        exit_z  : z-score threshold to exit trade (default 0.5)

    Returns:
        DataFrame with zscore, signal, position columns added
    """
    df = data.copy()

    rolling_mean  = df["Close"].rolling(window).mean()
    rolling_std   = df["Close"].rolling(window).std()
    df["zscore"]  = (df["Close"] - rolling_mean) / rolling_std

    # State machine
    signal   = pd.Series(0, index=df.index)
    in_trade = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(df)):
        z = df["zscore"].iloc[i]

        if pd.isna(z):
            signal.iloc[i] = 0
            continue

        if in_trade == 0:
            if z < -entry_z:
                in_trade = 1       # enter long — price is unusually low
            elif z > entry_z:
                in_trade = -1      # enter short — price is unusually high
        elif in_trade == 1:
            if z > -exit_z:
                in_trade = 0       # exit long — reverted to mean
        elif in_trade == -1:
            if z < exit_z:
                in_trade = 0       # exit short — reverted to mean

        signal.iloc[i] = in_trade

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(window=30, entry_z=2.0, exit_z=0.5):
    return {"window": window, "entry_z": entry_z, "exit_z": exit_z}
