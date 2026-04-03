import pandas as pd

STRATEGY_NAME = "ROC_Momentum"


def generate_signals(data, window=20, threshold=0.0):
    """
    Rate of Change (ROC) Momentum strategy.

    ROC measures the percentage change in price over N periods.
    Positive ROC = upward momentum → long.
    Negative ROC = downward momentum → short.

    ROC = (Close - Close[N periods ago]) / Close[N periods ago] * 100

    Signal logic:
        Long  (position= 1) when ROC >  threshold
        Short (position=-1) when ROC < -threshold
        Flat  (position= 0) when ROC is within threshold band

    Args:
        data      : DataFrame with 'Close' column
        window    : lookback period for ROC (default 20)
        threshold : minimum ROC magnitude to trigger signal (default 0.0)
                    e.g. threshold=2.0 means only trade when |ROC| > 2%

    Returns:
        DataFrame with roc, signal, position columns added
    """
    df = data.copy()

    df["roc"] = df["Close"].pct_change(periods=window) * 100

    df["signal"] = 0
    df.loc[df["roc"] >  threshold, "signal"] =  1
    df.loc[df["roc"] < -threshold, "signal"] = -1

    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(window=20, threshold=0.0):
    return {"window": window, "threshold": threshold}
