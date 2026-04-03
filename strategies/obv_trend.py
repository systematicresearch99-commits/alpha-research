import pandas as pd
import numpy as np

STRATEGY_NAME = "OBV_Trend"


def _compute_obv(df):
    """
    On-Balance Volume.
    OBV adds volume on up days, subtracts on down days.
    Cumulative running total — direction matters, not absolute value.
    """
    direction = np.sign(df["Close"].diff())
    direction.iloc[0] = 0
    obv = (direction * df["Volume"]).cumsum()
    return obv


def generate_signals(data, obv_ma_period=20):
    """
    OBV (On-Balance Volume) Trend strategy.

    OBV confirms price trends with volume. When OBV is rising faster
    than its moving average, volume is flowing into the asset → bullish.
    When OBV falls below its MA → bearish volume divergence.

    Signal logic:
        Long  (position= 1) when OBV > OBV_MA  (volume confirming uptrend)
        Short (position=-1) when OBV < OBV_MA  (volume confirming downtrend)

    Args:
        data          : DataFrame with 'Close', 'Volume' columns
        obv_ma_period : MA period applied to OBV for smoothing (default 20)

    Returns:
        DataFrame with obv, obv_ma, signal, position columns added
    """
    df = data.copy()

    df["obv"]    = _compute_obv(df)
    df["obv_ma"] = df["obv"].rolling(obv_ma_period).mean()

    df["signal"] = 0
    df.loc[df["obv"] > df["obv_ma"], "signal"] =  1
    df.loc[df["obv"] < df["obv_ma"], "signal"] = -1

    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(obv_ma_period=20):
    return {"obv_ma_period": obv_ma_period}
