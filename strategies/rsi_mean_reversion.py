import pandas as pd
import numpy as np

STRATEGY_NAME = "RSI_MeanReversion"


def _compute_rsi(series, period=14):
    """Wilder's RSI — computed from scratch so you know the math."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # Wilder smoothing (exponential with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signals(data, rsi_period=14, oversold=30, overbought=55):
    """
    RSI Mean Reversion strategy.

    Signal logic:
        Enter long  (signal=1) when RSI crosses BELOW oversold threshold
        Exit long   (signal=0) when RSI crosses ABOVE overbought threshold
        Never short — flat (0) when not in a trade

    This captures the "snap back" after extreme selling.

    Args:
        data        : DataFrame with 'Close' column
        rsi_period  : RSI lookback period (default 14)
        oversold    : RSI level to enter long (default 30)
        overbought  : RSI level to exit long  (default 55)

    Returns:
        DataFrame with rsi, signal, position columns added
    """
    df = data.copy()
    df["rsi"] = _compute_rsi(df["Close"], period=rsi_period)

    # State machine: hold position until exit condition met
    signal = pd.Series(0, index=df.index)
    in_trade = False

    for i in range(len(df)):
        if not in_trade and df["rsi"].iloc[i] < oversold:
            in_trade = True
        elif in_trade and df["rsi"].iloc[i] > overbought:
            in_trade = False
        signal.iloc[i] = 1 if in_trade else 0

    df["signal"]   = signal
    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(rsi_period=14, oversold=30, overbought=55):
    return {
        "rsi_period": rsi_period,
        "oversold":   oversold,
        "overbought": overbought,
    }
    