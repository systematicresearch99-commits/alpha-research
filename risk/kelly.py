"""
risk/kelly.py
-------------
Kelly Criterion Position Sizing.

Computes the mathematically optimal position size based on the
strategy's historical win rate and average win/loss ratio.
Uses a rolling window so sizing adapts as the strategy's recent
performance changes.

Standalone usage:
    from risk.kelly import apply
    df = apply(df, window=60, kelly_fraction=0.5)

Interface:
    Input  : df with 'position' (0/1/-1) and 'Close' columns
    Output : df with 'sized_position', 'kelly_f', 'win_rate', 'win_loss_ratio' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "Kelly"
MODULE_TYPE = "sizing"


def _rolling_kelly(df, window):
    """
    Compute rolling Kelly fraction from trade returns.

    Kelly f = W - (1-W)/R
    where W = win rate, R = avg win / avg loss ratio
    """
    # Daily returns when in a position
    returns = df["Close"].pct_change()
    in_position = df["position"].shift(1).fillna(0).abs() > 0
    trade_returns = returns.where(in_position, other=np.nan)

    win_rate      = pd.Series(np.nan, index=df.index)
    win_loss_ratio = pd.Series(np.nan, index=df.index)
    kelly_f       = pd.Series(0.0, index=df.index)

    for i in range(window, len(df)):
        window_returns = trade_returns.iloc[i - window:i].dropna()
        if len(window_returns) < 10:
            continue

        wins   = window_returns[window_returns > 0]
        losses = window_returns[window_returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            continue

        w = len(wins) / len(window_returns)
        r = wins.mean() / abs(losses.mean())

        f = w - (1 - w) / r   # Kelly formula
        f = max(f, 0)          # Never negative — Kelly says don't trade if f<0

        win_rate.iloc[i]       = w
        win_loss_ratio.iloc[i] = r
        kelly_f.iloc[i]        = f

    return kelly_f, win_rate, win_loss_ratio


def apply(df, window=60, kelly_fraction=0.5, max_position=1.0):
    """
    Kelly Criterion Position Sizing.

    Full Kelly is theoretically optimal but extremely aggressive in
    practice — drawdowns are brutal. kelly_fraction scales it down:
    - 0.5 = Half Kelly (recommended — much smoother equity curve)
    - 0.25 = Quarter Kelly (conservative)
    - 1.0 = Full Kelly (aggressive, high variance)

    Size is recomputed on a rolling basis so it adapts to recent
    strategy performance rather than using fixed historical stats.

    Args:
        df             : DataFrame with 'position' and 'Close' columns
        window         : rolling lookback for win rate / win-loss ratio (default 60)
        kelly_fraction : fraction of full Kelly to use (default 0.5 = half Kelly)
        max_position   : maximum allowed position size (default 1.0)

    Returns:
        DataFrame with 'kelly_f', 'win_rate', 'win_loss_ratio', 'sized_position' added
    """
    df = df.copy()

    df["kelly_f"], df["win_rate"], df["win_loss_ratio"] = _rolling_kelly(df, window)

    # Apply fractional Kelly and direction
    direction = df["position"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    raw_size  = (df["kelly_f"] * kelly_fraction).clip(upper=max_position)

    df["sized_position"] = (direction * raw_size).clip(
        lower=-max_position, upper=max_position
    )

    return df


def get_params(window=60, kelly_fraction=0.5, max_position=1.0):
    return {"window": window, "kelly_fraction": kelly_fraction, "max_position": max_position}
