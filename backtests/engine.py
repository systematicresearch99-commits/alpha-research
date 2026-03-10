import numpy as np
import pandas as pd


def run_backtest(df, transaction_cost=0.001):
    """
    Core vectorized backtest engine.

    Args:
        df               : DataFrame with at least ['Close', 'position'] columns
                           (position should already be shifted to avoid lookahead)
        transaction_cost : cost per trade as a fraction (default 0.1%)

    Returns:
        df with added columns:
            returns, trade, strategy_returns, equity_curve, buy_hold_equity
    """
    df = df.copy()

    # Daily returns of the underlying
    df["returns"] = df["Close"].pct_change()

    # Detect position changes (entries/exits)
    df["trade"] = df["position"].diff().abs()

    # Strategy daily returns = position * underlying return - transaction cost on trade days
    df["strategy_returns"] = df["returns"] * df["position"]
    df["strategy_returns"] -= df["trade"] * transaction_cost

    # Equity curves
    df["equity_curve"]    = (1 + df["strategy_returns"]).cumprod()
    df["buy_hold_equity"] = (1 + df["returns"]).cumprod()

    return df