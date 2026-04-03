import pandas as pd
import numpy as np

STRATEGY_NAME = "Dual_Momentum"


def generate_signals(data, lookback=252, risk_free_rate=0.0):
    """
    Dual Momentum strategy (Gary Antonacci).

    Combines:
    1. Absolute Momentum — is the asset beating the risk-free rate?
       If yes, asset has positive absolute momentum → eligible to hold.
    2. Relative Momentum — is the asset's return positive over lookback?
       (Simplified single-asset version: compares asset to cash/risk-free)

    Signal logic:
        Long  (position= 1) when N-period return > risk_free_rate
                              (asset outperforms cash — stay invested)
        Flat  (position= 0) when N-period return <= risk_free_rate
                              (move to safety — do NOT short per Antonacci's design)

    This is intentionally long-only. The flat periods represent moving
    to a safe asset (bonds/cash), not shorting. Antonacci's research
    shows avoiding the short side improves risk-adjusted returns.

    Args:
        data           : DataFrame with 'Close' column
        lookback       : momentum lookback in trading days (default 252 = 1 year)
        risk_free_rate : annualized risk-free rate as decimal (default 0.0)
                         e.g. 0.04 = 4% annualized → daily = 0.04/252

    Returns:
        DataFrame with momentum_return, signal, position columns added
    """
    df = data.copy()

    # Convert annualized risk-free rate to lookback-period equivalent
    period_rf = (1 + risk_free_rate) ** (lookback / 252) - 1

    df["momentum_return"] = df["Close"].pct_change(periods=lookback)

    df["signal"] = 0
    df.loc[df["momentum_return"] > period_rf, "signal"] = 1

    # Shift 1 to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(lookback=252, risk_free_rate=0.0):
    return {"lookback": lookback, "risk_free_rate": risk_free_rate}
