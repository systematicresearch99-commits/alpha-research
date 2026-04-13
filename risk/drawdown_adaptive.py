"""
risk/drawdown_adaptive.py
-------------------------
Drawdown-Adaptive Position Sizing.

Automatically shrinks position size as the equity curve drawdown
deepens. Protects capital during losing streaks by reducing exposure
when things are going wrong, and restores full size as equity recovers.

Standalone usage:
    from risk.drawdown_adaptive import apply
    df = apply(df, max_drawdown_limit=0.20, floor=0.1)

Interface:
    Input  : df with 'position' (0/1/-1), 'strategy_returns', 'equity_curve' columns
             NOTE: requires a backtest to have been run first (engine.py)
             so equity_curve is available. In run_risk.py this is handled
             by running a base backtest before applying risk modules.
    Output : df with 'sized_position', 'drawdown', 'dd_scalar' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "DrawdownAdaptive"
MODULE_TYPE = "sizing"


def apply(df, max_drawdown_limit=0.20, floor=0.10, recovery_factor=1.0):
    """
    Drawdown-Adaptive Position Sizing.

    Computes current drawdown from equity peak.
    Scales position size linearly from 1.0 (at 0% drawdown) down
    to floor (at max_drawdown_limit drawdown).

    scalar = 1.0 - (current_dd / max_drawdown_limit) * (1 - floor)
    scalar is clamped to [floor, 1.0]

    recovery_factor controls how quickly size is restored:
    - 1.0 = size recovers as fast as drawdown recovers (symmetric)
    - 0.5 = size recovers at half the speed of equity (more cautious)

    Args:
        df                 : DataFrame with 'position' and 'equity_curve' columns
        max_drawdown_limit : drawdown level at which position hits floor (default 0.20 = 20%)
        floor              : minimum position scalar at max drawdown (default 0.10 = 10%)
        recovery_factor    : speed of size recovery vs equity recovery (default 1.0)

    Returns:
        DataFrame with 'drawdown', 'dd_scalar', 'sized_position' added
    """
    df = df.copy()

    # Build equity curve if not already present
    if "equity_curve" not in df.columns:
        raise ValueError(
            "drawdown_adaptive requires 'equity_curve' column. "
            "Run the base backtest (engine.py) before applying this module, "
            "or use run_risk.py which handles this automatically."
        )

    equity             = df["equity_curve"]
    rolling_peak       = equity.cummax()
    df["drawdown"]     = (equity - rolling_peak) / rolling_peak  # negative values

    # Scalar: 1.0 at peak, floor at max_drawdown_limit
    dd_abs             = df["drawdown"].abs()
    raw_scalar         = 1.0 - (dd_abs / max_drawdown_limit) * (1.0 - floor)
    df["dd_scalar"]    = raw_scalar.clip(lower=floor, upper=1.0) ** (1.0 / recovery_factor)

    # Shift 1 — use yesterday's drawdown to size today
    dd_scalar_lagged   = df["dd_scalar"].shift(1).fillna(1.0)

    source_col = "sized_position" if "sized_position" in df.columns else "position"
    direction  = df[source_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Preserve existing magnitude if already sized, else use scalar directly
    existing_mag = df[source_col].abs()
    new_mag      = existing_mag * dd_scalar_lagged

    df["sized_position"] = direction * new_mag

    return df


def get_params(max_drawdown_limit=0.20, floor=0.10, recovery_factor=1.0):
    return {
        "max_drawdown_limit": max_drawdown_limit,
        "floor":              floor,
        "recovery_factor":    recovery_factor,
    }
