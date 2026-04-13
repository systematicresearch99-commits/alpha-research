"""
risk/regime_sizer.py
--------------------
Regime-Aware Position Sizing.

Uses HMM regime labels (from run_regime.py) to scale position size
up or down depending on which regime the market is currently in.
Lets you be aggressive in favourable regimes and defensive in hostile ones.

Standalone usage:
    from risk.regime_sizer import apply
    df = apply(df, regime_scalars={0: 1.0, 1: 0.5, 2: 0.25, 3: 0.0})

Interface:
    Input  : df with 'position' (0/1/-1) and 'regime' columns
             NOTE: 'regime' column is added by RegimeAnalyzer.analyze()
             In run_risk.py --regime flag triggers the HMM and attaches labels.
    Output : df with 'sized_position', 'regime_scalar' columns added
"""

import pandas as pd
import numpy as np

MODULE_NAME = "RegimeSizer"
MODULE_TYPE = "sizing"

# Default scalars — you'll want to tune these after running regime analysis
# to understand which regime is which for your specific ticker + period.
# 0 = typically low-vol bull, 1 = moderate, 2 = choppy, 3 = high-vol bear
DEFAULT_SCALARS = {
    0: 1.0,    # Full size — favourable regime
    1: 0.75,   # Slightly reduced
    2: 0.50,   # Half size — uncertain
    3: 0.0,    # Flat — hostile regime (e.g. high-vol bear)
}


def apply(df, regime_scalars=None, max_position=1.0):
    """
    Regime-Aware Position Sizing.

    Maps each HMM regime label to a position scalar. Strategy signal
    direction is preserved but magnitude is scaled by the regime scalar.

    Regime labels are integers (0 to n_regimes-1). The mapping between
    label number and market condition (bull/bear/choppy etc.) depends on
    your HMM training — check RegimeAnalyzer output to assign scalars.

    Args:
        df             : DataFrame with 'position' and 'regime' columns
        regime_scalars : dict mapping regime int → size scalar (0.0 to 1.0)
                         defaults to DEFAULT_SCALARS if None
        max_position   : maximum allowed position size (default 1.0)

    Returns:
        DataFrame with 'regime_scalar', 'sized_position' added
    """
    df = df.copy()

    if "regime" not in df.columns:
        raise ValueError(
            "regime_sizer requires a 'regime' column. "
            "Use --regime flag in run_risk.py to attach HMM regime labels, "
            "or run run_regime.py first and pass the labeled df."
        )

    scalars = regime_scalars if regime_scalars is not None else DEFAULT_SCALARS

    # Map regime labels to scalars — unknown regimes default to 0.5
    df["regime_scalar"] = df["regime"].map(scalars).fillna(0.5)

    # Shift 1 — use yesterday's regime to size today
    scalar_lagged = df["regime_scalar"].shift(1).fillna(1.0)

    source_col   = "sized_position" if "sized_position" in df.columns else "position"
    direction    = df[source_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    existing_mag = df[source_col].abs()

    new_mag = existing_mag * scalar_lagged
    df["sized_position"] = (direction * new_mag).clip(
        lower=-max_position, upper=max_position
    )

    return df


def get_params(regime_scalars=None, max_position=1.0):
    return {
        "regime_scalars": regime_scalars or DEFAULT_SCALARS,
        "max_position":   max_position,
    }
