"""
features/position_bridge.py
----------------------------
Position Bridge — converts continuous alpha scores into positions
that the existing backtest engine can consume.

Pipeline position:
    feature_engine → alpha_model → position_bridge → backtest

This is the ONLY place in the alpha pipeline where a discrete
decision is made. Everything upstream is continuous.

The bridge does three things:
    1. Dead zone  — scores too close to zero → flat (0)
                    avoids overtrading on noise
    2. Sizing     — score magnitude → position size in (-1, 1)
                    stronger signal = larger position
    3. Shift      — shift by 1 bar to prevent lookahead bias

Modes:
    "continuous"  : position = alpha_score (after dead zone)
                    fractional positions between -1 and 1
    "discrete"    : position = sign(alpha_score) if |score| > threshold
                    pure -1/0/1, like existing strategies
    "tiered"      : position = small/medium/full based on score magnitude
                    e.g. |score| > 0.3 → 0.5, |score| > 0.6 → 1.0
"""

import pandas as pd
import numpy as np

BRIDGE_NAME = "PositionBridge"


def apply(
    df,
    mode            = "continuous",
    dead_zone       = 0.1,
    max_position    = 1.0,
    long_only       = False,
    tiers           = None,
):
    """
    Convert alpha_score column to position column.

    Args:
        df           : DataFrame with 'alpha_score' column
        mode         : "continuous" | "discrete" | "tiered" (default "continuous")
        dead_zone    : scores with |alpha_score| < dead_zone → flat (default 0.1)
        max_position : maximum absolute position size (default 1.0)
        long_only    : if True, negative scores → flat instead of short (default False)
        tiers        : list of (threshold, size) pairs for tiered mode
                       default: [(0.3, 0.5), (0.6, 1.0)]

    Returns:
        DataFrame with 'position' column added (shifted 1 bar, no lookahead)
    """
    if "alpha_score" not in df.columns:
        raise ValueError("alpha_score column not found. Run AlphaModel.predict() first.")

    df  = df.copy()
    raw = df["alpha_score"].copy()

    # Apply dead zone — zero out weak signals
    raw = raw.where(raw.abs() >= dead_zone, other=0.0)

    if mode == "continuous":
        pos = raw.clip(-max_position, max_position)

    elif mode == "discrete":
        pos = raw.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    elif mode == "tiered":
        if tiers is None:
            tiers = [(0.3, 0.5), (0.6, 1.0)]
        tiers_sorted = sorted(tiers, key=lambda x: x[0])

        def _tier(score):
            if score == 0:
                return 0.0
            direction = 1.0 if score > 0 else -1.0
            magnitude = abs(score)
            size = 0.0
            for threshold, tier_size in tiers_sorted:
                if magnitude >= threshold:
                    size = tier_size
            return direction * size

        pos = raw.apply(_tier)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: continuous | discrete | tiered")

    # Long-only filter
    if long_only:
        pos = pos.clip(lower=0)

    # Shift 1 bar to avoid lookahead — signal on close of day N
    # executes at open of day N+1
    df["signal"]   = pos
    df["position"] = pos.shift(1).fillna(0)

    return df


def get_params(mode="continuous", dead_zone=0.1, max_position=1.0,
               long_only=False):
    return {
        "mode":         mode,
        "dead_zone":    dead_zone,
        "max_position": max_position,
        "long_only":    long_only,
    }

    