"""
risk/risk_manager.py
--------------------
Risk Manager — chains multiple risk modules into a single pipeline.

This is the central coordinator used by run_risk.py. It handles:
  - Applying modules in the correct order (sizing first, stops last)
  - Stacking (one combined backtest with all modules applied)
  - Comparing (separate backtests per module, side-by-side metrics)
  - Computing strategy_returns from sized_position for the engine

Module application order (when stacking):
  1. Sizing modules  — determine how large the position is
     (fixed_fractional, atr_sizing, kelly, vol_target,
      drawdown_adaptive, regime_sizer, signal_strength)
  2. Stop modules    — hard overrides that can zero the position
     (fixed_stop, atr_trailing_stop, time_stop)

Usage (internal — called by run_risk.py):
    from risk.risk_manager import apply_stack, apply_compare
"""

import pandas as pd
import numpy as np

# ── Module registry ────────────────────────────────────────────────────────────
from risk.fixed_fractional   import apply as ff_apply,    get_params as ff_params,    MODULE_NAME as FF_NAME,    MODULE_TYPE as FF_TYPE
from risk.atr_sizing         import apply as atr_sz_apply, get_params as atr_sz_params, MODULE_NAME as ATR_SZ_NAME, MODULE_TYPE as ATR_SZ_TYPE
from risk.kelly              import apply as kelly_apply,  get_params as kelly_params,  MODULE_NAME as KELLY_NAME,  MODULE_TYPE as KELLY_TYPE
from risk.vol_target         import apply as vt_apply,     get_params as vt_params,     MODULE_NAME as VT_NAME,     MODULE_TYPE as VT_TYPE
from risk.drawdown_adaptive  import apply as dd_apply,     get_params as dd_params,     MODULE_NAME as DD_NAME,     MODULE_TYPE as DD_TYPE
from risk.regime_sizer       import apply as rs_apply,     get_params as rs_params,     MODULE_NAME as RS_NAME,     MODULE_TYPE as RS_TYPE
from risk.signal_strength    import apply as ss_apply,     get_params as ss_params,     MODULE_NAME as SS_NAME,     MODULE_TYPE as SS_TYPE
from risk.fixed_stop         import apply as fs_apply,     get_params as fs_params,     MODULE_NAME as FS_NAME,     MODULE_TYPE as FS_TYPE
from risk.atr_trailing_stop  import apply as ats_apply,    get_params as ats_params,    MODULE_NAME as ATS_NAME,    MODULE_TYPE as ATS_TYPE
from risk.time_stop          import apply as ts_apply,     get_params as ts_params,     MODULE_NAME as TS_NAME,     MODULE_TYPE as TS_TYPE

REGISTRY = {
    "fixed_fractional":  (ff_apply,     ff_params,     FF_NAME,     FF_TYPE),
    "atr_sizing":        (atr_sz_apply, atr_sz_params, ATR_SZ_NAME, ATR_SZ_TYPE),
    "kelly":             (kelly_apply,  kelly_params,  KELLY_NAME,  KELLY_TYPE),
    "vol_target":        (vt_apply,     vt_params,     VT_NAME,     VT_TYPE),
    "drawdown_adaptive": (dd_apply,     dd_params,     DD_NAME,     DD_TYPE),
    "regime_sizer":      (rs_apply,     rs_params,     RS_NAME,     RS_TYPE),
    "signal_strength":   (ss_apply,     ss_params,     SS_NAME,     SS_TYPE),
    "fixed_stop":        (fs_apply,     fs_params,     FS_NAME,     FS_TYPE),
    "atr_trailing_stop": (ats_apply,    ats_params,    ATS_NAME,    ATS_TYPE),
    "time_stop":         (ts_apply,     ts_params,     TS_NAME,     TS_TYPE),
}

# Correct application order — sizing before stops
SIZING_MODULES = {"fixed_fractional", "atr_sizing", "kelly", "vol_target",
                  "drawdown_adaptive", "regime_sizer", "signal_strength"}
STOP_MODULES   = {"fixed_stop", "atr_trailing_stop", "time_stop"}


def _ordered(module_keys):
    """Sort module keys: sizing modules first, then stop modules."""
    sizing = [k for k in module_keys if k in SIZING_MODULES]
    stops  = [k for k in module_keys if k in STOP_MODULES]
    return sizing + stops


def _recompute_returns(df, cost_bps=10):
    """
    Recompute strategy_returns and equity_curve from sized_position.
    Used after risk modules modify position sizing.
    """
    df = df.copy()
    daily_ret = df["Close"].pct_change()

    # Transaction cost on position changes
    position_change = df["sized_position"].diff().abs()
    cost            = position_change * (cost_bps / 10_000)

    df["strategy_returns"] = df["sized_position"].shift(1) * daily_ret - cost
    df["equity_curve"]     = (1 + df["strategy_returns"].fillna(0)).cumprod()

    return df


def apply_stack(df, module_keys, module_kwargs=None):
    """
    Stack multiple risk modules — apply all to one backtest sequentially.

    Sizing modules run first, stop modules run last.
    Returns a single DataFrame with all risk adjustments applied.

    Args:
        df           : DataFrame after base backtest (has 'position', 'equity_curve')
        module_keys  : list of module keys to apply e.g. ["atr_sizing", "atr_trailing_stop"]
        module_kwargs: dict of {module_key: {param: value}} for per-module params

    Returns:
        df with sized_position, strategy_returns, equity_curve recomputed
    """
    module_kwargs = module_kwargs or {}
    ordered_keys  = _ordered(module_keys)

    result = df.copy()
    for key in ordered_keys:
        if key not in REGISTRY:
            raise ValueError(f"Unknown risk module '{key}'. Choose from: {list(REGISTRY)}")
        apply_fn, _, name, _ = REGISTRY[key]
        kwargs = module_kwargs.get(key, {})
        print(f"  [risk] Applying {name}  {kwargs if kwargs else ''}")
        result = apply_fn(result, **kwargs)

    # Recompute returns from the final sized_position
    result = _recompute_returns(result)
    return result


def apply_compare(df, module_keys, module_kwargs=None):
    """
    Compare risk modules side by side — separate backtest per module.

    Args:
        df           : DataFrame after base backtest (has 'position', 'equity_curve')
        module_keys  : list of module keys to compare
        module_kwargs: dict of {module_key: {param: value}}

    Returns:
        dict of {module_key: df_with_risk_applied}
    """
    module_kwargs = module_kwargs or {}
    results       = {}

    for key in module_keys:
        if key not in REGISTRY:
            raise ValueError(f"Unknown risk module '{key}'. Choose from: {list(REGISTRY)}")
        apply_fn, _, name, _ = REGISTRY[key]
        kwargs = module_kwargs.get(key, {})
        print(f"  [risk] Running {name}  {kwargs if kwargs else ''}")
        result          = apply_fn(df.copy(), **kwargs)
        result          = _recompute_returns(result)
        results[key]    = result

    return results


def list_modules():
    """Print all available risk modules."""
    print("\n── Risk Modules ─────────────────────────────")
    print("  Sizing:")
    for k in sorted(SIZING_MODULES):
        if k in REGISTRY:
            print(f"    {k:<22} {REGISTRY[k][2]}")
    print("  Stops:")
    for k in sorted(STOP_MODULES):
        if k in REGISTRY:
            print(f"    {k:<22} {REGISTRY[k][2]}")
    print()
