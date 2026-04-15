"""
robustness/walk_forward.py
--------------------------
Walk-Forward Testing.

Splits the data into rolling train/test windows and runs the strategy
on each out-of-sample window independently. The key question:
does the strategy produce consistent results on data it has never seen?

A strategy that only works in-sample is overfit.
A robust strategy produces positive out-of-sample results consistently.

Standalone usage:
    from robustness.walk_forward import run
    results = run(df, generate_signals_fn, n_splits=5, test_pct=0.2)
"""

import pandas as pd
import numpy as np
from backtests.engine  import run_backtest
from utils.performance import calculate_metrics

MODULE_NAME = "WalkForward"


def run(data, generate_signals_fn, n_splits=5, test_pct=0.2,
        min_train_pct=0.4, **strategy_kwargs):
    """
    Walk-Forward Test.

    Divides data into n_splits sequential windows. Each window has a
    train period and an immediately following test period. The strategy
    is run on the test period only — train period is never evaluated,
    it just represents "data available at that point in time."

    Window structure (n_splits=5, test_pct=0.2):

    |-----T1-----|--t1--|
          |-----T2-----|--t2--|
                |-----T3-----|--t3--|
                      |-----T4-----|--t4--|
                            |-----T5-----|--t5--|

    T = train window (not evaluated — represents lookback)
    t = test window  (evaluated — true out-of-sample)

    Args:
        data               : full OHLCV DataFrame
        generate_signals_fn: strategy's generate_signals function
        n_splits           : number of walk-forward windows (default 5)
        test_pct           : fraction of each window used for testing (default 0.2)
        min_train_pct      : minimum fraction of total data for first train (default 0.4)
        **strategy_kwargs  : passed to generate_signals_fn

    Returns:
        dict with:
            'window_results' : list of per-window metrics dicts
            'oos_equity'     : concatenated out-of-sample equity curve
            'summary'        : aggregated metrics across all OOS windows
            'consistency'    : fraction of windows with positive Sharpe
    """
    n = len(data)

    # Build window boundaries
    # Start of test windows spaced evenly from min_train_pct to end
    usable_start = int(n * min_train_pct)
    test_size    = int(n * test_pct)
    step         = max(test_size, (n - usable_start - test_size) // max(n_splits - 1, 1))

    windows = []
    for i in range(n_splits):
        test_start = usable_start + i * step
        test_end   = test_start + test_size
        if test_end > n:
            break
        train_start = 0
        train_end   = test_start
        windows.append((train_start, train_end, test_start, test_end))

    if not windows:
        raise ValueError("Not enough data for the requested number of splits.")

    window_results = []
    oos_pieces     = []

    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        test_data = data.iloc[te_s:te_e].copy()

        # Run strategy on test window
        # NOTE: we use the full data up to test_end for signal generation
        # so indicators have proper lookback, but only evaluate test period
        full_window = data.iloc[tr_s:te_e].copy()
        try:
            df_signals = generate_signals_fn(full_window, **strategy_kwargs)
            df_test    = df_signals.iloc[te_s - tr_s:].copy()
            df_test    = run_backtest(df_test)
            metrics    = calculate_metrics(df_test)
        except Exception as e:
            print(f"  [wf] Window {idx+1} failed: {e}")
            continue

        window_info = {
            "window":       idx + 1,
            "train_start":  data.index[tr_s],
            "train_end":    data.index[tr_e - 1],
            "test_start":   data.index[te_s],
            "test_end":     data.index[te_e - 1],
            "n_test_bars":  te_e - te_s,
            **metrics,
        }
        window_results.append(window_info)
        oos_pieces.append(df_test)

    if not window_results:
        raise ValueError("All walk-forward windows failed.")

    # Concatenate OOS equity curves (chain them — each starts from prior end)
    oos_equity = _chain_equity(oos_pieces)

    # Aggregate summary
    sharpes      = [w["Sharpe Ratio"]      for w in window_results if not np.isnan(w.get("Sharpe Ratio", np.nan))]
    returns      = [w["Total Return"]      for w in window_results if not np.isnan(w.get("Total Return", np.nan))]
    drawdowns    = [w["Max Drawdown"]      for w in window_results if not np.isnan(w.get("Max Drawdown", np.nan))]
    consistency  = sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0

    summary = {
        "Avg OOS Sharpe":      round(np.mean(sharpes),   4) if sharpes   else np.nan,
        "Std OOS Sharpe":      round(np.std(sharpes),    4) if sharpes   else np.nan,
        "Avg OOS Return":      round(np.mean(returns),   4) if returns   else np.nan,
        "Avg OOS MaxDrawdown": round(np.mean(drawdowns), 4) if drawdowns else np.nan,
        "Consistency":         round(consistency,        4),
        "Windows Tested":      len(window_results),
        "Windows Profitable":  sum(1 for r in returns if r > 0),
    }

    return {
        "window_results": window_results,
        "oos_equity":     oos_equity,
        "summary":        summary,
        "consistency":    consistency,
        "windows":        windows,
    }


def _chain_equity(dfs):
    """
    Chain equity curves from multiple windows end-to-end.
    Each window's equity starts from where the prior window ended.
    """
    if not dfs:
        return pd.Series(dtype=float)

    pieces = []
    running_equity = 1.0

    for df in dfs:
        if "equity_curve" not in df.columns:
            continue
        eq     = df["equity_curve"].dropna()
        scaled = eq * running_equity
        running_equity = scaled.iloc[-1]
        pieces.append(scaled)

    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def print_results(results, strategy_name="Strategy"):
    """Pretty-print walk-forward results."""
    summary = results["summary"]
    windows = results["window_results"]

    print(f"\n{'═'*65}")
    print(f"  Walk-Forward Results  —  {strategy_name}")
    print(f"{'═'*65}")

    print(f"\n  Per-Window Out-of-Sample Performance")
    print(f"  {'─'*60}")
    print(f"  {'Win':>3}  {'Test Start':>12}  {'Test End':>12}  "
          f"{'Sharpe':>7}  {'Return':>8}  {'MaxDD':>8}  {'Trades':>6}")
    print(f"  {'─'*60}")

    for w in windows:
        sharpe = w.get("Sharpe Ratio", float("nan"))
        ret    = w.get("Total Return", float("nan"))
        mdd    = w.get("Max Drawdown", float("nan"))
        trades = w.get("Num Trades",   0)
        flag   = "✓" if (not np.isnan(sharpe) and sharpe > 0) else "✗"
        print(f"  {w['window']:>3}  "
              f"{str(w['test_start'].date()):>12}  "
              f"{str(w['test_end'].date()):>12}  "
              f"{sharpe:>7.3f}  "
              f"{ret*100:>7.1f}%  "
              f"{mdd*100:>7.1f}%  "
              f"{trades:>6}  {flag}")

    print(f"\n  Summary")
    print(f"  {'─'*40}")
    for k, v in summary.items():
        if isinstance(v, float):
            if "Return" in k or "Drawdown" in k:
                print(f"    {k:<28} {v*100:>8.2f}%")
            elif "Consistency" in k:
                print(f"    {k:<28} {v*100:>8.1f}%  ({summary['Windows Profitable']}/{summary['Windows Tested']} windows profitable)")
            else:
                print(f"    {k:<28} {v:>8.4f}")
        else:
            print(f"    {k:<28} {str(v):>8}")

    print(f"\n{'═'*65}\n")
