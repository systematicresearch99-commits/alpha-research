"""
robustness/param_sensitivity.py
--------------------------------
Parameter Sensitivity Analysis.

Tests a grid of parameter values around your chosen settings and asks:
does performance hold up when parameters change slightly, or does it
collapse — indicating the result was curve-fit to one lucky combination?

A robust strategy shows a smooth "hill" of performance across the
parameter space. A fragile strategy shows a sharp spike — one lucky
combination surrounded by poor results.

Standalone usage:
    from robustness.param_sensitivity import run
    results = run(data, generate_signals_fn, param_grid)
"""

import pandas as pd
import numpy as np
import itertools
from backtests.engine  import run_backtest
from utils.performance import calculate_metrics

MODULE_NAME = "ParamSensitivity"


def run(data, generate_signals_fn, param_grid, metric="Sharpe Ratio"):
    """
    Parameter Sensitivity Analysis.

    Tests every combination in param_grid and records performance.
    Shows how sensitive results are to parameter choice.

    Args:
        data                : full OHLCV DataFrame
        generate_signals_fn : strategy's generate_signals function
        param_grid          : dict of {param_name: [list of values to test]}
                              e.g. {"rsi_period": [10,12,14,16,18],
                                    "oversold":   [25,28,30,32,35]}
        metric              : primary metric to rank by (default "Sharpe Ratio")

    Returns:
        dict with:
            'results_df'     : DataFrame of all param combinations + metrics
            'best_params'    : params with highest metric value
            'baseline_params': first param combination (your default)
            'sensitivity'    : std of metric across all combinations
            'robustness_pct' : % of combinations with positive Sharpe
    """
    # Build all combinations
    param_names  = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"  [sensitivity] Testing {len(combinations)} parameter combinations...")

    rows = []
    for combo in combinations:
        kwargs = dict(zip(param_names, combo))
        try:
            df  = generate_signals_fn(data.copy(), **kwargs)
            df  = run_backtest(df)
            m   = calculate_metrics(df)
            row = {**kwargs,
                   "Sharpe Ratio":    m.get("Sharpe Ratio",    np.nan),
                   "Total Return":    m.get("Total Return",    np.nan),
                   "Max Drawdown":    m.get("Max Drawdown",    np.nan),
                   "Annualized Return": m.get("Annualized Return", np.nan),
                   "Win Rate":        m.get("Win Rate",        np.nan),
                   "Num Trades":      m.get("Num Trades",      0),
                   "Sortino Ratio":   m.get("Sortino Ratio",   np.nan),
                   "Calmar Ratio":    m.get("Calmar Ratio",    np.nan),
            }
        except Exception as e:
            row = {**kwargs,
                   "Sharpe Ratio": np.nan, "Total Return": np.nan,
                   "Max Drawdown": np.nan, "Annualized Return": np.nan,
                   "Win Rate": np.nan, "Num Trades": 0,
                   "Sortino Ratio": np.nan, "Calmar Ratio": np.nan,
                   "_error": str(e)}
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Compute sensitivity stats
    metric_vals  = results_df[metric].dropna()
    sensitivity  = metric_vals.std()
    robustness   = (metric_vals > 0).mean()
    best_idx     = results_df[metric].idxmax()
    best_params  = {k: results_df.loc[best_idx, k] for k in param_names}
    best_metrics = {
        "Sharpe Ratio":  results_df.loc[best_idx, "Sharpe Ratio"],
        "Total Return":  results_df.loc[best_idx, "Total Return"],
        "Max Drawdown":  results_df.loc[best_idx, "Max Drawdown"],
    }

    return {
        "results_df":      results_df.sort_values(metric, ascending=False),
        "best_params":     best_params,
        "best_metrics":    best_metrics,
        "param_names":     param_names,
        "metric":          metric,
        "sensitivity":     round(sensitivity, 4),
        "robustness_pct":  round(robustness,  4),
        "n_combinations":  len(combinations),
        "metric_mean":     round(metric_vals.mean(), 4),
        "metric_std":      round(metric_vals.std(),  4),
        "metric_min":      round(metric_vals.min(),  4),
        "metric_max":      round(metric_vals.max(),  4),
    }


def print_results(results, strategy_name="Strategy", top_n=10):
    """Pretty-print parameter sensitivity results."""
    df         = results["results_df"]
    param_names = results["param_names"]
    metric      = results["metric"]

    print(f"\n{'═'*65}")
    print(f"  Parameter Sensitivity  —  {strategy_name}")
    print(f"  Primary metric: {metric}")
    print(f"{'═'*65}")

    print(f"\n  Distribution of {metric} across {results['n_combinations']} combinations")
    print(f"  {'─'*45}")
    print(f"    Mean     {results['metric_mean']:>8.4f}")
    print(f"    Std      {results['metric_std']:>8.4f}   ← lower = more robust")
    print(f"    Min      {results['metric_min']:>8.4f}")
    print(f"    Max      {results['metric_max']:>8.4f}")
    print(f"    Positive {results['robustness_pct']*100:>7.1f}%  ← higher = more robust")

    # Interpretation
    print(f"\n  Robustness Assessment")
    print(f"  {'─'*45}")
    pct  = results["robustness_pct"]
    std  = results["metric_std"]
    diff = results["metric_max"] - results["metric_mean"]

    if pct >= 0.7 and std < 0.3:
        verdict = "ROBUST — consistent performance across parameter space"
    elif pct >= 0.5 and std < 0.5:
        verdict = "MODERATELY ROBUST — some sensitivity, generally positive"
    elif diff > 1.0:
        verdict = "FRAGILE — sharp spike suggests curve-fitting"
    else:
        verdict = "WEAK — strategy doesn't work across parameter ranges"

    print(f"    {verdict}")
    print(f"    Best params: {results['best_params']}")
    print(f"    Best Sharpe: {results['best_metrics']['Sharpe Ratio']:.4f}  "
          f"vs Mean: {results['metric_mean']:.4f}")

    # Top N combinations
    print(f"\n  Top {top_n} Parameter Combinations")
    print(f"  {'─'*60}")
    param_header = "  ".join(f"{p:>12}" for p in param_names)
    print(f"  {param_header}  {'Sharpe':>8}  {'Return':>8}  {'MaxDD':>8}  {'Trades':>6}")
    print(f"  {'─'*60}")

    for _, row in df.head(top_n).iterrows():
        param_vals = "  ".join(f"{row[p]:>12}" for p in param_names)
        sharpe = row["Sharpe Ratio"]
        ret    = row["Total Return"]
        mdd    = row["Max Drawdown"]
        trades = row["Num Trades"]
        print(f"  {param_vals}  {sharpe:>8.4f}  {ret*100:>7.1f}%  {mdd*100:>7.1f}%  {trades:>6}")

    print(f"\n{'═'*65}\n")
