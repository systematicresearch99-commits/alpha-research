"""
robustness/monte_carlo.py
--------------------------
Monte Carlo Simulation.

Answers the question: could this strategy's performance have happened
by chance? Shuffles the trade returns thousands of times and builds a
distribution of what random luck looks like. If your actual Sharpe is
in the top 5% of random outcomes, you have a statistically significant edge.

Also runs equity curve simulations — draws random paths from the return
distribution to show the range of possible futures.

Standalone usage:
    from robustness.monte_carlo import run
    results = run(df, n_simulations=1000)
"""

import pandas as pd
import numpy as np
from utils.performance import calculate_metrics

MODULE_NAME = "MonteCarlo"


def run(df, n_simulations=1000, confidence_levels=(0.05, 0.25, 0.75, 0.95),
        random_seed=42):
    """
    Monte Carlo Robustness Test.

    Two analyses:

    1. Trade shuffle test — randomly shuffles the ORDER of trade returns
       n_simulations times and measures what Sharpe/Return looks like by
       chance. Your actual result is compared against this distribution.
       p-value = fraction of simulations that beat your actual result.
       p < 0.05 means less than 5% chance this was luck → significant edge.

    2. Equity path simulation — samples from the return distribution with
       replacement to simulate n_simulations possible futures. Shows the
       range of outcomes you might expect going forward.

    Args:
        df               : DataFrame with strategy_returns and equity_curve columns
        n_simulations    : number of Monte Carlo runs (default 1000)
        confidence_levels: quantiles for path simulation output
        random_seed      : reproducibility seed

    Returns:
        dict with:
            'shuffle_sharpes'  : array of Sharpe ratios from shuffled runs
            'shuffle_returns'  : array of total returns from shuffled runs
            'actual_sharpe'    : your strategy's actual Sharpe
            'actual_return'    : your strategy's actual total return
            'p_value_sharpe'   : fraction of shuffles that beat actual Sharpe
            'p_value_return'   : fraction of shuffles that beat actual Return
            'is_significant'   : True if p_value_sharpe < 0.05
            'sim_paths'        : equity path simulations dict {quantile: series}
            'sim_final_returns': distribution of final equity values
            'confidence_bands' : dict of {pct: final_equity_value}
    """
    np.random.seed(random_seed)

    returns    = df["strategy_returns"].dropna()
    actual_m   = calculate_metrics(df)
    actual_sharpe = actual_m.get("Sharpe Ratio", np.nan)
    actual_return = actual_m.get("Total Return", np.nan)

    # ── 1. Trade Shuffle Test ─────────────────────────────────────────────────
    shuffle_sharpes = []
    shuffle_returns = []

    ret_array = returns.values.copy()

    for _ in range(n_simulations):
        shuffled = ret_array.copy()
        np.random.shuffle(shuffled)

        eq    = (1 + shuffled).cumprod()
        tr    = eq[-1] - 1
        n_yr  = len(shuffled) / 252
        ann_r = (1 + tr) ** (1 / n_yr) - 1 if n_yr > 0 else np.nan
        vol   = shuffled.std() * np.sqrt(252)
        sr    = (shuffled.mean() * 252) / vol if vol > 0 else np.nan

        shuffle_sharpes.append(sr)
        shuffle_returns.append(tr)

    shuffle_sharpes = np.array(shuffle_sharpes)
    shuffle_returns = np.array(shuffle_returns)

    p_value_sharpe = (shuffle_sharpes > actual_sharpe).mean()
    p_value_return = (shuffle_returns > actual_return).mean()
    is_significant = p_value_sharpe < 0.05

    # ── 2. Equity Path Simulation ─────────────────────────────────────────────
    n_bars         = len(returns)
    sim_final      = []
    sim_paths_raw  = []

    for _ in range(n_simulations):
        sampled = np.random.choice(ret_array, size=n_bars, replace=True)
        path    = (1 + sampled).cumprod()
        sim_final.append(path[-1])
        sim_paths_raw.append(path)

    sim_paths_array = np.array(sim_paths_raw)
    sim_final       = np.array(sim_final)

    # Build quantile paths
    sim_paths = {}
    for q in confidence_levels:
        sim_paths[q] = pd.Series(
            np.quantile(sim_paths_array, q, axis=0),
            index=returns.index,
        )

    confidence_bands = {
        q: round(float(np.quantile(sim_final, q)), 4)
        for q in confidence_levels
    }

    # Percentile rank of actual strategy vs simulations
    actual_pct_rank = (sim_final < (1 + actual_return)).mean()

    return {
        "shuffle_sharpes":   shuffle_sharpes,
        "shuffle_returns":   shuffle_returns,
        "actual_sharpe":     actual_sharpe,
        "actual_return":     actual_return,
        "p_value_sharpe":    round(p_value_sharpe, 4),
        "p_value_return":    round(p_value_return,  4),
        "is_significant":    is_significant,
        "sim_paths":         sim_paths,
        "sim_final_returns": sim_final,
        "confidence_bands":  confidence_bands,
        "actual_pct_rank":   round(actual_pct_rank, 4),
        "n_simulations":     n_simulations,
        "shuffle_sharpe_mean": round(shuffle_sharpes.mean(), 4),
        "shuffle_sharpe_std":  round(shuffle_sharpes.std(),  4),
        "shuffle_sharpe_95p":  round(np.percentile(shuffle_sharpes, 95), 4),
    }


def print_results(results, strategy_name="Strategy"):
    """Pretty-print Monte Carlo results."""
    print(f"\n{'═'*65}")
    print(f"  Monte Carlo Results  —  {strategy_name}")
    print(f"  {results['n_simulations']} simulations")
    print(f"{'═'*65}")

    # Trade shuffle significance test
    print(f"\n  Edge Significance Test  (trade shuffle)")
    print(f"  {'─'*50}")
    print(f"    Actual Sharpe         {results['actual_sharpe']:>10.4f}")
    print(f"    Random mean Sharpe    {results['shuffle_sharpe_mean']:>10.4f}")
    print(f"    Random 95th pct       {results['shuffle_sharpe_95p']:>10.4f}")
    print(f"    p-value (Sharpe)      {results['p_value_sharpe']:>10.4f}  "
          f"← fraction of random runs that beat you")
    print(f"    p-value (Return)      {results['p_value_return']:>10.4f}")

    sig = results["is_significant"]
    print(f"\n    Verdict: {'SIGNIFICANT EDGE (p < 0.05)' if sig else 'NOT SIGNIFICANT (p >= 0.05) — may be luck'}")

    if not sig:
        print(f"    Warning: {results['p_value_sharpe']*100:.1f}% of random shuffles "
              f"beat your strategy. This suggests the returns")
        print(f"    could be explained by luck rather than a genuine edge.")

    # Equity path simulation
    bands = results["confidence_bands"]
    print(f"\n  Future Path Simulation  (bootstrap resampling)")
    print(f"  {'─'*50}")
    print(f"    Your strategy ended at  {1 + results['actual_return']:.2f}x")
    print(f"    Simulated outcomes ({results['n_simulations']} paths):")
    for q, val in sorted(bands.items()):
        label = f"{int(q*100)}th percentile"
        print(f"      {label:<22} {val:.2f}x")
    print(f"    Actual result ranks at  {results['actual_pct_rank']*100:.1f}th percentile "
          f"of simulated paths")

    print(f"\n{'═'*65}\n")
