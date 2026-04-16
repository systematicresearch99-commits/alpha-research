"""
run_robustness.py — Robustness Testing Runner
==============================================
Tests whether a strategy's performance is genuine or curve-fitted.

Usage
-----
    # Run all three robustness tests
    python run_robustness.py --strategy atr --ticker BTC-USD --start 2018-01-01 --test all

    # Walk-forward only
    python run_robustness.py --strategy atr --ticker BTC-USD --test walk_forward

    # Parameter sensitivity — uses default grid for the strategy
    python run_robustness.py --strategy rsi --ticker BTC-USD --test param_sensitivity

    # Monte Carlo
    python run_robustness.py --strategy dual_momentum --ticker BTC-USD --test monte_carlo

    # Multiple strategies, same test
    python run_robustness.py --strategies atr dual_momentum sma --ticker BTC-USD --test walk_forward --overlay

    # Multiple tickers
    python run_robustness.py --strategy atr --tickers BTC-USD ETH-USD SOL-USD --test walk_forward --overlay

Tests
-----
    walk_forward      : out-of-sample rolling windows — consistency check
    param_sensitivity : grid of params — are results stable across settings?
    monte_carlo       : shuffle trades — could this be luck?
    all               : run all three
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary
from utils.plotting     import block
from backtests.engine   import run_backtest

from robustness.walk_forward     import run as wf_run,   print_results as wf_print
from robustness.param_sensitivity import run as ps_run,  print_results as ps_print
from robustness.monte_carlo      import run as mc_run,   print_results as mc_print

# ── Strategy imports ───────────────────────────────────────────────────────────
from strategies.sma_crossover       import generate_signals as sma_signals,      get_params as sma_params,      STRATEGY_NAME as SMA_NAME
from strategies.ema_crossover       import generate_signals as ema_signals,      get_params as ema_params,      STRATEGY_NAME as EMA_NAME
from strategies.macd                import generate_signals as macd_signals,     get_params as macd_params,     STRATEGY_NAME as MACD_NAME
from strategies.donchian_breakout   import generate_signals as donchian_signals, get_params as donchian_params, STRATEGY_NAME as DONCHIAN_NAME
from strategies.rsi_mean_reversion  import generate_signals as rsi_signals,      get_params as rsi_params,      STRATEGY_NAME as RSI_NAME
from strategies.bollinger_reversion import generate_signals as boll_signals,     get_params as boll_params,     STRATEGY_NAME as BOLL_NAME
from strategies.zscore_reversion    import generate_signals as zscore_signals,   get_params as zscore_params,   STRATEGY_NAME as ZSCORE_NAME
from strategies.stochastic          import generate_signals as stoch_signals,    get_params as stoch_params,    STRATEGY_NAME as STOCH_NAME
from strategies.roc_momentum        import generate_signals as roc_signals,      get_params as roc_params,      STRATEGY_NAME as ROC_NAME
from strategies.dual_momentum       import generate_signals as dual_signals,     get_params as dual_params,     STRATEGY_NAME as DUAL_NAME
from strategies.atr_breakout        import generate_signals as atr_signals,      get_params as atr_params,      STRATEGY_NAME as ATR_NAME
from strategies.keltner_breakout    import generate_signals as keltner_signals,  get_params as keltner_params,  STRATEGY_NAME as KELTNER_NAME
from strategies.obv_trend           import generate_signals as obv_signals,      get_params as obv_params,      STRATEGY_NAME as OBV_NAME
from strategies.vwap_reversion      import generate_signals as vwap_signals,     get_params as vwap_params,     STRATEGY_NAME as VWAP_NAME
from strategies.kalman_mispricing   import generate_signals as kalman_signals,   get_params as kalman_params,   STRATEGY_NAME as KALMAN_NAME

STRATEGIES = {
    "sma":           (sma_signals,      sma_params,      SMA_NAME),
    "ema":           (ema_signals,      ema_params,      EMA_NAME),
    "macd":          (macd_signals,     macd_params,     MACD_NAME),
    "donchian":      (donchian_signals, donchian_params, DONCHIAN_NAME),
    "rsi":           (rsi_signals,      rsi_params,      RSI_NAME),
    "bollinger":     (boll_signals,     boll_params,     BOLL_NAME),
    "zscore":        (zscore_signals,   zscore_params,   ZSCORE_NAME),
    "stochastic":    (stoch_signals,    stoch_params,    STOCH_NAME),
    "roc":           (roc_signals,      roc_params,      ROC_NAME),
    "dual_momentum": (dual_signals,     dual_params,     DUAL_NAME),
    "atr":           (atr_signals,      atr_params,      ATR_NAME),
    "keltner":       (keltner_signals,  keltner_params,  KELTNER_NAME),
    "obv":           (obv_signals,      obv_params,      OBV_NAME),
    "vwap":          (vwap_signals,     vwap_params,     VWAP_NAME),
    "kalman":        (kalman_signals,   kalman_params,   KALMAN_NAME),
}

# ── Default parameter grids per strategy ──────────────────────────────────────
DEFAULT_PARAM_GRIDS = {
    "sma":          {"short_window": [10,15,20,25,30],  "long_window": [40,50,60,70,80]},
    "ema":          {"short_window": [8,10,12,15,20],   "long_window": [20,26,30,35,40]},
    "macd":         {"fast": [8,10,12,14,16],           "slow": [22,24,26,28,30],       "signal_period": [7,9,11]},
    "donchian":     {"window": [10,15,20,25,30,40,50]},
    "rsi":          {"rsi_period": [10,12,14,16,18],    "oversold": [25,28,30,32,35],   "overbought": [50,55,60,65,70]},
    "bollinger":    {"window": [15,20,25,30],            "num_std": [1.5,2.0,2.5,3.0]},
    "zscore":       {"window": [20,30,40,50],            "entry_z": [1.5,2.0,2.5,3.0],  "exit_z": [0.3,0.5,0.75]},
    "stochastic":   {"k_period": [10,14,18],             "oversold": [15,20,25],          "overbought": [75,80,85]},
    "roc":          {"window": [10,15,20,30,40],         "threshold": [0.0,0.5,1.0,2.0]},
    "dual_momentum":{"lookback": [126,180,252,365]},
    "atr":          {"atr_period": [10,14,20],           "multiplier": [1.5,2.0,2.5,3.0]},
    "keltner":      {"ema_period": [15,20,25],           "multiplier": [1.5,2.0,2.5,3.0]},
    "obv":          {"obv_ma_period": [10,15,20,25,30]},
    "vwap":         {"window": [10,15,20,30],            "entry_pct": [0.01,0.02,0.03,0.05]},
    "kalman":       {"entry_z": [1.0,1.2,1.5,1.8,2.0],  "exit_z": [0.2,0.3,0.5]},
}

# ── Style (mirrors plotting.py dark theme) ────────────────────────────────────
STYLE = {
    "bg":       "#0f0f0f", "panel_bg": "#161616",
    "text":     "#e0e0e0", "grid":     "#2a2a2a",
    "green":    "#00e676", "red":      "#ef5350",
    "blue":     "#4fc3f7", "purple":   "#ab47bc",
    "orange":   "#ffa726", "grey":     "#78909c",
}
COLORS = ["#4fc3f7","#ab47bc","#66bb6a","#ffa726","#ef5350","#26c6da"]


def _style_ax(fig, axes):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in axes:
        ax.set_facecolor(STYLE["panel_bg"])
        ax.tick_params(colors=STYLE["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(color=STYLE["grid"], linewidth=0.5, alpha=0.6)
        ax.title.set_color(STYLE["text"])
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        except Exception:
            pass


# ── Chart functions ────────────────────────────────────────────────────────────

def _plot_walk_forward(wf_results, base_df, title):
    fig = plt.figure(figsize=(14, 10), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                           height_ratios=[2, 1.5, 1])
    ax_eq     = fig.add_subplot(gs[0, :])
    ax_sharpe = fig.add_subplot(gs[1, 0])
    ax_ret    = fig.add_subplot(gs[1, 1])
    ax_cons   = fig.add_subplot(gs[2, :])
    _style_ax(fig, [ax_eq, ax_sharpe, ax_ret, ax_cons])

    # Full equity curve (in-sample) vs OOS chained
    if "equity_curve" in base_df.columns:
        ax_eq.plot(base_df.index, base_df["equity_curve"],
                   color=STYLE["grey"], linewidth=0.8, linestyle="--",
                   alpha=0.6, label="Full period (in-sample)")

    oos_eq = wf_results["oos_equity"]
    if not oos_eq.empty:
        ax_eq.plot(oos_eq.index, oos_eq.values,
                   color=STYLE["blue"], linewidth=1.2, label="OOS chained equity")

    # Shade test windows
    colors_w = [STYLE["green"], STYLE["orange"], STYLE["purple"],
                STYLE["blue"], STYLE["red"]]
    for i, w in enumerate(wf_results["window_results"]):
        c = colors_w[i % len(colors_w)]
        ax_eq.axvspan(w["test_start"], w["test_end"], alpha=0.08, color=c)

    ax_eq.axhline(1.0, color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_eq.set_ylabel("Equity", fontsize=8)
    ax_eq.set_title("In-Sample vs Out-of-Sample Equity  (shaded = test windows)", fontsize=9)
    ax_eq.legend(fontsize=7, facecolor=STYLE["panel_bg"],
                 labelcolor=STYLE["text"], edgecolor=STYLE["grid"])

    # Per-window Sharpe bar chart
    windows  = wf_results["window_results"]
    w_labels = [f"W{w['window']}" for w in windows]
    sharpes  = [w.get("Sharpe Ratio", 0) or 0 for w in windows]
    bar_colors = [STYLE["green"] if s > 0 else STYLE["red"] for s in sharpes]
    ax_sharpe.bar(w_labels, sharpes, color=bar_colors, alpha=0.8)
    ax_sharpe.axhline(0, color=STYLE["grid"], linewidth=0.8, linestyle="--")
    ax_sharpe.set_ylabel("OOS Sharpe", fontsize=8)
    ax_sharpe.set_title("OOS Sharpe per Window", fontsize=9)

    # Per-window return bar chart
    returns    = [w.get("Total Return", 0) or 0 for w in windows]
    bar_colors2 = [STYLE["green"] if r > 0 else STYLE["red"] for r in returns]
    ax_ret.bar(w_labels, [r*100 for r in returns], color=bar_colors2, alpha=0.8)
    ax_ret.axhline(0, color=STYLE["grid"], linewidth=0.8, linestyle="--")
    ax_ret.set_ylabel("OOS Return %", fontsize=8)
    ax_ret.set_title("OOS Return per Window", fontsize=9)
    ax_ret.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Summary text
    s   = wf_results["summary"]
    con = s["Consistency"]
    ax_cons.axis("off")
    summary_text = (
        f"Consistency: {con*100:.0f}%  ({s['Windows Profitable']}/{s['Windows Tested']} profitable)     "
        f"Avg OOS Sharpe: {s['Avg OOS Sharpe']:.3f}  ±{s['Std OOS Sharpe']:.3f}     "
        f"Avg OOS Return: {s['Avg OOS Return']*100:.1f}%     "
        f"Avg OOS MaxDD: {s['Avg OOS MaxDrawdown']*100:.1f}%"
    )
    verdict = "ROBUST" if con >= 0.6 and s["Avg OOS Sharpe"] > 0 else \
              "MARGINAL" if con >= 0.4 else "FRAGILE"
    color = STYLE["green"] if verdict == "ROBUST" else \
            STYLE["orange"] if verdict == "MARGINAL" else STYLE["red"]
    ax_cons.text(0.5, 0.65, summary_text, ha="center", va="center",
                 color=STYLE["text"], fontsize=8,
                 transform=ax_cons.transAxes)
    ax_cons.text(0.5, 0.25, f"Verdict: {verdict}", ha="center", va="center",
                 color=color, fontsize=12, fontweight="bold",
                 transform=ax_cons.transAxes)

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def _plot_param_sensitivity(ps_results, title):
    df          = ps_results["results_df"]
    param_names = ps_results["param_names"]
    metric      = ps_results["metric"]
    n_params    = len(param_names)

    fig = plt.figure(figsize=(14, 9), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)
    fig.patch.set_facecolor(STYLE["bg"])

    # Top row: one subplot per parameter
    n_top_cols = max(n_params, 2)
    gs_top = gridspec.GridSpec(1, n_top_cols, figure=fig,
                               top=0.88, bottom=0.52, hspace=0.5, wspace=0.35)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               top=0.44, bottom=0.08, hspace=0.5, wspace=0.35)

    for i, param in enumerate(param_names):
        ax = fig.add_subplot(gs_top[0, i % n_top_cols])
        _style_ax(fig, [ax])
        grouped = df.groupby(param)[metric].mean().reset_index()
        ax.bar(grouped[param].astype(str), grouped[metric],
               color=STYLE["blue"], alpha=0.8)
        ax.axhline(ps_results["metric_mean"], color=STYLE["orange"],
                   linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel(f"Avg {metric}", fontsize=8)
        ax.set_title(f"{param} sensitivity", fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax_dist = fig.add_subplot(gs_bot[0, 0])
    _style_ax(fig, [ax_dist])
    metric_vals = df[metric].dropna()
    n_bins = max(5, min(30, len(metric_vals) // 2 + 1))
    ax_dist.hist(metric_vals, bins=n_bins,
                 color=STYLE["blue"], alpha=0.7, edgecolor=STYLE["grid"])
    ax_dist.axvline(ps_results.get("actual_sharpe", ps_results["metric_max"]),
                    color=STYLE["green"], linewidth=1.5, linestyle="--", label="Best params")
    ax_dist.axvline(ps_results["metric_mean"], color=STYLE["orange"],
                    linewidth=1.5, linestyle="--", label="Mean")
    ax_dist.axvline(0, color=STYLE["red"], linewidth=1, linestyle="-", alpha=0.5)
    ax_dist.set_xlabel(metric, fontsize=8)
    ax_dist.set_ylabel("Count", fontsize=8)
    ax_dist.set_title(f"Distribution across {ps_results['n_combinations']} combos", fontsize=9)
    ax_dist.legend(fontsize=7, facecolor=STYLE["panel_bg"],
                   labelcolor=STYLE["text"], edgecolor=STYLE["grid"])

    ax_v = fig.add_subplot(gs_bot[0, 1])
    _style_ax(fig, [ax_v])
    ax_v.axis("off")
    pct     = ps_results["robustness_pct"]
    verdict = "ROBUST" if pct >= 0.7 and ps_results["metric_std"] < 0.3 else               "MODERATELY ROBUST" if pct >= 0.5 else "FRAGILE"
    color   = STYLE["green"] if verdict == "ROBUST" else               STYLE["orange"] if "MODERATE" in verdict else STYLE["red"]
    ax_v.text(0.5, 0.72, f"Verdict: {verdict}", ha="center", va="center",
              color=color, fontsize=12, fontweight="bold",
              transform=ax_v.transAxes)
    ax_v.text(0.5, 0.42,
              f"{pct*100:.0f}% of {ps_results['n_combinations']} combos profitable\n"
              f"Sharpe std: {ps_results['metric_std']:.3f}  (lower = more robust)\n"
              f"Best: {ps_results['metric_max']:.3f}  Mean: {ps_results['metric_mean']:.3f}",
              ha="center", va="center", color=STYLE["text"], fontsize=9,
              transform=ax_v.transAxes)

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def _plot_monte_carlo(mc_results, df, title):
    fig = plt.figure(figsize=(14, 10), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax_paths  = fig.add_subplot(gs[0, :])
    ax_sharpe = fig.add_subplot(gs[1, 0])
    ax_final  = fig.add_subplot(gs[1, 1])
    _style_ax(fig, [ax_paths, ax_sharpe, ax_final])

    # Equity path simulation — confidence bands
    returns = df["strategy_returns"].dropna()
    sim_paths = mc_results["sim_paths"]
    bands_sorted = sorted(sim_paths.items())

    # Fill between confidence bands
    qs = [q for q, _ in bands_sorted]
    if len(qs) >= 2:
        ax_paths.fill_between(
            sim_paths[qs[0]].index,
            sim_paths[qs[0]].values,
            sim_paths[qs[-1]].values,
            color=STYLE["blue"], alpha=0.15, label="5th–95th pct range"
        )
    if len(qs) >= 4:
        ax_paths.fill_between(
            sim_paths[qs[1]].index,
            sim_paths[qs[1]].values,
            sim_paths[qs[-2]].values,
            color=STYLE["blue"], alpha=0.25, label="25th–75th pct range"
        )

    # Median path
    if 0.5 in sim_paths:
        ax_paths.plot(sim_paths[0.5].index, sim_paths[0.5].values,
                      color=STYLE["blue"], linewidth=1.0, linestyle="--",
                      label="Median sim path", alpha=0.8)

    # Actual equity curve
    if "equity_curve" in df.columns:
        ax_paths.plot(df["equity_curve"].dropna().index,
                      df["equity_curve"].dropna().values,
                      color=STYLE["green"], linewidth=1.5, label="Actual strategy")

    ax_paths.axhline(1.0, color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_paths.set_ylabel("Equity", fontsize=8)
    ax_paths.set_title(f"Equity Path Simulation  ({mc_results['n_simulations']} paths)", fontsize=9)
    ax_paths.legend(fontsize=7, facecolor=STYLE["panel_bg"],
                    labelcolor=STYLE["text"], edgecolor=STYLE["grid"])

    # Shuffle Sharpe distribution
    shuffle_sharpes = mc_results["shuffle_sharpes"]
    actual_sharpe   = mc_results["actual_sharpe"]

    # Auto-detect degenerate case — low-frequency strategies produce near-identical
    # shuffled Sharpes (most returns are 0 when flat), so bins=50 crashes.
    sharpe_range = shuffle_sharpes.max() - shuffle_sharpes.min()
    n_bins = max(5, min(50, int(len(shuffle_sharpes) / 10))) if sharpe_range > 1e-6 else 1
    low_freq = sharpe_range < 1e-6

    ax_sharpe.hist(shuffle_sharpes, bins=n_bins, color=STYLE["purple"],
                   alpha=0.7, edgecolor=STYLE["grid"], label="Random shuffles")
    ax_sharpe.axvline(actual_sharpe, color=STYLE["green"], linewidth=2,
                      linestyle="--", label=f"Actual ({actual_sharpe:.3f})")
    ax_sharpe.axvline(0, color=STYLE["red"], linewidth=1, alpha=0.5)
    ax_sharpe.set_xlabel("Sharpe Ratio", fontsize=8)
    ax_sharpe.set_ylabel("Count", fontsize=8)
    ax_sharpe.set_title("Trade Shuffle Test — Sharpe Distribution", fontsize=9)
    ax_sharpe.legend(fontsize=7, facecolor=STYLE["panel_bg"],
                     labelcolor=STYLE["text"], edgecolor=STYLE["grid"])

    # Add p-value annotation
    pval  = mc_results["p_value_sharpe"]
    color = STYLE["green"] if pval < 0.05 else STYLE["red"]
    ax_sharpe.text(0.97, 0.95, f"p = {pval:.3f}\n{'SIGNIFICANT' if pval < 0.05 else 'NOT SIG.'}",
                   ha="right", va="top", transform=ax_sharpe.transAxes,
                   color=color, fontsize=8, fontweight="bold")

    if low_freq:
        ax_sharpe.text(0.5, 0.5,
                       "⚠ Low-frequency strategy\nShuffle test not reliable\n(<30 trades)",
                       ha="center", va="center", transform=ax_sharpe.transAxes,
                       color=STYLE["orange"], fontsize=8, alpha=0.9)

    # Final equity distribution
    sim_final = mc_results["sim_final_returns"]
    actual_eq = 1 + mc_results["actual_return"]
    final_range = sim_final.max() - sim_final.min()
    n_bins_final = max(5, min(50, int(len(sim_final) / 10))) if final_range > 1e-6 else 1
    ax_final.hist(sim_final, bins=n_bins_final, color=STYLE["orange"],
                  alpha=0.7, edgecolor=STYLE["grid"], label="Simulated final equity")
    ax_final.axvline(actual_eq, color=STYLE["green"], linewidth=2,
                     linestyle="--", label=f"Actual ({actual_eq:.2f}x)")
    ax_final.axvline(1.0, color=STYLE["red"], linewidth=1, alpha=0.5)
    ax_final.set_xlabel("Final Equity (x)", fontsize=8)
    ax_final.set_ylabel("Count", fontsize=8)
    ax_final.set_title("Distribution of Final Equity Outcomes", fontsize=9)
    ax_final.legend(fontsize=7, facecolor=STYLE["panel_bg"],
                    labelcolor=STYLE["text"], edgecolor=STYLE["grid"])

    rank = mc_results["actual_pct_rank"]
    ax_final.text(0.97, 0.95, f"Actual at\n{rank*100:.0f}th pct",
                  ha="right", va="top", transform=ax_final.transAxes,
                  color=STYLE["green"], fontsize=8)

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def _plot_overlay_wf(all_wf_results, title):
    """Overlay walk-forward OOS equity curves for multiple strategies/tickers."""
    fig = plt.figure(figsize=(14, 8), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax_eq   = fig.add_subplot(gs[0, :])
    ax_sr   = fig.add_subplot(gs[1, 0])
    ax_cons = fig.add_subplot(gs[1, 1])
    _style_ax(fig, [ax_eq, ax_sr, ax_cons])
    fig.patch.set_facecolor(STYLE["bg"])

    labels    = []
    sharpes   = []
    consist   = []

    for i, (label, wf_res) in enumerate(all_wf_results.items()):
        color  = COLORS[i % len(COLORS)]
        oos_eq = wf_res["oos_equity"]
        if not oos_eq.empty:
            ax_eq.plot(oos_eq.index, oos_eq.values,
                       color=color, linewidth=1.2, label=label)

        labels.append(label)
        sharpes.append(wf_res["summary"]["Avg OOS Sharpe"])
        consist.append(wf_res["summary"]["Consistency"] * 100)

    ax_eq.axhline(1.0, color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_eq.set_ylabel("OOS Equity", fontsize=8)
    ax_eq.set_title("Out-of-Sample Equity Curves", fontsize=9)
    ax_eq.legend(fontsize=7, facecolor=STYLE["panel_bg"],
                 labelcolor=STYLE["text"], edgecolor=STYLE["grid"])

    bar_colors = [STYLE["green"] if s > 0 else STYLE["red"] for s in sharpes]
    ax_sr.bar(labels, sharpes, color=bar_colors, alpha=0.8)
    ax_sr.axhline(0, color=STYLE["grid"], linewidth=0.8, linestyle="--")
    ax_sr.set_ylabel("Avg OOS Sharpe", fontsize=8)
    ax_sr.set_title("Avg OOS Sharpe Comparison", fontsize=9)
    plt.setp(ax_sr.xaxis.get_majorticklabels(), rotation=20, ha="right")

    ax_cons.bar(labels, consist, color=COLORS[:len(labels)], alpha=0.8)
    ax_cons.axhline(60, color=STYLE["orange"], linewidth=1,
                    linestyle="--", alpha=0.7, label="60% threshold")
    ax_cons.set_ylabel("Consistency %", fontsize=8)
    ax_cons.set_title("% Windows Profitable", fontsize=9)
    ax_cons.set_ylim(0, 100)
    plt.setp(ax_cons.xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.show(block=False)
    plt.pause(0.1)
    return fig


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_robustness(
    strategy_key, ticker, start, tests,
    end=None, source="yfinance", n_splits=5, n_simulations=1000,
    param_grid=None, show_chart=True, **strategy_kwargs
):
    """
    Full robustness pipeline for one strategy × one ticker.

    Args:
        strategy_key    : strategy key from STRATEGIES dict
        ticker          : ticker symbol
        start           : start date string
        tests           : list of test names e.g. ["walk_forward", "monte_carlo"]
                          or ["all"] to run everything
        end             : end date string (optional)
        source          : data source
        n_splits        : walk-forward windows (default 5)
        n_simulations   : Monte Carlo runs (default 1000)
        param_grid      : custom param grid for sensitivity test
        show_chart      : whether to pop up charts
        **strategy_kwargs : strategy parameters

    Returns:
        dict of {test_name: results}
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    if "all" in tests:
        tests = ["walk_forward", "param_sensitivity", "monte_carlo"]

    print(f"\n{'='*65}")
    print(f"  Robustness Tests  —  {strat_name}  [{ticker}]")
    print(f"  Tests: {', '.join(tests)}")
    print(f"{'='*65}")

    print(f"\n[robustness] Loading {ticker} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    print(f"[robustness] {len(data)} rows  ({data.index[0].date()} → {data.index[-1].date()})")

    # Base backtest for reference
    df_base = gen_signals(data.copy(), **strategy_kwargs)
    df_base = run_backtest(df_base)
    base_m  = calculate_metrics(df_base)
    print_summary(base_m, strategy_name=f"{strat_name}  [{ticker}]  (full period baseline)")

    all_results = {}

    # ── Walk-Forward ──────────────────────────────────────────────────────────
    if "walk_forward" in tests:
        print(f"\n[robustness] Running walk-forward ({n_splits} splits)...")
        wf_res = wf_run(data, gen_signals, n_splits=n_splits, **strategy_kwargs)
        wf_print(wf_res, strategy_name=f"{strat_name}  [{ticker}]")
        all_results["walk_forward"] = wf_res

        if show_chart:
            _plot_walk_forward(
                wf_res, df_base,
                title=f"Walk-Forward  |  {strat_name}  |  {ticker}"
            )

    # ── Parameter Sensitivity ─────────────────────────────────────────────────
    if "param_sensitivity" in tests:
        grid = param_grid or DEFAULT_PARAM_GRIDS.get(strategy_key, {})
        if not grid:
            print(f"[robustness] No param grid for {strategy_key} — skipping sensitivity")
        else:
            print(f"\n[robustness] Running parameter sensitivity...")
            ps_res = ps_run(data, gen_signals, grid)
            ps_res["actual_sharpe"] = base_m.get("Sharpe Ratio", np.nan)
            ps_print(ps_res, strategy_name=f"{strat_name}  [{ticker}]")
            all_results["param_sensitivity"] = ps_res

            if show_chart:
                _plot_param_sensitivity(
                    ps_res,
                    title=f"Parameter Sensitivity  |  {strat_name}  |  {ticker}"
                )

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    if "monte_carlo" in tests:
        print(f"\n[robustness] Running Monte Carlo ({n_simulations} simulations)...")
        mc_res = mc_run(df_base, n_simulations=n_simulations)
        mc_print(mc_res, strategy_name=f"{strat_name}  [{ticker}]")
        all_results["monte_carlo"] = mc_res

        if show_chart:
            _plot_monte_carlo(
                mc_res, df_base,
                title=f"Monte Carlo  |  {strat_name}  |  {ticker}"
            )

    return all_results


# ── Multi-run pipelines ────────────────────────────────────────────────────────

def run_multi_strategy_robustness(
    strategy_keys, ticker, start, tests, overlay=False, **kwargs
):
    """Multiple strategies × one ticker."""
    print(f"\n[robustness] Multi-strategy: {len(strategy_keys)} strategies on {ticker}")
    all_wf = {}
    for key in strategy_keys:
        res = run_robustness(key, ticker, start, tests,
                             show_chart=not overlay, **kwargs)
        label = STRATEGIES[key][2]
        if "walk_forward" in res:
            all_wf[label] = res["walk_forward"]

    if overlay and "walk_forward" in tests and all_wf:
        _plot_overlay_wf(all_wf,
                         title=f"Walk-Forward Comparison  |  {ticker}")
    return all_wf


def run_multi_ticker_robustness(
    strategy_key, tickers, start, tests, overlay=False, **kwargs
):
    """One strategy × multiple tickers."""
    strat_name = STRATEGIES[strategy_key][2]
    print(f"\n[robustness] Multi-ticker: {strat_name} on {len(tickers)} tickers")
    all_wf = {}
    for ticker in tickers:
        res = run_robustness(strategy_key, ticker, start, tests,
                             show_chart=not overlay, **kwargs)
        if "walk_forward" in res:
            all_wf[ticker] = res["walk_forward"]

    if overlay and "walk_forward" in tests and all_wf:
        _plot_overlay_wf(all_wf,
                         title=f"Walk-Forward Comparison  |  {strat_name}")
    return all_wf


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaByProcess — Robustness Testing Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_robustness.py --strategy atr --ticker BTC-USD --start 2018-01-01 --test all
  python run_robustness.py --strategy rsi --ticker BTC-USD --test walk_forward param_sensitivity
  python run_robustness.py --strategy dual_momentum --ticker BTC-USD --test monte_carlo
  python run_robustness.py --strategies atr dual_momentum sma --ticker BTC-USD --test walk_forward --overlay
  python run_robustness.py --strategy atr --tickers BTC-USD ETH-USD --test walk_forward --overlay
        """
    )

    strat_group = parser.add_mutually_exclusive_group()
    strat_group.add_argument("--strategy",   default=None,      help="Single strategy key")
    strat_group.add_argument("--strategies", nargs="+",         help="Multiple strategy keys")

    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker",    default=None,      help="Single ticker")
    ticker_group.add_argument("--tickers",   nargs="+",         help="Multiple tickers")

    parser.add_argument("--test",     nargs="+", required=True,
                        help="Tests to run: walk_forward param_sensitivity monte_carlo all")
    parser.add_argument("--start",    default="2018-01-01")
    parser.add_argument("--end",      default=None)
    parser.add_argument("--source",   default="yfinance")
    parser.add_argument("--splits",   default=5,    type=int,   help="Walk-forward splits")
    parser.add_argument("--sims",     default=1000, type=int,   help="Monte Carlo simulations")
    parser.add_argument("--overlay",  action="store_true",      help="Overlay results on one chart")

    parser.add_argument("--obs-noise",  default=1.0,  type=float)
    parser.add_argument("--proc-noise", default=0.01, type=float)
    parser.add_argument("--entry-z",    default=1.5,  type=float)
    parser.add_argument("--exit-z",     default=0.3,  type=float)
    parser.add_argument("--stop-z",     default=3.5,  type=float)

    args = parser.parse_args()

    strategies = args.strategies or ([args.strategy] if args.strategy else ["sma"])
    tickers    = args.tickers    or ([args.ticker]    if args.ticker    else ["BTC-USD"])

    strategy_kwargs = {}
    if "kalman" in strategies:
        strategy_kwargs = dict(
            obs_noise_var=args.obs_noise, proc_noise_var=args.proc_noise,
            entry_z=args.entry_z, exit_z=args.exit_z, stop_loss_z=args.stop_z,
        )

    common = dict(start=args.start, end=args.end, source=args.source,
                  n_splits=args.splits, n_simulations=args.sims,
                  **strategy_kwargs)

    if len(strategies) == 1 and len(tickers) == 1:
        run_robustness(strategies[0], tickers[0], tests=args.test, **common)

    elif len(strategies) > 1 and len(tickers) == 1:
        run_multi_strategy_robustness(
            strategies, tickers[0], tests=args.test,
            overlay=args.overlay, **common
        )

    elif len(strategies) == 1 and len(tickers) > 1:
        run_multi_ticker_robustness(
            strategies[0], tickers, tests=args.test,
            overlay=args.overlay, **common
        )

    else:
        for ticker in tickers:
            run_multi_strategy_robustness(
                strategies, ticker, tests=args.test,
                overlay=args.overlay, **common
            )

    block()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 65)
        print("  AlphaByProcess — Robustness Demo")
        print("  ATR Breakout on BTC-USD 2018→2024")
        print("=" * 65)

        run_robustness("atr", "BTC-USD", "2018-01-01",
                       tests=["walk_forward", "param_sensitivity", "monte_carlo"])
        block()
    else:
        main()

        