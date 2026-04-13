"""
utils/plotting.py
-----------------
Charting utilities for AlphaByProcess.

Every run produces a single figure with 4 panels:
  1. Price chart with buy/sell signal markers
  2. Equity curve
  3. Drawdown curve
  4. Rolling Sharpe (left) | Position over time (right)

For run_risk.py compare mode an extra panel is added:
  5. All equity curves overlaid

All charts are non-blocking — execution continues after plt.show().
Charts are never saved to disk.

Usage:
    from utils.plotting import plot_run, plot_compare

    # Single run
    plot_run(df, title="MACD on SPY 2015-2024")

    # Compare mode (risk runner)
    plot_compare(results_dict, base_df, title="MACD SPY — Risk Module Comparison")
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Non-interactive backend fallback — switch to TkAgg or Qt if available
try:
    matplotlib.use("TkAgg")
except Exception:
    pass


# ── Style ──────────────────────────────────────────────────────────────────────
STYLE = {
    "bg":          "#0f0f0f",
    "panel_bg":    "#161616",
    "text":        "#e0e0e0",
    "grid":        "#2a2a2a",
    "price":       "#8888aa",
    "equity":      "#4fc3f7",
    "baseline":    "#78909c",
    "drawdown":    "#ef5350",
    "sharpe":      "#ab47bc",
    "position":    "#66bb6a",
    "buy":         "#00e676",
    "sell":        "#ff1744",
    "flat":        "#546e7a",
}

COMPARE_COLORS = [
    "#4fc3f7", "#ab47bc", "#66bb6a", "#ffa726",
    "#ef5350", "#26c6da", "#d4e157", "#ec407a",
]


def _apply_style(fig, axes):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in axes:
        ax.set_facecolor(STYLE["panel_bg"])
        ax.tick_params(colors=STYLE["text"], labelsize=8)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(color=STYLE["grid"], linewidth=0.5, alpha=0.7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _plot_price_signals(ax, df):
    """Panel 1: Price with buy/sell markers."""
    ax.plot(df.index, df["Close"], color=STYLE["price"], linewidth=0.8, label="Price")

    # Detect position changes for entry/exit markers
    pos      = df["position"].fillna(0)
    prev_pos = pos.shift(1).fillna(0)
    entries  = df.index[((prev_pos == 0) | (prev_pos == -1)) & (pos == 1)]
    exits    = df.index[((prev_pos == 0) | (prev_pos == 1))  & (pos == -1)]
    closes   = df.index[(prev_pos != 0) & (pos == 0)]

    if len(entries) > 0:
        ax.scatter(entries, df.loc[entries, "Close"],
                   marker="^", color=STYLE["buy"],  s=35, zorder=5, label="Long entry")
    if len(exits) > 0:
        ax.scatter(exits,   df.loc[exits,   "Close"],
                   marker="v", color=STYLE["sell"], s=35, zorder=5, label="Short entry")
    if len(closes) > 0:
        ax.scatter(closes,  df.loc[closes,  "Close"],
                   marker="x", color=STYLE["flat"], s=20, zorder=5, label="Exit")

    ax.set_ylabel("Price", fontsize=8)
    ax.legend(fontsize=7, loc="upper left",
              facecolor=STYLE["panel_bg"], labelcolor=STYLE["text"],
              edgecolor=STYLE["grid"])


def _plot_equity(ax, df, label="Strategy", color=None):
    """Panel 2: Equity curve."""
    color = color or STYLE["equity"]
    ax.plot(df.index, df["equity_curve"], color=color, linewidth=1.2, label=label)
    ax.axhline(1.0, color=STYLE["grid"], linewidth=0.8, linestyle="--")
    ax.set_ylabel("Equity", fontsize=8)
    ax.legend(fontsize=7, loc="upper left",
              facecolor=STYLE["panel_bg"], labelcolor=STYLE["text"],
              edgecolor=STYLE["grid"])


def _plot_drawdown(ax, df):
    """Panel 3: Drawdown curve."""
    equity   = df["equity_curve"]
    drawdown = (equity / equity.cummax() - 1) * 100
    ax.fill_between(df.index, drawdown, 0,
                    color=STYLE["drawdown"], alpha=0.5, linewidth=0)
    ax.plot(df.index, drawdown, color=STYLE["drawdown"], linewidth=0.8)
    ax.set_ylabel("Drawdown %", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))


def _plot_rolling_sharpe(ax, df, window=60):
    """Panel 4a: Rolling Sharpe ratio."""
    ret    = df["strategy_returns"].fillna(0)
    roll_s = (ret.rolling(window).mean() * 252) / \
             (ret.rolling(window).std() * np.sqrt(252))
    ax.plot(df.index, roll_s, color=STYLE["sharpe"], linewidth=0.8)
    ax.axhline(0, color=STYLE["grid"], linewidth=0.8, linestyle="--")
    ax.axhline(1, color=STYLE["sharpe"], linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylabel(f"Sharpe ({window}d)", fontsize=8)


def _plot_position(ax, df):
    """Panel 4b: Position over time."""
    pos_col = "sized_position" if "sized_position" in df.columns else "position"
    pos     = df[pos_col].fillna(0)
    ax.fill_between(df.index, pos, 0,
                    where=(pos > 0), color=STYLE["buy"],  alpha=0.5, linewidth=0)
    ax.fill_between(df.index, pos, 0,
                    where=(pos < 0), color=STYLE["sell"], alpha=0.5, linewidth=0)
    ax.plot(df.index, pos, color=STYLE["position"], linewidth=0.6)
    ax.axhline(0, color=STYLE["grid"], linewidth=0.6)
    ax.set_ylabel("Position", fontsize=8)


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_run(df, title="Strategy"):
    """
    Single-run chart — 4 panels.

    Args:
        df    : DataFrame after backtest with equity_curve, strategy_returns,
                position, Close columns
        title : window title and suptitle
    """
    fig = plt.figure(figsize=(14, 10), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(
        4, 2,
        figure     = fig,
        hspace     = 0.45,
        wspace     = 0.25,
        height_ratios = [2, 1.5, 1, 1],
    )

    ax_price  = fig.add_subplot(gs[0, :])
    ax_equity = fig.add_subplot(gs[1, :])
    ax_dd     = fig.add_subplot(gs[2, :])
    ax_sharpe = fig.add_subplot(gs[3, 0])
    ax_pos    = fig.add_subplot(gs[3, 1])

    _apply_style(fig, [ax_price, ax_equity, ax_dd, ax_sharpe, ax_pos])

    _plot_price_signals(ax_price,  df)
    _plot_equity(ax_equity,        df)
    _plot_drawdown(ax_dd,          df)
    _plot_rolling_sharpe(ax_sharpe, df)
    _plot_position(ax_pos,         df)

    ax_price.set_title("Price  +  Signals",  fontsize=9)
    ax_equity.set_title("Equity Curve",      fontsize=9)
    ax_dd.set_title("Drawdown",              fontsize=9)
    ax_sharpe.set_title("Rolling Sharpe",    fontsize=9)
    ax_pos.set_title("Position",             fontsize=9)

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def plot_compare(results_dict, base_df, title="Risk Module Comparison"):
    """
    Compare mode chart — 5 panels (standard 4 + overlaid equity curves).

    Args:
        results_dict : {module_key: df} from apply_compare()
        base_df      : base backtest df (no risk applied)
        title        : window title
    """
    # Use base_df for price/drawdown/sharpe/position panels
    fig = plt.figure(figsize=(14, 13), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(
        5, 2,
        figure        = fig,
        hspace        = 0.50,
        wspace        = 0.25,
        height_ratios = [2, 1.5, 1.5, 1, 1],
    )

    ax_price   = fig.add_subplot(gs[0, :])
    ax_compare = fig.add_subplot(gs[1, :])
    ax_dd      = fig.add_subplot(gs[2, :])
    ax_sharpe  = fig.add_subplot(gs[3, 0])
    ax_pos     = fig.add_subplot(gs[3, 1])

    _apply_style(fig, [ax_price, ax_compare, ax_dd, ax_sharpe, ax_pos])

    # Panel 1: price + signals from base
    _plot_price_signals(ax_price, base_df)

    # Panel 2: overlaid equity curves
    ax_compare.plot(base_df.index, base_df["equity_curve"],
                    color=STYLE["baseline"], linewidth=1.0,
                    linestyle="--", label="Baseline (no risk)", alpha=0.8)

    for i, (key, df) in enumerate(results_dict.items()):
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        label = key.replace("_", " ").title()
        ax_compare.plot(df.index, df["equity_curve"],
                        color=color, linewidth=1.2, label=label)

    ax_compare.axhline(1.0, color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_compare.set_ylabel("Equity", fontsize=8)
    ax_compare.legend(fontsize=7, loc="upper left",
                      facecolor=STYLE["panel_bg"], labelcolor=STYLE["text"],
                      edgecolor=STYLE["grid"])

    # Panels 3-5: use base_df for context
    _plot_drawdown(ax_dd,           base_df)
    _plot_rolling_sharpe(ax_sharpe, base_df)
    _plot_position(ax_pos,          base_df)

    ax_price.set_title("Price  +  Signals",          fontsize=9)
    ax_compare.set_title("Equity Curves (overlaid)",  fontsize=9)
    ax_dd.set_title("Drawdown  (baseline)",           fontsize=9)
    ax_sharpe.set_title("Rolling Sharpe  (baseline)", fontsize=9)
    ax_pos.set_title("Position  (baseline)",          fontsize=9)

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def plot_regime_run(df, results, title="Regime Analysis"):
    """
    Regime-aware chart — standard 4 panels + regime shading on price and equity.

    Args:
        df      : DataFrame with equity_curve, position, Close, regime columns
        results : regime_results dict from RegimeAnalyzer.analyze()
        title   : window title
    """
    REGIME_COLORS = ["#1a237e", "#1b5e20", "#e65100", "#b71c1c"]
    REGIME_ALPHA  = 0.15

    fig = plt.figure(figsize=(14, 10), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(
        4, 2,
        figure        = fig,
        hspace        = 0.45,
        wspace        = 0.25,
        height_ratios = [2, 1.5, 1, 1],
    )

    ax_price  = fig.add_subplot(gs[0, :])
    ax_equity = fig.add_subplot(gs[1, :])
    ax_dd     = fig.add_subplot(gs[2, :])
    ax_sharpe = fig.add_subplot(gs[3, 0])
    ax_pos    = fig.add_subplot(gs[3, 1])

    _apply_style(fig, [ax_price, ax_equity, ax_dd, ax_sharpe, ax_pos])

    # Shade regimes on price and equity panels
    if "regime" in df.columns:
        unique_regimes = sorted(df["regime"].dropna().unique(), key=str)
        regime_color_map = {r: REGIME_COLORS[i % len(REGIME_COLORS)]
                            for i, r in enumerate(unique_regimes)}

        for regime_id in unique_regimes:
            mask  = df["regime"] == regime_id
            color = regime_color_map[regime_id]
            for ax in [ax_price, ax_equity]:
                ax.fill_between(df.index, 0, 1,
                                where=mask, color=color, alpha=REGIME_ALPHA,
                                transform=ax.get_xaxis_transform())

        # Regime legend
        legend_elements = [
            Line2D([0], [0], color=regime_color_map[r],
                   lw=6, alpha=0.5, label=str(r))
            for r in unique_regimes
        ]
        ax_price.legend(handles=legend_elements, fontsize=7, loc="upper left",
                        facecolor=STYLE["panel_bg"], labelcolor=STYLE["text"],
                        edgecolor=STYLE["grid"])

    _plot_price_signals(ax_price,   df)
    _plot_equity(ax_equity,         df)
    _plot_drawdown(ax_dd,           df)
    _plot_rolling_sharpe(ax_sharpe, df)
    _plot_position(ax_pos,          df)

    ax_price.set_title("Price  +  Signals  +  Regimes", fontsize=9)
    ax_equity.set_title("Equity Curve",                  fontsize=9)
    ax_dd.set_title("Drawdown",                          fontsize=9)
    ax_sharpe.set_title("Rolling Sharpe",                fontsize=9)
    ax_pos.set_title("Position",                         fontsize=9)

    plt.show(block=False)
    plt.pause(0.1)
    return fig


def block():
    """
    Call at the end of a script to keep all non-blocking charts open.
    Without this, charts close immediately when the script exits.

    Usage:
        from utils.plotting import block
        # ... all your run_strategy() calls ...
        block()   # keeps windows open until user closes them
    """
    plt.show(block=True)


def plot_overlay(results_dict, title="Strategy Comparison"):
    """
    Overlay chart — multiple strategies or tickers on one figure.
    4 panels: overlaid equity curves, overlaid drawdowns,
    overlaid rolling Sharpe, overlaid positions + summary table.

    Args:
        results_dict : {label: df} — label is e.g. "MACD | HDFCBANK.NS"
        title        : window title
    """
    fig = plt.figure(figsize=(14, 12), num=title)
    fig.suptitle(title, color=STYLE["text"], fontsize=11, y=0.98)

    gs = gridspec.GridSpec(
        4, 2,
        figure        = fig,
        hspace        = 0.45,
        wspace        = 0.25,
        height_ratios = [2, 1.5, 1, 1],
    )

    ax_equity = fig.add_subplot(gs[0, :])
    ax_dd     = fig.add_subplot(gs[1, :])
    ax_sharpe = fig.add_subplot(gs[2, 0])
    ax_pos    = fig.add_subplot(gs[2, 1])
    ax_table  = fig.add_subplot(gs[3, :])

    _apply_style(fig, [ax_equity, ax_dd, ax_sharpe, ax_pos, ax_table])

    for i, (label, df) in enumerate(results_dict.items()):
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]

        ax_equity.plot(df.index, df["equity_curve"],
                       color=color, linewidth=1.2, label=label)

        equity   = df["equity_curve"]
        drawdown = (equity / equity.cummax() - 1) * 100
        ax_dd.plot(df.index, drawdown, color=color, linewidth=0.8, alpha=0.8)
        ax_dd.fill_between(df.index, drawdown, 0, color=color, alpha=0.08)

        ret    = df["strategy_returns"].fillna(0)
        roll_s = (ret.rolling(60).mean() * 252) / (ret.rolling(60).std() * np.sqrt(252))
        ax_sharpe.plot(df.index, roll_s, color=color, linewidth=0.8)

        pos_col = "sized_position" if "sized_position" in df.columns else "position"
        ax_pos.plot(df.index, df[pos_col].fillna(0),
                    color=color, linewidth=0.6, alpha=0.8)

    ax_equity.axhline(1.0, color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_dd.axhline(0,       color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_sharpe.axhline(0,   color=STYLE["grid"], linewidth=0.6, linestyle="--")
    ax_sharpe.axhline(1,   color=STYLE["grid"], linewidth=0.4, linestyle=":", alpha=0.5)
    ax_pos.axhline(0,      color=STYLE["grid"], linewidth=0.6)

    ax_equity.set_ylabel("Equity",       fontsize=8)
    ax_dd.set_ylabel("Drawdown %",       fontsize=8)
    ax_sharpe.set_ylabel("Sharpe (60d)", fontsize=8)
    ax_pos.set_ylabel("Position",        fontsize=8)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    ax_equity.set_title("Equity Curves",  fontsize=9)
    ax_dd.set_title("Drawdowns",          fontsize=9)
    ax_sharpe.set_title("Rolling Sharpe", fontsize=9)
    ax_pos.set_title("Position",          fontsize=9)

    ax_equity.legend(fontsize=7, loc="upper left",
                     facecolor=STYLE["panel_bg"], labelcolor=STYLE["text"],
                     edgecolor=STYLE["grid"])

    # Summary metrics table
    from utils.performance import calculate_metrics
    rows = []
    cols = ["Label", "Sharpe", "Return", "MaxDD", "WinRate", "Trades"]
    for label, df in results_dict.items():
        m = calculate_metrics(df)
        rows.append([
            label,
            f"{m.get('Sharpe Ratio', float('nan')):.3f}",
            f"{m.get('Total Return', 0)*100:.1f}%",
            f"{m.get('Max Drawdown', 0)*100:.1f}%",
            f"{m.get('Win Rate', 0)*100:.1f}%",
            str(m.get('Num Trades', 0)),
        ])

    ax_table.axis("off")
    tbl = ax_table.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(STYLE["panel_bg"] if r > 0 else STYLE["grid"])
        cell.set_edgecolor(STYLE["grid"])
        cell.set_text_props(color=STYLE["text"])

    plt.show(block=False)
    plt.pause(0.1)
    return fig


