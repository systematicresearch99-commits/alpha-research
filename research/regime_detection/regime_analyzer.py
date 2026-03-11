"""
research/regime_detection/regime_analyzer.py
=============================================
Slices backtest results by regime and reuses the framework's
existing calculate_metrics() to answer:

  "Do strategies perform differently across regimes?"

This module is purely additive — it never modifies engine.py,
performance.py, or any existing framework file. It takes the
output DataFrame that run_backtest() already returns and adds
a 'regime' column, then re-runs metrics per regime slice.

Usage
-----
    from research.regime_detection.regime_analyzer import RegimeAnalyzer

    analyzer = RegimeAnalyzer(detector, index_prices)
    results  = analyzer.analyze(df_backtest, prices)   # df from run_backtest()
    analyzer.print_regime_report(results)
"""

import numpy as np
import pandas as pd

from utils.performance import calculate_metrics
from research.regime_detection.hmm_model import RegimeDetector, REGIME_ORDER


class RegimeAnalyzer:
    """
    Attaches regime labels to a backtest DataFrame and
    computes per-regime performance metrics.

    Parameters
    ----------
    detector      : fitted RegimeDetector instance
    index_prices  : pd.Series — benchmark close prices used for feature computation
    """

    def __init__(self, detector: RegimeDetector, index_prices: pd.Series):
        if not detector._is_fitted:
            raise ValueError("RegimeDetector must be fitted before passing to RegimeAnalyzer.")
        self.detector     = detector
        self.index_prices = index_prices

    def attach_regimes(self, df: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """
        Add 'regime' column to a backtest DataFrame.

        Parameters
        ----------
        df     : output of run_backtest() — must have strategy_returns, equity_curve, position
        prices : close prices for the same asset (used by feature engine)

        Returns
        -------
        df with 'regime' column added (rows before feature warmup period are NaN)
        """
        regimes = self.detector.predict(prices, self.index_prices)

        df = df.copy()
        df["regime"] = regimes  # aligns on index; pre-warmup rows get NaN
        return df

    def analyze(self, df: pd.DataFrame, prices: pd.Series) -> dict:
        """
        Full regime analysis pipeline.

        1. Attach regime labels to backtest df
        2. Split df into per-regime slices
        3. Run calculate_metrics() on each slice (reuses framework's metric engine)
        4. Also compute overall metrics for comparison baseline

        Parameters
        ----------
        df     : output of run_backtest()
        prices : close prices of the asset

        Returns
        -------
        dict with keys:
            'overall'     : metrics dict for full period
            'by_regime'   : {regime_name: metrics_dict, ...}
            'df_labeled'  : df with 'regime' column attached
            'regime_summary' : DataFrame of time-in-regime stats
        """
        df_labeled = self.attach_regimes(df, prices)

        # Drop warmup rows that don't have a regime yet
        df_clean = df_labeled.dropna(subset=["regime"])

        # Overall baseline (same as a normal run_strategy call)
        overall_metrics = calculate_metrics(df_clean)

        # Per-regime metrics
        by_regime = {}
        for regime in REGIME_ORDER:
            slice_df = df_clean[df_clean["regime"] == regime].copy()

            if len(slice_df) < 5:
                # Not enough data in this regime to compute meaningful metrics
                by_regime[regime] = {"note": f"Too few observations ({len(slice_df)} days)"}
                continue

            # Re-base equity curve for this slice so metrics are self-contained
            slice_df = _rebase_equity(slice_df)
            by_regime[regime] = calculate_metrics(slice_df)

        # Regime time stats
        regime_summary = self.detector.regime_summary(
            df_labeled["regime"].dropna(), prices
        )

        return {
            "overall":        overall_metrics,
            "by_regime":      by_regime,
            "df_labeled":     df_labeled,
            "regime_summary": regime_summary,
        }

    def print_regime_report(self, results: dict, strategy_name: str = "Strategy"):
        """
        Pretty-print the regime analysis results to console.
        Mirrors the print_summary() style from performance.py.
        """
        COL_W  = 14   # fixed width for every data column
        MET_W  = 24   # fixed width for metric label column

        regimes = [r for r in REGIME_ORDER if r in results["by_regime"]]
        n_cols  = 1 + len(regimes)          # Overall + one per regime
        line_w  = MET_W + 2 + COL_W * n_cols
        line    = "─" * line_w
        dline   = "═" * line_w

        print(f"\n{dline}")
        print(f"  REGIME ANALYSIS  —  {strategy_name}")
        print(dline)

        # ── Time distribution ─────────────────────────────────────────────────
        print("\n  Time Distribution")
        print(line)
        print(results["regime_summary"].to_string())

        # ── Transition matrix ─────────────────────────────────────────────────
        print(f"\n  HMM Transition Matrix")
        print(line)
        print(self.detector.get_transition_matrix().to_string())

        # ── Performance table ─────────────────────────────────────────────────
        key_metrics = [
            "Total Return", "Annualized Return", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Win Rate", "Num Trades",
        ]

        print(f"\n  Performance by Regime")
        print(line)

        # Header — all columns built the same way, no special-casing
        cols    = ["Overall"] + regimes
        header  = f"  {'Metric':<{MET_W}}" + "".join(f"{c:>{COL_W}}" for c in cols)
        print(header)
        print(line)

        # Data rows — every cell uses the same rjust(COL_W)
        for metric in key_metrics:
            row = f"  {metric:<{MET_W}}"

            # Overall
            row += _fmt_metric(metric, results["overall"].get(metric)).rjust(COL_W)

            # Per-regime
            for regime in regimes:
                rm = results["by_regime"][regime]
                if "note" in rm:
                    row += "—".rjust(COL_W)
                else:
                    row += _fmt_metric(metric, rm.get(metric)).rjust(COL_W)

            print(row)

        print(f"\n{dline}\n")

    def compare_strategies(
        self,
        results_map: dict,
    ):
        """
        Compare multiple strategy results side-by-side per regime.

        Parameters
        ----------
        results_map : {strategy_name: results_dict}
                      where each results_dict is from analyze()

        Prints a table: rows = regimes, cols = strategies, cell = Sharpe
        """
        line  = "─" * 60
        dline = "═" * 60

        strategies = list(results_map.keys())
        print(f"\n{dline}")
        print("  STRATEGY vs REGIME COMPARISON  (Sharpe Ratio)")
        print(dline)

        header = f"  {'Regime':<14}" + "".join(f"{s:>14}" for s in strategies)
        print(header)
        print(line)

        for regime in REGIME_ORDER:
            row = f"  {regime:<14}"
            for strat in strategies:
                rm = results_map[strat]["by_regime"].get(regime, {})
                if "note" in rm or not rm:
                    row += "      —       "
                else:
                    sharpe = rm.get("Sharpe Ratio", np.nan)
                    row += f"{sharpe:>14.3f}" if not np.isnan(sharpe) else "      —       "
            print(row)

        print(line)
        print("  Overall")
        row = f"  {'(full period)':<14}"
        for strat in strategies:
            sharpe = results_map[strat]["overall"].get("Sharpe Ratio", np.nan)
            row += f"{sharpe:>14.3f}" if not np.isnan(sharpe) else "      —       "
        print(row)
        print(f"\n{dline}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rebase_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebase equity_curve and buy_hold_equity to start at 1.0 for a slice.
    This makes calculate_metrics() return regime-relative returns, not
    cumulative returns from the start of the full backtest.
    """
    df = df.copy()
    if "equity_curve" in df.columns and not df["equity_curve"].empty:
        start = df["equity_curve"].iloc[0]
        if start and start != 0:
            df["equity_curve"] = df["equity_curve"] / start
    if "buy_hold_equity" in df.columns and not df["buy_hold_equity"].empty:
        start = df["buy_hold_equity"].iloc[0]
        if start and start != 0:
            df["buy_hold_equity"] = df["buy_hold_equity"] / start
    return df


def _fmt_metric(metric: str, value) -> str:
    """Format a metric value as % or float for display, matching performance.py style."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    pct_metrics = [
        "Total Return", "Annualized Return", "Annualized Vol",
        "Max Drawdown", "Win Rate", "Avg Win", "Avg Loss",
    ]
    if isinstance(value, float):
        if any(m in metric for m in pct_metrics):
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"
    return str(value)


    