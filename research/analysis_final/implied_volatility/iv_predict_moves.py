"""
iv_predict_moves.py — Does Implied Volatility Predict Extreme Moves?
=====================================================================
Research script for @AlphaByProcess.
Tests whether elevated VIX (implied volatility) predicts subsequent
large S&P 500 moves, or merely reflects already-realized volatility.

Steps:
    1. Load data  — ^GSPC, ^VIX, SPY, TLT, VIX9D (optional)
    2. Define IV flags — absolute (VIX > 25) and relative (VIX > 1.5× 60d mean)
    3. Event study — mean |CAR| at t+1, t+5, t+20
    4. Conditional probability — P(|SP500| > threshold | IV_flag) vs baseline
    5. Lead/lag analysis — VIX in [-10, +10] window around large moves
    6. Term structure slope — VIX9D/VIX backwardation signal (if available)
    7. Regime distribution — IV flag events by HMM regime
    8. Realized vs Implied vol — post-flag realized vol vs VIX level

Usage:
    python research/analysis/iv_predict_moves/iv_predict_moves.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

# Try framework imports — fall back to standalone mode gracefully
try:
    from utils.data_loader import load_data
    from utils.performance import calculate_metrics, print_summary
    from backtests.engine  import run_backtest
    _FRAMEWORK = True
except ImportError:
    _FRAMEWORK = False
    print("[init] Framework utils not found — running in standalone mode (yfinance direct).")

# ── Config ────────────────────────────────────────────────────────────────────
START           = "2004-01-01"
END             = None               # today
IV_ABS_THRESH   = 25                 # VIX > 25 = absolute spike
IV_REL_MULT     = 1.5                # VIX > 1.5× 60d mean = relative spike
IV_ROLL_WINDOW  = 60                 # rolling mean window for relative threshold
MOVE_THRESHOLDS = [0.015, 0.025]     # |return| > 1.5% and 2.5%
EXCLUSION_DAYS  = 5                  # min gap between IV flag events
HORIZONS        = [1, 5, 20]        # forward return windows (days)
BOOTSTRAP_N     = 1000
LEADLAG_WINDOW  = 10                 # days before/after large move

FINDINGS_DIR = os.path.join(os.path.dirname(__file__), "findings")


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def _yf_load(ticker, start, end):
    """Thin wrapper — download a single ticker via yfinance, return Close series."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    # yfinance multi-level columns after 0.2.x
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Close"]].rename(columns={"Close": ticker})


def load_all_data(start=START, end=END):
    """
    Load ^GSPC (S&P 500), ^VIX, SPY, TLT.
    Attempt VIX9D — note availability starts ~2011.
    Returns merged DataFrame aligned on trading dates.
    """
    print("[data] Loading ^GSPC...")
    if _FRAMEWORK:
        sp500 = load_data("^GSPC", start=start, end=end, source="yfinance", ohlcv=False)
        sp500.columns = ["Close"]
    else:
        sp500 = _yf_load("^GSPC", start, end).rename(columns={"^GSPC": "Close"})

    print("[data] Loading ^VIX...")
    if _FRAMEWORK:
        vix = load_data("^VIX", start=start, end=end, source="yfinance", ohlcv=False)
        vix.columns = ["VIX"]
    else:
        vix = _yf_load("^VIX", start, end).rename(columns={"^VIX": "VIX"})

    print("[data] Loading TLT (HMM benchmark)...")
    if _FRAMEWORK:
        tlt = load_data("TLT", start=start, end=end, source="yfinance", ohlcv=False)
        tlt.columns = ["TLT"]
    else:
        tlt = _yf_load("TLT", start, end).rename(columns={"TLT": "TLT"})

    # Merge on shared trading days
    df = sp500.join(vix, how="inner").join(tlt, how="left")
    df["sp500_ret"] = df["Close"].pct_change()
    df["abs_ret"]   = df["sp500_ret"].abs()
    df.dropna(subset=["sp500_ret", "VIX"], inplace=True)

    print(f"[data] Base data: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    # Attempt VIX9D — shorter history, optional
    print("[data] Attempting VIX9D (from ~2011)...")
    try:
        if _FRAMEWORK:
            vix9d = load_data("^VIX9D", start=start, end=end, source="yfinance", ohlcv=False)
            vix9d.columns = ["VIX9D"]
        else:
            vix9d = _yf_load("^VIX9D", start, end).rename(columns={"^VIX9D": "VIX9D"})
        df = df.join(vix9d, how="left")
        df["vix_slope"] = df["VIX9D"] / df["VIX"]  # > 1 = backwardation
        n_vix9d = df["VIX9D"].notna().sum()
        print(f"[data] VIX9D loaded — {n_vix9d} observations")
    except Exception as e:
        print(f"[data] VIX9D not available: {e} — term structure step will be skipped.")
        df["VIX9D"]     = np.nan
        df["vix_slope"] = np.nan

    return df


# ── 2. IV Flag Identification ─────────────────────────────────────────────────

def identify_iv_flags(df,
                       abs_thresh=IV_ABS_THRESH,
                       rel_mult=IV_REL_MULT,
                       roll_window=IV_ROLL_WINDOW,
                       exclusion_days=EXCLUSION_DAYS):
    """
    Flag IV spike events using two methods:
      - Absolute:  VIX > abs_thresh
      - Relative:  VIX > rel_mult × rolling mean(VIX, roll_window)

    Applies exclusion_days non-overlap window (keep first occurrence).
    Flags are set on day t; forward returns are computed from t+1.
    No look-ahead bias.
    """
    df = df.copy()

    # Raw flags
    df["vix_roll_mean"]     = df["VIX"].rolling(roll_window).mean()
    df["abs_flag_raw"]      = (df["VIX"] > abs_thresh).astype(int)
    df["rel_flag_raw"]      = (df["VIX"] > rel_mult * df["vix_roll_mean"]).astype(int)

    # Also compute large-move flags on S&P 500 (used in lead/lag analysis)
    for thresh in MOVE_THRESHOLDS:
        col = f"large_move_{int(thresh*1000)}bps"
        df[col] = (df["abs_ret"] > thresh).astype(int)

    # Apply exclusion window
    for raw_col, out_col in [("abs_flag_raw", "abs_iv_flag"),
                              ("rel_flag_raw", "rel_iv_flag")]:
        flags   = df[raw_col].values.copy()
        cleaned = np.zeros(len(flags), dtype=int)
        last    = -exclusion_days - 1
        for i in range(len(flags)):
            if flags[i] == 1 and (i - last) > exclusion_days:
                cleaned[i] = 1
                last = i
        df[out_col] = cleaned

    n_abs = df["abs_iv_flag"].sum()
    n_rel = df["rel_iv_flag"].sum()
    print(f"[flags] Absolute IV flags (VIX > {abs_thresh}):        {n_abs}")
    print(f"[flags] Relative IV flags (VIX > {rel_mult}× {roll_window}d mean): {n_rel}")

    for thresh in MOVE_THRESHOLDS:
        col = f"large_move_{int(thresh*1000)}bps"
        print(f"[flags] Large moves (|ret| > {thresh*100:.1f}%):             {df[col].sum()}")

    return df


# ── 3. Event Study ────────────────────────────────────────────────────────────

def event_study(df, flag_col, horizons=HORIZONS, n_bootstrap=BOOTSTRAP_N):
    """
    For each IV flag event, compute forward |return| at each horizon.
    CAR here = mean(|r_t| - |r_mean|) — abnormal absolute return.

    Returns dict with results per horizon.
    """
    mean_abs    = df["abs_ret"].mean()
    flag_dates  = df.index[df[flag_col] == 1]
    results     = {}

    for h in horizons:
        abs_cars  = []
        raw_abs   = []

        for flag_date in flag_dates:
            loc = df.index.get_loc(flag_date)
            end = loc + h + 1
            if end > len(df):
                continue
            window_abs = df["abs_ret"].iloc[loc+1 : loc+h+1]
            raw_abs.append(window_abs.mean())
            abs_cars.append((window_abs - mean_abs).mean())

        abs_cars = np.array(abs_cars)
        raw_abs  = np.array(raw_abs)

        # Bootstrap baseline (non-flag windows)
        non_flag_idx = np.where(df[flag_col].values == 0)[0]
        baseline_cars = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(non_flag_idx[non_flag_idx + h < len(df)])
            w   = df["abs_ret"].iloc[idx+1 : idx+h+1]
            baseline_cars.append((w - mean_abs).mean())
        baseline_cars = np.array(baseline_cars)

        t_stat, p_val = stats.ttest_1samp(abs_cars, 0)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))

        results[h] = {
            "abs_cars":        abs_cars,
            "mean_abs_car":    abs_cars.mean(),
            "mean_raw_abs":    raw_abs.mean(),
            "t_stat":          t_stat,
            "p_val":           p_val,
            "n_events":        len(abs_cars),
            "baseline_mean":   baseline_cars.mean(),
            "baseline_std":    baseline_cars.std(),
        }

        print(f"[event_study] {flag_col} t+{h:>2}: "
              f"mean_|CAR|={abs_cars.mean():+.4f}  "
              f"t={t_stat:+.2f}  p={p_val:.3f} {sig}  n={len(abs_cars)}")

    return results


# ── 4. Conditional Probability ────────────────────────────────────────────────

def conditional_probability(df, flag_col, horizons=HORIZONS,
                             move_thresholds=MOVE_THRESHOLDS):
    """
    Compute P(|SP500| > move_thresh | IV_flag) vs unconditional baseline
    at each horizon and move threshold.

    Returns nested dict: results[move_thresh][horizon]
    """
    flag_dates = df.index[df[flag_col] == 1]
    results    = {}

    for move_thresh in move_thresholds:
        results[move_thresh] = {}
        thresh_label = f"{int(move_thresh*1000)}bps"

        for h in horizons:
            post_flag_abs = []
            for flag_date in flag_dates:
                loc = df.index.get_loc(flag_date)
                end = loc + h + 1
                if end > len(df):
                    continue
                # max |return| in the forward window
                max_abs = df["abs_ret"].iloc[loc+1 : loc+h+1].max()
                post_flag_abs.append(max_abs)

            post_flag_abs = np.array(post_flag_abs)

            # Unconditional baseline — rolling windows of same length
            all_max_abs = []
            for i in range(len(df) - h):
                mx = df["abs_ret"].iloc[i+1 : i+h+1].max()
                all_max_abs.append(mx)
            all_max_abs = np.array(all_max_abs)

            p_large_given_flag = (post_flag_abs > move_thresh).mean()
            p_large_baseline   = (all_max_abs > move_thresh).mean()
            lift = ((p_large_given_flag - p_large_baseline) / p_large_baseline
                    if p_large_baseline > 0 else np.nan)

            results[move_thresh][h] = {
                "p_large_given_flag": p_large_given_flag,
                "p_large_baseline":   p_large_baseline,
                "lift":               lift,
                "n_events":           len(post_flag_abs),
            }

            print(f"[cond_prob] {flag_col}  thresh={thresh_label}  t+{h:>2}: "
                  f"P(large|flag)={p_large_given_flag:.3f}  "
                  f"P(large)={p_large_baseline:.3f}  "
                  f"lift={lift:+.3f}")

    return results


# ── 5. Lead/Lag Analysis ──────────────────────────────────────────────────────

def lead_lag_analysis(df, move_thresh=MOVE_THRESHOLDS[1], window=LEADLAG_WINDOW):
    """
    Compute mean VIX in [-window, +window] days around large |SP500| move events.

    Reveals whether IV leads (predictive) or lags (reactive) large moves.

    Returns DataFrame: rows = days relative to event (-window to +window),
                       cols = mean_vix, median_vix, mean_vix_non_event.
    """
    large_move_dates = df.index[df["abs_ret"] > move_thresh]
    print(f"[lead_lag] {len(large_move_dates)} large move events "
          f"(|ret| > {move_thresh*100:.1f}%)")

    rows = []
    for t in range(-window, window + 1):
        vix_vals = []
        for evt_date in large_move_dates:
            loc = df.index.get_loc(evt_date)
            target_loc = loc + t
            if 0 <= target_loc < len(df):
                vix_vals.append(df["VIX"].iloc[target_loc])
        rows.append({
            "days_relative": t,
            "mean_vix":      np.mean(vix_vals),
            "median_vix":    np.median(vix_vals),
            "n":             len(vix_vals),
        })

    result = pd.DataFrame(rows).set_index("days_relative")

    # Baseline: mean VIX on non-large-move days
    non_event_vix = df.loc[df["abs_ret"] <= move_thresh, "VIX"].mean()
    result["non_event_mean_vix"] = non_event_vix

    # Symmetry test: is pre-event mean VIX > post-event mean VIX?
    pre_mean  = result.loc[-window:-1, "mean_vix"].mean()
    post_mean = result.loc[1:window, "mean_vix"].mean()
    event_day = result.loc[0, "mean_vix"]
    print(f"[lead_lag] Pre-event mean VIX  (t-{window} to t-1): {pre_mean:.2f}")
    print(f"[lead_lag] Event day VIX        (t=0):              {event_day:.2f}")
    print(f"[lead_lag] Post-event mean VIX  (t+1 to t+{window}): {post_mean:.2f}")
    print(f"[lead_lag] Non-event baseline:                      {non_event_vix:.2f}")

    return result


# ── 6. Term Structure Slope ───────────────────────────────────────────────────

def term_structure_analysis(df, move_thresholds=MOVE_THRESHOLDS,
                             horizons=HORIZONS):
    """
    Tests whether VIX9D/VIX slope (backwardation vs contango) provides
    incremental predictive signal beyond raw VIX level.

    backwardation (slope > 1): short-dated IV > long-dated IV — imminent stress priced.
    contango      (slope < 1): normal upward-sloping term structure.

    Skipped if VIX9D is unavailable.
    """
    if df["VIX9D"].isna().all():
        print("[term_struct] VIX9D unavailable — skipping.")
        return None

    available = df.dropna(subset=["VIX9D", "vix_slope"])
    print(f"[term_struct] VIX9D sample: {len(available)} rows "
          f"({available.index[0].date()} → {available.index[-1].date()})")

    results = {}
    for move_thresh in move_thresholds:
        results[move_thresh] = {}
        for h in horizons:
            # Backwardation flag: slope > 1 (VIX9D > VIX)
            back_mask  = available["vix_slope"] > 1.0
            back_dates = available.index[back_mask]
            norm_dates = available.index[~back_mask]

            def _fwd_large(dates):
                flags = []
                for d in dates:
                    loc = available.index.get_loc(d)
                    end = loc + h + 1
                    if end > len(available):
                        continue
                    mx = available["abs_ret"].iloc[loc+1 : loc+h+1].max()
                    flags.append(mx > move_thresh)
                return np.array(flags)

            back_flags = _fwd_large(back_dates)
            norm_flags = _fwd_large(norm_dates)

            p_back = back_flags.mean() if len(back_flags) > 0 else np.nan
            p_norm = norm_flags.mean() if len(norm_flags) > 0 else np.nan
            lift   = ((p_back - p_norm) / p_norm) if p_norm > 0 else np.nan

            results[move_thresh][h] = {
                "p_large_backwardation": p_back,
                "p_large_contango":      p_norm,
                "lift":                  lift,
                "n_backwardation":       len(back_flags),
                "n_contango":            len(norm_flags),
            }

            thresh_label = f"{int(move_thresh*1000)}bps"
            print(f"[term_struct] thresh={thresh_label}  t+{h:>2}: "
                  f"P(large|back)={p_back:.3f}  "
                  f"P(large|contango)={p_norm:.3f}  "
                  f"lift={lift:+.3f}")

    return results


# ── 7. Regime Distribution ────────────────────────────────────────────────────

def regime_distribution(df, save_path=None):
    """
    Fit 4-state HMM on ^GSPC vs TLT, attach regime labels,
    count IV flag events per regime vs baseline.

    Returns dict with regime distribution results.
    Gracefully skips if HMM framework not available.
    """
    try:
        from research.regime_detection.hmm_model import RegimeDetector
    except ImportError:
        print("[regime_dist] HMM framework not available — skipping.")
        return None

    if df["TLT"].isna().all():
        print("[regime_dist] TLT data unavailable — skipping regime analysis.")
        return None

    print("[regime_dist] Fitting HMM (4 states)...")
    sp500_prices = df["Close"]
    tlt_prices   = df["TLT"]

    detector = RegimeDetector(n_regimes=4, n_iter=1000, window=20, random_state=42)
    detector.fit(sp500_prices, tlt_prices)
    regime_labels = detector.predict(sp500_prices, tlt_prices)

    all_regimes     = ["Bull", "Transition", "Bear", "Crisis"]
    baseline_counts = regime_labels.value_counts().reindex(all_regimes, fill_value=0)
    baseline_pct    = (baseline_counts / baseline_counts.sum() * 100).round(1)

    df_r = df.copy()
    df_r["regime"] = regime_labels

    results = {"baseline_pct": baseline_pct, "regime_labels": regime_labels,
               "detector": detector}

    for flag_col, label in [("abs_iv_flag", "abs"), ("rel_iv_flag", "rel")]:
        shock_regimes = df_r.loc[
            (df_r[flag_col] == 1) & df_r["regime"].notna(), "regime"
        ]
        counts = shock_regimes.value_counts().reindex(all_regimes, fill_value=0)
        pct    = (counts / counts.sum() * 100).round(1)

        results[f"{label}_counts"] = counts
        results[f"{label}_pct"]    = pct
        results[f"{label}_n"]      = len(shock_regimes)

        print(f"\n[regime_dist] {flag_col} — {len(shock_regimes)} events with regime labels:")
        for regime in all_regimes:
            n    = counts[regime]
            p    = pct[regime]
            b    = baseline_pct[regime]
            lift = p - b
            sign = "+" if lift >= 0 else ""
            print(f"  {regime:<12} {n:>3} events  {p:>5.1f}%  "
                  f"(baseline {b:.1f}%  lift {sign}{lift:.1f}pp)")

    if save_path:
        _plot_regime_distribution(results, all_regimes, save_path)

    return results


# ── 8. Realized vs Implied Vol ────────────────────────────────────────────────

def realized_vs_implied(df, flag_col, horizons=HORIZONS):
    """
    For each IV flag event, compare:
      - VIX level on flag date (implied vol, annualised %)
      - Realized vol over next h days (annualised)

    If realized > implied → IV underpriced → options markets were cheap → predictive.
    If realized < implied → IV overpriced → options markets were expensive → reactive.

    Returns DataFrame: rows = flag events, cols per horizon.
    """
    flag_dates = df.index[df[flag_col] == 1]
    rows = []

    for flag_date in flag_dates:
        loc = df.index.get_loc(flag_date)
        iv_level = df["VIX"].iloc[loc]  # annualised %

        row = {"date": flag_date, "vix_level": iv_level}
        for h in horizons:
            end = loc + h + 1
            if end > len(df):
                row[f"realized_vol_t{h}"] = np.nan
                row[f"iv_minus_realized_t{h}"] = np.nan
                continue
            fwd_rets = df["sp500_ret"].iloc[loc+1 : loc+h+1]
            # Annualise: std × sqrt(252)
            rv = fwd_rets.std() * np.sqrt(252) * 100  # convert to %
            row[f"realized_vol_t{h}"] = rv
            row[f"iv_minus_realized_t{h}"] = iv_level - rv  # positive → IV expensive
        rows.append(row)

    result = pd.DataFrame(rows).set_index("date")

    for h in horizons:
        col = f"iv_minus_realized_t{h}"
        if col in result.columns:
            mean_diff = result[col].mean()
            pct_expensive = (result[col] > 0).mean()
            print(f"[rv_vs_iv] {flag_col} t+{h:>2}: "
                  f"mean(IV-RV)={mean_diff:+.2f}pp  "
                  f"IV expensive {pct_expensive:.1%} of the time")

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

DARK_BG  = "#0d0d0d"
GRID_COL = "#2a2a2a"
TEXT_COL = "white"
DIM_COL  = "#a3a3a3"
COLORS   = {"abs": "#3b82f6", "rel": "#f59e0b",
            "baseline": "#4b5563", "large": "#ef4444",
            "VIX": "#8b5cf6"}


def plot_event_study(abs_results, rel_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("IV Spike Event Study — Mean Abnormal |Return| Post-Flag",
                 color=TEXT_COL, fontsize=13, fontweight="bold")

    for ax, res, label, color in [
        (axes[0], abs_results, f"Absolute flag (VIX > {IV_ABS_THRESH})", COLORS["abs"]),
        (axes[1], rel_results, f"Relative flag (VIX > {IV_REL_MULT}× 60d mean)", COLORS["rel"]),
    ]:
        ax.set_facecolor(DARK_BG)
        horizons = sorted(res.keys())
        mean_cars = [res[h]["mean_abs_car"] for h in horizons]
        base_means = [res[h]["baseline_mean"] for h in horizons]
        p_vals    = [res[h]["p_val"] for h in horizons]

        x = np.arange(len(horizons))
        w = 0.35
        ax.bar(x - w/2, mean_cars,  w, label="IV flag windows", color=color, alpha=0.9)
        ax.bar(x + w/2, base_means, w, label="Baseline (non-flag)", color=COLORS["baseline"], alpha=0.7)
        ax.axhline(0, color=DIM_COL, linewidth=0.8)

        for i, (h, p) in enumerate(zip(horizons, p_vals)):
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            if sig:
                y_pos = max(mean_cars[i], 0) + 0.0002
                ax.text(x[i] - w/2, y_pos, sig, ha="center", color=TEXT_COL, fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in horizons], color=TEXT_COL)
        ax.tick_params(colors=TEXT_COL)
        ax.spines[list(ax.spines)].set_visible(False)
        ax.set_title(label, color=DIM_COL, fontsize=10)
        ax.set_ylabel("Mean |Abnormal Return|", color=TEXT_COL)
        ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, framealpha=0.8, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_conditional_prob(abs_cp, rel_cp, save_path):
    move_thresholds = sorted(abs_cp.keys())
    horizons        = sorted(abs_cp[move_thresholds[0]].keys())

    fig, axes = plt.subplots(len(move_thresholds), 2,
                              figsize=(14, 5 * len(move_thresholds)),
                              facecolor=DARK_BG)
    if len(move_thresholds) == 1:
        axes = [axes]

    fig.suptitle("Conditional Probability: P(Large Move | IV Flag) vs Baseline",
                 color=TEXT_COL, fontsize=13, fontweight="bold")

    for row_idx, thresh in enumerate(move_thresholds):
        thresh_label = f"|ret| > {thresh*100:.1f}%"
        for col_idx, (res, flag_label, color) in enumerate([
            (abs_cp[thresh], f"Absolute flag — {thresh_label}", COLORS["abs"]),
            (rel_cp[thresh], f"Relative flag — {thresh_label}", COLORS["rel"]),
        ]):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor(DARK_BG)
            x = np.arange(len(horizons))
            w = 0.35

            p_flags   = [res[h]["p_large_given_flag"] for h in horizons]
            p_baselines = [res[h]["p_large_baseline"] for h in horizons]
            lifts     = [res[h]["lift"] for h in horizons]

            ax.bar(x - w/2, p_flags,      w, label="P(large | IV flag)", color=color, alpha=0.9)
            ax.bar(x + w/2, p_baselines,  w, label="P(large) baseline",  color=COLORS["baseline"], alpha=0.7)

            for i, lift in enumerate(lifts):
                sign = "+" if lift >= 0 else ""
                ax.text(x[i] - w/2, p_flags[i] + 0.01,
                        f"{sign}{lift:.0%}", ha="center", color=TEXT_COL, fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels([f"t+{h}" for h in horizons], color=TEXT_COL)
            ax.tick_params(colors=TEXT_COL)
            ax.spines[list(ax.spines)].set_visible(False)
            ax.set_ylim(0, min(1.0, max(p_flags + p_baselines) * 1.3))
            ax.set_title(flag_label, color=DIM_COL, fontsize=10)
            ax.set_ylabel("Probability", color=TEXT_COL)
            ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, framealpha=0.8, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_lead_lag(lead_lag_df, move_thresh, save_path):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    days = lead_lag_df.index.tolist()
    ax.plot(days, lead_lag_df["mean_vix"], color=COLORS["VIX"],
            linewidth=2, label="Mean VIX around large moves")
    ax.fill_between(days, lead_lag_df["mean_vix"], alpha=0.15, color=COLORS["VIX"])
    ax.axhline(lead_lag_df["non_event_mean_vix"].iloc[0],
               color=COLORS["baseline"], linewidth=1.5, linestyle="--",
               label="Non-event day mean VIX")
    ax.axvline(0, color=COLORS["large"], linewidth=1.5, linestyle="--",
               label=f"Large move day (|ret| > {move_thresh*100:.1f}%)")

    ax.set_xticks(days)
    ax.set_xticklabels([str(d) for d in days], color=TEXT_COL, fontsize=8)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[list(ax.spines)].set_visible(False)
    ax.set_xlabel("Days relative to large move event", color=TEXT_COL)
    ax.set_ylabel("Mean VIX", color=TEXT_COL)
    ax.set_title("IV Lead/Lag Around Large S&P 500 Moves\n"
                 "(Pre-event → predictive  |  Post-event peak → reactive)",
                 color=TEXT_COL, fontsize=12)
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, framealpha=0.8, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_realized_vs_implied(rv_df_abs, rv_df_rel, save_path):
    horizons = [h for h in HORIZONS if f"iv_minus_realized_t{h}" in rv_df_abs.columns]
    if not horizons:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Realized Volatility vs Implied Volatility Post-Flag\n"
                 "(Positive = IV expensive / overpriced  |  Negative = IV cheap / underpriced)",
                 color=TEXT_COL, fontsize=12, fontweight="bold")

    for ax, rv_df, flag_label, color in [
        (axes[0], rv_df_abs, f"Absolute flag (VIX > {IV_ABS_THRESH})", COLORS["abs"]),
        (axes[1], rv_df_rel, f"Relative flag (VIX > {IV_REL_MULT}× 60d mean)", COLORS["rel"]),
    ]:
        ax.set_facecolor(DARK_BG)
        means = [rv_df[f"iv_minus_realized_t{h}"].mean() for h in horizons]
        stds  = [rv_df[f"iv_minus_realized_t{h}"].std() / np.sqrt(rv_df[f"iv_minus_realized_t{h}"].notna().sum())
                 for h in horizons]

        x = np.arange(len(horizons))
        bar_colors = [COLORS["large"] if m > 0 else COLORS["abs"] for m in means]
        ax.bar(x, means, color=bar_colors, alpha=0.85,
               yerr=stds, error_kw={"color": TEXT_COL, "capsize": 4})
        ax.axhline(0, color=DIM_COL, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in horizons], color=TEXT_COL)
        ax.tick_params(colors=TEXT_COL)
        ax.spines[list(ax.spines)].set_visible(False)
        ax.set_title(flag_label, color=DIM_COL, fontsize=10)
        ax.set_ylabel("Mean (VIX − Realized Vol) in pp", color=TEXT_COL)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def _plot_regime_distribution(results, all_regimes, save_path):
    REGIME_COLORS = {
        "Bull": "#22c55e", "Transition": "#f59e0b",
        "Bear": "#ef4444", "Crisis": "#8b5cf6",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK_BG)
    fig.suptitle("IV Flag Distribution by HMM Regime vs Baseline\n"
                 "(TLT as benchmark — equity/bond correlation signal)",
                 color=TEXT_COL, fontsize=13, fontweight="bold", y=1.02)

    baseline = results["baseline_pct"].values
    x        = np.arange(len(all_regimes))
    colors   = [REGIME_COLORS[r] for r in all_regimes]
    width    = 0.35

    for ax_idx, (label, flag_key, title) in enumerate([
        ("abs", "abs_pct", f"Absolute Flags (n={results.get('abs_n','?')})\nvs Baseline"),
        ("rel", "rel_pct", f"Relative Flags (n={results.get('rel_n','?')})\nvs Baseline"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(DARK_BG)
        flag_pct = results[flag_key].values
        ax.bar(x - width/2, baseline,  width, label="Baseline % of days",
               color="#3b82f6", alpha=0.7)
        ax.bar(x + width/2, flag_pct, width, label=f"{label.capitalize()} IV flag %",
               color=colors, alpha=0.9)
        for i, (fp, b) in enumerate(zip(flag_pct, baseline)):
            lift = fp - b
            sign = "+" if lift >= 0 else ""
            ax.text(x[i] + width/2, fp + 0.5, f"{sign}{lift:.1f}pp",
                    ha="center", color=TEXT_COL, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(all_regimes, color=TEXT_COL, fontsize=9)
        ax.tick_params(colors=TEXT_COL)
        ax.spines[list(ax.spines)].set_visible(False)
        ax.set_ylabel("% of events", color=TEXT_COL)
        ax.set_title(title, color=DIM_COL, fontsize=10)
        ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, framealpha=0.8, fontsize=8)

    # Panel 3: absolute counts
    ax = axes[2]
    ax.set_facecolor(DARK_BG)
    abs_counts = results.get("abs_counts", pd.Series(0, index=all_regimes)).values
    rel_counts = results.get("rel_counts", pd.Series(0, index=all_regimes)).values
    ax.bar(x - width/2, abs_counts, width, label="Absolute IV flags", color=colors, alpha=0.9)
    ax.bar(x + width/2, rel_counts, width, label="Relative IV flags",  color=colors, alpha=0.5, hatch="//")
    for i, (a, r) in enumerate(zip(abs_counts, rel_counts)):
        if a > 0:
            ax.text(x[i] - width/2, a + 0.3, str(a), ha="center", color=TEXT_COL, fontsize=9)
        if r > 0:
            ax.text(x[i] + width/2, r + 0.3, str(r), ha="center", color=TEXT_COL, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(all_regimes, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[list(ax.spines)].set_visible(False)
    ax.set_ylabel("Event count", color=TEXT_COL)
    ax.set_title("Absolute Counts\nAbsolute vs Relative flags", color=DIM_COL, fontsize=10)
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, framealpha=0.8, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FINDINGS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  Does Implied Volatility Predict Extreme Moves?")
    print("  @AlphaByProcess  |  research/analysis/iv_predict_moves")
    print("="*60)

    # ── 1. Load ───────────────────────────────────────────────────
    df = load_all_data()

    # ── 2. Flags ──────────────────────────────────────────────────
    df = identify_iv_flags(df)

    # ── 3. Event Study ────────────────────────────────────────────
    print("\n── Event Study ──────────────────────────────────────────")
    abs_es = event_study(df, flag_col="abs_iv_flag")
    rel_es = event_study(df, flag_col="rel_iv_flag")

    # ── 4. Conditional Probability ────────────────────────────────
    print("\n── Conditional Probability ──────────────────────────────")
    abs_cp = conditional_probability(df, flag_col="abs_iv_flag")
    rel_cp = conditional_probability(df, flag_col="rel_iv_flag")

    # ── 5. Lead/Lag Analysis ──────────────────────────────────────
    print("\n── Lead/Lag Analysis ────────────────────────────────────")
    ll_df = lead_lag_analysis(df, move_thresh=MOVE_THRESHOLDS[1])

    # ── 6. Term Structure ─────────────────────────────────────────
    print("\n── Term Structure Slope (VIX9D/VIX) ────────────────────")
    ts_results = term_structure_analysis(df)

    # ── 7. Regime Distribution ────────────────────────────────────
    print("\n── Regime Distribution ──────────────────────────────────")
    regime_dist = regime_distribution(
        df, save_path=os.path.join(FINDINGS_DIR, "regime_distribution.png")
    )

    # ── 8. Realized vs Implied Vol ────────────────────────────────
    print("\n── Realized vs Implied Volatility ──────────────────────")
    rv_abs = realized_vs_implied(df, flag_col="abs_iv_flag")
    rv_rel = realized_vs_implied(df, flag_col="rel_iv_flag")

    # ── Plots ─────────────────────────────────────────────────────
    print("\n── Saving Plots ─────────────────────────────────────────")
    plot_event_study(abs_es, rel_es,
                     os.path.join(FINDINGS_DIR, "event_study.png"))
    plot_conditional_prob(abs_cp, rel_cp,
                          os.path.join(FINDINGS_DIR, "conditional_prob.png"))
    plot_lead_lag(ll_df, MOVE_THRESHOLDS[1],
                  os.path.join(FINDINGS_DIR, "lead_lag.png"))
    plot_realized_vs_implied(rv_abs, rv_rel,
                             os.path.join(FINDINGS_DIR, "realized_vs_implied.png"))

    print("\n" + "="*60)
    print("  Done. All outputs saved to findings/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

    