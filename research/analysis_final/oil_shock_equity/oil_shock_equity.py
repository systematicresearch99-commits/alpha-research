"""
oil_shock_equity.py — Oil Price Shocks → Equity Market Drawdowns
=================================================================
Research script for @AlphaByProcess.
Tests whether sudden WTI oil price shocks predict negative S&P 500 returns.

Steps:
    1. Load data  — WTI (CL=F), S&P 500 (^GSPC), sector tickers
    2. Identify shocks — daily >5%, weekly >10%
    3. Event study — CAR at t+1, t+5, t+20
    4. Conditional probability — P(SP500<0 | shock) vs baseline
    5. Sector analysis — XLE, XLK, XLU, Airlines (AAL+DAL+UAL avg)
    6. Strategy backtest — calls strategies/oil_shock_short.py + backtests/engine.py

Usage:
    python research/analysis_final/oil_shock_equity/oil_shock_equity.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from backtests.engine   import run_backtest
from strategies.oil_shock_short import generate_signals, STRATEGY_NAME

# ── Config ────────────────────────────────────────────────────────────────────
START          = "2000-01-01"
END            = None           # today
DAILY_THRESH   = 0.05           # >5% daily WTI return = shock
WEEKLY_THRESH  = 0.10           # >10% 5-day WTI return = shock
EXCLUSION_DAYS = 5              # min gap between shock events
HORIZONS       = [1, 5, 20]    # forward return windows (days)
BOOTSTRAP_N    = 1000

FINDINGS_DIR   = os.path.join(os.path.dirname(__file__), "findings")

SECTOR_TICKERS = {
    "Energy (XLE)":    "XLE",
    "Tech (XLK)":      "XLK",
    "Utilities (XLU)": "XLU",
    "Airlines (avg)":  ["AAL", "DAL", "UAL"],
}


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def load_all_data(start=START, end=END):
    """
    Load WTI, S&P 500, and sector tickers.
    Returns merged DataFrame aligned on trading dates.
    """
    print("[data] Loading WTI crude oil (CL=F)...")
    wti = load_data("CL=F", start=start, end=end, source="yfinance", ohlcv=False)
    wti.columns = ["WTI"]

    print("[data] Loading S&P 500 (^GSPC)...")
    sp500 = load_data("^GSPC", start=start, end=end, source="yfinance", ohlcv=False)
    sp500.columns = ["Close"]

    # Inner join — only dates both markets traded
    df = sp500.join(wti, how="inner")
    df["sp500_ret"] = df["Close"].pct_change()
    df["wti_ret"]   = df["WTI"].pct_change()
    df.dropna(inplace=True)

    print(f"[data] Base data: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    # Sector data
    sectors = {}
    for label, ticker in SECTOR_TICKERS.items():
        if isinstance(ticker, list):
            # Average multiple tickers (airlines)
            frames = []
            for t in ticker:
                try:
                    s = load_data(t, start=start, end=end, source="yfinance", ohlcv=False)
                    s.columns = [t]
                    frames.append(s)
                    print(f"[data] Loaded {t}")
                except Exception as e:
                    print(f"[data] Warning: {t} failed — {e}")
            if frames:
                combined = pd.concat(frames, axis=1).mean(axis=1)
                sectors[label] = combined.pct_change()
        else:
            try:
                s = load_data(ticker, start=start, end=end, source="yfinance", ohlcv=False)
                sectors[label] = s["Close"].pct_change()
                print(f"[data] Loaded {ticker}")
            except Exception as e:
                print(f"[data] Warning: {ticker} failed — {e}")

    sector_df = pd.DataFrame(sectors)
    df = df.join(sector_df, how="left")
    df.dropna(subset=["sp500_ret", "wti_ret"], inplace=True)

    return df


# ── 2. Shock Identification ───────────────────────────────────────────────────

def identify_shocks(df, daily_thresh=DAILY_THRESH, weekly_thresh=WEEKLY_THRESH,
                    exclusion_days=EXCLUSION_DAYS):
    """
    Flag oil shock events.

    daily_shock  : WTI 1-day return > daily_thresh
    weekly_shock : WTI 5-day rolling return > weekly_thresh

    Overlapping events within exclusion_days window are dropped (keep first).
    """
    df = df.copy()

    # Raw flags
    df["daily_shock_raw"]  = (df["wti_ret"] > daily_thresh).astype(int)
    df["weekly_ret_wti"]   = df["wti_ret"].rolling(5).sum()
    df["weekly_shock_raw"] = (df["weekly_ret_wti"] > weekly_thresh).astype(int)

    # Apply exclusion window — keep first shock, suppress next exclusion_days bars
    for shock_col, out_col in [("daily_shock_raw", "daily_shock"),
                                ("weekly_shock_raw", "weekly_shock")]:
        flags   = df[shock_col].values.copy()
        cleaned = np.zeros(len(flags), dtype=int)
        last    = -exclusion_days - 1
        for i in range(len(flags)):
            if flags[i] == 1 and (i - last) > exclusion_days:
                cleaned[i] = 1
                last = i
        df[out_col] = cleaned

    n_daily  = df["daily_shock"].sum()
    n_weekly = df["weekly_shock"].sum()
    print(f"[shocks] Daily shocks  (>{daily_thresh*100:.0f}%/day):   {n_daily}")
    print(f"[shocks] Weekly shocks (>{weekly_thresh*100:.0f}%/5-day): {n_weekly}")

    return df


# ── 3. Event Study ────────────────────────────────────────────────────────────

def event_study(df, shock_col, horizons=HORIZONS, n_bootstrap=BOOTSTRAP_N):
    """
    For each shock event compute forward S&P 500 returns at each horizon.
    Compute CAR = sum of (r_t - r_mean) over window.
    Test vs. non-shock baseline using t-test and bootstrap.

    Returns dict with results per horizon.
    """
    mean_ret    = df["sp500_ret"].mean()
    shock_dates = df.index[df[shock_col] == 1]
    results     = {}

    for h in horizons:
        cars       = []
        raw_rets   = []

        for shock_date in shock_dates:
            loc = df.index.get_loc(shock_date)
            end = loc + h + 1
            if end > len(df):
                continue
            window_rets  = df["sp500_ret"].iloc[loc+1 : loc+h+1]
            raw_ret      = window_rets.sum()
            car          = (window_rets - mean_ret).sum()
            cars.append(car)
            raw_rets.append(raw_ret)

        cars     = np.array(cars)
        raw_rets = np.array(raw_rets)

        # Baseline: random windows of same length (non-shock)
        non_shock_idx = np.where(df[shock_col].values == 0)[0]
        baseline_cars = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx  = rng.choice(non_shock_idx[non_shock_idx + h < len(df)])
            w    = df["sp500_ret"].iloc[idx+1 : idx+h+1]
            baseline_cars.append((w - mean_ret).sum())
        baseline_cars = np.array(baseline_cars)

        t_stat, p_val = stats.ttest_1samp(cars, 0)

        results[h] = {
            "cars":           cars,
            "raw_rets":       raw_rets,
            "mean_car":       cars.mean(),
            "mean_raw_ret":   raw_rets.mean(),
            "std_car":        cars.std(),
            "t_stat":         t_stat,
            "p_val":          p_val,
            "n_events":       len(cars),
            "baseline_mean":  baseline_cars.mean(),
            "baseline_std":   baseline_cars.std(),
        }

        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        print(f"[event_study] {shock_col} t+{h:>2}: "
              f"mean_CAR={cars.mean():+.4f}  "
              f"t={t_stat:+.2f}  p={p_val:.3f} {sig}  n={len(cars)}")

    return results


# ── 4. Conditional Probability ────────────────────────────────────────────────

def conditional_probability(df, shock_col, horizons=HORIZONS):
    """
    Compute P(SP500 < 0 | shock) vs P(SP500 < 0) at each horizon.
    Reports lift = [P(neg|shock) - P(neg)] / P(neg).
    """
    results     = {}
    shock_dates = df.index[df[shock_col] == 1]

    for h in horizons:
        post_shock_rets = []
        for shock_date in shock_dates:
            loc = df.index.get_loc(shock_date)
            end = loc + h + 1
            if end > len(df):
                continue
            fwd = df["sp500_ret"].iloc[loc+1 : loc+h+1].sum()
            post_shock_rets.append(fwd)

        post_shock_rets = np.array(post_shock_rets)

        # Rolling baseline windows of same length
        all_fwd = []
        for i in range(len(df) - h):
            all_fwd.append(df["sp500_ret"].iloc[i+1 : i+h+1].sum())
        all_fwd = np.array(all_fwd)

        p_neg_given_shock = (post_shock_rets < 0).mean()
        p_neg_baseline    = (all_fwd < 0).mean()
        lift              = (p_neg_given_shock - p_neg_baseline) / p_neg_baseline if p_neg_baseline > 0 else np.nan

        results[h] = {
            "p_neg_given_shock": p_neg_given_shock,
            "p_neg_baseline":    p_neg_baseline,
            "lift":              lift,
            "n_events":          len(post_shock_rets),
        }

        print(f"[cond_prob] {shock_col} t+{h:>2}: "
              f"P(neg|shock)={p_neg_given_shock:.3f}  "
              f"P(neg)={p_neg_baseline:.3f}  "
              f"lift={lift:+.3f}")

    return results


# ── 5. Sector Analysis ────────────────────────────────────────────────────────

def sector_analysis(df, shock_col, horizons=HORIZONS):
    """
    Compute mean post-shock returns for each sector vs S&P 500 baseline.
    Returns DataFrame: rows = sectors, cols = horizons.
    """
    sector_cols = [c for c in df.columns if c in
                   ["Energy (XLE)", "Tech (XLK)", "Utilities (XLU)", "Airlines (avg)"]]

    if not sector_cols:
        print("[sector] No sector data available — skipping.")
        return None

    shock_dates = df.index[df[shock_col] == 1]
    results     = {}

    for col in sector_cols + ["sp500_ret"]:
        if col not in df.columns:
            continue
        row = {}
        for h in horizons:
            fwd_rets = []
            for shock_date in shock_dates:
                loc = df.index.get_loc(shock_date)
                end = loc + h + 1
                if end > len(df):
                    continue
                fwd = df[col].iloc[loc+1 : loc+h+1].sum()
                fwd_rets.append(fwd)
            row[f"t+{h}"] = np.mean(fwd_rets) if fwd_rets else np.nan
        results[col] = row

    result_df = pd.DataFrame(results).T
    result_df.index.name = "Sector"

    print(f"\n[sector] Post-shock mean returns ({shock_col}):")
    print(result_df.round(4).to_string())

    return result_df

def regime_distribution(df, benchmark_ticker="TLT", feature_window=20,
                         save_path=None):
    """
    Fit HMM regime detector on ^GSPC vs TLT, attach regime labels to
    all trading days, then count how many daily and weekly shock events
    fell in each regime.
 
    Uses TLT (not SPY) as benchmark so index_correlation feature has
    genuine discriminatory power — equity/bond correlation varies
    meaningfully across Bull/Bear/Crisis/Transition regimes.
 
    Returns dict with:
        regime_labels    : pd.Series of regime per trading day
        daily_dist       : pd.Series — shock count per regime (daily)
        weekly_dist      : pd.Series — shock count per regime (weekly)
        daily_pct        : pd.Series — % of daily shocks per regime
        weekly_pct       : pd.Series — % of weekly shocks per regime
        baseline_pct     : pd.Series — % of all trading days per regime
    """
    from research.regime_detection.hmm_model import RegimeDetector
 
    print(f"[regime_dist] Loading benchmark {benchmark_ticker}...")
    try:
        benchmark = load_data(benchmark_ticker, start=START, end=END,
                              source="yfinance", ohlcv=False)
    except Exception as e:
        print(f"[regime_dist] Failed to load {benchmark_ticker}: {e}")
        print("[regime_dist] Skipping regime distribution analysis.")
        return None
 
    # Fit HMM on full price history
    # Asset = ^GSPC Close (already in df), Benchmark = TLT
    sp500_prices = df["Close"]
 
    print(f"[regime_dist] Fitting HMM (4 states, window={feature_window})...")
    detector = RegimeDetector(
        n_regimes  = 4,
        n_iter     = 1000,
        window     = feature_window,
        random_state = 42,
    )
    detector.fit(sp500_prices, benchmark["Close"])
 
    # Predict regime for every trading day in the sample
    regime_labels = detector.predict(sp500_prices, benchmark["Close"])
    print(f"[regime_dist] Regime labels computed: {len(regime_labels)} days")
 
    # Regime distribution across all trading days (baseline)
    all_regimes    = ["Bull", "Transition", "Bear", "Crisis"]
    baseline_counts = regime_labels.value_counts().reindex(all_regimes, fill_value=0)
    baseline_pct    = (baseline_counts / baseline_counts.sum() * 100).round(1)
 
    # Merge regime labels onto main df
    df_with_regime = df.copy()
    df_with_regime["regime"] = regime_labels
 
    # Count shock events per regime
    results = {}
    for shock_col, label in [("daily_shock", "daily"), ("weekly_shock", "weekly")]:
        shock_dates = df_with_regime.index[df_with_regime[shock_col] == 1]
 
        # Get regime label on each shock date
        # Some shock dates may predate the HMM warmup window — drop those
        shock_regimes = df_with_regime.loc[
            df_with_regime.index.isin(shock_dates) &
            df_with_regime["regime"].notna(),
            "regime"
        ]
 
        counts = shock_regimes.value_counts().reindex(all_regimes, fill_value=0)
        pct    = (counts / counts.sum() * 100).round(1)
 
        results[f"{label}_counts"] = counts
        results[f"{label}_pct"]    = pct
        results[f"{label}_n"]      = len(shock_regimes)
 
        print(f"\n[regime_dist] {shock_col} — {len(shock_regimes)} events with regime labels:")
        for regime in all_regimes:
            n   = counts[regime]
            p   = pct[regime]
            b   = baseline_pct[regime]
            lift = p - b
            sign = "+" if lift >= 0 else ""
            print(f"  {regime:<12} {n:>3} events  {p:>5.1f}%  "
                  f"(baseline {b:.1f}%  lift {sign}{lift:.1f}pp)")
 
    results["regime_labels"] = regime_labels
    results["baseline_pct"]  = baseline_pct
    results["detector"]      = detector





# ── 6. Strategy Backtest ──────────────────────────────────────────────────────

def run_strategy_backtest(df, shock_col="daily_shock", hold_days=3):
    """
    Build signal via oil_shock_short.py, run via engine.py, report via performance.py.
    Returns (df_result, metrics).
    """
    # Prepare dataframe for strategy — needs Close (S&P 500) + shock_col
    strat_df = df[["Close", shock_col]].copy()

    # generate_signals already shifts position by 1 — no lookahead
    strat_df = generate_signals(strat_df, shock_col=shock_col, hold_days=hold_days)
    strat_df = run_backtest(strat_df)

    metrics = calculate_metrics(strat_df)
    print_summary(metrics, strategy_name=f"{STRATEGY_NAME}  [^GSPC | {shock_col} | hold={hold_days}d]")

    return strat_df, metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_event_study(daily_results, weekly_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("Event Study — Cumulative Abnormal Returns Post Oil Shock",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, results, label in zip(axes,
                                   [daily_results, weekly_results],
                                   ["Daily Shock (>5%)", "Weekly Shock (>10%)"]):
        ax.set_facecolor("#0d0d0d")
        horizons   = list(results.keys())
        mean_cars  = [results[h]["mean_car"] for h in horizons]
        std_cars   = [results[h]["std_car"]  for h in horizons]
        p_vals     = [results[h]["p_val"]    for h in horizons]
        baselines  = [results[h]["baseline_mean"] for h in horizons]

        x = np.arange(len(horizons))
        colors = ["#ef4444" if v < 0 else "#22c55e" for v in mean_cars]

        bars = ax.bar(x, mean_cars, color=colors, alpha=0.85, width=0.5, zorder=3)
        ax.errorbar(x, mean_cars, yerr=std_cars, fmt="none",
                    color="white", alpha=0.5, capsize=4, linewidth=1)
        ax.scatter(x, baselines, color="#facc15", s=60, zorder=5,
                   label="Baseline (random)", marker="D")

        ax.axhline(0, color="white", linewidth=0.5, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in horizons], color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.yaxis.label.set_color("white")
        ax.set_ylabel("Mean CAR", color="white")
        ax.set_title(label, color="#a3a3a3", fontsize=10)
        ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)

        for i, (bar, p) in enumerate(zip(bars, p_vals)):
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            if sig:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.001,
                        sig, ha="center", color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_conditional_prob(daily_results, weekly_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("Conditional Probability — P(SP500 < 0 | Oil Shock) vs Baseline",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, results, label in zip(axes,
                                   [daily_results, weekly_results],
                                   ["Daily Shock (>5%)", "Weekly Shock (>10%)"]):
        ax.set_facecolor("#0d0d0d")
        horizons  = list(results.keys())
        p_shock   = [results[h]["p_neg_given_shock"] for h in horizons]
        p_base    = [results[h]["p_neg_baseline"]    for h in horizons]
        x         = np.arange(len(horizons))
        width     = 0.3

        ax.bar(x - width/2, p_base,  width, label="P(neg) baseline", color="#3b82f6", alpha=0.85)
        ax.bar(x + width/2, p_shock, width, label="P(neg | shock)",  color="#ef4444", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in horizons], color="white")
        ax.set_ylabel("Probability", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.set_ylim(0, 0.75)
        ax.set_title(label, color="#a3a3a3", fontsize=10)
        ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)

        # Lift annotations
        for i, h in enumerate(horizons):
            lift = results[h]["lift"]
            ax.text(i + width/2, p_shock[i] + 0.02,
                    f"{lift:+.1%}", ha="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_sector_heatmap(daily_sector, weekly_sector, save_path):
    if daily_sector is None and weekly_sector is None:
        print("[plot] No sector data — skipping heatmap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("Sector Returns Post Oil Shock",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, data, label in zip(axes,
                                [daily_sector, weekly_sector],
                                ["Daily Shock (>5%)", "Weekly Shock (>10%)"]):
        if data is None:
            ax.set_visible(False)
            continue
        ax.set_facecolor("#0d0d0d")
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
        vals = data.values.astype(float)
        vmax = np.nanmax(np.abs(vals))
        im   = ax.imshow(vals, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, color="white")
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, color="white")
        ax.tick_params(colors="white")
        ax.set_title(label, color="#a3a3a3", fontsize=10)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2%}", ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold")

        plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_equity_curve(df_daily, df_weekly, metrics_daily, metrics_weekly, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle("Strategy Equity Curve — Oil Shock Short vs Buy & Hold S&P 500",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, df, metrics, label in zip(
        axes,
        [df_daily, df_weekly],
        [metrics_daily, metrics_weekly],
        ["Daily Shock (>5%) | hold=3d", "Weekly Shock (>10%) | hold=5d"]
    ):
        ax.set_facecolor("#0d0d0d")
        ax.plot(df.index, df["equity_curve"],    color="#ef4444", linewidth=1.5,
                label="Oil Shock Short")
        ax.plot(df.index, df["buy_hold_equity"], color="#3b82f6", linewidth=1.0,
                alpha=0.6, label="Buy & Hold SPX")
        ax.axhline(1, color="white", linewidth=0.4, alpha=0.3, linestyle="--")

        sharpe = metrics.get("Sharpe Ratio", np.nan)
        mdd    = metrics.get("Max Drawdown", np.nan)
        tr     = metrics.get("Total Return", np.nan)
        ax.set_title(f"{label}\nSharpe={sharpe:.2f} | MDD={mdd:.1%} | Return={tr:.1%}",
                     color="#a3a3a3", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.set_ylabel("Equity", color="white")
        ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def regime_distribution(df, benchmark_ticker="TLT", feature_window=20,
                        save_path=None):
    """
    Fit HMM regime detector on ^GSPC vs TLT, attach regime labels to
    all trading days, then count how many daily and weekly shock events
    fell in each regime.

    Uses TLT (not SPY) as benchmark so index_correlation feature has
    genuine discriminatory power — equity/bond correlation varies
    meaningfully across Bull/Bear/Crisis/Transition regimes.

    Returns dict with regime counts, percentages, and baseline distribution.
    """
    from research.regime_detection.hmm_model import RegimeDetector

    print(f"[regime_dist] Loading benchmark {benchmark_ticker}...")
    try:
        benchmark = load_data(benchmark_ticker, start=START, end=END,
                              source="yfinance", ohlcv=False)
    except Exception as e:
        print(f"[regime_dist] Failed to load {benchmark_ticker}: {e}")
        print("[regime_dist] Skipping regime distribution analysis.")
        return None

    sp500_prices = df["Close"]

    print(f"[regime_dist] Fitting HMM (4 states, window={feature_window})...")
    detector = RegimeDetector(
        n_regimes    = 4,
        n_iter       = 1000,
        window       = feature_window,
        random_state = 42,
    )
    detector.fit(sp500_prices, benchmark["Close"])

    regime_labels = detector.predict(sp500_prices, benchmark["Close"])
    print(f"[regime_dist] Regime labels computed: {len(regime_labels)} days")

    all_regimes     = ["Bull", "Transition", "Bear", "Crisis"]
    baseline_counts = regime_labels.value_counts().reindex(all_regimes, fill_value=0)
    baseline_pct    = (baseline_counts / baseline_counts.sum() * 100).round(1)

    df_with_regime          = df.copy()
    df_with_regime["regime"] = regime_labels

    results = {}
    for shock_col, label in [("daily_shock", "daily"), ("weekly_shock", "weekly")]:
        shock_regimes = df_with_regime.loc[
            (df_with_regime[shock_col] == 1) &
            df_with_regime["regime"].notna(),
            "regime"
        ]

        counts = shock_regimes.value_counts().reindex(all_regimes, fill_value=0)
        pct    = (counts / counts.sum() * 100).round(1)

        results[f"{label}_counts"] = counts
        results[f"{label}_pct"]    = pct
        results[f"{label}_n"]      = len(shock_regimes)

        print(f"\n[regime_dist] {shock_col} — {len(shock_regimes)} events with regime labels:")
        for regime in all_regimes:
            n    = counts[regime]
            p    = pct[regime]
            b    = baseline_pct[regime]
            lift = p - b
            sign = "+" if lift >= 0 else ""
            print(f"  {regime:<12} {n:>3} events  {p:>5.1f}%  "
                  f"(baseline {b:.1f}%  lift {sign}{lift:.1f}pp)")

    results["regime_labels"] = regime_labels
    results["baseline_pct"]  = baseline_pct
    results["detector"]      = detector

    if save_path:
        _plot_regime_distribution(results, all_regimes, save_path)

    return results


def _plot_regime_distribution(results, all_regimes, save_path):
    """
    Three-panel plot:
      Left   — grouped bar: shock % vs baseline % per regime (daily shocks)
      Center — grouped bar: shock % vs baseline % per regime (weekly shocks)
      Right  — stacked bar: absolute shock counts, daily vs weekly side by side
    """
    import matplotlib.pyplot as plt
    import numpy as np
 
    REGIME_COLORS = {
        "Bull":       "#22c55e",
        "Transition": "#f59e0b",
        "Bear":       "#ef4444",
        "Crisis":     "#8b5cf6",
    }
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle(
        "Shock Event Distribution by HMM Regime vs Baseline\n"
        "(TLT as benchmark — equity/bond correlation signal)",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )
 
    baseline = results["baseline_pct"].values
    x        = np.arange(len(all_regimes))
    colors   = [REGIME_COLORS[r] for r in all_regimes]
    width    = 0.35
 
    # ── Panel 1: Daily shock % vs baseline ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d0d0d")
    daily_pct = results["daily_pct"].values
    b1 = ax.bar(x - width/2, baseline,  width, label="Baseline % of days",
                color="#3b82f6", alpha=0.7)
    b2 = ax.bar(x + width/2, daily_pct, width, label="Daily shock %",
                color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(all_regimes, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_ylabel("% of events", color="white")
    ax.set_title(f"Daily Shocks (n={results['daily_n']})\nvs Baseline",
                 color="#a3a3a3", fontsize=10)
    ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8, fontsize=8)
 
    # Lift annotations
    for i, (d, b) in enumerate(zip(daily_pct, baseline)):
        lift = d - b
        sign = "+" if lift >= 0 else ""
        ax.text(x[i] + width/2, d + 0.5, f"{sign}{lift:.1f}pp",
                ha="center", color="white", fontsize=8)
 
    # ── Panel 2: Weekly shock % vs baseline ──────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#0d0d0d")
    weekly_pct = results["weekly_pct"].values
    ax.bar(x - width/2, baseline,   width, label="Baseline % of days",
           color="#3b82f6", alpha=0.7)
    ax.bar(x + width/2, weekly_pct, width, label="Weekly shock %",
           color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(all_regimes, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_ylabel("% of events", color="white")
    ax.set_title(f"Weekly Shocks (n={results['weekly_n']})\nvs Baseline",
                 color="#a3a3a3", fontsize=10)
    ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8, fontsize=8)
 
    for i, (w, b) in enumerate(zip(weekly_pct, baseline)):
        lift = w - b
        sign = "+" if lift >= 0 else ""
        ax.text(x[i] + width/2, w + 0.5, f"{sign}{lift:.1f}pp",
                ha="center", color="white", fontsize=8)
 
    # ── Panel 3: Absolute counts, daily vs weekly ─────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#0d0d0d")
    daily_counts  = results["daily_counts"].values
    weekly_counts = results["weekly_counts"].values
    ax.bar(x - width/2, daily_counts,  width, label="Daily shocks",
           color=colors, alpha=0.9)
    ax.bar(x + width/2, weekly_counts, width, label="Weekly shocks",
           color=colors, alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(all_regimes, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_ylabel("Event count", color="white")
    ax.set_title("Absolute Counts\nDaily vs Weekly", color="#a3a3a3", fontsize=10)
    ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8, fontsize=8)
 
    for i, (d, w) in enumerate(zip(daily_counts, weekly_counts)):
        if d > 0:
            ax.text(x[i] - width/2, d + 0.3, str(d),
                    ha="center", color="white", fontsize=9)
        if w > 0:
            ax.text(x[i] + width/2, w + 0.3, str(w),
                    ha="center", color="white", fontsize=9)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[plot] Saved → {save_path}")
 


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FINDINGS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  Oil Price Shocks → Equity Market Drawdowns")
    print("  @AlphaByProcess  |  research/analysis_final/oil_shock_equity")
    print("="*60)

    # ── 1. Load ───────────────────────────────────────────────────
    df = load_all_data()

    # ── 2. Shocks ─────────────────────────────────────────────────
    df = identify_shocks(df)

    # ── 3. Event Study ────────────────────────────────────────────
    print("\n── Event Study ──────────────────────────────────────────")
    daily_es  = event_study(df, shock_col="daily_shock")
    weekly_es = event_study(df, shock_col="weekly_shock")

    # ── 4. Conditional Probability ────────────────────────────────
    print("\n── Conditional Probability ──────────────────────────────")
    daily_cp  = conditional_probability(df, shock_col="daily_shock")
    weekly_cp = conditional_probability(df, shock_col="weekly_shock")

    # ── 5. Sector Analysis ────────────────────────────────────────
    print("\n── Sector Analysis ──────────────────────────────────────")
    daily_sec  = sector_analysis(df, shock_col="daily_shock")
    weekly_sec = sector_analysis(df, shock_col="weekly_shock")


    # ── 5. Regime Detection ────────────────────────────────────────
    print("\n── Regime Distribution ──────────────────────────────────")
    regime_dist = regime_distribution(df, save_path=os.path.join(FINDINGS_DIR, "regime_distribution.png"))
    
    # ── 6. Strategy Backtest ──────────────────────────────────────
    print("\n── Strategy Backtest ────────────────────────────────────")
    df_daily_bt,  metrics_daily  = run_strategy_backtest(df, shock_col="daily_shock",  hold_days=3)
    df_weekly_bt, metrics_weekly = run_strategy_backtest(df, shock_col="weekly_shock", hold_days=5)

    # ── Plots ─────────────────────────────────────────────────────
    print("\n── Saving Plots ─────────────────────────────────────────")
    plot_event_study(daily_es, weekly_es,
                     os.path.join(FINDINGS_DIR, "event_study.png"))
    plot_conditional_prob(daily_cp, weekly_cp,
                          os.path.join(FINDINGS_DIR, "conditional_prob.png"))
    plot_sector_heatmap(daily_sec, weekly_sec,
                        os.path.join(FINDINGS_DIR, "sector_heatmap.png"))
    plot_equity_curve(df_daily_bt, df_weekly_bt, metrics_daily, metrics_weekly,
                      os.path.join(FINDINGS_DIR, "strategy_equity_curve.png"))

    print("\n" + "="*60)
    print("  Done. All outputs saved to findings/")
    print("="*60 + "\n")





if __name__ == "__main__":
    main()