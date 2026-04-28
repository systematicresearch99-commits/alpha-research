"""
pcp_arbitrage_efficiency.py — Testing Arbitrage Efficiency in Options Markets
==============================================================================
Research script for @AlphaByProcess.
Tests whether S&P 500 options prices satisfy put-call parity, and whether
observed deviations are exploitable after transaction costs or merely reflect
structural premia and market frictions.

Put-Call Parity: C - P = S - K·e^(-r·T)

Steps:
    1. Load data  — ^GSPC, ^IRX, ^VIX, SPY (options chains + dividend yield), TLT
    2. Assemble matched pairs — ATM (C, P) at identical (K, T)
    3. Compute deviations — δ_raw, δ_adj (cost-adjusted), δ_norm (% of spot)
    4. Arbitrage bounds test — frequency and magnitude of bound violations
    5. Deviation magnitude analysis — autocorrelation, half-life of mean reversion
    6. Regime conditioning — deviations by HMM state (Bull/Transition/Bear/Crisis)
    7. VIX conditioning — mean |δ| by VIX quintile, OLS β
    8. Term structure comparison — deviations by DTE bucket
    9. Simulated arbitrage P&L — simplified exploitation estimate

Usage:
    python research/analysis/pcp_arbitrage_efficiency/pcp_arbitrage_efficiency.py
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

try:
    from utils.data_loader import load_data
    from utils.performance import calculate_metrics, print_summary
    _FRAMEWORK = True
except ImportError:
    _FRAMEWORK = False
    print("[init] Framework utils not found — running in standalone mode (yfinance direct).")

# ── Config ────────────────────────────────────────────────────────────────────
START               = "2010-01-01"
END                 = None               # today
MIN_PREMIUM         = 0.05               # filter near-zero-premium legs
MAX_STRIKE_PCT_DEV  = 0.05               # ATM filter: |K/S - 1| < 5%
DTE_BUCKETS         = [0, 14, 30, 60, 999]
DTE_LABELS          = ["0–14d", "15–30d", "31–60d", "60d+"]
BA_PROXY_FALLBACK   = 0.002              # 0.2% of mid fallback if spread unavailable
HALFLIFE_LAGS       = 20                 # max lags for AR(1) half-life estimation
VIX_QUINTILES       = 5

FINDINGS_DIR = os.path.join(os.path.dirname(__file__), "findings")

# ── Dark theme constants ──────────────────────────────────────────────────────
DARK_BG  = "#0f0f0f"
TEXT_COL = "#e5e5e5"
DIM_COL  = "#888888"
COLORS = {
    "pos":      "#22c55e",   # calls rich
    "neg":      "#ef4444",   # puts rich (structural)
    "neutral":  "#3b82f6",
    "adj":      "#f59e0b",
    "regime":   {"Bull": "#22c55e", "Transition": "#f59e0b",
                 "Bear": "#ef4444", "Crisis": "#8b5cf6"},
}


# ── 1. Data Loading ───────────────────────────────────────────────────────────

def _yf_load(ticker, start, end, col="Close"):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[[col]].rename(columns={col: ticker})


def load_market_data(start=START, end=END):
    """
    Load spot (^GSPC), risk-free rate (^IRX), VIX (^VIX),
    SPY (for dividend yield proxy), TLT (HMM benchmark).
    Returns merged DataFrame on trading dates.
    """
    print("[data] Loading ^GSPC (spot)...")
    sp500 = _yf_load("^GSPC", start, end).rename(columns={"^GSPC": "Close"})

    print("[data] Loading ^IRX (risk-free rate)...")
    irx = _yf_load("^IRX", start, end).rename(columns={"^IRX": "IRX"})

    print("[data] Loading ^VIX...")
    vix = _yf_load("^VIX", start, end).rename(columns={"^VIX": "VIX"})

    print("[data] Loading TLT (HMM benchmark)...")
    tlt = _yf_load("TLT", start, end).rename(columns={"TLT": "TLT"})

    df = sp500.join(irx, how="inner").join(vix, how="left").join(tlt, how="left")

    # Risk-free: ^IRX is annualised %, convert to continuously compounded decimal
    df["r_ann"] = df["IRX"] / 100.0
    df["sp500_ret"] = df["Close"].pct_change()
    df.dropna(subset=["Close", "r_ann", "VIX"], inplace=True)

    print(f"[data] Market data: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_options_chain(ticker="SPY", spot=None):
    """
    Fetch current options chain for SPY using yfinance.
    Returns DataFrame of matched (C, P) pairs at same (K, T).

    Note: yfinance only provides current chain snapshots.
    For historical analysis, this function is called once and results cached.
    A full historical study requires CBOE DataShop or OptionMetrics — document
    this limitation clearly in findings.
    """
    print(f"[options] Fetching options chain for {ticker}...")
    tkr = yf.Ticker(ticker)

    # Always use the ticker's own spot price for ATM filtering —
    # never the index level (^GSPC ~7174 vs SPY ~560 would zero out all pairs)
    try:
        spy_info  = tkr.fast_info
        spy_spot  = spy_info.last_price
        if spy_spot is None or spy_spot <= 0:
            raise ValueError("fast_info returned invalid price")
        print(f"[options] {ticker} spot (fast_info): {spy_spot:.2f}")
    except Exception:
        # Fallback: pull last close from recent history
        spy_hist = tkr.history(period="2d")
        spy_spot = float(spy_hist["Close"].iloc[-1])
        print(f"[options] {ticker} spot (history fallback): {spy_spot:.2f}")

    # Override the caller-supplied spot with the correct ticker spot
    spot = spy_spot

    expirations = tkr.options

    if not expirations:
        print(f"[options] No expiry dates available for {ticker}.")
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()
    records = []

    for exp_str in expirations:
        try:
            exp_date = pd.Timestamp(exp_str)
            T_days = (exp_date - today).days
            if T_days <= 0:
                continue
            T = T_days / 365.0

            chain = tkr.option_chain(exp_str)
            calls = chain.calls[["strike", "lastPrice", "bid", "ask"]].copy()
            puts  = chain.puts[["strike", "lastPrice", "bid", "ask"]].copy()

            calls.columns = ["K", "C_last", "C_bid", "C_ask"]
            puts.columns  = ["K", "P_last", "P_bid", "P_ask"]

            merged = calls.merge(puts, on="K", suffixes=("_c", "_p"))

            # Compute mid quotes
            merged["C_mid"] = (merged["C_bid"] + merged["C_ask"]) / 2
            merged["P_mid"] = (merged["P_bid"] + merged["P_ask"]) / 2

            # BA proxy: sum of half-spreads
            merged["C_half_ba"] = (merged["C_ask"] - merged["C_bid"]) / 2
            merged["P_half_ba"] = (merged["P_ask"] - merged["P_bid"]) / 2
            merged["ba_proxy"]  = merged["C_half_ba"] + merged["P_half_ba"]

            merged["T"]       = T
            merged["T_days"]  = T_days
            merged["expiry"]  = exp_str
            merged["snap_date"] = today

            records.append(merged)
        except Exception as e:
            print(f"[options] Skipped expiry {exp_str}: {e}")
            continue

    if not records:
        return pd.DataFrame()

    df_opts = pd.concat(records, ignore_index=True)

    # Filter near-zero-premium legs
    df_opts = df_opts[
        (df_opts["C_mid"] > MIN_PREMIUM) &
        (df_opts["P_mid"] > MIN_PREMIUM)
    ].copy()

    # ATM filter: |K/S - 1| < MAX_STRIKE_PCT_DEV
    if spot is not None:
        df_opts["moneyness"] = df_opts["K"] / spot - 1
        df_opts = df_opts[df_opts["moneyness"].abs() < MAX_STRIKE_PCT_DEV].copy()

    print(f"[options] Matched pairs after ATM filter: {len(df_opts)}")
    return df_opts, spot   # return ticker spot — callers must NOT use ^GSPC level


# ── 2. PCP Deviation Calculation ──────────────────────────────────────────────

def compute_pcp_deviations(df_opts, spot, r_ann, div_yield=0.0):
    """
    Compute put-call parity deviations for each matched (C, P, K, T) pair.

    PCP identity (with continuous dividend yield q):
        C - P = S·e^(-q·T) - K·e^(-r·T)

    δ_raw  = (C_mid - P_mid) - (S·e^(-q·T) - K·e^(-r·T))
    δ_adj  = |δ_raw| - ba_proxy     (cost-adjusted; > 0 = exploitable)
    δ_norm = δ_raw / S              (normalised by spot)
    """
    df = df_opts.copy()
    S = spot
    r = r_ann
    q = div_yield

    # PCP theoretical value: S·e^(-q·T) - K·e^(-r·T)
    df["pcp_theory"] = S * np.exp(-q * df["T"]) - df["K"] * np.exp(-r * df["T"])

    # Observed synthetic: C - P
    df["synthetic"] = df["C_mid"] - df["P_mid"]

    df["delta_raw"]  = df["synthetic"] - df["pcp_theory"]
    df["delta_norm"] = df["delta_raw"] / S
    df["delta_adj"]  = df["delta_raw"].abs() - df["ba_proxy"].fillna(BA_PROXY_FALLBACK * S)
    df["exploitable"] = (df["delta_adj"] > 0).astype(int)
    df["sign"]        = np.sign(df["delta_raw"])   # +1 = calls rich, -1 = puts rich

    return df


def summarise_deviations(df_pcp, label="All"):
    """Print basic summary statistics for deviations."""
    d = df_pcp["delta_norm"].dropna()
    print(f"\n[pcp | {label}]")
    print(f"  N pairs:            {len(d)}")
    print(f"  Mean δ_norm:        {d.mean()*100:.4f}%")
    print(f"  Median δ_norm:      {d.median()*100:.4f}%")
    print(f"  Std δ_norm:         {d.std()*100:.4f}%")
    print(f"  5th / 95th pct:     {d.quantile(0.05)*100:.4f}%  /  {d.quantile(0.95)*100:.4f}%")
    n_exploit = df_pcp["exploitable"].sum()
    print(f"  Exploitable (δ_adj > 0): {n_exploit} / {len(df_pcp)}  ({n_exploit/len(df_pcp)*100:.1f}%)")
    sign_neg = (df_pcp["sign"] < 0).sum()
    print(f"  Puts rich (δ < 0):  {sign_neg} / {len(df_pcp)}  ({sign_neg/len(df_pcp)*100:.1f}%)")
    return d


# ── 3. Arbitrage Bounds Test ──────────────────────────────────────────────────

def arbitrage_bounds_test(df_pcp, label=""):
    """
    Classical no-arbitrage bounds test.
    Violation: |δ_raw| > ba_proxy
    Reports frequency and average magnitude of violations.
    """
    df = df_pcp.copy()
    violations = df[df["exploitable"] == 1]
    n_total = len(df)
    n_viol  = len(violations)

    print(f"\n[bounds | {label}] Violation frequency: {n_viol}/{n_total} ({n_viol/n_total*100:.1f}%)")
    if n_viol > 0:
        mag = violations["delta_adj"].mean()
        print(f"  Mean |δ_adj| for violations: {mag:.4f} ({mag/df['delta_norm'].mean():.1f}× mean δ)")
        pos_viol = (violations["delta_raw"] > 0).sum()
        neg_viol = (violations["delta_raw"] < 0).sum()
        print(f"  Sign: calls rich violations: {pos_viol} | puts rich: {neg_viol}")

    return {
        "n_total": n_total,
        "n_violations": n_viol,
        "violation_rate": n_viol / n_total if n_total > 0 else 0,
        "mean_viol_magnitude": violations["delta_adj"].mean() if n_viol > 0 else 0,
    }


# ── 4. Deviation Magnitude Analysis ──────────────────────────────────────────

def deviation_magnitude_analysis(df_pcp):
    """
    Autocorrelation and half-life of mean reversion for δ_norm.
    Requires a time-indexed series of deviations.
    (Meaningful when called on a time series of snapshots — see note below.)

    Note: with a single snapshot this step reports cross-sectional dispersion only.
    For time-series mean reversion, re-run with multiple daily snapshots and
    concatenate df_pcp across dates.
    """
    d = df_pcp["delta_norm"].dropna().values

    if len(d) < 10:
        print("[reversion] Too few observations for mean reversion analysis.")
        return {}

    # AR(1) regression: Δδ_t = α + β·δ_{t-1}
    delta_lag  = d[:-1]
    delta_diff = np.diff(d)
    slope, intercept, r_val, p_val, se = stats.linregress(delta_lag, delta_diff)

    # Half-life: ln(2) / |β| (only meaningful if β < 0)
    half_life = np.log(2) / abs(slope) if slope < 0 else np.inf

    print(f"\n[reversion] AR(1) β = {slope:.4f}  (p = {p_val:.4f})")
    print(f"  Half-life of mean reversion: {half_life:.1f} observations")
    if half_life < 2:
        print("  → Deviations are fleeting — not exploitable at daily frequency")
    elif half_life < 10:
        print("  → Moderate persistence — borderline exploitable")
    else:
        print("  → Long half-life — structural premium or illiquidity")

    # Lag-1 autocorrelation
    autocorr = pd.Series(d).autocorr(lag=1)
    print(f"  Lag-1 autocorrelation: {autocorr:.4f}")

    return {
        "ar1_beta": slope,
        "ar1_pval": p_val,
        "half_life": half_life,
        "autocorr_lag1": autocorr,
    }


# ── 5. Regime Conditioning ────────────────────────────────────────────────────

def regime_conditioning(df_pcp, df_market):
    """
    Attach HMM regime labels to options snapshot dates and compare
    mean |δ_norm| and violation rate across Bull/Transition/Bear/Crisis.

    Falls back gracefully if HMM module unavailable — uses VIX quintile
    as a regime proxy instead.
    """
    print("\n[regime] Attaching regime labels to options data...")

    # Attempt framework HMM
    hmm_available = False
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from research.regime_detection.hmm_model import RegimeDetector
        from research.regime_detection.features import compute_features

        features = compute_features(df_market)
        detector = RegimeDetector(n_states=4)
        detector.fit(features)
        df_market = detector.assign_regimes(df_market, features)
        hmm_available = True
        print("[regime] HMM regime labels assigned.")
    except Exception as e:
        print(f"[regime] HMM not available ({e}) — using VIX quartile proxy.")
        # Use historical VIX quartiles on df_market (full time series) so regime
        # labels vary across dates. The single today-snapshot maps to exactly one
        # regime label — printed below with a clear note.
        df_market["regime"] = pd.qcut(
            df_market["VIX"], q=4,
            labels=["Bull", "Transition", "Bear", "Crisis"]
        )

    # Map regime to each options snapshot date
    snap_dates = df_pcp["snap_date"].unique()
    regime_map = {}
    for d in snap_dates:
        try:
            regime_map[d] = df_market.loc[df_market.index <= d, "regime"].iloc[-1]
        except Exception:
            regime_map[d] = "Unknown"

    df_pcp = df_pcp.copy()
    df_pcp["regime"] = df_pcp["snap_date"].map(regime_map)

    results = []
    for regime in ["Bull", "Transition", "Bear", "Crisis"]:
        sub = df_pcp[df_pcp["regime"] == regime]
        if len(sub) == 0:
            continue
        results.append({
            "regime": regime,
            "n": len(sub),
            "mean_abs_delta_norm": sub["delta_norm"].abs().mean() * 100,
            "mean_delta_norm":     sub["delta_norm"].mean() * 100,
            "violation_rate":      sub["exploitable"].mean() * 100,
            "puts_rich_pct":       (sub["sign"] < 0).mean() * 100,
        })
        print(f"  {regime:12s} | n={len(sub):4d} | mean |δ|={sub['delta_norm'].abs().mean()*100:.4f}% "
              f"| exploitable={sub['exploitable'].mean()*100:.1f}%")

    if len(df_pcp["snap_date"].unique()) == 1:
        print(f"  [note] Single snapshot — all {len(df_pcp)} pairs map to one regime.")
        print(f"  Regime variation requires a multi-day panel (run daily, concatenate).")

    return pd.DataFrame(results), df_pcp


# ── 6. VIX Conditioning ───────────────────────────────────────────────────────

def vix_conditioning(df_pcp, df_market):
    """
    Bin observations by VIX quintile. Report mean |δ_norm| per quintile.
    OLS: |δ_norm| ~ β₀ + β₁·VIX

    Single-snapshot mode: when all pairs share one snap_date, VIX is constant
    across the cross-section. VIX quintile binning is replaced by DTE quintile
    binning (which does vary cross-sectionally). OLS uses historical VIX from
    df_market vs daily historical |δ| proxy (rolling std of GSPC returns as
    a realized-vol proxy) — documented clearly. Full VIX conditioning requires
    a multi-day panel; re-run daily and concatenate df_pcp to enable it.
    """
    print("\n[vix] VIX quintile conditioning...")

    df_pcp = df_pcp.copy()

    # Attach VIX on each snapshot date
    df_pcp["vix_on_date"] = df_pcp["snap_date"].map(
        lambda d: df_market.loc[df_market.index <= d, "VIX"].iloc[-1]
        if len(df_market.loc[df_market.index <= d]) > 0 else np.nan
    )

    n_unique_vix = df_pcp["vix_on_date"].nunique()

    # ── Single-snapshot: VIX is constant across all rows ──────────────────────
    if n_unique_vix <= 1:
        current_vix = df_pcp["vix_on_date"].iloc[0]
        print(f"  [note] Single snapshot — VIX constant at {current_vix:.2f} across all pairs.")
        print(f"  Substituting DTE quintile binning (varies cross-sectionally).")
        print(f"  Full VIX quintile analysis requires a multi-day panel.")

        # Bin by DTE instead — this does vary across pairs
        df_pcp["vix_quintile"] = pd.qcut(
            df_pcp["T_days"],
            q=VIX_QUINTILES,
            labels=[f"DTE-Q{i+1}" for i in range(VIX_QUINTILES)],
            duplicates="drop"
        )

        results = []
        for q_label in df_pcp["vix_quintile"].cat.categories:
            sub = df_pcp[df_pcp["vix_quintile"] == q_label]
            dte_mid = sub["T_days"].median()
            mean_abs = sub["delta_norm"].abs().mean() * 100
            results.append({"quintile": q_label, "vix_median": current_vix,
                             "dte_median": dte_mid,
                             "mean_abs_delta_norm": mean_abs, "n": len(sub)})
            print(f"  {q_label} (DTE ≈ {dte_mid:.0f}d, VIX={current_vix:.1f}) "
                  f"| mean |δ| = {mean_abs:.4f}%  n={len(sub)}")

        # OLS: |δ_norm| ~ T_days (cross-sectional term structure slope)
        y = df_pcp["delta_norm"].abs() * 100
        x = df_pcp["T_days"].astype(float)
        if x.nunique() > 1:
            slope, intercept, r_val, p_val, se = stats.linregress(x, y)
            print(f"\n  OLS: |δ_norm| ~ DTE  →  β₁ = {slope:.5f}  "
                  f"(p = {p_val:.4f})  R² = {r_val**2:.4f}")
            print(f"  [note] β₁ here measures term structure slope, not VIX sensitivity.")
            print(f"  Re-run daily to accumulate panel for true VIX conditioning.")
        else:
            slope, p_val = 0.0, 1.0
            print("  OLS skipped — insufficient DTE variation.")

        return pd.DataFrame(results), slope, p_val

    # ── Multi-snapshot (panel) mode: true VIX quintile binning ────────────────
    df_pcp["vix_quintile"] = pd.qcut(
        df_pcp["vix_on_date"].fillna(df_pcp["vix_on_date"].median()),
        q=VIX_QUINTILES,
        labels=[f"Q{i+1}" for i in range(VIX_QUINTILES)],
        duplicates="drop"
    )

    results = []
    for q_label in df_pcp["vix_quintile"].cat.categories:
        sub = df_pcp[df_pcp["vix_quintile"] == q_label]
        vix_mid = sub["vix_on_date"].median()
        mean_abs = sub["delta_norm"].abs().mean() * 100
        results.append({"quintile": q_label, "vix_median": vix_mid,
                         "mean_abs_delta_norm": mean_abs, "n": len(sub)})
        print(f"  {q_label} (VIX ≈ {vix_mid:.1f}) | mean |δ| = {mean_abs:.4f}%  n={len(sub)}")

    y = df_pcp["delta_norm"].abs() * 100
    x = df_pcp["vix_on_date"].fillna(df_pcp["vix_on_date"].median())
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    print(f"\n  OLS: |δ_norm| ~ VIX  →  β₁ = {slope:.5f}  (p = {p_val:.4f})  R² = {r_val**2:.4f}")
    if p_val < 0.05 and slope > 0:
        print("  → VIX positively predicts deviation magnitude (significant)")
    else:
        print("  → VIX does not significantly predict deviations")

    return pd.DataFrame(results), slope, p_val


# ── 7. Term Structure Comparison ─────────────────────────────────────────────

def term_structure_comparison(df_pcp):
    """
    Compare |δ_norm| across DTE buckets.
    Expectation: larger deviations at very short and very long expiries.
    """
    print("\n[term] Term structure comparison by DTE bucket...")

    df = df_pcp.copy()
    df["dte_bucket"] = pd.cut(
        df["T_days"], bins=DTE_BUCKETS, labels=DTE_LABELS, right=True
    )

    results = []
    for label in DTE_LABELS:
        sub = df[df["dte_bucket"] == label]
        if len(sub) == 0:
            continue
        mean_abs = sub["delta_norm"].abs().mean() * 100
        viol_rate = sub["exploitable"].mean() * 100
        results.append({
            "dte_bucket": label,
            "n": len(sub),
            "mean_abs_delta_norm": mean_abs,
            "violation_rate": viol_rate,
        })
        print(f"  {label:8s} | n={len(sub):4d} | mean |δ| = {mean_abs:.4f}%  "
              f"exploitable = {viol_rate:.1f}%")

    return pd.DataFrame(results)


# ── 8. Simulated Arbitrage P&L ────────────────────────────────────────────────

def simulate_arb_pnl(df_pcp, spot, r_ann, label=""):
    """
    For each pair where δ_adj > 0 (exploitable after costs):
    Compute simplified P&L assuming mid-quote execution at snapshot
    and hold to expiry.

    Strategy:
      If δ_raw > 0 (calls rich): sell call, buy put, buy stock, borrow K·e^(-rT)
      If δ_raw < 0 (puts rich):  buy call, sell put, short stock, lend K·e^(-rT)

    P&L at expiry = |δ_raw| − ba_proxy  (= δ_adj)
    Simplified: ignores path-dependent costs, margin, carry on stock leg.
    """
    print(f"\n[arb pnl | {label}] Simulated arbitrage P&L...")
    arb = df_pcp[df_pcp["exploitable"] == 1].copy()

    if len(arb) == 0:
        print("  No exploitable deviations found.")
        return {}

    # P&L = δ_adj (cost-adjusted deviation = net capture after spread)
    pnl = arb["delta_adj"]

    mean_pnl   = pnl.mean()
    win_rate   = (pnl > 0).mean()
    mean_norm  = (pnl / spot).mean() * 100

    print(f"  N exploitable pairs:  {len(arb)}")
    print(f"  Mean P&L:             {mean_pnl:.4f} ({mean_norm:.4f}% of spot)")
    print(f"  Win rate:             {win_rate*100:.1f}%")
    print(f"  Std P&L:              {pnl.std():.4f}")
    if pnl.std() > 0:
        sharpe = mean_pnl / pnl.std() * np.sqrt(252)
        print(f"  Annualised Sharpe:    {sharpe:.2f}  (assumes daily, for illustration)")
    else:
        print("  Sharpe: undefined (zero variance)")

    print("  [caveat] Mid-quote execution assumed. Real slippage reduces P&L.")
    return {
        "n_arb": len(arb),
        "mean_pnl": mean_pnl,
        "win_rate": win_rate,
        "mean_pnl_norm": mean_norm,
    }


# ── 9. Plots ──────────────────────────────────────────────────────────────────

def _apply_dark(ax, title, xlabel, ylabel):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT_COL)
    ax.set_ylabel(ylabel, color=TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.spines[list(ax.spines)].set_visible(False)


def plot_deviation_distribution(df_pcp, save_path):
    """Histogram of δ_norm and δ_adj."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("PCP Deviation Distribution\n"
                 "(Negative = puts rich — structural equity put premium)",
                 color=TEXT_COL, fontsize=12, fontweight="bold")

    d_norm = df_pcp["delta_norm"].dropna() * 100
    d_adj  = df_pcp["delta_adj"].dropna() * 100

    for ax, data, color, title, xlabel in [
        (axes[0], d_norm, COLORS["neutral"],
         "δ_norm (% of spot)", "δ_norm (%)"),
        (axes[1], d_adj,  COLORS["adj"],
         "δ_adj after BA cost filter", "δ_adj"),
    ]:
        ax.set_facecolor(DARK_BG)
        ax.hist(data, bins=50, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(data.mean(), color=COLORS["pos"], linewidth=1.5,
                   linestyle="--", label=f"Mean = {data.mean():.4f}%")
        ax.axvline(0, color=DIM_COL, linewidth=1.0, linestyle=":")
        _apply_dark(ax, title, xlabel, "Frequency")
        ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_regime_conditioning(regime_df, save_path):
    """Bar chart: mean |δ_norm| and violation rate by HMM regime."""
    if regime_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("PCP Deviations by Market Regime (HMM)\n"
                 "(Stress regimes expected to show wider deviations)",
                 color=TEXT_COL, fontsize=12, fontweight="bold")

    regimes = regime_df["regime"].tolist()
    x = np.arange(len(regimes))
    colors = [COLORS["regime"].get(r, COLORS["neutral"]) for r in regimes]

    for ax, col, ylabel, title in [
        (axes[0], "mean_abs_delta_norm", "Mean |δ_norm| (%)", "Mean |δ_norm| by Regime"),
        (axes[1], "violation_rate",      "Exploitable (%)",   "Violation Rate by Regime"),
    ]:
        ax.set_facecolor(DARK_BG)
        vals = regime_df[col].values
        ax.bar(x, vals, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.3f}", ha="center", color=TEXT_COL, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, color=TEXT_COL)
        _apply_dark(ax, title, "Regime", ylabel)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_vix_quintile(vix_results_df, slope, p_val, save_path):
    """Bar chart: mean |δ_norm| by VIX quintile with OLS annotation."""
    if vix_results_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    x = np.arange(len(vix_results_df))
    vals = vix_results_df["mean_abs_delta_norm"].values
    vix_meds = vix_results_df["vix_median"].values

    bars = ax.bar(x, vals, color=COLORS["neutral"], alpha=0.85)
    for i, (v, vm) in enumerate(zip(vals, vix_meds)):
        ax.text(i, v + 0.0002, f"{v:.4f}%\n(VIX≈{vm:.0f})",
                ha="center", color=TEXT_COL, fontsize=8)

    ax.axhline(vals.mean(), color=COLORS["adj"], linewidth=1.2, linestyle="--",
               label=f"Mean across quintiles")

    sig_str = f"β₁={slope:.5f}  p={'<0.001' if p_val < 0.001 else f'{p_val:.3f}'}"
    ax.text(0.97, 0.95, sig_str, transform=ax.transAxes,
            ha="right", va="top", color=DIM_COL, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(vix_results_df["quintile"].values, color=TEXT_COL)
    _apply_dark(ax,
        "Mean |δ_norm| by VIX Quintile\n(Q1 = low vol, Q5 = high vol)",
        "VIX Quintile", "Mean |δ_norm| (%)")
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_term_structure(term_df, save_path):
    """Bar chart: mean |δ_norm| by DTE bucket."""
    if term_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    x = np.arange(len(term_df))
    vals  = term_df["mean_abs_delta_norm"].values
    viols = term_df["violation_rate"].values

    ax.bar(x - 0.2, vals,  0.4, color=COLORS["neutral"], alpha=0.85, label="|δ_norm| (%)")
    ax.bar(x + 0.2, viols, 0.4, color=COLORS["adj"],     alpha=0.85, label="Exploitable (%)")

    for i, (v, vl) in enumerate(zip(vals, viols)):
        ax.text(i - 0.2, v + 0.0003, f"{v:.4f}%", ha="center", color=TEXT_COL, fontsize=8)
        ax.text(i + 0.2, vl + 0.3,   f"{vl:.1f}%", ha="center", color=TEXT_COL, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(term_df["dte_bucket"].values, color=TEXT_COL)
    _apply_dark(ax,
        "PCP Deviations by Expiry Bucket\n(Larger deviations expected at extremes)",
        "DTE Bucket", "Value")
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_strike_vs_deviation(df_pcp, spot, save_path):
    """Scatter: moneyness (K/S - 1) vs δ_norm — shows ITM/OTM asymmetry."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    if "moneyness" not in df_pcp.columns:
        df_pcp = df_pcp.copy()
        df_pcp["moneyness"] = df_pcp["K"] / spot - 1

    pos = df_pcp[df_pcp["delta_norm"] >= 0]
    neg = df_pcp[df_pcp["delta_norm"] <  0]

    ax.scatter(pos["moneyness"] * 100, pos["delta_norm"] * 100,
               color=COLORS["pos"], alpha=0.5, s=12, label="Calls rich (δ > 0)")
    ax.scatter(neg["moneyness"] * 100, neg["delta_norm"] * 100,
               color=COLORS["neg"], alpha=0.5, s=12, label="Puts rich (δ < 0)")
    ax.axhline(0, color=DIM_COL, linewidth=0.8, linestyle=":")
    ax.axvline(0, color=DIM_COL, linewidth=0.8, linestyle=":")

    _apply_dark(ax,
        "Moneyness vs PCP Deviation\n(Negative δ = puts systematically expensive)",
        "Moneyness (K/S − 1) %", "δ_norm (%)")
    ax.legend(facecolor="#1a1a1a", labelcolor=TEXT_COL, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[plot] Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FINDINGS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  Testing Arbitrage Efficiency in Options Markets")
    print("  Put-Call Parity Deviation Study")
    print("  @AlphaByProcess  |  research/analysis/pcp_arbitrage_efficiency")
    print("="*60)

    # ── 1. Market data ────────────────────────────────────────────
    df_market  = load_market_data()
    gspc_level = df_market["Close"].iloc[-1]   # ^GSPC index level — context only
    r_ann      = df_market["r_ann"].iloc[-1]

    print(f"\n[config] ^GSPC level (context):  {gspc_level:.2f}")
    print(f"[config] Risk-free rate (^IRX):   {r_ann*100:.4f}% p.a.")

    # ── 2. Options chain ──────────────────────────────────────────
    # fetch_options_chain resolves SPY's own spot internally.
    # Do NOT pass ^GSPC level — SPY trades ~$560, ^GSPC ~7174.
    # Passing GSPC as spot causes K/spot ~ 0.07 → all pairs fail ATM filter.
    df_opts, spy_spot = fetch_options_chain(ticker="SPY")

    if df_opts.empty:
        print("[ERROR] Could not fetch options chain. Exiting.")
        return

    print(f"[config] SPY spot (used for PCP): {spy_spot:.2f}")

    # ── 3. PCP deviations ─────────────────────────────────────────
    print("\n── PCP Deviation Calculation ────────────────────────────")
    # All PCP math uses spy_spot — strikes are in SPY price terms
    div_yield = 0.013   # SPY trailing yield ~1.3%; update dynamically if needed
    df_pcp = compute_pcp_deviations(df_opts, spot=spy_spot, r_ann=r_ann,
                                     div_yield=div_yield)
    summarise_deviations(df_pcp, label="Full sample")

    # ── 4. Arbitrage bounds ───────────────────────────────────────
    print("\n── Arbitrage Bounds Test ─────────────────────────────────")
    bounds_results = arbitrage_bounds_test(df_pcp, label="Full sample")

    # ── 5. Deviation magnitude / mean reversion ───────────────────
    print("\n── Deviation Magnitude Analysis ─────────────────────────")
    reversion_results = deviation_magnitude_analysis(df_pcp)

    # ── 6. Regime conditioning ────────────────────────────────────
    print("\n── Regime Conditioning ───────────────────────────────────")
    regime_df, df_pcp = regime_conditioning(df_pcp, df_market)

    # ── 7. VIX conditioning ───────────────────────────────────────
    print("\n── VIX Quintile Conditioning ─────────────────────────────")
    vix_df, vix_slope, vix_pval = vix_conditioning(df_pcp, df_market)

    # ── 8. Term structure ─────────────────────────────────────────
    print("\n── Term Structure (DTE Buckets) ──────────────────────────")
    term_df = term_structure_comparison(df_pcp)

    # ── 9. Simulated arb P&L ──────────────────────────────────────
    print("\n── Simulated Arbitrage P&L ───────────────────────────────")
    arb_pnl = simulate_arb_pnl(df_pcp, spot=spy_spot, r_ann=r_ann)

    # ── Plots ─────────────────────────────────────────────────────
    print("\n── Saving Plots ─────────────────────────────────────────")
    plot_deviation_distribution(
        df_pcp,
        os.path.join(FINDINGS_DIR, "deviation_distribution.png"))
    plot_regime_conditioning(
        regime_df,
        os.path.join(FINDINGS_DIR, "regime_conditioning.png"))
    plot_vix_quintile(
        vix_df, vix_slope, vix_pval,
        os.path.join(FINDINGS_DIR, "vix_quintile.png"))
    plot_term_structure(
        term_df,
        os.path.join(FINDINGS_DIR, "expiry_bucket.png"))
    plot_strike_vs_deviation(
        df_pcp, spy_spot,
        os.path.join(FINDINGS_DIR, "moneyness_vs_deviation.png"))

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  Total matched pairs:       {len(df_pcp)}")
    print(f"  Mean δ_norm:               {df_pcp['delta_norm'].mean()*100:.4f}%")
    print(f"  Exploitable violations:    {bounds_results['n_violations']} "
          f"({bounds_results['violation_rate']*100:.1f}%)")
    if reversion_results:
        hl = reversion_results.get('half_life', float('inf'))
        hl_str = f"{hl:.1f}" if hl != float('inf') else "∞"
        print(f"  AR(1) half-life:           {hl_str} observations")
    print(f"  VIX → |δ| OLS β:          {vix_slope:.5f}  (p={vix_pval:.4f})")
    print("\n  [note] yfinance options = current snapshot only.")
    print("  For time-series analysis, loop this script daily and")
    print("  concatenate df_pcp across dates into a panel dataset.")
    print("  Full historical study: CBOE DataShop / OptionMetrics.")
    print("="*60)
    print("  Done. All outputs saved to findings/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

    