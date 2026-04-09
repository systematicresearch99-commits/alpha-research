"""
Entropy and Information Flow | @AlphaByProcess
research/analysis/entropy_information_flow/entropy_analysis.py

Run from inside the project folder:
  cd research/analysis/entropy_information_flow
  python entropy_analysis.py

Expects:
  ../../../data/raw/entropy_prices_daily.csv

Outputs:
  findings/01_entropy_timeseries.csv
  findings/01_mann_kendall.csv
  findings/02_entropy_vs_vol.csv
  findings/02_crosscorr_lead_lag.csv
  findings/03_entropy_vs_returns.csv
  findings/04_regime_entropy.csv
  findings/04_wilcoxon_tests.csv
  findings/05_india_vs_us.csv
  findings/05_bootstrap_test.csv
  findings/charts/fig1_entropy_timeseries.png
  findings/charts/fig2_entropy_vs_vol.png
  findings/charts/fig3_lead_lag.png
  findings/charts/fig4_entropy_regimes.png
  findings/charts/fig5_india_vs_us.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_IN      = "../../../data/raw/entropy_prices_daily.csv"
FINDINGS_DIR = "findings"
CHARTS_DIR   = "findings/charts"

os.makedirs(FINDINGS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR,   exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────

INDIA_SERIES = ["Nifty50", "IT", "Pharma", "FMCG", "Bank",
                "Auto", "Metal", "Realty", "Energy"]
US_SERIES    = ["SP500", "US_Tech", "US_Health", "US_Energy", "US_Financials",
                "US_ConsDisc", "US_ConsStap", "US_Industrial", "US_Materials"]
ALL_SERIES   = INDIA_SERIES + US_SERIES

# Entropy parameters
WINDOW       = 60          # rolling window (trading days)
SHANNON_BINS = 20          # bins for Shannon discretisation (tested: 10, 20, 50)
APEN_M       = 2           # ApEn / SampEn template length
APEN_R_COEF  = 0.2         # tolerance for ApEn  = APEN_R_COEF × std(series)
SAMPEN_R_COEF = 0.5        # tolerance for SampEn — needs wider window; daily returns are tiny

# ── Regime labels ─────────────────────────────────────────────────────────────

def assign_regime(date):
    if date < pd.Timestamp("2020-01-01"):
        return "pre_covid"
    elif date < pd.Timestamp("2022-01-01"):
        return "covid_recovery"
    else:
        return "rate_hike"

# ══════════════════════════════════════════════════════════════════════════════
# ENTROPY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(series, n_bins=SHANNON_BINS):
    """
    Shannon entropy on discretised return distribution.
    H = -sum(p * log2(p))
    Returns NaN if fewer than n_bins / 2 valid observations.
    """
    s = series.dropna()
    if len(s) < n_bins // 2:
        return np.nan
    counts, _ = np.histogram(s, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def approximate_entropy(series, m=APEN_M, r_coef=APEN_R_COEF):
    """
    Approximate Entropy (ApEn) — Pincus (1991).
    Tolerance r = r_coef * std(series).
    Low ApEn = regular/predictable. High ApEn = irregular/unpredictable.
    """
    u = np.array(series.dropna(), dtype=float)
    N = len(u)
    if N < 2 * m + 10:
        return np.nan
    r = r_coef * np.std(u, ddof=1)
    if r == 0:
        return np.nan

    def phi(m_val):
        templates = np.array([u[i:i + m_val] for i in range(N - m_val + 1)])
        count = np.array([
            np.sum(np.max(np.abs(templates - templates[i]), axis=1) <= r)
            for i in range(len(templates))
        ])
        return np.sum(np.log(count / (N - m_val + 1))) / (N - m_val + 1)

    return float(phi(m) - phi(m + 1))


def sample_entropy(series, m=APEN_M, r_coef=SAMPEN_R_COEF):
    """
    Sample Entropy (SampEn) — Richman & Moorman (2000).
    Eliminates self-matching bias; more robust on shorter windows than ApEn.

    r_coef is set to 0.5 (vs 0.2 for ApEn) because daily equity returns have
    very small absolute values — a tight tolerance causes A → 0 → log(0) = inf.
    Guards: return NaN if B == 0 or A == 0 to prevent inf propagation.
    """
    u = np.array(series.dropna(), dtype=float)
    N = len(u)
    if N < 2 * m + 10:
        return np.nan
    r = r_coef * np.std(u, ddof=1)
    if r == 0:
        return np.nan

    def count_matches(m_val, exclude_self=True):
        templates = np.array([u[i:i + m_val] for i in range(N - m_val)])
        total = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            if exclude_self:
                dists[i] = np.inf
            total += np.sum(dists <= r)
        return total

    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0 or A == 0:   # guard: both cases produce inf without this
        return np.nan
    result = float(-np.log(A / B))
    return result if np.isfinite(result) else np.nan  # final inf safety net


def rolling_entropy(series, window=WINDOW):
    """Compute all three entropy measures over a rolling window."""
    shannon = series.rolling(window).apply(shannon_entropy, raw=False)
    apen    = series.rolling(window).apply(approximate_entropy, raw=False)
    sampen  = series.rolling(window).apply(sample_entropy, raw=False)
    return shannon, apen, sampen


def mann_kendall(series):
    """
    Non-parametric Mann-Kendall trend test.
    Returns: tau (Kendall correlation), p-value, direction string.
    """
    s = series.dropna().values
    n = len(s)
    if n < 4:
        return np.nan, np.nan, "insufficient data"
    s_stat = 0
    for i in range(n - 1):
        s_stat += np.sum(np.sign(s[i + 1:] - s[i]))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s_stat > 0:
        z = (s_stat - 1) / np.sqrt(var_s)
    elif s_stat < 0:
        z = (s_stat + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s_stat / (0.5 * n * (n - 1))
    direction = "upward" if tau > 0 else "downward"
    return round(tau, 4), round(p, 4), direction


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD PANEL
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 2 — BUILDING PANEL")
print("=" * 60)

prices  = pd.read_csv(DATA_IN, index_col=0, parse_dates=True)
prices  = prices[ALL_SERIES]
returns = prices.pct_change().dropna()
returns["regime"] = returns.index.map(assign_regime)

print(f"  Daily observations : {len(returns)}")
print(f"  Date range         : {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"  Series             : {len(ALL_SERIES)}")
print(f"  Regime counts      : {returns['regime'].value_counts().to_dict()}")
print(f"\n  ⚠  Computing rolling entropy for {len(ALL_SERIES)} series × 3 measures.")
print(f"     Window = {WINDOW}d | ApEn/SampEn m={APEN_M}, r={APEN_R_COEF}×std")
print(f"     This may take several minutes — profiling ApEn on first series...\n")

import time
_test = returns[ALL_SERIES[0]].iloc[:200]
t0 = time.time()
_ = _test.rolling(WINDOW).apply(approximate_entropy, raw=False)
elapsed = time.time() - t0
est_total = elapsed * len(ALL_SERIES) * 3
print(f"  ApEn probe: {elapsed:.1f}s for 200 obs → est. total ≈ {est_total/60:.1f} min")
print(f"  Proceeding...\n")

# Build entropy panel
entropy_records = {}
for i, col in enumerate(ALL_SERIES):
    print(f"  [{i+1:02d}/{len(ALL_SERIES)}] {col:<20}", end="", flush=True)
    t0 = time.time()
    sh, ap, sa = rolling_entropy(returns[col])
    entropy_records[f"{col}_shannon"] = sh
    entropy_records[f"{col}_apen"]    = ap
    entropy_records[f"{col}_sampen"]  = sa
    print(f" ✓  ({time.time()-t0:.1f}s)")

entropy = pd.DataFrame(entropy_records, index=returns.index)
entropy["regime"] = returns["regime"]
entropy = entropy.dropna(subset=[f"{ALL_SERIES[0]}_shannon"])   # drop pre-window rows

# Realised volatility (same rolling window)
for col in ALL_SERIES:
    entropy[f"{col}_vol"] = returns[col].rolling(WINDOW).std() * np.sqrt(252)

print(f"\n  Entropy panel built: {len(entropy)} rows × {len(entropy.columns)} columns")
print("✓ Panel built\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — ENTROPY TIME SERIES + MANN-KENDALL TREND
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 1 — ENTROPY TIME SERIES & TREND")
print("=" * 60)

mk_results = []
ts_records = []

print(f"\n  {'Series':<22} {'Measure':<10} {'τ':>8} {'p':>8} {'Sig':>5} {'Direction'}")
print("  " + "-" * 65)

for col in ALL_SERIES:
    for measure in ["shannon", "apen", "sampen"]:
        series = entropy[f"{col}_{measure}"]
        tau, pval, direction = mann_kendall(series)
        sig = "✓" if pval is not np.nan and pval < 0.05 else " "
        mk_results.append({
            "series": col, "measure": measure,
            "tau": tau, "pval": pval, "direction": direction,
            "significant": pval < 0.05 if pval is not np.nan else False,
        })
        print(f"  {col:<22} {measure:<10} {tau:>+8.3f} {pval:>8.4f} {sig:>5}  {direction}")

    # Aggregate for time series CSV
    for idx in entropy.index:
        ts_records.append({
            "date": idx,
            "series": col,
            "shannon": entropy.loc[idx, f"{col}_shannon"],
            "apen":    entropy.loc[idx, f"{col}_apen"],
            "sampen":  entropy.loc[idx, f"{col}_sampen"],
            "regime":  entropy.loc[idx, "regime"],
        })

# Summary by measure
print(f"\n  Trend summary by measure:")
mk_df = pd.DataFrame(mk_results)
for measure in ["shannon", "apen", "sampen"]:
    sub = mk_df[mk_df["measure"] == measure]
    n_up  = (sub["direction"] == "upward").sum()
    n_dn  = (sub["direction"] == "downward").sum()
    n_sig = sub["significant"].sum()
    print(f"    {measure:<10}  ↑ {n_up}/{len(sub)}  ↓ {n_dn}/{len(sub)}  "
          f"significant: {n_sig}/{len(sub)}")

# Save
pd.DataFrame(ts_records).to_csv(f"{FINDINGS_DIR}/01_entropy_timeseries.csv", index=False)
mk_df.to_csv(f"{FINDINGS_DIR}/01_mann_kendall.csv", index=False)
print("✓ Test 1 complete\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — ENTROPY vs VOLATILITY + LEAD-LAG
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 2 — ENTROPY vs VOLATILITY")
print("=" * 60)

ev_results = []
print(f"\n  {'Series':<22} {'Measure':<10} {'ρ (full)':>10} {'p':>8} "
      f"{'ρ Pre':>8} {'ρ COVID':>8} {'ρ Rate':>8}")
print("  " + "-" * 80)

for col in ALL_SERIES:
    vol_col = f"{col}_vol"
    for measure in ["shannon", "apen", "sampen"]:
        ent_col = f"{col}_{measure}"
        row = {"series": col, "measure": measure}

        # Full sample
        x = entropy[ent_col].dropna()
        y = entropy.loc[x.index, vol_col].dropna()
        idx = x.index.intersection(y.index)
        rho_full, p_full = stats.spearmanr(x[idx], y[idx]) if len(idx) > 5 else (np.nan, np.nan)
        row.update({"rho_full": round(rho_full, 4), "pval_full": round(p_full, 4)})

        # By regime
        rho_str = ""
        for regime, key in [("pre_covid", "rho_pre"), ("covid_recovery", "rho_covid"),
                             ("rate_hike", "rho_rate")]:
            mask = entropy["regime"] == regime
            xi = entropy.loc[mask, ent_col].dropna()
            yi = entropy.loc[xi.index, vol_col].dropna()
            ii = xi.index.intersection(yi.index)
            if len(ii) > 5:
                rho_r, p_r = stats.spearmanr(xi[ii], yi[ii])
                row[key] = round(rho_r, 4)
                row[f"pval_{key[4:]}"] = round(p_r, 4)
                rho_str += f"  {rho_r:>+7.3f}"
            else:
                row[key] = np.nan
                rho_str += f"  {'N/A':>7}"

        ev_results.append(row)
        sig = "*" if p_full < 0.05 else " "
        print(f"  {col:<22} {measure:<10} {rho_full:>+9.3f}{sig}  {p_full:>7.4f}{rho_str}")

ev_df = pd.DataFrame(ev_results)

# Lead-lag cross-correlation: does entropy LEAD volatility?
print(f"\n  Lead-lag cross-correlation (entropy leads/lags vol):")
print(f"  Negative lag = entropy leads volatility\n")
print(f"  {'Series':<22} {'Measure':<10}", end="")
LAGS = list(range(-10, 11))
for lag in [-5, 0, 5]:
    print(f"  lag{lag:+d}ρ", end="")
print()
print("  " + "-" * 75)

ll_records = []
for col in ["Nifty50", "SP500"]:   # headline indices only for brevity
    vol_col = f"{col}_vol"
    for measure in ["shannon", "apen", "sampen"]:
        ent_col = f"{col}_{measure}"
        row = {"series": col, "measure": measure}
        line = f"  {col:<22} {measure:<10}"
        for lag in LAGS:
            ent_s = entropy[ent_col].dropna()
            vol_s = entropy[vol_col].dropna()
            if lag >= 0:
                x = ent_s.iloc[:len(ent_s)-lag] if lag > 0 else ent_s
                y = vol_s.shift(-lag).dropna()
            else:
                x = ent_s.shift(abs(lag)).dropna()
                y = vol_s
            idx = x.index.intersection(y.index)
            if len(idx) > 10:
                rho, _ = stats.spearmanr(x[idx], y[idx])
            else:
                rho = np.nan
            row[f"lag_{lag:+d}"] = round(rho, 4) if not np.isnan(rho) else np.nan
            if lag in [-5, 0, 5]:
                line += f"  {rho:>+7.3f}"
        ll_records.append(row)
        print(line)

ev_df.to_csv(f"{FINDINGS_DIR}/02_entropy_vs_vol.csv", index=False)
pd.DataFrame(ll_records).to_csv(f"{FINDINGS_DIR}/02_crosscorr_lead_lag.csv", index=False)
print("✓ Test 2 complete\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — ENTROPY vs FORWARD RETURNS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 3 — ENTROPY vs FORWARD RETURNS")
print("=" * 60)

FORWARD_HORIZONS = [21, 63]   # ~1 month, ~3 months in trading days

er_results = []
print(f"\n  {'Series':<22} {'Measure':<10} {'ρ fwd1m':>10} {'p':>8} {'ρ fwd3m':>10} {'p':>8}")
print("  " + "-" * 75)

for col in ALL_SERIES:
    for measure in ["shannon", "apen", "sampen"]:
        ent_col = f"{col}_{measure}"
        row = {"series": col, "measure": measure}
        line = f"  {col:<22} {measure:<10}"
        for horizon, tag in zip(FORWARD_HORIZONS, ["1m", "3m"]):
            fwd_ret = returns[col].rolling(horizon).sum().shift(-horizon)
            ent_s   = entropy[ent_col]
            idx     = ent_s.dropna().index.intersection(fwd_ret.dropna().index)
            if len(idx) > 20:
                rho, pval = stats.spearmanr(ent_s[idx], fwd_ret[idx])
            else:
                rho, pval = np.nan, np.nan
            row[f"rho_fwd{tag}"]  = round(rho, 4)  if not np.isnan(rho)  else np.nan
            row[f"pval_fwd{tag}"] = round(pval, 4) if not np.isnan(pval) else np.nan
            sig = "*" if not np.isnan(pval) and pval < 0.05 else " "
            line += f"  {rho:>+9.3f}{sig}  {pval:>7.4f}"
        er_results.append(row)
        print(line)

er_df = pd.DataFrame(er_results)
er_df.to_csv(f"{FINDINGS_DIR}/03_entropy_vs_returns.csv", index=False)
print("✓ Test 3 complete\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — ENTROPY AS REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 4 — ENTROPY REGIME CLASSIFIER")
print("=" * 60)

# Wilcoxon rank-sum test across adjacent regime transitions
regime_order   = ["pre_covid", "covid_recovery", "rate_hike"]
regime_labels  = {"pre_covid": "Pre-COVID", "covid_recovery": "COVID Recovery",
                  "rate_hike": "Rate Hike"}
wilcoxon_rows  = []
regime_entropy = []

print(f"\n  Mean entropy by regime (Nifty50 + SP500 headline):\n")
print(f"  {'Series':<12} {'Measure':<10} {'Pre-COVID':>12} {'COVID Rec':>12} {'Rate Hike':>12}")
print("  " + "-" * 62)

for col in ["Nifty50", "SP500"]:
    for measure in ["shannon", "apen", "sampen"]:
        ent_col = f"{col}_{measure}"
        line = f"  {col:<12} {measure:<10}"
        reg_means = {}
        for regime in regime_order:
            vals = entropy.loc[entropy["regime"] == regime, ent_col].dropna()
            mean_val = vals.mean()
            reg_means[regime] = {"mean": mean_val, "n": len(vals), "vals": vals}
            line += f"  {mean_val:>11.4f}"
            regime_entropy.append({"series": col, "measure": measure,
                                   "regime": regime, "mean_entropy": round(mean_val, 5),
                                   "n": len(vals)})
        print(line)

        # Wilcoxon tests: Pre→COVID, COVID→Rate
        pairs = [("pre_covid", "covid_recovery"), ("covid_recovery", "rate_hike")]
        for r1, r2 in pairs:
            v1 = reg_means[r1]["vals"]
            v2 = reg_means[r2]["vals"]
            if len(v1) > 5 and len(v2) > 5:
                stat, pval = stats.ranksums(v1, v2)
                sig = "✓" if pval < 0.05 else "✗"
                wilcoxon_rows.append({
                    "series": col, "measure": measure,
                    "regime_a": r1, "regime_b": r2,
                    "statistic": round(stat, 4), "pval": round(pval, 4),
                    "significant": pval < 0.05,
                })
                direction = "↑" if reg_means[r2]["mean"] > reg_means[r1]["mean"] else "↓"
                print(f"    Wilcoxon ({regime_labels[r1]} → {regime_labels[r2]}): "
                      f"p = {pval:.4f} {sig} entropy {direction}")

# Threshold-based regime classifier using Nifty50 Shannon entropy
print(f"\n  Threshold classifier (Nifty50 Shannon entropy):")
sh_series = entropy["Nifty50_shannon"].dropna()
lo_thresh = sh_series.quantile(0.33)
hi_thresh = sh_series.quantile(0.67)

def classify_entropy(val):
    if val <= lo_thresh:   return "Low"
    elif val <= hi_thresh: return "Medium"
    else:                  return "High"

entropy["entropy_class"] = sh_series.map(classify_entropy)
cross = pd.crosstab(entropy["entropy_class"], entropy["regime"],
                    normalize="index").round(3)
print(f"\n  Entropy Class → Regime distribution:")
print(cross.to_string())

pd.DataFrame(regime_entropy).to_csv(f"{FINDINGS_DIR}/04_regime_entropy.csv", index=False)
pd.DataFrame(wilcoxon_rows).to_csv(f"{FINDINGS_DIR}/04_wilcoxon_tests.csv", index=False)
print("\n✓ Test 4 complete\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — INDIA vs US STRUCTURAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 5 — INDIA vs US STRUCTURAL COMPARISON")
print("=" * 60)

def market_mean_entropy(series_list, measure):
    vals = []
    for col in series_list:
        vals.extend(entropy[f"{col}_{measure}"].dropna().tolist())
    return np.array(vals)

iu_results = []
print(f"\n  {'Measure':<10} {'India mean':>12} {'US mean':>12} {'Diff':>10} {'p (boot)':>10} {'Sig':>5}")
print("  " + "-" * 62)

bootstrap_rows = []
N_BOOT = 1000

for measure in ["shannon", "apen", "sampen"]:
    india_vals = market_mean_entropy(INDIA_SERIES, measure)
    us_vals    = market_mean_entropy(US_SERIES,    measure)
    india_mean = india_vals.mean()
    us_mean    = us_vals.mean()
    obs_diff   = india_mean - us_mean

    # Bootstrap
    combined = np.concatenate([india_vals, us_vals])
    n_india  = len(india_vals)
    boot_diffs = np.array([
        np.random.choice(combined, n_india, replace=True).mean() -
        np.random.choice(combined, len(us_vals), replace=True).mean()
        for _ in range(N_BOOT)
    ])
    p_boot = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    sig = "✓" if p_boot < 0.05 else "✗"

    iu_results.append({
        "measure": measure,
        "india_mean": round(india_mean, 5),
        "us_mean": round(us_mean, 5),
        "diff": round(obs_diff, 5),
        "bootstrap_pval": round(p_boot, 4),
        "ci_lo_95": round(ci_lo, 5),
        "ci_hi_95": round(ci_hi, 5),
        "significant": p_boot < 0.05,
    })
    bootstrap_rows.append({
        "measure": measure, "observed_diff": round(obs_diff, 5),
        "bootstrap_pval": round(p_boot, 4),
        "ci_lo_95": round(ci_lo, 5), "ci_hi_95": round(ci_hi, 5),
        "significant": p_boot < 0.05,
    })
    print(f"  {measure:<10} {india_mean:>12.4f} {us_mean:>12.4f} {obs_diff:>+10.4f} "
          f"{p_boot:>10.4f} {sig:>5}")

# Regime-conditioned gap
print(f"\n  India vs US gap by regime (Shannon entropy):")
print(f"  {'Regime':<22} {'India':>10} {'US':>10} {'Gap':>10}")
print("  " + "-" * 55)

for regime in regime_order:
    mask = entropy["regime"] == regime
    india_m = np.mean([entropy.loc[mask, f"{c}_shannon"].mean() for c in INDIA_SERIES])
    us_m    = np.mean([entropy.loc[mask, f"{c}_shannon"].mean() for c in US_SERIES])
    print(f"  {regime_labels[regime]:<22} {india_m:>10.4f} {us_m:>10.4f} {india_m-us_m:>+10.4f}")

pd.DataFrame(iu_results).to_csv(f"{FINDINGS_DIR}/05_india_vs_us.csv", index=False)
pd.DataFrame(bootstrap_rows).to_csv(f"{FINDINGS_DIR}/05_bootstrap_test.csv", index=False)
print("\n✓ Test 5 complete\n")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("VISUALISATIONS")
print("=" * 60)

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor":  "#0d1117",
    "axes.edgecolor":   "#30363d", "axes.labelcolor": "#c9d1d9",
    "text.color":       "#c9d1d9", "xtick.color":     "#8b949e",
    "ytick.color":      "#8b949e", "grid.color":      "#21262d",
    "grid.linewidth":   0.5,       "font.family":     "monospace",
    "axes.spines.top":  False,     "axes.spines.right": False,
})

ACCENT  = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"
MUTED   = "#8b949e"

REGIME_COLORS = {
    "pre_covid":       ACCENT,
    "covid_recovery":  ACCENT3,
    "rate_hike":       ACCENT2,
}

def add_regime_spans(ax, entropy_df):
    """Shade background by regime across a time-series axis."""
    regime_alpha = 0.08
    for regime, color in REGIME_COLORS.items():
        mask   = entropy_df["regime"] == regime
        blocks = entropy_df.index[mask]
        if blocks.empty:
            continue
        start = blocks[0]
        for i in range(1, len(blocks)):
            if (blocks[i] - blocks[i-1]).days > 10:
                ax.axvspan(start, blocks[i-1], alpha=regime_alpha, color=color)
                start = blocks[i]
        ax.axvspan(start, blocks[-1], alpha=regime_alpha, color=color)

# ── Fig 1 — Entropy Time Series ───────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
measures  = ["shannon", "apen", "sampen"]
m_labels  = ["Shannon Entropy (H)", "Approximate Entropy (ApEn)",
             "Sample Entropy (SampEn)"]

for ax, measure, label in zip(axes, measures, m_labels):
    for col, color in [("Nifty50", ACCENT), ("SP500", ACCENT2)]:
        s = entropy[f"{col}_{measure}"].dropna()
        ax.plot(s.index, s.values, color=color, linewidth=0.8, alpha=0.9,
                label=col)
    add_regime_spans(ax, entropy)
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

regime_changes = entropy[entropy["regime"] != entropy["regime"].shift(1)].index[1:]
for rc in regime_changes:
    for ax in axes:
        ax.axvline(rc, color=MUTED, linewidth=1, linestyle="--", alpha=0.5)

axes[0].set_title("Entropy Time Series — Nifty 50 vs S&P 500 (2012–2026)",
                  fontsize=12, pad=12, color="#e6edf3", fontweight="bold")
axes[-1].set_xlabel("Date", fontsize=9)

for regime, label in [("pre_covid","Pre-COVID"), ("covid_recovery","COVID"),
                      ("rate_hike","Rate Hike")]:
    subset  = entropy[entropy["regime"] == regime]
    if subset.empty: continue
    mid_idx = subset.index[len(subset)//2]
    axes[0].text(mid_idx, axes[0].get_ylim()[1] * 0.97, label,
                 ha="center", va="top", color=MUTED, fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/fig1_entropy_timeseries.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  ✓ Fig 1 — Entropy time series")

# ── Fig 2 — Entropy vs Volatility (scatter, by regime) ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, measure, label in zip(axes, measures, m_labels):
    for regime, color in REGIME_COLORS.items():
        mask    = entropy["regime"] == regime
        x_vals  = entropy.loc[mask, f"Nifty50_{measure}"].dropna()
        y_vals  = entropy.loc[x_vals.index, "Nifty50_vol"].dropna()
        idx     = x_vals.index.intersection(y_vals.index)
        ax.scatter(x_vals[idx], y_vals[idx], color=color, alpha=0.3, s=5,
                   label=regime.replace("_", " ").title())
    rho_row = ev_df[(ev_df["series"] == "Nifty50") & (ev_df["measure"] == measure)]
    if not rho_row.empty:
        rho = rho_row["rho_full"].values[0]
        p   = rho_row["pval_full"].values[0]
        sig = "*" if p < 0.05 else ""
        ax.text(0.05, 0.93, f"ρ = {rho:+.3f}{sig}", transform=ax.transAxes,
                fontsize=9, color="#e6edf3")
    ax.set_title(f"Nifty50 — {label}", fontsize=9, color="#e6edf3", fontweight="bold")
    ax.set_xlabel("Entropy", fontsize=8)
    ax.set_ylabel("Realised Volatility (ann.)", fontsize=8)
    ax.legend(fontsize=7)

plt.suptitle("Entropy vs Realised Volatility — by Regime",
             fontsize=12, color="#e6edf3", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/fig2_entropy_vs_vol.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  ✓ Fig 2 — Entropy vs volatility")

# ── Fig 3 — Lead-Lag Cross-Correlation ───────────────────────────────────────
ll_df  = pd.read_csv(f"{FINDINGS_DIR}/02_crosscorr_lead_lag.csv")
lag_cols = [c for c in ll_df.columns if c.startswith("lag_")]
lag_vals = [int(c.replace("lag_", "")) for c in lag_cols]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, series_name in zip(axes, ["Nifty50", "SP500"]):
    for measure, color in zip(measures, [ACCENT, ACCENT3, ACCENT2]):
        row = ll_df[(ll_df["series"] == series_name) & (ll_df["measure"] == measure)]
        if row.empty:
            continue
        rhos = row[lag_cols].values[0].astype(float)
        ax.plot(lag_vals, rhos, color=color, linewidth=1.5, marker="o",
                markersize=3, label=measure)
    ax.axhline(0,  color=MUTED, linewidth=0.8, alpha=0.5)
    ax.axvline(0,  color=MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"{series_name} — Entropy leads/lags Volatility",
                 fontsize=10, color="#e6edf3", fontweight="bold")
    ax.set_xlabel("Lag (days) — negative = entropy leads", fontsize=8)
    ax.set_ylabel("Spearman ρ", fontsize=8)
    ax.legend(fontsize=8)
    ax.text(0.05, 0.05, "← entropy leads vol  |  vol leads entropy →",
            transform=ax.transAxes, fontsize=7, color=MUTED)

plt.suptitle("Lead-Lag Cross-Correlation: Entropy vs Realised Volatility",
             fontsize=12, color="#e6edf3", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/fig3_lead_lag.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  ✓ Fig 3 — Lead-lag cross-correlation")

# ── Fig 4 — Entropy by Regime ─────────────────────────────────────────────────
rd_df = pd.read_csv(f"{FINDINGS_DIR}/04_regime_entropy.csv")
rd_hl = rd_df[rd_df["series"].isin(["Nifty50", "SP500"])]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
regime_names = {"pre_covid": "Pre-COVID", "covid_recovery": "COVID Rec.",
                "rate_hike": "Rate Hike"}

for ax, measure, label in zip(axes, measures, m_labels):
    sub = rd_hl[rd_hl["measure"] == measure]
    x   = np.arange(len(regime_order))
    w   = 0.3
    for i, (series_name, color) in enumerate([("Nifty50", ACCENT), ("SP500", ACCENT2)]):
        vals = [sub[(sub["series"] == series_name) &
                    (sub["regime"] == r)]["mean_entropy"].values
                for r in regime_order]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        bars = ax.bar(x + i * w, vals, w, color=color, alpha=0.85,
                      edgecolor="none", label=series_name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*0.01,
                    f"{val:.3f}", ha="center", fontsize=7, color="#e6edf3")
    ax.set_xticks(x + w/2)
    ax.set_xticklabels([regime_names[r] for r in regime_order], fontsize=8)
    ax.set_title(label, fontsize=9, color="#e6edf3", fontweight="bold")
    ax.legend(fontsize=8)

plt.suptitle("Mean Entropy by Macro Regime — Nifty50 vs S&P 500",
             fontsize=12, color="#e6edf3", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/fig4_entropy_regimes.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  ✓ Fig 4 — Entropy by regime")

# ── Fig 5 — India vs US Structural Comparison ────────────────────────────────
iu_df   = pd.read_csv(f"{FINDINGS_DIR}/05_india_vs_us.csv")
boot_df = pd.read_csv(f"{FINDINGS_DIR}/05_bootstrap_test.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, measure, label in zip(axes, measures, m_labels):
    row     = iu_df[iu_df["measure"] == measure].iloc[0]
    br      = boot_df[boot_df["measure"] == measure].iloc[0]
    vals    = [row["india_mean"], row["us_mean"]]
    colors  = [ACCENT, ACCENT2]
    bars    = ax.bar(["India", "US"], vals, color=colors, width=0.4, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*0.01,
                f"{val:.4f}", ha="center", fontsize=10, color="#e6edf3")
    sig_str = "✓ sig" if br["significant"] else "✗ not sig"
    ax.text(0.5, 0.92, f"Bootstrap p = {br['bootstrap_pval']:.3f} ({sig_str})",
            transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    ax.text(0.5, 0.83, f"Diff: {row['diff']:+.4f}  CI: [{br['ci_lo_95']:+.4f}, {br['ci_hi_95']:+.4f}]",
            transform=ax.transAxes, ha="center", fontsize=7, color=MUTED)
    ax.set_title(label, fontsize=9, color="#e6edf3", fontweight="bold")

plt.suptitle("India vs US — Mean Entropy Comparison (2012–2026)",
             fontsize=12, color="#e6edf3", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/fig5_india_vs_us.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("  ✓ Fig 5 — India vs US")

print("\n" + "=" * 60)
print("ALL DONE")
print(f"  CSVs   → {FINDINGS_DIR}/")
print(f"  Charts → {CHARTS_DIR}/")
print("=" * 60)