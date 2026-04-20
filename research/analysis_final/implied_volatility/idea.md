# Does Implied Volatility Predict Extreme Moves?
**Project:** iv_predict_moves
**Author:** @AlphaByProcess
**Date:** 2026-04
**Status:** Design

---

## Research Question
Does elevated implied volatility (IV) predict large subsequent price moves in the S&P 500?

This tests whether options markets are forward-looking (pricing in genuine uncertainty)
or backward-looking (reacting to realized volatility that has already occurred).

---

## Hypothesis
Elevated IV raises the conditional probability of a large subsequent |return|.

P(|r_SP500| > threshold | IV > IV_threshold) > P(|r_SP500| > threshold)

A rejection of the null means IV carries predictive content for move magnitude.

---

## Data
| Series              | Ticker   | Source   | Period     |
|---------------------|----------|----------|------------|
| S&P 500             | ^GSPC    | yfinance | 2004–2026  |
| VIX (30d IV proxy)  | ^VIX     | yfinance | 2004–2026  |
| S&P 500 ETF         | SPY      | yfinance | 2004–2026  |
| HMM Benchmark       | TLT      | yfinance | 2004–2026  |

**Note on IV proxy:** VIX is used as the implied volatility measure throughout.
VIX represents the 30-day forward implied volatility of the S&P 500 derived
from SPX options. It is a forward-looking consensus estimate from options markets.
This is the cleanest, longest available IV series for this asset.

**Note on look-ahead bias:** IV threshold flags are set on day t.
Forward returns are computed starting t+1. No same-day contamination.

**Note on TLT:** Used as HMM benchmark consistent with oil_shock_equity.
Equity/bond correlation varies meaningfully across regimes — genuine
discriminatory power for Bull/Transition/Bear/Crisis state separation.

---

## IV Threshold Definition
Two thresholds tested independently and compared:

- **Spike threshold:** VIX > 25 (absolute — historically elevated)
- **Relative threshold:** VIX > 1.5× its own 60-day rolling mean (regime-relative spike)

Rationale: Absolute threshold captures periods of genuine market fear.
Relative threshold captures sudden local spikes regardless of the regime baseline.
Comparing both tests whether it's the level of IV or the suddenness of the spike
that carries predictive content.

IV flags set on close of day t. Forward returns computed from open of t+1.

---

## Move Definition
Large move = |daily return| > threshold

Two thresholds tested:
- 1.5% (moderate — ~1.5σ historically)
- 2.5% (large — ~2.5σ historically)

Both directions included (large up or large down).
Directional analysis (up vs. down separately) included as robustness check.

---

## Methodology

### Step 1 — IV Flag Identification
Flag all dates where VIX crosses the absolute threshold (>25).
Flag all dates where VIX > 1.5× 60d rolling mean (relative spike).
Apply 5-day exclusion window to remove overlapping events (keep first).
Report: total flag count, distribution by year, distribution by regime.

### Step 2 — Event Study
For each IV flag, compute forward |return| at:
- t+1 (next day)
- t+5 (1 week)
- t+20 (1 month)

Compute average |CAR| = mean(|r_t| - |r_mean|) over window.
Compare to non-flag baseline.
Test statistical significance: t-test + bootstrap (n=1000).

Also compute directional split: what fraction of post-flag large moves
were positive vs negative? Tests if IV predicts direction or only magnitude.

### Step 3 — Conditional Probability
Compute:
  P(|SP500| > move_threshold | IV_flag) — fraction of flag dates followed by large move
  P(|SP500| > move_threshold) — unconditional baseline

Test at t+1, t+5, t+20 horizons.
Report lift: [P(large|IV_flag) - P(large)] / P(large)
Test at both move thresholds (1.5% and 2.5%).

### Step 4 — IV Lead/Lag Analysis
Key test: does IV lead moves, lag them, or both?

Compute:
  a) Mean VIX in the N days BEFORE a large move (pre-move IV)
  b) Mean VIX in the N days AFTER a large move (post-move IV)
  c) Same for non-large-move days

If IV peaks after large moves → reactive.
If IV peaks before large moves → predictive.
If both → mixed (pricing + updating).

Plot mean VIX in [-10, +10] day window around large move events.

### Step 5 — VIX Term Structure Slope (Robustness)
VIX9D (9-day) vs VIX (30-day) slope as additional signal.

When short-dated IV > long-dated IV (inverted term structure / backwardation),
markets are pricing imminent stress — a more specific predictive signal
than raw VIX level.

Slope = VIX9D / VIX — values > 1 indicate backwardation.
Compute conditional probability of large moves when slope > 1 vs slope < 1.
Requires VIX9D data from CBOE/yfinance (available from ~2011).

**Note:** VIX9D availability shorter than VIX — report separately with
explicit caveat on reduced sample size.

### Step 6 — Regime Distribution
Fit 4-state HMM (GaussianHMM, hmmlearn) on ^GSPC vs TLT.
Features: volatility, autocorrelation, index_correlation, skewness (rolling 20d).
States auto-labeled by volatility rank: Bull / Transition / Bear / Crisis.
Count IV flag events per regime. Compare to baseline % of all trading days.
Report lift per regime.

Rationale: If IV spikes concentrate in Bear/Crisis regimes, any predictive
relationship may be driven by regime-recovery dynamics (as seen in oil shock study).
This is the key confound to rule out — or confirm.

### Step 7 — Trading Strategy
Rule: If IV_flag (absolute or relative) → buy straddle or long volatility for next 5 days.
Simplified backtest using long SPY + short SPY to approximate straddle P&L
by tracking realized vs implied vol spread (VIX - subsequent 5d realized vol).

Metrics: mean realized vol in post-flag vs non-flag windows.
If realized vol consistently exceeds VIX post-flag → IV was underpriced → predictive.
If realized vol consistently undershoots → IV was overpriced → reactive.

Strategy flag uses SPY for execution, ^GSPC for event definition.

---

## Null Hypothesis
IV carries no predictive content for subsequent move magnitude.
P(|SP500| > threshold | IV_flag) = P(|SP500| > threshold)

---

## Expected Findings (pre-experiment)
IV is partially predictive but largely reactive.

Prior literature (Giot 2005, Whaley 2009) finds VIX mean-reverts and
large moves cluster in the days immediately following VIX spikes, but
the directionality is mixed. Expecting:
- Significant lift at t+1 (clustering)
- Decay at t+5 and t+20
- Regime confound: Bear/Crisis regimes over-represented in IV flag dates
- Term structure slope (VIX9D/VIX) may add incremental signal at t+1

---

## Limitations
- VIX measures 30-day IV — not forward-looking for t+1 specifically
- VIX9D data begins ~2011 — limits term structure robustness test
- No single-name or sector IV analysis (index only)
- Straddle backtest is approximated (no true options data)
- Regime-conditional analysis may have small samples in Bull regime
- Threshold sensitivity partially tested (2 thresholds) but not grid-searched
- No transaction costs in backtest

---

## Output Files
- findings.md — written research summary
- findings/event_study.png — |CAR| chart across horizons
- findings/conditional_prob.png — bar chart, conditional vs unconditional
- findings/lead_lag.png — VIX around large move events (±10 days)
- findings/term_structure.png — backwardation vs normal IV conditional probs
- findings/regime_distribution.png — IV flag distribution by HMM regime
- findings/realized_vs_implied.png — post-flag realized vol vs VIX

---

## Framework Integration
- Strategy file: strategies/iv_spike_long_vol.py
- Backtest engine: backtests/engine.py
- Data loader: utils/data_loader.py
- Performance: utils/performance.py (calculate_metrics + print_summary)
- HMM: research/regime_detection/hmm_model.py (RegimeDetector)
- HMM features: research/regime_detection/features.py
- Registered in run.py as "iv_predict" key via run_iv_predict() wrapper

---

## Notes
- No look-ahead bias: IV flag on day t, forward returns from t+1
- Run absolute and relative thresholds as separate event sets, report both
- Both move thresholds (1.5% and 2.5%) reported at each horizon
- Lead/lag analysis is the core diagnostic — do not omit
- All plots saved to findings/ subfolder
- Script is fully modular — each step is an independent function
- VIX9D term structure test reported separately with sample size caveat

