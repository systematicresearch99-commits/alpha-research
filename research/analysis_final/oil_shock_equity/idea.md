# Oil Price Shocks → Equity Market Drawdowns
**Project:** oil_shock_equity
**Author:** @AlphaByProcess
**Date:** 2026-03
**Status:** Complete

---

## Research Question
Do sudden oil price shocks predict negative equity market returns in the short term?

This tests the macro narrative that energy cost shocks transmit into equity selloffs
through earnings compression, consumer spending reduction, and risk-off sentiment.

---

## Hypothesis
Large oil price increases raise the conditional probability of negative S&P 500 returns.

P(r_SP500 < 0 | ΔOil > x) > P(r_SP500 < 0)

A rejection of the null means oil shocks carry predictive content for equity direction.

---

## Data
| Series         | Ticker  | Source   | Period     |
|----------------|---------|----------|------------|
| WTI Crude Oil  | CL=F    | yfinance | 2000–2026  |
| S&P 500        | ^GSPC   | yfinance | 2000–2026  |
| Energy sector  | XLE     | yfinance | 2000–2026  |
| Tech sector    | XLK     | yfinance | 2000–2026  |
| Utilities      | XLU     | yfinance | 2000–2026  |
| Airlines       | AAL+DAL+UAL (avg) | yfinance | 2000–2026 |
| HMM Benchmark  | TLT     | yfinance | 2002–2026  |

**Note on airlines:** JETS ETF only available from 2015.
Using equal-weighted average of AAL, DAL, UAL for full-sample consistency.
DAL bankrupt 2005, UAL bankrupt 2002 — both restructured and relisted.
Data gaps handled via forward-fill with explicit caveat in findings.

**Note on TLT:** Used as HMM benchmark instead of SPY. When asset = ^GSPC,
using SPY as benchmark produces ~1.0 index_correlation feature (uninformative).
TLT equity/bond correlation varies meaningfully across regimes — genuine
discriminatory power for Bull/Transition/Bear/Crisis state separation.
TLT data begins July 2002 — ~10 shock events pre-TLT excluded from regime
distribution count with explicit notation.

---

## Shock Definition
Two thresholds tested independently and compared:

- **Daily shock:** WTI 1-day return > +5%
- **Weekly shock:** WTI 5-day rolling return > +10%

Rationale: Daily captures sudden supply disruptions or geopolitical events.
Weekly captures sustained pressure buildup. Comparing both tests whether
shock speed affects equity transmission.

Negative oil shocks (crashes) excluded from primary analysis — separate
directional question. May be added as robustness check.

---

## Methodology

### Step 1 — Shock Identification
Flag all dates meeting daily or weekly threshold.
Remove overlapping events within a 5-day window (keep first occurrence).
Report: total shock count, distribution by year, distribution by regime.

Counts (final): 94 daily shock events, 74 weekly shock events.
After regime label join (TLT warmup): 84 daily, 66 weekly with labels.

### Step 2 — Event Study
For each shock event, compute S&P 500 forward returns:
- t+1 (next day)
- t+5 (1 week)
- t+20 (1 month)

Compute Cumulative Abnormal Return (CAR):
  CAR = Σ(r_t - r_mean) over window

Compare CAR distribution to non-shock baseline.
Test statistical significance: t-test + bootstrap (n=1000).

### Step 3 — Conditional Probability
Compute:
  P(SP500 < 0 | OilShock) — fraction of shock events followed by negative returns
  P(SP500 < 0) — unconditional baseline

Test at t+1, t+5, t+20 horizons.
Report lift: [P(SP500<0|shock) - P(SP500<0)] / P(SP500<0)

### Step 4 — Sector Analysis + Regime Distribution

#### 4a. Sector Analysis
Post-shock returns for: XLE, XLK, XLU, Airlines (AAL+DAL+UAL avg)
Compare each sector's CAR to S&P 500 CAR.
Expected direction (hypothesis): XLE ↑, Airlines ↓, XLK ↓, XLU neutral/mixed.
Actual result: all sectors positive at t+20 — narrative not supported.

#### 4b. Regime Distribution (added after team review)
Fit 4-state HMM (GaussianHMM, hmmlearn) on ^GSPC vs TLT.
Features: volatility, autocorrelation, index_correlation, skewness (rolling 20d).
States auto-labeled by volatility rank: Bull / Transition / Bear / Crisis.
Count shock events per regime. Compare to baseline % of all trading days.
Report lift per regime to quantify over/underrepresentation.

Rationale for addition: sector results showed uniformly positive post-shock returns,
contradicting the cost-transmission narrative. Team flagged that the structural
explanation ("shocks occur in risk-on environments") was asserted not shown.
Regime distribution table converts the assertion into an empirical finding.

Key result: 63% of daily and 68% of weekly shock events fell in Bear/Crisis regimes
(baseline: 36% of trading days). Crisis alone: 33% of shocks vs 15% baseline (+18pp).
This explains positive post-shock returns — Bear/Crisis recovery dynamics dominate
oil cost-transmission signal in the forward return window.

### Step 5 — Trading Strategy
Rule: If daily oil shock → short ^GSPC for next 3 days.
      If weekly oil shock → short ^GSPC for next 5 days.

Backtest via backtests/engine.py.
Metrics: Total return, Sharpe ratio, Max drawdown, Win rate, # trades.
Benchmark: Buy-and-hold S&P 500.

Strategy failure has two independent causes:
1. Payoff asymmetry: avg loss > avg win despite 60% win rate
2. Adverse regime selection: signal fires most in Bear/Crisis where
   short-side recovery probability is highest

---

## Actual Findings (post-experiment)
- Hypothesis rejected: oil shocks do not predict negative equity returns in aggregate
- CARs near-zero and statistically insignificant at all horizons for both shock types
- Post-shock sector returns uniformly positive at t+20 — cost-transmission narrative
  not supported in the data
- Oil shocks concentrate in Bear/Crisis regimes (63–68%), not bull markets
- Positive post-shock returns reflect regime-recovery dynamics, not shock absorption
- 60% strategy win rate confirms directional signal exists but payoff asymmetry
  and adverse regime selection make it unexploitable as raw short rule
- Correct next test: regime-conditional event study isolating Bull-regime shocks
  (n=14 daily, n=9 weekly — currently too small for reliable estimates)

---

## Null Hypothesis
Oil shocks carry no predictive content for S&P 500 direction.
P(SP500 < 0 | OilShock) = P(SP500 < 0)

**Result:** Cannot reject null in aggregate. Conditional probability results
inconsistent across horizons and shock types. No stable signal.

---

## Limitations
- Regime-conditional event study incomplete — Bull-regime shock sample too small
- WTI as oil proxy — Brent may differ slightly but highly correlated
- Airline data has restructuring gaps (UAL 2002, DAL 2005)
- No transaction costs in backtest beyond 0.1% per trade
- Strategy shorting index assumes liquid, costless short — unrealistic for retail
- Threshold sensitivity untested — fixed at 5%/day and 10%/5-day
- Negative oil shocks (crashes) excluded from primary analysis
- TLT data begins July 2002 — ~10 early shock events excluded from regime count

---

## Output Files
- findings.md — written research summary
- findings/event_study.png — CAR chart across horizons
- findings/conditional_prob.png — bar chart, conditional vs unconditional
- findings/sector_heatmap.png — sector returns post-shock
- findings/strategy_equity_curve.png — backtest equity curve
- findings/regime_distribution.png — shock event distribution by HMM regime

---

## Framework Integration
- Strategy file: strategies/oil_shock_short.py
- Backtest engine: backtests/engine.py
- Data loader: utils/data_loader.py (called twice: CL=F + ^GSPC)
- Performance: utils/performance.py (calculate_metrics + print_summary)
- HMM: research/regime_detection/hmm_model.py (RegimeDetector)
- HMM features: research/regime_detection/features.py
- NOT saved to SQLite research log (experimental run)
- Registered in run.py as "oil_shock" key via run_oil_shock() wrapper

---

## Notes
- Keep shock events non-overlapping (5-day exclusion window)
- Run daily and weekly shocks as separate event sets, report both
- All plots saved to findings/ subfolder
- Script is fully modular — each step is an independent function
- TLT chosen over SPY as HMM benchmark — see Data section for rationale
- Regime distribution added after v1 team review — not in original design
- Article version: oil_shock_equity_article_v2.docx