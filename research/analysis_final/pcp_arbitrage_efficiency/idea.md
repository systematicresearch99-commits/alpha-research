# Testing Arbitrage Efficiency in Options Markets
**Project:** pcp_arbitrage_efficiency
**Author:** @AlphaByProcess
**Date:** 2026-04
**Status:** Design

---

## Research Question
Are S&P 500 options prices internally consistent with put-call parity?

Put-call parity (PCP) is a no-arbitrage identity derived from pure financial
mathematics — it must hold in any frictionless, complete market. Persistent
deviations imply either (a) pricing inefficiency, (b) transaction-cost bounds
being exceeded, or (c) data/microstructure artefacts.

This project tests how often, by how much, and under what conditions
the identity breaks — and whether deviations are exploitable after costs.

---

## Hypothesis
PCP deviations in liquid index options are bounded by transaction costs
and do not represent exploitable arbitrage opportunities in normal regimes.
In stress regimes, deviations widen and become more persistent.

H₀: Mean |deviation| = 0 after adjusting for bid-ask spread proxy
H₁: Mean |deviation| > 0, clustering in Bear/Crisis regimes

---

## Theoretical Framework

### Put-Call Parity Identity
For European options on a non-dividend-paying asset:

    C − P = S − K·e^(−r·T)

Where:
  C = call price (mid-quote)
  P = put price (mid-quote)
  S = spot price of underlying
  K = strike price
  r = risk-free rate (annualised)
  T = time to expiry (years)

Rearranged as the **synthetic forward**:
    F_synthetic = C − P + K·e^(−r·T)
    F_market    = S·e^(r·T)    (cost-of-carry forward)

Deviation:
    δ = (C − P) − (S − K·e^(−r·T))

δ > 0  → calls expensive relative to puts (bullish skew / demand pressure)
δ < 0  → puts expensive relative to calls (typical in equity markets = put premium)

**Note on dividends:** SPX options are on a dividend-paying index.
Adjusted PCP: C − P = S·e^(−q·T) − K·e^(−r·T) where q = continuous dividend yield.
Dividend yield proxied from SPY trailing yield — introduce as robustness check.

---

## Data
| Series              | Ticker / Source          | Period      | Notes                          |
|---------------------|--------------------------|-------------|--------------------------------|
| S&P 500 index       | ^GSPC — yfinance         | 2010–2026   | Spot price S                   |
| SPX options chain   | CBOE via yfinance        | 2010–2026   | C, P, K, T for matched pairs   |
| Risk-free rate      | ^IRX (13-week T-bill)    | 2010–2026   | Annualised, used as r           |
| SPY dividend yield  | SPY — yfinance           | 2010–2026   | Proxy for q (robustness)       |
| VIX                 | ^VIX — yfinance          | 2010–2026   | Regime context                 |
| HMM Benchmark       | TLT — yfinance           | 2010–2026   | Bull/Transition/Bear/Crisis    |

**Note on options data:** yfinance returns live/recent options chains only.
For historical chains, CBOE DataShop or WRDS OptionMetrics is ideal but
requires institutional access. Workaround: use synthetic historical PCP
reconstruction from index futures vs options via public CBOE data where
available, or limit the study to ATM options over available snapshots.
This limitation is documented explicitly in the script and findings.

**Note on ATM selection:** PCP is most cleanly tested at-the-money (ATM)
or near-ATM strikes where both C and P are liquid. Deep ITM/OTM pairs
have wide bid-ask spreads that inflate apparent deviations. Strategy: select
the strike closest to current spot for each expiry tested.

**Note on expiries:** Focus on front-month (≤30 DTE) and second-month (30–60 DTE).
Weekly options included as separate sub-sample — typically wider spreads.

---

## Deviation Definition

### Raw Deviation
    δ_raw = (C_mid − P_mid) − (S − K·e^(−r·T))

### Transaction-Cost-Adjusted Deviation
    δ_adj = |δ_raw| − BA_proxy

Where BA_proxy = 0.5 × (C_ask − C_bid) + 0.5 × (P_ask − P_bid)
(sum of half-spreads for call and put leg)

δ_adj > 0 → deviation survives transaction cost filter → potential arbitrage
δ_adj ≤ 0 → deviation absorbed by spreads → consistent with efficiency

### Normalised Deviation
    δ_norm = δ_raw / S     (as % of spot price)

Allows comparison across time periods and spot levels.

---

## Methodology

### Step 1 — Data Assembly and Matching
For each available options snapshot:
  - Select ATM strike (|K − S| minimised) for each available expiry
  - Match call and put at identical (K, T) — required for valid PCP test
  - Compute r from ^IRX (annualised, continuously compounded)
  - Compute T in years from snapshot date to expiry
  - Filter: T > 0, both C and P > 0.05 (remove nearly-zero-premium legs)

Report: total matched pairs, distribution by expiry bucket, by year.

### Step 2 — PCP Deviation Calculation
For each matched (C, P, S, K, r, T) tuple:
  - Compute δ_raw, δ_adj, δ_norm
  - Classify: [within BA bounds | exploitable deviation]
  - Sign: positive (calls rich) vs negative (puts rich)

Baseline statistics:
  - Mean, median, std, 5th/95th pct of δ_norm
  - % of observations with |δ_adj| > 0 (exploitable after costs)
  - Time series of δ_norm — does it trend or mean-revert?

### Step 3 — Arbitrage Bounds Test
Classical no-arbitrage bounds (from Hull 2018):

Lower bound: C − P ≥ S·e^(−q·T) − K·e^(−r·T) − BA_proxy
Upper bound: C − P ≤ S·e^(−q·T) − K·e^(−r·T) + BA_proxy

Report:
  - Frequency of bound violations (raw, then cost-adjusted)
  - Magnitude of violations conditional on occurring
  - Are violations one-sided (puts systematically expensive)?

One-sided test: equity markets are known to have a persistent put premium
(negative δ_norm on average). Test whether observed sign is consistent
with the equity risk premium / demand-pressure hypothesis or exceeds
what transaction costs can explain.

### Step 4 — Deviation Magnitude Analysis
For cost-adjusted deviations that exceed the BA proxy:
  - Distribution of |δ_adj| — is it fat-tailed?
  - Autocorrelation: AR(1) of δ_norm — are deviations persistent?
  - Half-life of mean reversion (OLS regression: Δδ = α + β·δ_{t-1})

If half-life < 1 day → deviations fleeting, not exploitable
If half-life > 1 day → potential slow-moving inefficiency or structural premium

### Step 5 — Regime Conditioning
Fit 4-state HMM (GaussianHMM, hmmlearn) on ^GSPC vs TLT.
Features: volatility, autocorrelation, index_correlation, skewness (rolling 20d).
States auto-labeled by volatility rank: Bull / Transition / Bear / Crisis.

For each regime, report:
  - Mean |δ_norm| — do deviations widen in stress?
  - % of exploitable deviations (δ_adj > 0)
  - Sign distribution — does the put premium intensify in Bear/Crisis?

Hypothesis: deviations widen in Bear/Crisis due to liquidity withdrawal,
margin pressure, and demand-pressure for put protection. If confirmed,
PCP deviations are not random noise but regime-correlated risk signals.

### Step 6 — VIX Conditioning
Independently of regime (to avoid regime label noise):
  - Bin observations by VIX quintile (Q1 = low vol, Q5 = high vol)
  - Report mean |δ_norm| per VIX quintile
  - OLS: |δ_norm| ~ β₀ + β₁·VIX + ε
    Test: is β₁ positive and significant?

If VIX predicts deviation magnitude → IV encodes information about
market friction beyond just volatility expectation.

### Step 7 — Term Structure Comparison (Robustness)
Compare PCP deviations across expiry buckets:
  - 0–14 DTE (weekly / very short)
  - 15–30 DTE (front month)
  - 31–60 DTE (second month)
  - 60+ DTE (LEAPS — lower liquidity)

Expectation: deviations larger for very short (event-driven demand) and
very long (lower liquidity) expiries. Front-month closest to zero.

### Step 8 — Simulated Arbitrage P&L
For each observation where δ_adj > 0 (apparent arbitrage):
  - Construct the synthetic: buy cheap side, sell expensive side
  - P&L at expiry = δ_adj (if held to expiry, ignoring path)
  - P&L at t+1 day (if deviation reverts quickly)

Assumptions: mid-quote execution (optimistic), no early exercise risk
(European — SPX options are European), no pin risk.
Report: mean P&L, % profitable, Sharpe of strategy.

Caveat: this is a simplified PnL estimate. Real execution requires
simultaneous leg fill; slippage and fill risk not modelled.

---

## Null Hypothesis
PCP deviations are zero after transaction costs:
    E[δ_adj] = 0    for all regimes, VIX levels, and expiry buckets.

---

## Expected Findings (pre-experiment)
1. Raw deviations are non-zero but small in magnitude (< 0.5% of spot)
2. After BA adjustment, most deviations are absorbed — consistent with efficiency
3. Persistent negative δ (puts rich) — structural equity put premium
4. Deviations widen significantly in Bear/Crisis regimes and high-VIX environments
5. Very short (weekly) and long (LEAPS) expiries show larger deviations
6. Half-life of mean reversion < 1 day — deviations not exploitable in practice

Prior literature (Kamara & Miller 1995, Ofek et al. 2004):
PCP holds for index options within transaction costs. Violations
concentrate when short-sale constraints bind or when capital is scarce.

---

## Limitations
- Historical options data availability: yfinance provides limited historical
  chains; full study requires CBOE DataShop or OptionMetrics
- Execution: mid-quote assumption overstates achievable P&L
- No early exercise premium: SPX is European so this is valid, but
  SPY (American) would require adjustment
- Dividend yield proxy (SPY trailing) introduces estimation error
- Regime sample sizes may be small for Crisis state
- No microstructure model — bid-ask proxy is simplified
- Transaction costs exclude clearing fees, margin requirements, capital costs

---

## Output Files
- findings.md — written research summary
- findings/deviation_timeseries.png — δ_norm over time with VIX overlay
- findings/deviation_distribution.png — histogram of δ_norm, δ_adj
- findings/arbitrage_bounds.png — bound violation frequency by year
- findings/regime_conditioning.png — mean |δ| by HMM regime
- findings/vix_quintile.png — mean |δ| by VIX quintile (bar chart)
- findings/expiry_bucket.png — deviation by DTE bucket
- findings/mean_reversion.png — AR(1) half-life scatter / regression

---

## Framework Integration
- Strategy file: strategies/pcp_arbitrage.py
- Backtest engine: backtests/engine.py
- Data loader: utils/data_loader.py
- Performance: utils/performance.py (calculate_metrics + print_summary)
- HMM: research/regime_detection/hmm_model.py (RegimeDetector)
- HMM features: research/regime_detection/features.py
- Registered in run.py as "pcp_arb" key via run_pcp_arb() wrapper

---

## Notes
- No look-ahead bias: all inputs (C, P, S, K, r, T) are known at snapshot time
- ATM selection uses spot price known at snapshot — no forward-looking filter
- VIX conditioning and regime conditioning run independently — report both
- All plots saved to findings/ subfolder
- Script fully modular — each step is an independent function
- Data limitation (historical chains) documented inline — do not suppress this caveat
- Dividend adjustment included as robustness check, not primary test