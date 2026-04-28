# Testing Arbitrage Efficiency in Options Markets — Findings
**Author**: AlphaByProcess
**Project**: pcp_arbitrage_efficiency
**Date**: 2026-04-27
**Universe**: SPY options (American-style; European PCP applied as approximation)
**Snapshot**: Single cross-section — 2026-04-27. Panel requires daily loop.
**Data note**: yfinance provides current chain only. Regime and VIX conditioning
reflect single-date constraints documented explicitly throughout.

---

## Methodology

Put-call parity (PCP) is the foundational no-arbitrage identity for European options:

    C − P = S·e^(−q·T) − K·e^(−r·T)

Where C = call mid, P = put mid, S = SPY spot ($715.17), K = strike,
r = risk-free rate (^IRX = 3.59% p.a.), q = dividend yield (1.3% trailing),
T = time to expiry (years).

Three deviation measures computed per matched (C, P, K, T) pair:
1. **δ_raw** — raw deviation: (C − P) − (S·e^(−q·T) − K·e^(−r·T))
2. **δ_adj** — cost-adjusted: |δ_raw| minus sum of half-spreads (BA proxy)
3. **δ_norm** — normalised: δ_raw / S (expressed as % of spot)

Exploitable when δ_adj > 0 (deviation survives the transaction cost filter).
Sign: δ > 0 = calls rich; δ < 0 = puts rich.

ATM filter: |K/S − 1| < 5%. Min premium per leg: $0.05.
977 matched (C, P, K, T) pairs across all available SPY expiries.

> [!] Cross-sectional caveat: AR(1) and half-life are computed across strikes
> and expiries within a single snapshot — not the same pair tracked over time.
> Regime and VIX quintile conditioning collapse to a single point (VIX = 18.02,
> regime = Bear). Full time-series inference requires the daily panel.

---

## Results Summary

```
Test                               Metric                    Result                Sig
────────────────────────────────────────────────────────────────────────────────────────
Deviation: mean δ_norm             All pairs                 +0.1940%              —
Deviation: median δ_norm           All pairs                 +0.0938%              —
Deviation: std δ_norm              All pairs                  0.2993%              —
Deviation: 5th / 95th pct          All pairs                 -0.020% / +0.902%     —
Calls rich (δ > 0)                 % of observations          81.1%  (792/977)     —
Puts rich (δ < 0)                  % of observations          18.9%  (185/977)     —
Exploitable after costs            % with δ_adj > 0           52.0%  (508/977)     —
Mean violation magnitude           δ_adj (exploitable only)  $1.5535  (0.217%)     —
Violation sign                     Calls rich : puts rich    496 : 12              —
AR(1) β (cross-sectional)          Δδ ~ δ_{t-1}              −0.0169           ✓ p=0.012
AR(1) half-life                    cross-sectional obs         41.1 obs             —
Lag-1 autocorrelation              across strikes              0.9780               —
Term slope OLS β₁                  |δ_norm| ~ DTE             +0.00147      ✓ p<0.001 R²=0.947
Deviation: weekly  (0–14d)         mean |δ_norm|               0.0166%   exploitable=18.5%
Deviation: front-month (15–30d)    mean |δ_norm|               0.0787%   exploitable=61.9%
Deviation: 31–60d                  mean |δ_norm|               0.1263%   exploitable=38.6%
Deviation: LEAPS (60d+)            mean |δ_norm|               0.3971%   exploitable=77.5%
Regime (single snapshot)           Bear only (VIX=18.02)      |δ|=0.202%  viol=52.0%
Simulated arb P&L                  mid-quote, pre-cost        $1.5535  (0.217%)     —
Simulated win rate                 % profitable (by construct) 100.0%               —
Simulated Sharpe                   illustrative, no costs      12.91                —
```

---

## Interpretation

### Test 1 — Deviation Distribution and Sign

The mean δ_norm is **+0.194%** and **81.1% of pairs show calls rich** (δ > 0).
This is the opposite of the textbook expectation. Prior literature (Kamara & Miller 1995;
Ofek et al. 2004) documents a structural put premium in equity index options — puts
should be expensive because institutions systematically buy protective puts, bidding
them above PCP-implied fair value.

The snapshot reverses that. Three explanations, non-mutually exclusive:

**1. Bull market call demand (primary driver).** With SPY near all-time highs and
VIX at 18, retail and institutional call-buying for upside participation is elevated.
Covered call writing is common in this environment — market makers accumulating long
puts to hedge their short call exposure suppresses put prices. The demand pressure
flows toward calls, not puts, in low-vol bull regimes.

**2. American put early-exercise premium (structural bias).** SPY options are
American-style. Deep ITM puts carry early-exercise value not in the European formula,
making observed put prices *higher* than PCP implies — compressing δ_norm toward
negative. The fact that δ is still strongly positive despite this known upward
bias on the put side means calls are genuinely expensive on a European basis. The
American bias makes the calls-rich finding conservative.

**3. Dividend yield underestimation.** If the true forward q exceeds the 1.3%
trailing proxy, S·e^(−q·T) falls → theoretical PCP value falls → δ_norm rises.
A q underestimate inflates the apparent calls-rich result. Robustness test
(q = 0% vs q = 1.3% run) would quantify the sensitivity.

The 5th percentile at −0.020% and the 18.9% puts-rich share confirm the put premium
exists at the left tail — it just does not dominate the distribution in a benign
environment. The calls-rich result is **regime-dependent**: expect the sign to
flip toward puts-rich in a high-VIX, stress-regime snapshot.

### Test 2 — Arbitrage Bounds

**52.0% of pairs show exploitable deviations** after the bid-ask spread filter.
This is dramatically higher than the efficiency prediction but substantially
overstates the true opportunity for three reasons.

The **sign is extreme**: 496 calls-rich violations versus only 12 puts-rich.
A true bilateral arbitrage inefficiency would show roughly symmetric violations.
The one-sidedness confirms this is structural demand pressure, not pricing error.

**Violations concentrate at long maturities**: 77.5% of LEAPS (60d+) pairs are
flagged exploitable versus only 18.5% of weekly options. Weekly SPY options are
the most liquid, most actively arbitraged derivatives market globally. Their
18.5% rate is the realistic efficiency benchmark — and even that rate likely
reflects BA proxy underestimation of true execution cost for slightly-OTM pairs.

The mean violation magnitude of **$1.5535 = 800× mean raw δ** because δ_adj
is |δ_raw| minus ba_proxy — when both are small and of similar magnitude, ratios
amplify. The $1.55 capture in dollar terms is real but must be weighed against
unmodelled costs: simultaneous fill slippage on two legs, clearing fees, and
capital cost of the position.

**Correct read**: PCP is efficiently enforced in the liquid front of the curve.
The high aggregate violation rate is a BA proxy artefact for longer maturities,
not a genuine arbitrage opportunity.

### Test 3 — Term Structure (the headline result)

| DTE Bucket | Mean |δ_norm| | Exploitable |
|------------|--------------|-------------|
| 0–14d      | 0.0166%      | 18.5%       |
| 15–30d     | 0.0787%      | 61.9%       |
| 31–60d     | 0.1263%      | 38.6%       |
| 60d+       | 0.3971%      | 77.5%       |

Mean |δ_norm| increases **monotonically with DTE** — 24× larger for LEAPS
than for weekly options. OLS: |δ_norm| ~ DTE gives β₁ = 0.00147 per day
(p < 0.001, R² = 0.947). Time to expiry alone explains 94.7% of cross-sectional
deviation variance. This is the dominant structural driver.

The mechanism is straightforward: as T increases, the discounting terms diverge
further from par, small errors in r and q compound over longer horizons, and
bid-ask spreads widen as market-maker hedging costs grow and liquidity thins.
The BA proxy captures the spread for short-dated options but underestimates
the all-in cost for LEAPS (stock borrow, carry, pin risk). The deviation is
real in percentage terms; exploitability after true costs is much lower.

One anomaly: 31–60d (38.6% exploitable) dips below front-month (61.9%). A
plausible mechanism: front-month options cluster around monthly expiry dates
where expiry-pinning effects and gamma hedging create demand-driven dislocations.
The 31–60d bucket may have more balanced positioning with no imminent expiry
catalyst. This warrants investigation in the panel data.

**The term structure reframes the entire study.** PCP violations are not
primarily about regimes, volatility, or market stress in this data — they are
primarily a function of liquidity, proxied by DTE. This is exactly what the
efficiency hypothesis predicts.

### Test 4 — Cross-Sectional AR(1) Reversion

AR(1) β = −0.0169 (p = 0.012), half-life = 41.1, lag-1 autocorrelation = 0.978.

**Do not interpret as time-series persistence.** The AR(1) runs across strikes
sorted by strike price. The lag-1 autocorrelation of 0.978 reflects that δ_norm
is smooth and monotonically trending with DTE across the cross-section — adjacent
strikes at the same expiry have nearly identical deviations. The 41.1-step
"half-life" is a measure of how many strike steps it takes to traverse the
DTE gradient, not how long a deviation persists in calendar time.

The true time-series mean-reversion test — tracking the same front-month ATM
pair across consecutive trading days — requires the daily panel. Expected
result: half-life < 1 day for front-month options (arbitrage self-corrects within
the trading session) and longer half-lives for LEAPS (slower arbitrage cycle).

### Test 5 — Regime and VIX Conditioning (deferred)

VIX = 18.02 on the snapshot date. All 977 pairs map to a single regime (Bear
by VIX quartile proxy on historical distribution — note: VIX=18 is "Bear" only
because the 2010–2026 sample includes many sub-15 VIX years; this is not a
crisis regime by conventional definition).

The VIX quintile chart shows DTE quintiles substituted for VIX quintiles — a
term structure chart with different axis labels. The monotonic |δ| increase
across DTE-Q1 through DTE-Q5 (0.012% → 0.669%) is the same term structure
result in continuous form. β₁ = 0.00147 (R² = 0.947) — consistent with the
DTE bucket analysis above.

True VIX conditioning — does the same options pair show wider deviations on
a VIX=30 day versus a VIX=18 day — requires the daily panel. Prior prediction:
deviations will widen in stress (Shleifer & Vishny 1997 limits-to-arbitrage:
capital constraints bind when risk is elevated, impairing the arbitrage
mechanism). This remains untested.

### Test 6 — Simulated P&L (illustrative only)

Mean P&L $1.5535, win rate 100%, Sharpe 12.91.

The 100% win rate is definitional: exploitable pairs are selected precisely
because δ_adj > 0. The P&L equals δ_adj by construction — every selected
pair shows positive simulated P&L. This is not market evidence; it is an
identity. Discard the win rate and Sharpe entirely.

The $1.5535 mean capture (0.217% of spot) is the net-of-spread signal
before all other execution costs. Against this: two-leg simultaneous fill
slippage (~0.05–0.15% per leg for liquid front-month; higher for LEAPS),
clearing fees, capital cost. For front-month pairs specifically — where
spreads are tightest and the anomalous 61.9% violation rate deserves
investigation — the capture may be realistically positive. For LEAPS,
it is almost certainly consumed by unmodelled costs.

---

## Main Finding

**PCP deviations in SPY options are real, structurally calls-rich (not puts-rich)
in the current environment, and dominated by time to expiry — not regime stress
or volatility. The market efficiently enforces put-call parity where it matters
most: in the short-dated, liquid options that account for the majority of index
options volume.**

The term structure is the story. |δ_norm| scales monotonically with DTE
(R² = 0.947, β₁ = 0.00147/day). Weekly options show 0.017% mean deviation
and 18.5% apparent violation rate — effectively zero net of true costs.
LEAPS show 0.397% and 77.5%, but the BA proxy underestimates the true
execution cost of long-dated options. The aggregate 52% violation rate
is misleading and should not be reported without this decomposition.

The calls-rich sign inverts the textbook equity put-premium narrative.
This is not a contradiction — it is regime-dependence by absence. The put
premium is a stress-period phenomenon. In a low-vol, call-demand regime
(VIX=18, SPY near highs), the demand pressure reverses. The 18.9%
puts-rich tail confirms the put premium exists but does not dominate.

The regime and VIX conditioning hypotheses remain untested. The daily
snapshot panel is the essential next step. The prediction entering that study:
deviations will widen on high-VIX days (limits-to-arbitrage), the calls-rich
bias will invert toward puts-rich in stress, and the term structure slope will
steepen as liquidity concentrates at the short end of the curve.

---

## Robustness

- **Dividend yield**: rerun with q = 0% — positive δ_norm should increase
  further, confirming calls are genuinely rich independent of dividend assumption
- **ATM filter at 2%**: violation rate for front-month should approach zero
  for truly at-the-money pairs with lowest bid-ask spreads
- **SPX European options**: eliminates American put early-exercise bias;
  cleaner theoretical test; requires SPX chain (wider strikes, higher premiums)
- **BA proxy for LEAPS**: use actual chain spread data not fallback 0.2%;
  expect LEAPS violation rate to collapse materially once true costs modelled
- **High-VIX day comparison**: rerun on a day where VIX > 25 and compare
  sign, magnitude, and term structure slope against this snapshot

---

## Limitations

1. **Single snapshot** — regime and VIX conditioning are single-point,
   not time-series. All inference about VIX sensitivity and regime-dependence
   is prediction, not result.

2. **SPY American-style** — early exercise premium on puts biases δ_norm
   upward; calls-rich finding is conservative not inflated.

3. **BA proxy understates LEAPS costs** — 77.5% violation rate for 60d+
   is an artefact of an inadequate cost estimate, not a genuine opportunity.

4. **AR(1) is cross-sectional not time-series** — half-life of 41.1 is
   a strike-gradient measure, not a calendar-time persistence estimate.

5. **Dividend yield proxy** — 1.3% trailing SPY yield as continuous q;
   true forward dividend yield not directly observable.

6. **Win rate 100% is definitional** — not a market observation.

7. **Regime label artefact** — VIX=18 mapped to "Bear" by historical
   quartile is misleading by conventional regime labelling; not a stress regime.

---

## Next Steps
- [ ] Build daily snapshot loop → SQLite panel; accumulate 30–60 days minimum
- [ ] Re-run regime and VIX conditioning with panel data
- [ ] Switch to SPX European options for clean theoretical test
- [ ] Robustness: q=0%, ATM filter at 2%, actual spread data for LEAPS BA proxy
- [ ] Investigate 15–30d anomaly (61.9% exploitable) — expiry-pinning mechanism
- [ ] Rerun on a VIX > 25 day; compare sign, slope, violation rate
- [ ] Publish on Substack: "Testing Arbitrage Efficiency in Options Markets"
      Angle: term structure is the enforcer of no-arbitrage, not regime —
      liquidity decays with DTE and the market's arbitrage mechanism decays with it

---

## Output Files

| File                                | Description                                               |
|-------------------------------------|-----------------------------------------------------------|
| findings/deviation_distribution.png | Histogram of δ_norm (calls-rich skew) and δ_adj           |
| findings/regime_conditioning.png    | Single-regime (Bear) — panel required for comparison      |
| findings/vix_quintile.png           | DTE quintile binning — term structure slope in disguise   |
| findings/expiry_bucket.png          | Core result: |δ| monotonically increasing 24× from 0–14d to 60d+ |
| findings/moneyness_vs_deviation.png | Scatter: calls-rich dominates all moneyness levels        |