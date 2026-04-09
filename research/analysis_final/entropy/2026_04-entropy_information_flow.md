# Entropy and Information Flow in Financial Markets

**Date:** 2026-04-08
**Status:** Idea
**Author:** @AlphaByProcess

---

## Question

Are financial markets becoming more or less predictable over time — and
does that change vary by market structure, regime, or asset class?

---

## Hypothesis

> The information entropy of equity return distributions is non-stationary.
> Markets cycle between high-entropy (unpredictable, efficient) and
> low-entropy (structured, trend-dominated) regimes. The transition into
> low-entropy states precedes or coincides with identifiable macro regimes
> (volatility spikes, drawdowns, rate cycles).

**Null hypothesis:**
> Return entropy is stationary. Markets are uniformly efficient across
> time, and entropy measures carry no regime-predictive or return-predictive
> information.

**Secondary hypothesis:**
> Indian equity markets (Nifty sectors) exhibit structurally higher entropy
> than US equity markets (S&P 500 sectors) over the same period — consistent
> with the EM narrative of thinner liquidity, higher retail participation,
> and more fragmented information flow.

---

## Why It Matters

Entropy is a direct measure of market efficiency at a given point in time.
Low entropy means return distributions are compressed — price moves are
clustering, information is not being priced uniformly, and the market is
behaving in a structured (potentially exploitable) way. High entropy means
the market is close to random — the efficient market hypothesis holds locally.

Tracking entropy over time turns a static efficiency question into a dynamic
one. The practical implication: if entropy reliably drops before volatility
spikes or regime shifts, it functions as a leading indicator of market stress
— something neither VIX nor realised vol alone captures well.

The India vs US comparison gives this a publishable angle. If EM markets run
at structurally higher entropy, the standard assumption that EM = more
exploitable may be wrong. Disorder and predictability are not the same thing.

---

## Entropy Measures

Three measures are computed and compared for robustness. Each captures a
different aspect of information structure.

### Shannon Entropy (H)

Classical information entropy applied to discretized return distributions.

```
H = -∑ p(x) log p(x)
```

Returns are discretized into bins. H measures how uniformly distributed
returns are across those bins. Maximum H = uniform distribution (pure noise).
Minimum H = all mass in one bin (perfectly predictable direction).

**Implementation:** Rolling window (e.g. 60 trading days), returns binned
into N equal-width bins. Sensitive to bin count choice — must be tested
across bin sizes for robustness.

---

### Approximate Entropy (ApEn)

Measures the regularity or predictability of a time series without
discretization. Counts how often patterns of length m repeat as patterns
of length m+1. Low ApEn = highly regular, repetitive series. High ApEn =
irregular, unpredictable.

**Parameters:** m = 2 (template length), r = 0.2 × std(series) (tolerance).
Standard parameterisation from Pincus (1991).

**Advantage over Shannon:** Captures temporal structure (sequence order
matters), not just distributional shape. Two series with identical return
distributions but different autocorrelation structures will have different ApEn.

---

### Sample Entropy (SampEn)

A refinement of ApEn that eliminates self-matching bias and is less sensitive
to series length. Better suited to shorter rolling windows.

**Parameters:** m = 2, r = 0.2 × std(series). Same as ApEn for comparability.

**Why include both ApEn and SampEn:** ApEn is more widely cited in the
finance literature; SampEn is more statistically robust. Comparing them
tests whether the findings are measure-specific or structural.

---

## Tests

### Test 1 — Entropy Time Series: Are Markets Getting More or Less Predictable?

- Compute all three entropy measures on rolling 60-day windows for each series
- Plot entropy time series for Nifty 50 and S&P 500 (2012–present)
- Test for trend using Mann-Kendall test (non-parametric, appropriate for
  non-normal entropy series)
- Hypothesis: entropy is trending upward (markets becoming more efficient
  over time due to algo penetration, information diffusion)

**Benchmark:** If Mann-Kendall τ > 0 and significant → upward trend (more
efficient). If τ < 0 → markets becoming more structured/exploitable over time.

---

### Test 2 — Entropy vs Volatility

- Compute rolling realised volatility (std of daily returns, same 60-day window)
- Correlate entropy with volatility using Spearman rank correlation
- Regime-condition the correlation: Pre-COVID / COVID Recovery / Rate Hike

**Prior expectation:** Negative correlation — low entropy (structured
markets) tends to coincide with low volatility (trending, complacent phases).
High entropy tends to coincide with high volatility (panic, dispersed
information). But this is not guaranteed — test the direction empirically.

**Secondary test:** Does entropy lead or lag volatility? Run cross-correlation
at lags ±1 to ±10 days to check if entropy drops *before* volatility spikes.
If yes, this is the core tradable insight.

---

### Test 3 — Entropy vs Forward Returns

- Correlate current entropy (rolling 60-day) with forward 1-month and
  3-month returns
- Run for each Nifty sector and S&P 500 sector individually
- Spearman rank correlation; report ρ and p-value

**Prior expectation:** Weak negative correlation — low-entropy (structured)
environments may be associated with continuation (momentum), high-entropy
environments with mean reversion. But the relationship is likely sector- and
regime-specific. Report what the data shows.

---

### Test 4 — Entropy as a Regime Classifier

- Identify known regime inflection points: GFC aftermath (2012–13), demonetisation
  (Nov 2016), COVID crash (Feb–Mar 2020), COVID recovery (Apr 2020–Dec 2021),
  rate hike cycle (Jan 2022–present)
- Test whether entropy measures shift significantly at these transitions
  (Wilcoxon rank-sum test across adjacent regime windows)
- Build a simple threshold-based regime classifier using entropy alone:
  Low / Medium / High entropy regime
- Compare classifier output against the calendar-based regime labels

**Goal:** Determine whether entropy can endogenously identify regime transitions
without requiring a pre-labelled macro calendar.

---

### Test 5 — India vs US Structural Comparison

- Compute mean entropy (all three measures) for India sectors and US sectors
  over the full sample
- Bootstrap test (n=1000) for statistical significance of the India–US gap
- Regime-condition the comparison: is the gap consistent or driven by one period?

**Hypothesis:** India runs at higher Shannon entropy (more uniform, less
structured) but potentially lower ApEn/SampEn (more temporally regular at the
sequence level). The two types of entropy can diverge — Shannon measures
distributional spread, ApEn/SampEn measures sequential regularity.

---

## Data

| Index | Ticker (yfinance) | Market |
|---|---|---|
| Nifty 50 | `^NSEI` | India — Headline |
| Nifty IT | `^CNXIT` | India — Sector |
| Nifty Pharma | `^CNXPHARMA` | India — Sector |
| Nifty FMCG | `^CNXFMCG` | India — Sector |
| Nifty Bank | `^NSEBANK` | India — Sector |
| Nifty Auto | `^CNXAUTO` | India — Sector |
| Nifty Metal | `^CNXMETAL` | India — Sector |
| Nifty Realty | `^CNXREALTY` | India — Sector |
| Nifty Energy | `^CNXENERGY` | India — Sector |
| S&P 500 | `^GSPC` | US — Headline |
| S&P Tech | `XLK` | US — Sector |
| S&P Health | `XLV` | US — Sector |
| S&P Energy | `XLE` | US — Sector |
| S&P Financials | `XLF` | US — Sector |
| S&P Cons. Disc | `XLY` | US — Sector |
| S&P Cons. Stap | `XLP` | US — Sector |
| S&P Industrial | `XLI` | US — Sector |
| S&P Materials | `XLB` | US — Sector |

**Period:** January 2012 – present (daily prices, ~3,500 observations per series)
**Frequency:** Daily returns for entropy computation; monthly aggregation for
regime-level comparisons

> ⚠️ ApEn and SampEn are computationally expensive on long daily series.
> Profile runtime on a single series before running the full universe.
> Consider Numba or vectorised NumPy implementations if runtime exceeds 5 minutes.

> ⚠️ Shannon entropy is sensitive to bin count N. Test N = 10, 20, 50 and
> report which produces the most stable time series before committing to a
> single value.

---

## Open Questions

- Does the three-measure comparison converge (all tell the same story) or
  diverge (each captures a distinct dimension of market disorder)?
- Is the India–US entropy gap stable over time, or does it collapse during
  global stress events when both markets become correlated?
- Can entropy be combined with the sector rotation findings (persistence ratio)
  into a unified market structure indicator?
- Would intraday entropy (using 5-minute bar data) produce sharper regime
  signals than end-of-day?
