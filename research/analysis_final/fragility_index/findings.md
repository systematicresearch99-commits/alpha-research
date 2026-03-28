# Concentration vs Market Fragility Index — Findings
**Author**: AlphaByProcess  
**Period**: 2015-01-01 – 2025-12-31  
**Universe**: S&P 500 (US) + Nifty 500 (India)  
**Frequency**: Weekly signal, monthly validation  

---

## Methodology
The Market Fragility Score (MFS) is a composite of 4 percentile-ranked sub-signals:
1. **Top-5 Weight** — Concentration at the very top of the index
2. **Cross-Sectional Dispersion** — Return spread across constituents (rolling 21d)
3. **Breadth Deterioration** — % of stocks below 50-day MA
4. **Correlation Clustering** — Average pairwise correlation of top-50 stocks (rolling 63d)

Two versions tested:
- **MFS_equal**: Equal-weight composite
- **MFS_robust**: Inverse-correlation-weighted (down-weights redundant signals)

> [!] Survivorship bias caveat: constituent lists reflect current index membership.
> Point-in-time membership data (Compustat) would strengthen causal inference.

---

## Results Summary

Market                Test Q1 (Safe) Mean Q5 (Fragile) Mean               Stat p-value Significant
    US       Drawdown (1M)         -2.61%            -5.21%           KW=26.30  0.0000           ✓
    US       Drawdown (3M)         -6.40%            -8.86%           KW=31.16  0.0000           ✓
    US Vol Expansion (OLS)              —                 — β=0.0074, R²=0.171  0.0000           ✓
    US   Momentum Reversal          0.03%             0.10%             t=0.11  0.9127           ✗
    IN       Drawdown (1M)         -2.85%            -4.73%           KW=35.40  0.0000           ✓
    IN       Drawdown (3M)         -6.74%            -6.69%           KW=15.41  0.0039           ✓
    IN Vol Expansion (OLS)              —                 — β=0.0064, R²=0.141  0.0000           ✓
    IN   Momentum Reversal          0.38%             0.47%             t=0.12  0.9016           ✗

---

## Interpretation
*[To be filled post-run based on actual coefficient signs and significance levels]*

### Test 1 — Drawdown
- Does Q5 (highest fragility) show meaningfully worse forward drawdowns than Q1?
- Is the Kruskal-Wallis result statistically significant (p < 0.05)?

### Test 2 — Vol Expansion
- Is β positive and significant? (Higher MFS → higher forward vol)
- Does Granger test confirm temporal precedence?

### Test 3 — Momentum Reversal
- Does high fragility (MFS > 80th pct) coincide with more negative L/S returns?
- Is the difference in means statistically significant?

---

## Robustness
- MFS_robust vs MFS_equal convergence/divergence
- US vs India cross-market consistency

---

## Next Steps
- [ ] Add HMM regime conditioning (from existing regime_detection module)
- [ ] Test MFS as a factor in Fama-MacBeth cross-sectional regression
- [ ] Publish on Substack: "Concentration as an Early-Warning Signal"
