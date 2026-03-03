# Post-2022 Market Structure: Concentration vs Structural Dependence

## Abstract

This note investigates whether the post-2022 AI rally represented a structural shift in index dependence on mega-cap equities, or merely a dispersion-driven concentration of returns.

Two questions are examined:

1. Did cap-weighted index performance increasingly diverge from equal-weight performance?
2. Did statistical dependence on mega-cap leaders increase during the same period?

---

## Data & Instruments

- **SPY** — Cap-weighted S&P 500 proxy  
- **RSP** — Equal-weight S&P 500 proxy  
- **Top 5 Basket** — NVDA, AAPL, MSFT, AMZN, GOOGL  

**Regime split:** November 2022  

Daily data sourced via `yfinance`.

Top-5 basket weighted using a market-cap proxy:
`shares outstanding × price`, normalized within basket.

---

## Methodology

The following metrics were computed:

1. Cumulative return comparison (SPY vs RSP)
2. 6-month rolling relative performance (SPY / RSP)
3. 60-day rolling R² between SPY returns and weighted Top-5 basket returns

R² is used as a measure of structural explanatory power.

---

## Results

### Performance Divergence

**Pre-AI Regime**
- SPY: +54.1%
- RSP: +49.1%

**Post-AI Regime**
- SPY: +86.6%
- RSP: +54.4%

**Average 6-month relative acceleration**
- Pre-AI: 0.0061
- Post-AI: 0.0337

Performance concentration increased materially.

---

### Structural Dependence

**Average Rolling R² (SPY explained by weighted Top-5):**

- Pre-AI: 0.7220  
- Post-AI: 0.7119  

Statistical dependence did not increase.

---

## Interpretation

While cap-weighted performance significantly outpaced equal-weight post-2022, the explanatory power of mega-cap leaders over index returns did not increase.

This suggests that the post-2022 concentration episode was dispersion-driven rather than structurally driven.

The index was already highly correlated with its largest constituents prior to the AI rally.

---

## Caveat

RSP rebalances quarterly; part of the SPY–RSP divergence may reflect rebalancing drag rather than pure concentration effects.

---

## Reproducibility

Analysis script:

[View analysis script](../research/analysis/concentration_analysis.py)

Run locally:

```bash
python research/analysis/concentration_analysis.py