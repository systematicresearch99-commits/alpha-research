"""
Entropy Project — Data Ingestion | @AlphaByProcess
research/analysis_final/entropy/fetch_data.py

Run BEFORE entropy_analysis.py:
  cd research/analysis_final/entropy
  python fetch_data.py

Downloads daily OHLCV from yfinance for all India + US series.
Saves close prices to:
  ../../../data/raw/entropy_prices_daily.csv

Columns match ALL_SERIES in entropy_analysis.py exactly.
"""

import yfinance as yf
import pandas as pd
import os

# ── Output path ───────────────────────────────────────────────────────────────

OUT_DIR  = "../../../data/raw"
OUT_FILE = os.path.join(OUT_DIR, "entropy_prices_daily.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Ticker map ────────────────────────────────────────────────────────────────
# key   = column name used in entropy_analysis.py
# value = yfinance ticker

TICKER_MAP = {
    # India — headline + sectors
    "Nifty50":       "^NSEI",
    "IT":            "^CNXIT",
    "Pharma":        "^CNXPHARMA",
    "FMCG":          "^CNXFMCG",
    "Bank":          "^NSEBANK",
    "Auto":          "^CNXAUTO",
    "Metal":         "^CNXMETAL",
    "Realty":        "^CNXREALTY",
    "Energy":        "^CNXENERGY",
    # US — headline + sectors
    "SP500":         "^GSPC",
    "US_Tech":       "XLK",
    "US_Health":     "XLV",
    "US_Energy":     "XLE",
    "US_Financials": "XLF",
    "US_ConsDisc":   "XLY",
    "US_ConsStap":   "XLP",
    "US_Industrial": "XLI",
    "US_Materials":  "XLB",
}

START = "2012-01-01"
END   = None   # None = today

# ── Download ──────────────────────────────────────────────────────────────────

print("=" * 55)
print("ENTROPY DATA INGESTION")
print("=" * 55)
print(f"  Period   : {START} → present")
print(f"  Series   : {len(TICKER_MAP)}")
print(f"  Output   : {OUT_FILE}\n")

frames = {}
failed = []

for col_name, ticker in TICKER_MAP.items():
    try:
        raw = yf.download(ticker, start=START, end=END,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("empty response")
        close = raw["Close"].squeeze()
        close.name = col_name
        frames[col_name] = close
        print(f"  ✓  {col_name:<20} {ticker:<14} "
              f"{close.index[0].date()} → {close.index[-1].date()}  "
              f"({len(close)} obs)")
    except Exception as e:
        print(f"  ✗  {col_name:<20} {ticker:<14} FAILED: {e}")
        failed.append(col_name)

# ── Assemble & align ──────────────────────────────────────────────────────────

print(f"\n  Aligning on common trading calendar...")
prices = pd.DataFrame(frames)

# Forward-fill up to 3 days to handle India/US holiday mismatches,
# then drop rows where any series is still missing
prices = prices.ffill(limit=3)
before = len(prices)
prices = prices.dropna()
after  = len(prices)

print(f"  Rows before dropna : {before}")
print(f"  Rows after  dropna : {after}  (dropped {before - after} non-overlapping)")
print(f"  Date range         : {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"  Columns            : {list(prices.columns)}")

# ── Save ──────────────────────────────────────────────────────────────────────

prices.index.name = "Date"
prices.to_csv(OUT_FILE)
print(f"\n  Saved → {OUT_FILE}")

if failed:
    print(f"\n  ⚠  Failed tickers: {failed}")
    print(f"     Re-run or check yfinance coverage for these before proceeding.")
else:
    print(f"\n  All {len(TICKER_MAP)} series downloaded successfully.")

print("\n" + "=" * 55)
print("DONE — now run entropy_analysis.py")
print("=" * 55)

