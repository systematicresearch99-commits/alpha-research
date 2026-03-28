"""
Concentration vs Market Fragility Index
========================================
Research Question:
    Does a composite Market Fragility Score (MFS) — built from concentration
    and breadth metrics — serve as a leading indicator of:
        1. Index drawdowns
        2. Cross-sectional volatility expansion
        3. Factor (momentum) reversals

Universe  : S&P 500 (US) + Nifty 500 (India) — parallel study
Frequency : Weekly signal construction, monthly validation
Period    : 2015-01-01 to 2025-12-31
Author    : AlphaByProcess
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
START      = "2015-01-01"
END        = "2025-12-31"
TOP_N_CORR = 50        # constituents used for correlation clustering
ROLL_VOL   = 21        # rolling window for dispersion/vol (trading days)
ROLL_CORR  = 63        # rolling window for correlation clustering
MA_WINDOW  = 50        # breadth MA window
MOM_FORM   = 252       # momentum formation period (12M proxy)
MOM_SKIP   = 21        # skip last month
GRANGER_LAG= 4         # lags for Granger test (weeks)

OUTPUT_DIR = Path("research/analysis_final/fragility_index/findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# S&P 500 TICKERS  (representative 100-stock subset for tractability)
# Full point-in-time membership requires Compustat; we use current members
# and note survivorship-bias caveat in findings.md
# ─────────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","LLY",
    "V","UNH","XOM","MA","JNJ","PG","HD","AVGO","MRK","COST",
    "ABBV","CVX","CRM","BAC","NFLX","AMD","PEP","KO","WMT","TMO",
    "ADBE","ACN","MCD","ABT","CSCO","DIS","GE","VZ","DHR","TXN",
    "NEE","PM","CMCSA","AMGN","RTX","INTU","SPGI","HON","IBM","AMAT",
    "CAT","GS","MS","BLK","ISRG","LOW","BKNG","ELV","SYK","REGN",
    "PLD","VRTX","MDLZ","AXP","ADI","GILD","NOW","T","DE","LRCX",
    "ZTS","MMC","CB","MU","SCHW","CI","PANW","SO","DUK","EOG",
    "TJX","BSX","SLB","BDX","CL","AON","ITW","CME","PH","APD",
    "NOC","GD","ETN","FI","HCA","WM","KLAC","SNPS","CDNS","MCK"
]

# ─────────────────────────────────────────────
# NIFTY 500 TICKERS  (representative 80-stock subset — liquid large/mid cap)
# ─────────────────────────────────────────────
NIFTY_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "SBIN.NS","INFY.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "BAJFINANCE.NS","HCLTECH.NS","MARUTI.NS","SUNPHARMA.NS","ADANIENT.NS",
    "KOTAKBANK.NS","TITAN.NS","WIPRO.NS","ONGC.NS","NTPC.NS",
    "POWERGRID.NS","ULTRACEMCO.NS","ASIANPAINT.NS","AXISBANK.NS","M&M.NS",
    "NESTLEIND.NS","BAJAJFINSV.NS","TECHM.NS","JSWSTEEL.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","INDUSINDBK.NS","HINDALCO.NS","GRASIM.NS","COALINDIA.NS",
    "BPCL.NS","DRREDDY.NS","DIVISLAB.NS","CIPLA.NS","EICHERMOT.NS",
    "BRITANNIA.NS","TATACONSUM.NS","HEROMOTOCO.NS","SHREECEM.NS","APOLLOHOSP.NS",
    "SBILIFE.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS","ADANIPORTS.NS","HAVELLS.NS",
    "PIDILITIND.NS","DABUR.NS","GODREJCP.NS","MARICO.NS","BERGEPAINT.NS",
    "MUTHOOTFIN.NS","CHOLAFIN.NS","PIIND.NS","ALKEM.NS","TORNTPHARM.NS",
    "LTIM.NS","PERSISTENT.NS","MPHASIS.NS","COFORGE.NS","OFSS.NS",
    "DMART.NS","TRENT.NS","NAUKRI.NS","INDIGO.NS","IRCTC.NS",
    "ZOMATO.NS","POLICYBZR.NS","PAYTM.NS","DELHIVERY.NS","NYKAA.NS",
    "ADANIGREEN.NS","ADANITRANS.NS","ATGL.NS","AWL.NS","ABFRL.NS"
]

INDEX_TICKER = {"US": "SPY", "IN": "^NSEI"}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class DataLoader:
    def __init__(self, tickers: list, market: str):
        self.tickers = tickers
        self.market  = market
        self.prices  : pd.DataFrame = pd.DataFrame()
        self.mcap    : pd.DataFrame = pd.DataFrame()
        self.index_px: pd.Series    = pd.Series(dtype=float)

    def fetch(self) -> "DataLoader":
        log.info(f"[{self.market}] Fetching {len(self.tickers)} tickers …")
        raw = yf.download(
            self.tickers,
            start=START, end=END,
            auto_adjust=True,
            progress=False
        )
        # Adj close prices
        if isinstance(raw.columns, pd.MultiIndex):
            self.prices = raw["Close"].dropna(axis=1, how="all")
        else:
            self.prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})

        # Market cap approximation: price × shares_outstanding
        # yfinance fast_info gives shares; we fetch per-ticker
        log.info(f"[{self.market}] Fetching market caps …")
        shares_dict = {}
        for tk in tqdm(self.prices.columns, desc=f"[{self.market}] mcap"):
            try:
                info = yf.Ticker(tk).fast_info
                shares = getattr(info, "shares", None)
                if shares and shares > 0:
                    shares_dict[tk] = shares
            except Exception:
                pass

        # Build daily mcap = price × shares (static shares — standard approximation)
        mcap_df = pd.DataFrame(index=self.prices.index)
        for tk, sh in shares_dict.items():
            if tk in self.prices.columns:
                mcap_df[tk] = self.prices[tk] * sh

        self.mcap    = mcap_df.dropna(axis=1, how="all")
        self.prices  = self.prices[self.mcap.columns]   # align

        # Index
        idx_ticker = INDEX_TICKER[self.market]
        log.info(f"[{self.market}] Fetching index {idx_ticker} …")
        idx_raw = yf.download(idx_ticker, start=START, end=END,
                              auto_adjust=True, progress=False)
        self.index_px = idx_raw["Close"].squeeze()
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: FRAGILITY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
class FragilityBuilder:
    """
    Computes 4 sub-signals and composites them into weekly MFS.

    Sub-signals (all percentile-ranked, 0=safe, 1=fragile):
        S1: Top-5 weight concentration
        S2: Cross-sectional return dispersion  (higher = more fragile)
        S3: Breadth deterioration              (lower % above MA = more fragile)
        S4: Correlation clustering             (higher avg corr = more fragile)
    """
    def __init__(self, prices: pd.DataFrame, mcap: pd.DataFrame, market: str):
        self.prices = prices
        self.mcap   = mcap
        self.market = market
        self.returns= prices.pct_change()
        self.weekly_mfs : pd.DataFrame = pd.DataFrame()

    # ── Sub-signal helpers ────────────────────────────────────────────────────
    def _top5_weight(self) -> pd.Series:
        """Daily top-5 market-cap weight."""
        total = self.mcap.sum(axis=1)
        weights = self.mcap.div(total, axis=0)
        top5 = weights.apply(lambda row: row.nlargest(5).sum(), axis=1)
        return top5.rename("top5_weight")

    def _cs_dispersion(self) -> pd.Series:
        """Rolling 21-day cross-sectional std of daily constituent returns."""
        disp = self.returns.rolling(ROLL_VOL).std().mean(axis=1)
        return disp.rename("cs_dispersion")

    def _breadth(self) -> pd.Series:
        """Daily % of stocks above their 50-day MA (inverted → fragility)."""
        ma = self.prices.rolling(MA_WINDOW).mean()
        above = (self.prices > ma).sum(axis=1) / self.prices.shape[1]
        return (1 - above).rename("breadth_deterioration")

    def _corr_clustering(self) -> pd.Series:
        """Rolling 63-day avg pairwise correlation of top-50 by avg mcap."""
        top_cols = (
            self.mcap.mean().nlargest(min(TOP_N_CORR, self.mcap.shape[1])).index.tolist()
        )
        ret_top = self.returns[top_cols]

        avg_corr = []
        dates    = []
        for i in range(ROLL_CORR, len(ret_top)):
            window = ret_top.iloc[i - ROLL_CORR: i]
            corr_m = window.corr().values
            # upper triangle, excluding diagonal
            upper  = corr_m[np.triu_indices_from(corr_m, k=1)]
            avg_corr.append(np.nanmean(upper))
            dates.append(ret_top.index[i])

        return pd.Series(avg_corr, index=dates, name="corr_clustering")

    # ── Percentile rank helper ────────────────────────────────────────────────
    @staticmethod
    def _pct_rank(series: pd.Series, window: int = 252) -> pd.Series:
        """Rolling percentile rank — maps value to [0, 1]."""
        return series.rolling(window, min_periods=60).rank(pct=True)

    # ── Composite MFS ────────────────────────────────────────────────────────
    def build(self) -> "FragilityBuilder":
        log.info(f"[{self.market}] Building fragility sub-signals …")
        s1 = self._top5_weight()
        s2 = self._cs_dispersion()
        s3 = self._breadth()
        s4 = self._corr_clustering()

        # Align all to common dates
        df = pd.concat([s1, s2, s3, s4], axis=1).dropna()

        # Percentile rank each
        for col in df.columns:
            df[col] = self._pct_rank(df[col])

        df = df.dropna()

        # ── Equal-weight composite ────────────────────────────────────────
        df["MFS_equal"] = df[["top5_weight","cs_dispersion",
                               "breadth_deterioration","corr_clustering"]].mean(axis=1)

        # ── Correlation-based robustness: weight inversely to inter-signal corr
        # Lower correlation with other signals → more unique info → higher weight
        sub = df[["top5_weight","cs_dispersion","breadth_deterioration","corr_clustering"]]
        corr_matrix = sub.corr().abs()
        avg_corr_per_signal = corr_matrix.mean()
        inv_corr_weights = (1 / avg_corr_per_signal) / (1 / avg_corr_per_signal).sum()
        df["MFS_robust"] = sub.mul(inv_corr_weights).sum(axis=1)

        # Resample to weekly (Friday close)
        self.weekly_mfs = df.resample("W-FRI").last().dropna()
        log.info(f"[{self.market}] MFS built — {len(self.weekly_mfs)} weekly obs.")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: TARGET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
class TargetBuilder:
    """
    Builds 3 forward-looking targets at weekly frequency:
        T1: Forward 1M / 3M index max drawdown
        T2: Forward 1M cross-sectional realized vol
        T3: Forward 1M momentum factor return (L/S 12-1)
    """
    def __init__(self, prices: pd.DataFrame, index_px: pd.Series,
                 weekly_mfs: pd.DataFrame, market: str):
        self.prices     = prices
        self.index_px   = index_px
        self.weekly_mfs = weekly_mfs
        self.market     = market
        self.returns    = prices.pct_change()
        self.targets    : pd.DataFrame = pd.DataFrame()

    def _forward_drawdown(self, horizon_days: int) -> pd.Series:
        """Max drawdown of index over next horizon_days trading days."""
        idx_ret = self.index_px.pct_change()
        weekly_idx = self.index_px.resample("W-FRI").last()
        results = {}
        for dt in self.weekly_mfs.index:
            fwd = self.index_px.loc[dt:].iloc[:horizon_days]
            if len(fwd) < 5:
                continue
            peak = fwd.cummax()
            dd   = ((fwd - peak) / peak).min()
            results[dt] = dd
        return pd.Series(results, name=f"fwd_dd_{horizon_days}d")

    def _forward_cs_vol(self, horizon_days: int = 21) -> pd.Series:
        """Forward realized cross-sectional vol (avg constituent vol)."""
        results = {}
        for dt in self.weekly_mfs.index:
            fwd_ret = self.returns.loc[dt:].iloc[:horizon_days]
            if len(fwd_ret) < 5:
                continue
            # cross-sectional: std of constituent returns each day, then mean
            cs_vol = fwd_ret.std(axis=1).mean()
            results[dt] = cs_vol
        return pd.Series(results, name=f"fwd_cs_vol_{horizon_days}d")

    def _momentum_factor(self) -> pd.Series:
        """
        Weekly momentum factor return:
            Formation: past 12M returns, skip last 1M
            L/S: top-decile minus bottom-decile equal-weight
            Target: forward 1M return of this L/S portfolio
        """
        weekly_px  = self.prices.resample("W-FRI").last()
        weekly_ret = weekly_px.pct_change()

        results = {}
        for i, dt in enumerate(self.weekly_mfs.index):
            # Formation window: 52W back, skip last 4W
            form_end   = i
            form_start = i - 52
            skip_start = i - 4
            if form_start < 0:
                continue

            # Constituent returns over formation window (excl. skip)
            form_ret = weekly_px.iloc[form_start:skip_start]
            if len(form_ret) < 40:
                continue
            cum_ret = (1 + weekly_ret.iloc[form_start:skip_start]).prod() - 1

            # Decile sort
            n_decile = max(1, len(cum_ret) // 10)
            winners  = cum_ret.nlargest(n_decile).index
            losers   = cum_ret.nsmallest(n_decile).index

            # Forward 1M (4W) return
            fwd_window = weekly_ret.iloc[form_end: form_end + 4]
            if len(fwd_window) < 2:
                continue

            long_ret  = fwd_window[winners].mean(axis=1).sum()
            short_ret = fwd_window[losers].mean(axis=1).sum()
            ls_ret    = long_ret - short_ret
            results[dt] = ls_ret

        return pd.Series(results, name="fwd_momentum_ls")

    def build(self) -> "TargetBuilder":
        log.info(f"[{self.market}] Building prediction targets …")
        t1a = self._forward_drawdown(21)
        t1b = self._forward_drawdown(63)
        t2  = self._forward_cs_vol(21)
        t3  = self._momentum_factor()

        self.targets = pd.concat([t1a, t1b, t2, t3], axis=1)
        self.targets = self.weekly_mfs[["MFS_equal","MFS_robust"]].join(
            self.targets, how="inner"
        ).dropna()
        log.info(f"[{self.market}] Target df: {self.targets.shape}")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: DRAWDOWN TEST — Quintile Sort
# ═══════════════════════════════════════════════════════════════════════════════
class DrawdownTest:
    def __init__(self, targets: pd.DataFrame, market: str):
        self.df     = targets.copy()
        self.market = market
        self.results: dict = {}

    def run(self) -> "DrawdownTest":
        log.info(f"[{self.market}] Running drawdown quintile test …")
        self.df["MFS_quintile"] = pd.qcut(
            self.df["MFS_equal"], q=5, labels=[1, 2, 3, 4, 5]
        )
        for col in ["fwd_dd_21d", "fwd_dd_63d"]:
            stats_df = self.df.groupby("MFS_quintile")[col].agg(
                ["mean","median","std","count"]
            )
            self.results[col] = stats_df

            # Kruskal-Wallis test for significance
            groups = [grp[col].values for _, grp in self.df.groupby("MFS_quintile")]
            kw_stat, kw_p = stats.kruskal(*groups)
            log.info(f"  [{self.market}] {col} | KW stat={kw_stat:.2f}, p={kw_p:.4f}")
            self.results[f"{col}_kw"] = (kw_stat, kw_p)

        return self

    def plot_single(self, ax, col: str, label: str):
        """Quintile bar chart for one drawdown horizon."""
        means = self.results[col]["mean"] * 100
        ax.bar(means.index.astype(str), means.values,
               color=["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"],
               edgecolor="white", linewidth=0.5)
        kw_stat, kw_p = self.results[f"{col}_kw"]
        ax.set_title(f"[{self.market}] MFS Quintile vs {label}\n"
                     f"KW p={kw_p:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("MFS Quintile (1=Safe → 5=Fragile)")
        ax.set_ylabel("Avg Forward Max Drawdown (%)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_facecolor("#f8f9fa")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: VOL REGRESSION TEST
# ═══════════════════════════════════════════════════════════════════════════════
class VolRegressionTest:
    def __init__(self, targets: pd.DataFrame, market: str):
        self.df     = targets.copy()
        self.market = market
        self.ols_result  = None
        self.granger_res = None

    def run(self) -> "VolRegressionTest":
        log.info(f"[{self.market}] Running vol regression + Granger test …")
        sub = self.df[["MFS_equal","fwd_cs_vol_21d"]].dropna()

        # OLS: forward vol ~ MFS
        X = add_constant(sub["MFS_equal"])
        y = sub["fwd_cs_vol_21d"]
        self.ols_result = OLS(y, X).fit()
        log.info(f"  [{self.market}] OLS R²={self.ols_result.rsquared:.4f}, "
                 f"β={self.ols_result.params['MFS_equal']:.4f}, "
                 f"p={self.ols_result.pvalues['MFS_equal']:.4f}")

        # Granger causality: does MFS Granger-cause fwd vol?
        gc_df = sub[["fwd_cs_vol_21d","MFS_equal"]].dropna()
        self.granger_res = grangercausalitytests(gc_df, maxlag=GRANGER_LAG, verbose=False)
        return self

    def plot(self, ax):
        sub = self.df[["MFS_equal","fwd_cs_vol_21d"]].dropna()
        ax.scatter(sub["MFS_equal"], sub["fwd_cs_vol_21d"] * 100,
                   alpha=0.3, s=15, color="#3498db", edgecolors="none")
        # OLS line
        x_line = np.linspace(sub["MFS_equal"].min(), sub["MFS_equal"].max(), 100)
        y_line  = (self.ols_result.params["const"] +
                   self.ols_result.params["MFS_equal"] * x_line) * 100
        ax.plot(x_line, y_line, color="#e74c3c", linewidth=2)
        r2  = self.ols_result.rsquared
        p   = self.ols_result.pvalues["MFS_equal"]
        ax.set_title(f"[{self.market}] MFS vs Fwd CS Vol\nR²={r2:.3f}, p={p:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("MFS (Equal Weight)")
        ax.set_ylabel("Fwd 1M Cross-Sectional Vol (%)")
        ax.set_facecolor("#f8f9fa")

    def granger_summary(self) -> pd.DataFrame:
        rows = []
        for lag, res in self.granger_res.items():
            f_stat = res[0]["ssr_ftest"][0]
            p_val  = res[0]["ssr_ftest"][1]
            rows.append({"lag": lag, "F_stat": round(f_stat, 3), "p_value": round(p_val, 4)})
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: MOMENTUM CRASH EVENT STUDY
# ═══════════════════════════════════════════════════════════════════════════════
class MomentumCrashTest:
    def __init__(self, targets: pd.DataFrame, market: str):
        self.df     = targets.copy()
        self.market = market
        self.event_returns: pd.DataFrame = pd.DataFrame()
        self.non_event_returns: pd.DataFrame = pd.DataFrame()

    def run(self) -> "MomentumCrashTest":
        log.info(f"[{self.market}] Running momentum crash event study …")
        sub = self.df[["MFS_equal","fwd_momentum_ls"]].dropna()
        threshold = sub["MFS_equal"].quantile(0.80)

        high_fragility = sub[sub["MFS_equal"] >= threshold]
        low_fragility  = sub[sub["MFS_equal"] <  threshold]

        self.event_returns     = high_fragility["fwd_momentum_ls"]
        self.non_event_returns = low_fragility["fwd_momentum_ls"]

        t_stat, p_val = stats.ttest_ind(
            self.event_returns, self.non_event_returns, equal_var=False
        )
        log.info(f"  [{self.market}] High-fragility mean mom={self.event_returns.mean():.4f} | "
                 f"Low-fragility mean mom={self.non_event_returns.mean():.4f} | "
                 f"t={t_stat:.2f}, p={p_val:.4f}")
        self.t_stat = t_stat
        self.p_val  = p_val
        return self

    def plot(self, ax):
        data = [self.event_returns.values * 100,
                self.non_event_returns.values * 100]
        bp = ax.boxplot(data, patch_artist=True,
                        labels=["High Fragility\n(MFS > 80th pct)",
                                "Low Fragility\n(MFS < 80th pct)"],
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][1].set_facecolor("#2ecc71")
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"[{self.market}] Momentum L/S Return by Fragility\n"
                     f"t={self.t_stat:.2f}, p={self.p_val:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Fwd 1M Momentum L/S Return (%)")
        ax.set_facecolor("#f8f9fa")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 7: PLOTTER — MFS TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
def plot_mfs_timeseries(weekly_mfs: pd.DataFrame, market: str, ax):
    ax.plot(weekly_mfs.index, weekly_mfs["MFS_equal"],
            color="#2c3e50", linewidth=1.2, label="MFS Equal-Weight")
    ax.plot(weekly_mfs.index, weekly_mfs["MFS_robust"],
            color="#e74c3c", linewidth=1.0, linestyle="--",
            alpha=0.7, label="MFS Robust (corr-weighted)")
    ax.fill_between(weekly_mfs.index, weekly_mfs["MFS_equal"],
                    alpha=0.1, color="#2c3e50")
    ax.axhline(0.80, color="#e74c3c", linestyle=":", linewidth=0.8, label="80th pct")
    ax.set_title(f"[{market}] Market Fragility Score — Weekly", fontweight="bold")
    ax.set_ylabel("MFS (0=Safe, 1=Fragile)")
    ax.legend(fontsize=8)
    ax.set_facecolor("#f8f9fa")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 8: REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════════
def write_summary_table(results: dict):
    rows = []
    for market, res in results.items():
        dd_test    = res["dd"]
        vol_test   = res["vol"]
        mom_test   = res["mom"]

        # Drawdown
        for col, label in [("fwd_dd_21d","1M"), ("fwd_dd_63d","3M")]:
            kw_stat, kw_p = dd_test.results[f"{col}_kw"]
            q1_dd = dd_test.results[col]["mean"].iloc[0] * 100
            q5_dd = dd_test.results[col]["mean"].iloc[4] * 100
            rows.append({
                "Market": market, "Test": f"Drawdown ({label})",
                "Q1 (Safe) Mean": f"{q1_dd:.2f}%",
                "Q5 (Fragile) Mean": f"{q5_dd:.2f}%",
                "Stat": f"KW={kw_stat:.2f}",
                "p-value": f"{kw_p:.4f}",
                "Significant": "✓" if kw_p < 0.05 else "✗"
            })

        # Vol regression
        r2  = vol_test.ols_result.rsquared
        p   = vol_test.ols_result.pvalues["MFS_equal"]
        beta= vol_test.ols_result.params["MFS_equal"]
        rows.append({
            "Market": market, "Test": "Vol Expansion (OLS)",
            "Q1 (Safe) Mean": "—", "Q5 (Fragile) Mean": "—",
            "Stat": f"β={beta:.4f}, R²={r2:.3f}",
            "p-value": f"{p:.4f}",
            "Significant": "✓" if p < 0.05 else "✗"
        })

        # Momentum
        rows.append({
            "Market": market, "Test": "Momentum Reversal",
            "Q1 (Safe) Mean": f"{mom_test.non_event_returns.mean()*100:.2f}%",
            "Q5 (Fragile) Mean": f"{mom_test.event_returns.mean()*100:.2f}%",
            "Stat": f"t={mom_test.t_stat:.2f}",
            "p-value": f"{mom_test.p_val:.4f}",
            "Significant": "✓" if mom_test.p_val < 0.05 else "✗"
        })

    summary = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "summary_table.csv"
    summary.to_csv(out_path, index=False)
    log.info(f"Summary table saved → {out_path}")
    return summary


def write_findings_skeleton(summary: pd.DataFrame):
    text = f"""# Concentration vs Market Fragility Index — Findings
**Author**: AlphaByProcess  
**Period**: {START} – {END}  
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

{summary.to_string(index=False)}

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
"""
    path = Path("research/analysis_final/fragility_index/findings.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info(f"findings.md skeleton saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 9: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def run_market(tickers: list, market: str) -> dict:
    """Full pipeline for one market."""
    # Load
    loader = DataLoader(tickers, market).fetch()

    # Build MFS
    fb = FragilityBuilder(loader.prices, loader.mcap, market).build()

    # Build targets
    tb = TargetBuilder(loader.prices, loader.index_px, fb.weekly_mfs, market).build()

    # Tests
    dd_test  = DrawdownTest(tb.targets, market).run()
    vol_test = VolRegressionTest(tb.targets, market).run()
    mom_test = MomentumCrashTest(tb.targets, market).run()

    return {
        "loader"  : loader,
        "fb"      : fb,
        "tb"      : tb,
        "dd"      : dd_test,
        "vol"     : vol_test,
        "mom"     : mom_test
    }


def main():
    log.info("=" * 60)
    log.info("  FRAGILITY INDEX RESEARCH — START")
    log.info("=" * 60)

    markets = {
        "US": SP500_TICKERS,
        "IN": NIFTY_TICKERS
    }

    results = {}
    for market, tickers in markets.items():
        log.info(f"\n{'─'*40}\nProcessing {market}\n{'─'*40}")
        results[market] = run_market(tickers, market)

    # ── Master plot ──────────────────────────────────────────────────────────
    log.info("Generating master chart …")
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Market Fragility Index — Research Dashboard\nAlphaByProcess",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.45, wspace=0.35)

    # Row 0: MFS time series
    ax_ts_us = fig.add_subplot(gs[0, 0])
    ax_ts_in = fig.add_subplot(gs[0, 1])
    plot_mfs_timeseries(results["US"]["fb"].weekly_mfs, "US", ax_ts_us)
    plot_mfs_timeseries(results["IN"]["fb"].weekly_mfs, "IN", ax_ts_in)

    # Row 1: Drawdown 1M — US (left) | IN (right)
    ax_dd1_us = fig.add_subplot(gs[1, 0])
    ax_dd1_in = fig.add_subplot(gs[1, 1])
    results["US"]["dd"].plot_single(ax_dd1_us, "fwd_dd_21d", "1M Forward Drawdown")
    results["IN"]["dd"].plot_single(ax_dd1_in, "fwd_dd_21d", "1M Forward Drawdown")

    # Row 2: Drawdown 3M — US (left) | IN (right)
    ax_dd3_us = fig.add_subplot(gs[2, 0])
    ax_dd3_in = fig.add_subplot(gs[2, 1])
    results["US"]["dd"].plot_single(ax_dd3_us, "fwd_dd_63d", "3M Forward Drawdown")
    results["IN"]["dd"].plot_single(ax_dd3_in, "fwd_dd_63d", "3M Forward Drawdown")

    # Row 3: Vol regression scatter
    ax_vol_us = fig.add_subplot(gs[3, 0])
    ax_vol_in = fig.add_subplot(gs[3, 1])
    results["US"]["vol"].plot(ax_vol_us)
    results["IN"]["vol"].plot(ax_vol_in)

    # Row 4: Momentum crash boxplot
    ax_mom_us = fig.add_subplot(gs[4, 0])
    ax_mom_in = fig.add_subplot(gs[4, 1])
    results["US"]["mom"].plot(ax_mom_us)
    results["IN"]["mom"].plot(ax_mom_in)

    plt.savefig(OUTPUT_DIR / "master_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    log.info(f"Dashboard saved → {OUTPUT_DIR / 'master_dashboard.png'}")
    plt.close()

    # ── Granger tables ──────────────────────────────────────────────────────
    for market in ["US", "IN"]:
        gc_df = results[market]["vol"].granger_summary()
        gc_df.to_csv(OUTPUT_DIR / f"granger_{market}.csv", index=False)
        log.info(f"[{market}] Granger table:\n{gc_df.to_string()}")

    # ── Summary table + findings.md ─────────────────────────────────────────
    summary = write_summary_table(results)
    write_findings_skeleton(summary)

    log.info("\n" + "=" * 60)
    log.info("  ALL DONE — check research/analysis_final/fragility_index/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
    