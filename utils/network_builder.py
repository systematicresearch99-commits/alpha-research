"""
network_builder.py
------------------
Lead-Lag Network Builder
AlphaByProcess | ALPHA-RESEARCH

Builds a directed weighted graph of lead-lag relationships
between assets from a returns DataFrame.

For every pair (i, j), computes cross-correlation at lags
1, 2, 3, 5 days. The strongest lag becomes the directed edge
i → j with weight = peak correlation and lag = days.

Supports both rolling and expanding estimation windows.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ──────────────────────────────────────────────
# Core: pairwise lead-lag correlation
# ──────────────────────────────────────────────

def pairwise_lead_lag(returns: pd.DataFrame,
                      lags: list[int] = [1, 2, 3, 5]
                      ) -> pd.DataFrame:
    """
    Compute lead-lag correlations for all directed pairs (i → j).

    For each ordered pair (leader i, follower j), finds the lag at
    which r_i(t) best predicts r_j(t + lag).

    Parameters
    ----------
    returns : DataFrame of daily returns, shape (T, N)
    lags    : list of lag values in days to test

    Returns
    -------
    DataFrame with columns:
        leader, follower, best_lag, best_corr, corr_lag_1, corr_lag_2, ...
    One row per directed pair (i != j).
    """
    tickers = returns.columns.tolist()
    records = []

    for leader in tickers:
        for follower in tickers:
            if leader == follower:
                continue

            lag_corrs = {}
            for lag in lags:
                # r_leader(t) vs r_follower(t + lag)
                x = returns[leader].iloc[:-lag].values
                y = returns[follower].iloc[lag:].values
                if len(x) < 20:          # not enough data
                    lag_corrs[lag] = np.nan
                    continue
                corr = np.corrcoef(x, y)[0, 1]
                lag_corrs[lag] = corr

            valid = {k: v for k, v in lag_corrs.items() if not np.isnan(v)}
            if not valid:
                continue

            best_lag  = max(valid, key=lambda k: abs(valid[k]))
            best_corr = valid[best_lag]

            row = {
                "leader":    leader,
                "follower":  follower,
                "best_lag":  best_lag,
                "best_corr": best_corr,
            }
            for lag in lags:
                row[f"corr_lag_{lag}"] = lag_corrs.get(lag, np.nan)

            records.append(row)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Network scoring
# ──────────────────────────────────────────────

def score_assets(edge_df: pd.DataFrame,
                 min_corr: float = 0.1) -> pd.DataFrame:
    """
    Score each asset as leader or follower based on edge weights.

    leader_score  = sum of |best_corr| on outgoing edges
    follower_score = sum of |best_corr| on incoming edges
    net_score     = follower_score - leader_score
                    (positive = net follower, negative = net leader)

    Parameters
    ----------
    edge_df  : output of pairwise_lead_lag()
    min_corr : minimum |correlation| to include an edge (noise filter)

    Returns
    -------
    DataFrame indexed by ticker with leader_score, follower_score, net_score
    """
    filtered = edge_df[edge_df["best_corr"].abs() >= min_corr].copy()

    tickers = pd.unique(edge_df[["leader", "follower"]].values.ravel())
    scores  = pd.DataFrame(index=tickers,
                           columns=["leader_score", "follower_score", "net_score"],
                           dtype=float).fillna(0.0)

    for _, row in filtered.iterrows():
        scores.loc[row["leader"],   "leader_score"]   += abs(row["best_corr"])
        scores.loc[row["follower"], "follower_score"] += abs(row["best_corr"])

    scores["net_score"] = scores["follower_score"] - scores["leader_score"]
    return scores.sort_values("net_score", ascending=False)


# ──────────────────────────────────────────────
# Signal: predict follower next-bar direction
# ──────────────────────────────────────────────

def predict_follower_returns(returns_window: pd.DataFrame,
                              edge_df: pd.DataFrame,
                              min_corr: float = 0.1) -> pd.Series:
    """
    For each follower asset, predict the sign of its next-bar return
    using the weighted sum of its leaders' most recent returns.

    Prediction:
        pred_j = Σ_i  best_corr(i→j) * r_i(t - best_lag + 1 ... t)

    The sign of pred_j determines the position: +1 long, -1 short, 0 flat.

    Parameters
    ----------
    returns_window : DataFrame of returns ending at bar t (the lookback window)
    edge_df        : output of pairwise_lead_lag() estimated on the same window
    min_corr       : minimum |correlation| threshold

    Returns
    -------
    Series of predicted directions, indexed by ticker.
    Values: +1, -1, or 0 (if no strong leader found).
    """
    filtered = edge_df[edge_df["best_corr"].abs() >= min_corr].copy()
    tickers  = returns_window.columns.tolist()
    preds    = pd.Series(0.0, index=tickers)

    for follower in tickers:
        incoming = filtered[filtered["follower"] == follower]
        if incoming.empty:
            continue

        weighted_sum = 0.0
        total_weight = 0.0

        for _, edge in incoming.iterrows():
            leader   = edge["leader"]
            lag      = int(edge["best_lag"])
            corr     = edge["best_corr"]

            if leader not in returns_window.columns:
                continue
            if len(returns_window) < lag:
                continue

            # Use the leader's return at the appropriate lag
            leader_return = returns_window[leader].iloc[-lag]
            weighted_sum += corr * leader_return
            total_weight += abs(corr)

        if total_weight > 0:
            preds[follower] = weighted_sum / total_weight

    return preds


# ──────────────────────────────────────────────
# Rolling / Expanding network estimator
# ──────────────────────────────────────────────

class NetworkEstimator:
    """
    Wraps pairwise_lead_lag() with rolling or expanding window logic.

    Usage
    -----
    estimator = NetworkEstimator(window_type="rolling", window_size=60)
    edges_at_t = estimator.estimate(returns_up_to_t)
    """

    def __init__(self,
                 window_type: str  = "rolling",   # "rolling" | "expanding"
                 window_size: int  = 60,           # trading days (rolling only)
                 lags: list[int]   = [1, 2, 3, 5],
                 min_corr: float   = 0.1):
        if window_type not in ("rolling", "expanding"):
            raise ValueError("window_type must be 'rolling' or 'expanding'")
        self.window_type = window_type
        self.window_size = window_size
        self.lags        = lags
        self.min_corr    = min_corr

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate the network on the provided returns slice.
        Caller is responsible for passing the correct window.
        """
        if self.window_type == "rolling":
            window = returns.iloc[-self.window_size:]
        else:
            window = returns

        if len(window) < max(self.lags) + 20:
            return pd.DataFrame()   # not enough data yet

        return pairwise_lead_lag(window, lags=self.lags)
    
    