"""
lead_lag_network.py
-------------------
Lead-Lag Network Strategy
AlphaByProcess | ALPHA-RESEARCH

Core Idea
---------
Assets don't move simultaneously — there are leaders and followers.
For every directed pair (i → j), the strategy estimates at which lag
r_i(t) best predicts r_j(t + lag). Followers are traded based on
the weighted signal from their leaders.

Model
-----
  r_j(t+1) = f( r_i(t) )   for all leaders i of follower j

  pred_j = Σ_i  corr(i→j) * r_i(t - lag)   (weighted leader signal)
  position_j = sign(pred_j)  if |pred_j| > threshold else 0

Network
-------
  - Directed edges: i → j if r_i leads r_j at some lag k
  - Edge weight: peak cross-correlation across tested lags
  - Re-estimated on a rolling or expanding window each bar

Universe
--------
  Default: SPY, TLT, GLD, USO, XLK, XLE, XLF
  Configurable via tickers parameter.
"""

import numpy as np
import pandas as pd
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_builder import NetworkEstimator, predict_follower_returns

# ── Framework contract ────────────────────────────────────────────────────────
STRATEGY_NAME = "lead_lag_network"

DEFAULT_TICKERS = ["SPY", "TLT", "GLD", "USO", "XLK", "XLE", "XLF"]


def get_params(window_type:   str   = "rolling",
               window_size:   int   = 60,
               lags:          list  = None,
               min_corr:      float = 0.10,
               signal_threshold: float = 0.0,
               **kwargs) -> dict:
    """Return strategy parameters as a plain dict for store.save_run()."""
    return {
        "window_type":      window_type,
        "window_size":      window_size,
        "lags":             lags or [1, 2, 3, 5],
        "min_corr":         min_corr,
        "signal_threshold": signal_threshold,
    }
# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────
# Multi-ticker data loader
# ──────────────────────────────────────────────

def load_multi(tickers: list[str],
               start:   str,
               end:     Optional[str] = None) -> pd.DataFrame:
    """
    Load Close prices for multiple tickers and return a
    wide DataFrame (dates × tickers), aligned on common dates.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    # yfinance returns MultiIndex when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill().dropna()
    return prices


# ──────────────────────────────────────────────
# Core signal generation
# ──────────────────────────────────────────────

def generate_signals(data: pd.DataFrame,
                     window_type:      str   = "rolling",
                     window_size:      int   = 60,
                     lags:             list  = None,
                     min_corr:         float = 0.10,
                     signal_threshold: float = 0.0,
                     **kwargs) -> pd.DataFrame:
    """
    Framework entry point.

    Unlike single-asset strategies, data here is a wide DataFrame
    of Close prices (dates × tickers), as returned by load_multi().

    Returns a DataFrame in engine.py format with columns:
        Close      — proxy series (equal-weight portfolio of all assets)
        position   — aggregate net position (sum across assets, clipped to [-1, 1])
        plus per-asset columns: position_SPY, position_TLT, etc.

    The per-asset positions are the real output. The aggregate
    position + Close columns satisfy engine.py's interface so the
    standard backtest pipeline works unchanged.

    Parameters
    ----------
    data             : wide Close price DataFrame (dates × tickers)
    window_type      : "rolling" | "expanding"
    window_size      : lookback in trading days (rolling only)
    lags             : list of lags to test e.g. [1, 2, 3, 5]
    min_corr         : minimum |correlation| to use an edge
    signal_threshold : minimum |pred| to open a position (default 0 = any signal)
    """
    if lags is None:
        lags = [1, 2, 3, 5]

    returns  = data.pct_change().dropna()
    tickers  = returns.columns.tolist()
    n        = len(returns)

    estimator = NetworkEstimator(
        window_type = window_type,
        window_size = window_size,
        lags        = lags,
        min_corr    = min_corr,
    )

    # Pre-allocate position matrix
    positions = pd.DataFrame(0.0, index=returns.index, columns=tickers)

    min_bars = window_size if window_type == "rolling" else max(lags) + 20

    for t in range(min_bars, n):
        returns_so_far = returns.iloc[:t]          # no lookahead
        edge_df        = estimator.estimate(returns_so_far)

        if edge_df.empty:
            continue

        preds = predict_follower_returns(returns_so_far, edge_df, min_corr)

        # Equal weight: position = sign(pred) if |pred| > threshold
        for ticker in tickers:
            pred = preds.get(ticker, 0.0)
            if abs(pred) > signal_threshold:
                positions.loc[returns.index[t], ticker] = np.sign(pred)

    # Shift by 1 bar — act on today's signal tomorrow (no lookahead)
    positions_shifted = positions.shift(1).fillna(0.0)

    # Build output DataFrame
    # Close = equal-weight portfolio (for engine.py pct_change)
    eq_weight_close = data.mean(axis=1)

    out = pd.DataFrame(index=data.index)
    out["Close"]    = eq_weight_close

    # Aggregate position = mean across assets (keeps scale ≈ [-1, +1])
    out["position"] = positions_shifted.mean(axis=1)

    # Per-asset positions for detailed analysis
    for ticker in tickers:
        out[f"position_{ticker}"] = positions_shifted[ticker]

    # Carry forward last known position into price rows before signal starts
    out["position"] = out["position"].fillna(0.0)

    return out


# ──────────────────────────────────────────────
# Per-asset backtest (detailed breakdown)
# ──────────────────────────────────────────────

def backtest_per_asset(data: pd.DataFrame,
                       signals_df: pd.DataFrame,
                       transaction_cost: float = 0.001) -> pd.DataFrame:
    """
    Run a separate P&L calculation for each asset and return
    a summary DataFrame (one row per asset).

    Useful for understanding which leader-follower pairs are
    driving returns.

    Parameters
    ----------
    data             : wide Close price DataFrame (dates × tickers)
    signals_df       : output of generate_signals()
    transaction_cost : cost per trade as a fraction

    Returns
    -------
    DataFrame with per-asset: total_return, sharpe, num_trades
    """
    returns  = data.pct_change().fillna(0)
    tickers  = data.columns.tolist()
    records  = []

    for ticker in tickers:
        col = f"position_{ticker}"
        if col not in signals_df.columns:
            continue

        pos     = signals_df[col]
        ret     = returns[ticker]
        trades  = pos.diff().abs()

        strat_ret = pos * ret - trades * transaction_cost
        equity    = (1 + strat_ret).cumprod()

        total_ret = equity.iloc[-1] - 1
        ann_vol   = strat_ret.std() * np.sqrt(252)
        sharpe    = (strat_ret.mean() * 252) / (ann_vol + 1e-10)
        n_trades  = int((trades > 0).sum())

        records.append({
            "ticker":        ticker,
            "total_return":  round(total_ret, 4),
            "sharpe":        round(sharpe, 4),
            "ann_vol":       round(ann_vol, 4),
            "num_trades":    n_trades,
        })

    return pd.DataFrame(records).sort_values("sharpe", ascending=False)


# ──────────────────────────────────────────────
# Smoke test  (python lead_lag_network.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backtests.engine   import run_backtest
    from utils.performance  import calculate_metrics, print_summary

    TICKERS = ["SPY", "TLT", "GLD", "USO", "XLK", "XLE", "XLF"]
    START   = "2018-01-01"

    print(f"Loading {TICKERS} from {START}…")
    data = load_multi(TICKERS, start=START)
    print(f"{len(data)} rows loaded  ({data.index[0].date()} → {data.index[-1].date()})")

    print("\nGenerating signals (rolling, window=60)…")
    df_rolling = generate_signals(data, window_type="rolling", window_size=60)

    print("Running backtest (rolling)…")
    df_rolling = run_backtest(df_rolling)
    metrics_rolling = calculate_metrics(df_rolling)
    print_summary(metrics_rolling, strategy_name="Lead-Lag Network  [rolling-60]")

    print("\nGenerating signals (expanding)…")
    df_expanding = generate_signals(data, window_type="expanding")

    print("Running backtest (expanding)…")
    df_expanding = run_backtest(df_expanding)
    metrics_expanding = calculate_metrics(df_expanding)
    print_summary(metrics_expanding, strategy_name="Lead-Lag Network  [expanding]")

    print("\n── Per-asset breakdown (rolling) ────────")
    breakdown = backtest_per_asset(data, df_rolling)
    print(breakdown.to_string(index=False))


    