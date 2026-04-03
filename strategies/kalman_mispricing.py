"""
kalman_mispricing.py
--------------------
State-Space Mispricing Model
AlphaByProcess | ALPHA-RESEARCH

Model
-----
Observation:  y_t = x_t + ε_t     (ε ~ N(0, R))
State:        x_t = x_{t-1} + η_t  (η ~ N(0, Q))

x_t  = latent "fair value"  (random walk prior)
y_t  = observed price
ε_t  = observation noise
η_t  = state/process noise

Signal: z_t = (y_t - x̂_t) / √P_t   (normalised mispricing)

Trade:
  z_t < -entry_z  → BUY  (price below fair value)
  z_t >  entry_z  → SELL (price above fair value)
  |z_t| < exit_z  → EXIT (reversion complete)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ── Framework contract ────────────────────────────────────────────────────────
STRATEGY_NAME = "kalman_mispricing"


def get_params(obs_noise_var=1.0, proc_noise_var=0.01,
               entry_z=1.5, exit_z=0.3,
               stop_loss_z=3.5, **kwargs) -> dict:
    """Return strategy parameters as a plain dict for store.save_run()."""
    return {
        "obs_noise_var":  obs_noise_var,
        "proc_noise_var": proc_noise_var,
        "entry_z":        entry_z,
        "exit_z":         exit_z,
        "stop_loss_z":    stop_loss_z,
    }
# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────

@dataclass
class KalmanMispricingParams:
    # Kalman noise ratios
    obs_noise_var: float  = 1.0     # R  – observation noise variance
    proc_noise_var: float = 0.01    # Q  – process (state) noise variance

    # Signal thresholds (in units of σ-mispricing)
    entry_z: float = 1.5            # |z| > entry_z → open position
    exit_z:  float = 0.3            # |z| < exit_z  → close position

    # Risk / position sizing
    max_position: int   = 1         # +1 long, -1 short, 0 flat
    stop_loss_z: float  = 3.5       # hard stop if mispricing blows out

    # Kalman initialisation
    init_state: Optional[float] = None  # x̂_0  (None → use first price)
    init_cov: float = 1.0               # P_0


# ──────────────────────────────────────────────
# Core Kalman Filter
# ──────────────────────────────────────────────

def kalman_filter(prices: np.ndarray,
                  Q: float,
                  R: float,
                  x0: float,
                  P0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    One-dimensional Kalman filter over a price series.

    Returns
    -------
    fair_value : np.ndarray  – filtered state estimates x̂_t
    cov        : np.ndarray  – posterior covariance P_t
    """
    n = len(prices)
    fair_value = np.empty(n)
    cov        = np.empty(n)

    x_hat = x0
    P     = P0

    for t, y in enumerate(prices):
        # Predict
        x_pred = x_hat
        P_pred = P + Q

        # Update
        K     = P_pred / (P_pred + R)
        x_hat = x_pred + K * (y - x_pred)
        P     = (1 - K) * P_pred

        fair_value[t] = x_hat
        cov[t]        = P

    return fair_value, cov


# ──────────────────────────────────────────────
# Internal signal core (Series → DataFrame)
# ──────────────────────────────────────────────

def _signals_from_series(prices: pd.Series,
                          params: KalmanMispricingParams) -> pd.DataFrame:
    """
    Run Kalman filter on a price Series and return signal DataFrame.
    Position is shifted by 1 bar to prevent lookahead.

    Columns returned:
        Close, fair_value, cov, mispricing, z_score, signal, position
    """
    price_arr = prices.values
    x0 = params.init_state if params.init_state is not None else price_arr[0]

    fair_value, cov = kalman_filter(
        price_arr,
        Q  = params.proc_noise_var,
        R  = params.obs_noise_var,
        x0 = x0,
        P0 = params.init_cov,
    )

    mispricing = price_arr - fair_value
    z_score    = mispricing / np.sqrt(cov + 1e-10)

    # Raw signal: -1 / 0 / +1
    signal = np.zeros(len(prices))
    signal[z_score < -params.entry_z] = +1   # underpriced → long
    signal[z_score >  params.entry_z] = -1   # overpriced  → short

    # Position state machine: hold until exit threshold or stop
    raw_position = np.zeros(len(prices))
    pos = 0
    for t in range(len(prices)):
        if pos == 0:
            pos = signal[t]
        else:
            if abs(z_score[t]) < params.exit_z:
                pos = 0
            elif abs(z_score[t]) > params.stop_loss_z:
                pos = 0
            pos = np.clip(pos, -params.max_position, params.max_position)
        raw_position[t] = pos

    # Shift by 1 bar — engine.py uses position * same-bar return, so this
    # ensures we're acting on yesterday's signal, not today's.
    position_shifted = np.roll(raw_position, 1)
    position_shifted[0] = 0.0

    df = pd.DataFrame({
        "Close":      price_arr,      # capital-C: matches engine.py + performance.py
        "fair_value": fair_value,
        "cov":        cov,
        "mispricing": mispricing,
        "z_score":    z_score,
        "signal":     signal,
        "position":   position_shifted,
    }, index=prices.index)

    return df


# ──────────────────────────────────────────────
# Registry-compatible generate_signals
# ──────────────────────────────────────────────

def generate_signals(data: pd.DataFrame,
                     obs_noise_var:  float = 1.0,
                     proc_noise_var: float = 0.01,
                     entry_z:        float = 1.5,
                     exit_z:         float = 0.3,
                     stop_loss_z:    float = 3.5,
                     **kwargs) -> pd.DataFrame:
    """
    Framework entry point — mirrors the signature used by sma_crossover
    and rsi_mean_reversion so run.py can call it as:

        gen_signals(data, **strategy_kwargs)

    Args
    ----
    data : DataFrame with a 'Close' column (as returned by data_loader.load_data)
    All other args map directly to KalmanMispricingParams fields.

    Returns
    -------
    DataFrame ready for engine.run_backtest():
        Close, fair_value, cov, mispricing, z_score, signal, position
    """
    # Resolve Close column (handles both capitalisation conventions)
    if "Close" in data.columns:
        prices = data["Close"]
    elif "close" in data.columns:
        prices = data["close"]
    else:
        prices = data.iloc[:, 0]

    params = KalmanMispricingParams(
        obs_noise_var  = obs_noise_var,
        proc_noise_var = proc_noise_var,
        entry_z        = entry_z,
        exit_z         = exit_z,
        stop_loss_z    = stop_loss_z,
    )

    return _signals_from_series(prices, params)


# ──────────────────────────────────────────────
# Strategy Class  (engine.py object protocol)
# ──────────────────────────────────────────────

class KalmanMispricingStrategy:
    """
    Object-oriented wrapper — useful if you prefer the
    strategy.prepare() / strategy.get_position() protocol
    over the functional generate_signals() approach.
    """

    NAME = STRATEGY_NAME

    def __init__(self, params: Optional[KalmanMispricingParams] = None):
        self.params = params or KalmanMispricingParams()
        self._signals: Optional[pd.DataFrame] = None

    def prepare(self, data: pd.DataFrame) -> None:
        if "Close" in data.columns:
            prices = data["Close"]
        elif "close" in data.columns:
            prices = data["close"]
        else:
            prices = data.iloc[:, 0]
        self._signals = _signals_from_series(prices, self.params)

    def get_position(self, timestamp, data: pd.DataFrame) -> int:
        if self._signals is None:
            raise RuntimeError("Call prepare() before get_position().")
        if timestamp not in self._signals.index:
            return 0
        return int(self._signals.loc[timestamp, "position"])

    def get_signals(self) -> pd.DataFrame:
        if self._signals is None:
            raise RuntimeError("Call prepare() first.")
        return self._signals.copy()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    return float(dd.min())


# ──────────────────────────────────────────────
# Quick smoke-test  (python kalman_mispricing.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backtests.engine import run_backtest
    from utils.performance import calculate_metrics, print_summary

    print("Downloading SPY …")
    raw = yf.download("SPY", start="2018-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = generate_signals(raw,
                          obs_noise_var=2.0,
                          proc_noise_var=0.05,
                          entry_z=1.5,
                          exit_z=0.3,
                          stop_loss_z=3.5)

    df = run_backtest(df)
    metrics = calculate_metrics(df)
    print_summary(metrics, strategy_name="Kalman Mispricing  [SPY]")

    print("\n── Sample Signals (last 10 rows) ────────")
    print(df[["Close", "fair_value", "z_score", "position"]].tail(10).round(3))

    