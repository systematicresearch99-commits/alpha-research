"""
research/regime_detection/hmm_model.py
========================================
Hidden Markov Model wrapper for market regime detection.

Detects 4 regimes:
  Bull       — low vol, positive autocorr, high index corr, mild skew
  Bear       — elevated vol, negative autocorr, high index corr, negative skew
  Crisis     — very high vol, negative autocorr, corr breakdown, extreme neg skew
  Transition — mixed/intermediate signals between regime shifts

Auto-labels raw HMM states by volatility ranking post-fit so you
always get named regimes regardless of random initialization order.

Persistence: save/load via pickle so you don't retrain every run.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("Run: pip install hmmlearn")

from research.regime_detection.features import compute_features, normalize_features


# ── Regime metadata ──────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Bull":       "#2ecc71",
    "Bear":       "#e74c3c",
    "Crisis":     "#8e44ad",
    "Transition": "#f39c12",
}

REGIME_ORDER = ["Bull", "Transition", "Bear", "Crisis"]  # low → high vol


# ── Model ────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Trains a GaussianHMM on 4 market features and maps raw states
    to named regimes (Bull / Bear / Crisis / Transition).

    Compatible with the AlphaByProcess framework:
      - Takes pd.Series of close prices (same format as load_data output)
      - Returns pd.Series of regime labels aligned to price dates
    """

    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 1000,
        covariance_type: str = "full",
        random_state: int = 42,
        window: int = 20,
    ):
        self.n_regimes       = n_regimes
        self.window          = window
        self._feature_mean   = None
        self._feature_std    = None
        self._regime_map     = {}   # raw HMM state int → regime name str
        self._is_fitted      = False

        self.model = GaussianHMM(
            n_components    = n_regimes,
            covariance_type = covariance_type,
            n_iter          = n_iter,
            random_state    = random_state,
            verbose         = False,
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, prices: pd.Series, index_prices: pd.Series) -> "RegimeDetector":
        """
        Compute features → normalize → fit HMM → label regimes.

        Parameters
        ----------
        prices        : close prices of the asset being analyzed
        index_prices  : benchmark index close prices (e.g. SPY)
        """
        print("  [1/3] Computing features...")
        raw_features = compute_features(prices, index_prices, window=self.window)

        print("  [2/3] Fitting HMM...")
        norm_features, self._feature_mean, self._feature_std = normalize_features(raw_features)
        self.model.fit(norm_features.values)
        print(f"        Converged: {self.model.monitor_.converged}  "
              f"| Log-likelihood: {self.model.monitor_.history[-1]:.1f}")

        print("  [3/3] Labeling regimes by volatility rank...")
        self._label_regimes()
        self._is_fitted = True
        print(f"        Map: {self._regime_map}")
        return self

    def _label_regimes(self):
        """
        Auto-label states by ranking on volatility feature (index 0).
        Ascending vol order → Bull, Transition, Bear, Crisis.
        Works regardless of random init order in GaussianHMM.
        """
        vol_means = self.model.means_[:, 0]   # feature 0 = volatility
        ranked    = np.argsort(vol_means)      # indices sorted low→high vol

        labels = REGIME_ORDER[:self.n_regimes]
        self._regime_map = {int(ranked[i]): labels[i] for i in range(self.n_regimes)}

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, prices: pd.Series, index_prices: pd.Series) -> pd.Series:
        """
        Returns pd.Series of named regime labels aligned to feature dates.

        Parameters
        ----------
        prices        : close prices (can be same training data or new)
        index_prices  : benchmark close prices

        Returns
        -------
        pd.Series  index=dates, values='Bull'|'Bear'|'Crisis'|'Transition'
        """
        self._check_fitted()
        raw_features  = compute_features(prices, index_prices, window=self.window)
        norm_features, _, _ = normalize_features(
            raw_features, mean=self._feature_mean, std=self._feature_std
        )
        raw_states = self.model.predict(norm_features.values)
        return pd.Series(
            [self._regime_map[s] for s in raw_states],
            index=norm_features.index,
            name="regime",
        )

    def predict_proba(self, prices: pd.Series, index_prices: pd.Series) -> pd.DataFrame:
        """
        Posterior probability of each regime at each timestep.
        Useful for soft regime membership and transition analysis.

        Returns
        -------
        pd.DataFrame  shape (T, n_regimes), columns = regime names
        """
        self._check_fitted()
        raw_features  = compute_features(prices, index_prices, window=self.window)
        norm_features, _, _ = normalize_features(
            raw_features, mean=self._feature_mean, std=self._feature_std
        )
        proba = self.model.predict_proba(norm_features.values)
        cols  = [self._regime_map[i] for i in range(self.n_regimes)]
        return pd.DataFrame(proba, index=norm_features.index, columns=cols)

    # ── Regime statistics ─────────────────────────────────────────────────────

    def regime_summary(self, regimes: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """
        Summary statistics per regime:
          - count of days
          - % of total time
          - avg duration (consecutive days)
          - avg return per day in regime
          - avg volatility

        Parameters
        ----------
        regimes : pd.Series from predict()
        prices  : close prices (same asset)

        Returns
        -------
        pd.DataFrame  one row per regime
        """
        returns = prices.pct_change().reindex(regimes.index)
        df = pd.DataFrame({"regime": regimes, "return": returns})

        rows = []
        for regime in REGIME_ORDER[:self.n_regimes]:
            mask    = df["regime"] == regime
            r_slice = df.loc[mask, "return"]

            # Average consecutive run length
            runs, current = [], 0
            for val in mask:
                if val:
                    current += 1
                else:
                    if current > 0:
                        runs.append(current)
                        current = 0
            if current > 0:
                runs.append(current)

            rows.append({
                "Regime":       regime,
                "Days":         int(mask.sum()),
                "% Time":       round(mask.mean() * 100, 1),
                "Avg Duration": round(np.mean(runs), 1) if runs else 0,
                "Avg Daily Ret":round(r_slice.mean() * 100, 4),
                "Avg Vol (ann)":round(r_slice.std() * np.sqrt(252) * 100, 2),
            })

        return pd.DataFrame(rows).set_index("Regime")

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Return the HMM transition matrix with named regime labels.
        Entry [i,j] = probability of transitioning from regime i to regime j.
        """
        self._check_fitted()
        labels = [self._regime_map[i] for i in range(self.n_regimes)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        ).round(4)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Pickle the fitted detector to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[RegimeDetector] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        """Load a previously fitted detector from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[RegimeDetector] Loaded from {path}")
        return obj

    # ── Internal ─────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("RegimeDetector must be fitted before calling predict(). "
                               "Call .fit(prices, index_prices) first.")

                               