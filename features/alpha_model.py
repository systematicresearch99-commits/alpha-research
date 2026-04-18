"""
features/alpha_model.py
-----------------------
Linear Alpha Model — combines continuous features into a single
predictive alpha score using OLS regression weights.

Pipeline position:
    feature_engine → alpha_model → position_bridge → backtest

How it works:
    1. fit(train_df)  — regresses each feature against next-period
                        returns on the training window to learn weights
    2. predict(df)    — applies learned weights to produce alpha score
    3. Score is normalized to [-1, 1] for clean downstream use

Key design principles:
    - Fit only on training data, predict on unseen data
    - No lookahead bias — target is always next-period return
    - Weights are interpretable — positive = bullish feature,
      negative = bearish feature
    - Falls back gracefully if a feature has no predictive power

Usage:
    from features.alpha_model import AlphaModel
    from features.feature_engine import extract_features

    df    = extract_features(data)
    model = AlphaModel()
    model.fit(df.iloc[:train_end])
    df    = model.predict(df)
    # df now has 'alpha_score' column in [-1, 1]
"""

import pandas as pd
import numpy as np
from features.feature_engine import extract_features, get_feature_cols

# ── OLS helper ─────────────────────────────────────────────────────────────────

def _ols_weights(X, y):
    """
    Ordinary Least Squares: w = (X'X)^-1 X'y
    Pure numpy — no sklearn dependency.
    Returns weight vector aligned with X columns.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Add intercept column
    ones  = np.ones((X_arr.shape[0], 1))
    X_arr = np.hstack([ones, X_arr])

    # Remove rows with any NaN
    mask  = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr))
    X_arr = X_arr[mask]
    y_arr = y_arr[mask]

    if len(X_arr) < 10:
        return None, None

    try:
        w = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
        intercept = w[0]
        weights   = w[1:]
    except np.linalg.LinAlgError:
        return None, None

    return weights, intercept


def _normalize_score(series, method="tanh", scale=2.0):
    """
    Normalize raw alpha score to [-1, 1].

    method="tanh"   : smooth compression, never hits ±1 exactly
    method="clip"   : hard clip at ±scale std
    method="zscore" : rolling zscore then tanh
    """
    s = series.copy()

    if method == "tanh":
        std = s.std()
        if std > 0:
            s = np.tanh(s / (scale * std))
    elif method == "clip":
        std = s.std()
        if std > 0:
            s = (s / (scale * std)).clip(-1, 1)
    elif method == "zscore":
        mu  = s.rolling(60).mean()
        std = s.rolling(60).std()
        s   = ((s - mu) / std.replace(0, np.nan)).apply(np.tanh)

    return s.fillna(0)


# ── Alpha Model ────────────────────────────────────────────────────────────────

class AlphaModel:
    """
    Linear Alpha Model.

    Learns OLS weights mapping features → next-period returns,
    then applies those weights to score each bar.

    Attributes:
        weights      : dict {feature_name: weight}
        intercept    : float
        feature_cols : list of feature names used in fit
        is_fitted    : bool
        train_r2     : in-sample R² (informational only)
        train_ic     : information coefficient on training set
    """

    def __init__(self, forward_periods=1, normalize_method="tanh",
                 normalize_scale=2.0, min_obs=60):
        """
        Args:
            forward_periods  : how many periods ahead to predict (default 1)
            normalize_method : score normalization method (default "tanh")
            normalize_scale  : normalization scale factor (default 2.0)
            min_obs          : minimum observations required to fit (default 60)
        """
        self.forward_periods   = forward_periods
        self.normalize_method  = normalize_method
        self.normalize_scale   = normalize_scale
        self.min_obs           = min_obs

        self.weights      = {}
        self.intercept    = 0.0
        self.feature_cols = []
        self.is_fitted    = False
        self.train_r2     = np.nan
        self.train_ic     = np.nan


    def fit(self, df):
        """
        Fit OLS weights on training data.

        Target = forward return over forward_periods bars.
        Features = all available feature columns in df.

        Args:
            df : DataFrame after extract_features() — training period only

        Returns:
            self (for chaining)
        """
        feature_cols = get_feature_cols(df)
        if not feature_cols:
            raise ValueError("No feature columns found. Run extract_features() first.")

        # Target: forward return — shift features back by forward_periods
        # so feature at time t predicts return from t to t+forward_periods
        target = df["Close"].pct_change(self.forward_periods).shift(-self.forward_periods)

        X = df[feature_cols].copy()
        y = target

        # Drop warmup rows where features are NaN
        valid = X.dropna(how="any").index.intersection(y.dropna().index)
        X_fit = X.loc[valid]
        y_fit = y.loc[valid]

        if len(X_fit) < self.min_obs:
            raise ValueError(
                f"Only {len(X_fit)} valid observations after dropping NaN — "
                f"need at least {self.min_obs}. Use a longer training window."
            )

        weights, intercept = _ols_weights(X_fit, y_fit)

        if weights is None:
            raise ValueError("OLS fit failed — matrix may be singular.")

        self.feature_cols = feature_cols
        self.weights      = dict(zip(feature_cols, weights))
        self.intercept    = float(intercept)
        self.is_fitted    = True

        # Compute in-sample R² and IC
        raw_scores     = X_fit @ weights + intercept
        ss_res         = ((y_fit.values - raw_scores) ** 2).sum()
        ss_tot         = ((y_fit.values - y_fit.values.mean()) ** 2).sum()
        self.train_r2  = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        self.train_ic  = float(np.corrcoef(raw_scores, y_fit.values)[0, 1])

        return self


    def predict(self, df):
        """
        Apply learned weights to produce alpha score column.

        Args:
            df : DataFrame after extract_features() — any period

        Returns:
            DataFrame with 'alpha_score' column added, values in [-1, 1]
            Also adds 'alpha_raw' (pre-normalization) for diagnostics
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        out = df.copy()

        # Compute raw weighted sum
        raw = pd.Series(self.intercept, index=df.index)
        for col, w in self.weights.items():
            if col in df.columns:
                raw = raw + df[col].fillna(0) * w

        out["alpha_raw"]   = raw
        out["alpha_score"] = _normalize_score(
            raw,
            method=self.normalize_method,
            scale=self.normalize_scale,
        )

        return out


    def print_weights(self):
        """Print fitted weights sorted by absolute magnitude."""
        if not self.is_fitted:
            print("Model not fitted.")
            return

        print(f"\n── Alpha Model Weights ──────────────────────────")
        print(f"  Train R²  : {self.train_r2:.4f}")
        print(f"  Train IC  : {self.train_ic:.4f}  "
              f"({'positive' if self.train_ic > 0 else 'negative'} predictive correlation)")
        print(f"  Intercept : {self.intercept:.6f}")
        print(f"\n  {'Feature':<20}  {'Weight':>10}  {'Direction'}")
        print(f"  {'─'*45}")

        sorted_w = sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, w in sorted_w:
            direction = "↑ bullish" if w > 0 else "↓ bearish"
            print(f"  {feat:<20}  {w:>10.6f}  {direction}")
        print()


    def feature_importance(self):
        """
        Returns a Series of feature importances (absolute weights),
        normalized so they sum to 1.
        """
        if not self.is_fitted:
            return pd.Series()
        abs_w = {k: abs(v) for k, v in self.weights.items()}
        total = sum(abs_w.values())
        return pd.Series({k: v / total for k, v in abs_w.items()}).sort_values(ascending=False)

        