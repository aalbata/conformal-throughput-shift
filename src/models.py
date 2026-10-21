"""Base point and quantile predictors (all CPU-only, deterministic given seed)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PersistenceModel:
    """Naive baseline: predicted throughput = last observed throughput (feature dl_last)."""

    def __init__(self, dl_last_index: int):
        self.i = dl_last_index

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.i]


class RidgeModel:
    def __init__(self, seed: int):
        self.scaler = StandardScaler()
        self.m = Ridge(alpha=1.0, random_state=seed)

    def fit(self, X, y):
        self.m.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.scaler.transform(X))


class GBMPoint:
    def __init__(self, seed: int):
        self.m = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=300, learning_rate=0.1,
            max_depth=None, early_stopping=False, random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class GBMQuantiles:
    """Lower/upper quantile GBMs for CQR at miscoverage alpha (levels alpha/2, 1-alpha/2)."""

    def __init__(self, alpha: float, seed: int):
        self.lo = HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha / 2, max_iter=300, learning_rate=0.1,
            early_stopping=False, random_state=seed)
        self.hi = HistGradientBoostingRegressor(
            loss="quantile", quantile=1 - alpha / 2, max_iter=300, learning_rate=0.1,
            early_stopping=False, random_state=seed)

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def predict(self, X):
        return self.lo.predict(X), self.hi.predict(X)


class GaussianIntervals:
    """Non-conformal parametric baseline: pred +/- z * sigma with sigma from train residuals."""

    def __init__(self, alpha: float):
        from scipy.stats import norm
        self.z = float(norm.ppf(1 - alpha / 2))
        self.sigma_: float | None = None

    def fit_sigma(self, y_train: np.ndarray, pred_train: np.ndarray):
        self.sigma_ = float(np.std(y_train - pred_train, ddof=1))
        return self

    def interval(self, pred: np.ndarray):
        r = self.z * self.sigma_
        return np.clip(pred - r, 0.0, None), pred + r
