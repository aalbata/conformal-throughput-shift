"""Distribution-free prediction-interval methods.

Implements:
  - SplitConformal        : absolute-residual split conformal (Vovk et al., 2005)
  - CQR                   : conformalized quantile regression (Romano et al., 2019)
  - MondrianConformal     : group-conditional (Mondrian) split conformal (Vovk, 2003)
  - ACITrace              : adaptive conformal inference run online per test trace
                            (Gibbs & Candes, 2021)
  - MondrianACI           : proposed hybrid - group-specific score pools with
                            per-trace online alpha adaptation.

Throughput is non-negative, so all lower bounds are clipped at 0.

ACITrace and MondrianACI also provide run_delayed, which applies the feedback
of each instance when its label arrives (label time plus a reporting delay).
"""
from __future__ import annotations

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid quantile: ceil((n+1)(1-alpha))/n empirical quantile."""
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def _sorted_quantile(sorted_scores: np.ndarray, alpha: float) -> float:
    """_conformal_quantile for a pool that is already sorted in ascending order."""
    n = len(sorted_scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(sorted_scores[k - 1])


def _aci_radius(sorted_pool: np.ndarray, alpha_t: float) -> float:
    """ACI radius at level alpha_t: inf for alpha_t <= 0, 0 for alpha_t >= 1."""
    if alpha_t <= 0.0:
        return np.inf
    if alpha_t >= 1.0:
        return 0.0
    return _sorted_quantile(sorted_pool, alpha_t)


def feedback_schedule(issue_ts, label_ts, delay: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Order in which the labels of one trace are applied under the delay model.

    The label of instance i arrives at label_ts[i] + delay. Before the interval of
    instance j is issued at issue_ts[j], every earlier instance i < j whose label has
    arrived (arrival <= issue_ts[j]) and has not been applied yet is applied, in order
    of arrival with ties in issue order. issue_ts must be non-decreasing.

    Returns (order, step): order lists instance indices in application order and
    step[k] is the issue index before which label order[k] is applied; step[k] == n
    marks a label that arrives after the last issue and is never applied.
    """
    issue_ts = np.asarray(issue_ts, dtype=float)
    arrival = np.asarray(label_ts, dtype=float) + float(delay)
    n = len(issue_ts)
    if arrival.shape != issue_ts.shape:
        raise ValueError("issue_ts and label_ts must have the same length")
    if np.isnan(issue_ts).any() or np.isnan(arrival).any():
        raise ValueError("issue and label times must not be NaN")
    if n > 1 and np.any(np.diff(issue_ts) < 0):
        raise ValueError("issue_ts must be non-decreasing")
    first_issue = np.searchsorted(issue_ts, arrival, side="left")
    step = np.maximum(first_issue, np.arange(1, n + 1))
    order = np.lexsort((np.arange(n), arrival, step))
    return order, step[order]


def _run_delayed(pools: list, alpha: float, gamma: float, y, pred, issue_ts, label_ts,
                 delay: float):
    """Shared loop of ACITrace.run_delayed and MondrianACI.run_delayed."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(y)
    order, step = feedback_schedule(issue_ts, label_ts, delay)
    if len(order) != n:
        raise ValueError("y and issue_ts must have the same length")
    order, step = order.tolist(), step.tolist()
    alpha_t = alpha
    lo = np.empty_like(y)
    hi = np.empty_like(y)
    alphas = np.empty_like(y)
    errs = [0.0] * n
    k = 0
    max_outstanding = 0
    n_le0 = n_ge1 = n_ge1_covered = 0
    for t in range(n):
        while k < n and step[k] <= t:
            alpha_t = alpha_t + gamma * (alpha - errs[order[k]])
            k += 1
        if t - k > max_outstanding:
            max_outstanding = t - k
        r = _aci_radius(pools[t], alpha_t)
        lo[t] = max(pred[t] - r, 0.0) if np.isfinite(r) else 0.0
        hi[t] = pred[t] + r
        alphas[t] = alpha_t
        err = 1.0 if (y[t] < lo[t] or y[t] > hi[t]) else 0.0
        errs[t] = err
        if alpha_t <= 0.0:
            n_le0 += 1
        elif alpha_t >= 1.0:
            n_ge1 += 1
            if err == 0.0:
                n_ge1_covered += 1
    stats = {"max_outstanding": int(max_outstanding), "n_applied": int(k),
             "n_alpha_le0": int(n_le0), "n_alpha_ge1": int(n_ge1),
             "n_alpha_ge1_covered": int(n_ge1_covered)}
    return lo, hi, alphas, stats


class SplitConformal:
    """Symmetric absolute-residual split conformal around a point predictor."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.qhat_: float | None = None

    def calibrate(self, y_cal: np.ndarray, pred_cal: np.ndarray) -> "SplitConformal":
        self.qhat_ = _conformal_quantile(np.abs(y_cal - pred_cal), self.alpha)
        return self

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.clip(pred - self.qhat_, 0.0, None)
        hi = pred + self.qhat_
        return lo, hi


class CQR:
    """Conformalized quantile regression on top of lower/upper quantile predictors."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.qhat_: float | None = None

    @staticmethod
    def score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return np.maximum(lo - y, y - hi)

    def calibrate(self, y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray) -> "CQR":
        self.qhat_ = _conformal_quantile(self.score(y_cal, lo_cal, hi_cal), self.alpha)
        return self

    def interval(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(lo - self.qhat_, 0.0, None), hi + self.qhat_


class MondrianConformal:
    """Group-conditional split conformal: one conformal quantile per group.

    Groups with fewer than min_group_n calibration scores fall back to the
    pooled quantile (recorded in fallback_groups_).
    """

    def __init__(self, alpha: float, min_group_n: int = 50):
        self.alpha = alpha
        self.min_group_n = min_group_n
        self.group_qhat_: dict = {}
        self.pooled_qhat_: float | None = None
        self.fallback_groups_: set = set()

    def calibrate(self, y_cal, pred_cal, groups_cal) -> "MondrianConformal":
        scores = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
        groups_cal = np.asarray(groups_cal)
        self.pooled_qhat_ = _conformal_quantile(scores, self.alpha)
        for g in np.unique(groups_cal):
            s = scores[groups_cal == g]
            if len(s) >= self.min_group_n:
                self.group_qhat_[g] = _conformal_quantile(s, self.alpha)
            else:
                self.fallback_groups_.add(g)
        return self

    def interval(self, pred, groups) -> tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(pred, dtype=float)
        q = np.array([self.group_qhat_.get(g, self.pooled_qhat_) for g in np.asarray(groups)])
        return np.clip(pred - q, 0.0, None), pred + q


class ACITrace:
    """Adaptive conformal inference (Gibbs & Candes, 2021), run per test trace.

    alpha_t is updated online with step gamma using the realised miscoverage;
    the radius at effective level alpha_t is looked up in the (static)
    calibration-score pool. Following Gibbs & Candes, alpha_t is NOT clipped to
    (0, 1): alpha_t <= 0 yields an infinite (always-covering) interval and
    alpha_t >= 1 yields a zero-radius interval. This is what guarantees that
    long-run miscoverage converges to alpha under arbitrary distribution shift.
    """

    def __init__(self, alpha: float, gamma: float = 0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.cal_scores_: np.ndarray | None = None

    def calibrate(self, y_cal: np.ndarray, pred_cal: np.ndarray) -> "ACITrace":
        self.cal_scores_ = np.sort(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)))
        return self

    def _radius(self, alpha_t: float) -> float:
        return _aci_radius(self.cal_scores_, alpha_t)

    def run_delayed(self, y, pred, issue_ts, label_ts, delay: float = 0.0):
        """Run one trace under the delay model (see feedback_schedule).

        Returns (lo, hi, alpha_at_issue, stats); stats holds max_outstanding,
        n_applied, n_alpha_le0, n_alpha_ge1 and n_alpha_ge1_covered.
        """
        pools = [self.cal_scores_] * len(y)
        return _run_delayed(pools, self.alpha, self.gamma, y, pred, issue_ts, label_ts, delay)

    def run_trace(self, y: np.ndarray, pred: np.ndarray):
        """Sequentially produce intervals for one trace; returns (lo, hi, alpha_traj)."""
        y = np.asarray(y, dtype=float)
        pred = np.asarray(pred, dtype=float)
        alpha_t = self.alpha
        lo = np.empty_like(y)
        hi = np.empty_like(y)
        alphas = np.empty_like(y)
        for t in range(len(y)):
            r = self._radius(alpha_t)
            lo[t] = max(pred[t] - r, 0.0) if np.isfinite(r) else 0.0
            hi[t] = pred[t] + r
            alphas[t] = alpha_t
            err = 1.0 if (y[t] < lo[t] or y[t] > hi[t]) else 0.0
            alpha_t = alpha_t + self.gamma * (self.alpha - err)
        return lo, hi, alphas


class MondrianACI:
    """Proposed hybrid: group-specific calibration-score pools + per-trace ACI.

    Each test sample's radius is looked up in the score pool of its group
    (pooled fallback for thin groups), while alpha_t adapts online within the
    trace exactly as in ACI. Groups therefore correct systematic
    heteroskedasticity across contexts, and the online update corrects residual
    within-trace drift.
    """

    def __init__(self, alpha: float, gamma: float = 0.05, min_group_n: int = 50):
        self.alpha = alpha
        self.gamma = gamma
        self.min_group_n = min_group_n
        self.group_scores_: dict = {}
        self.pooled_scores_: np.ndarray | None = None

    def calibrate(self, y_cal, pred_cal, groups_cal) -> "MondrianACI":
        scores = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
        groups_cal = np.asarray(groups_cal)
        self.pooled_scores_ = np.sort(scores)
        for g in np.unique(groups_cal):
            s = scores[groups_cal == g]
            if len(s) >= self.min_group_n:
                self.group_scores_[g] = np.sort(s)
        return self

    def _radius(self, group, alpha_t: float) -> float:
        pool = self.group_scores_.get(group, self.pooled_scores_)
        return _aci_radius(pool, alpha_t)

    def run_delayed(self, y, pred, groups, issue_ts, label_ts, delay: float = 0.0):
        """Run one trace under the delay model (see feedback_schedule).

        Returns (lo, hi, alpha_at_issue, stats) as ACITrace.run_delayed.
        """
        pools = [self.group_scores_.get(g, self.pooled_scores_) for g in np.asarray(groups)]
        return _run_delayed(pools, self.alpha, self.gamma, y, pred, issue_ts, label_ts, delay)

    def run_trace(self, y, pred, groups):
        y = np.asarray(y, dtype=float)
        pred = np.asarray(pred, dtype=float)
        groups = np.asarray(groups)
        alpha_t = self.alpha
        lo = np.empty_like(y)
        hi = np.empty_like(y)
        alphas = np.empty_like(y)
        for t in range(len(y)):
            r = self._radius(groups[t], alpha_t)
            lo[t] = max(pred[t] - r, 0.0) if np.isfinite(r) else 0.0
            hi[t] = pred[t] + r
            alphas[t] = alpha_t
            err = 1.0 if (y[t] < lo[t] or y[t] > hi[t]) else 0.0
            alpha_t = alpha_t + self.gamma * (self.alpha - err)
        return lo, hi, alphas
