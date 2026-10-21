"""Statistical correctness tests for the conformal implementations."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import (ACITrace, CQR, MondrianACI, MondrianConformal,
                           SplitConformal, _conformal_quantile)

ALPHA = 0.1


def test_conformal_quantile_finite_sample():
    scores = np.arange(1, 100, dtype=float)  # n=99
    q = _conformal_quantile(scores, 0.1)
    # ceil(100*0.9)=90 -> 90th order statistic
    assert q == 90.0
    assert _conformal_quantile(np.array([]), 0.1) == np.inf


def test_split_conformal_marginal_coverage_exchangeable():
    rng = np.random.default_rng(0)
    covs = []
    for rep in range(50):
        y_cal = rng.normal(0, 1, 500)
        y_te = rng.normal(0, 1, 500)
        pred_cal = np.zeros(500)
        pred_te = np.zeros(500)
        sc = SplitConformal(ALPHA).calibrate(y_cal + 10, pred_cal + 10)
        lo, hi = sc.interval(pred_te + 10)
        covs.append(np.mean((y_te + 10 >= lo) & (y_te + 10 <= hi)))
    assert abs(np.mean(covs) - (1 - ALPHA)) < 0.02


def test_mondrian_group_coverage_heteroskedastic():
    """Pooled split conformal undercovers the noisy group; Mondrian fixes it."""
    rng = np.random.default_rng(1)
    n = 2000
    g_cal = np.array(["a"] * n + ["b"] * n)
    y_cal = np.concatenate([rng.normal(0, 0.5, n), rng.normal(0, 3.0, n)]) + 10
    g_te = np.array(["a"] * n + ["b"] * n)
    y_te = np.concatenate([rng.normal(0, 0.5, n), rng.normal(0, 3.0, n)]) + 10
    pred = np.full(2 * n, 10.0)
    mc = MondrianConformal(ALPHA).calibrate(y_cal, pred, g_cal)
    lo, hi = mc.interval(pred, g_te)
    cov_b = np.mean((y_te[g_te == "b"] >= lo[g_te == "b"]) & (y_te[g_te == "b"] <= hi[g_te == "b"]))
    assert abs(cov_b - (1 - ALPHA)) < 0.03
    sc = SplitConformal(ALPHA).calibrate(y_cal, pred)
    lo_s, hi_s = sc.interval(pred)
    cov_b_pooled = np.mean((y_te[g_te == "b"] >= lo_s[g_te == "b"]) & (y_te[g_te == "b"] <= hi_s[g_te == "b"]))
    assert cov_b_pooled < cov_b - 0.05  # pooled undercovers noisy group


def test_mondrian_thin_group_falls_back():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 100) + 5
    pred = np.full(100, 5.0)
    groups = np.array(["big"] * 95 + ["tiny"] * 5)
    mc = MondrianConformal(ALPHA, min_group_n=50).calibrate(y, pred, groups)
    assert "tiny" in mc.fallback_groups_
    lo, hi = mc.interval(np.array([5.0]), np.array(["tiny"]))
    assert np.isfinite(hi[0])


def test_aci_long_run_miscoverage_under_shift():
    """Under a variance shift that breaks split conformal, ACI's realised
    miscoverage still converges near alpha."""
    rng = np.random.default_rng(3)
    y_cal = rng.normal(0, 1, 1000) + 10
    pred_cal = np.full(1000, 10.0)
    y_te = rng.normal(0, 3.0, 5000) + 10  # 3x noisier than calibration
    pred_te = np.full(5000, 10.0)
    aci = ACITrace(ALPHA, gamma=0.05).calibrate(y_cal, pred_cal)
    lo, hi, _ = aci.run_trace(y_te, pred_te)
    miscov = np.mean((y_te < lo) | (y_te > hi))
    assert abs(miscov - ALPHA) < 0.03
    sc = SplitConformal(ALPHA).calibrate(y_cal, pred_cal)
    lo_s, hi_s = sc.interval(pred_te)
    miscov_split = np.mean((y_te < lo_s) | (y_te > hi_s))
    assert miscov_split > 2 * ALPHA  # split conformal badly undercovers here


def test_mondrian_aci_matches_aci_single_group():
    rng = np.random.default_rng(4)
    y_cal = rng.normal(0, 1, 500) + 5
    pred_cal = np.full(500, 5.0)
    y_te = rng.normal(0, 1, 300) + 5
    pred_te = np.full(300, 5.0)
    g = np.array(["only"] * 300)
    aci = ACITrace(ALPHA).calibrate(y_cal, pred_cal)
    maci = MondrianACI(ALPHA).calibrate(y_cal, pred_cal, np.array(["only"] * 500))
    lo_a, hi_a, _ = aci.run_trace(y_te, pred_te)
    lo_m, hi_m, _ = maci.run_trace(y_te, pred_te, g)
    np.testing.assert_allclose(lo_a, lo_m)
    np.testing.assert_allclose(hi_a, hi_m)


def test_cqr_coverage():
    rng = np.random.default_rng(5)
    n = 3000
    y_cal = rng.normal(0, 1, n) + 20
    y_te = rng.normal(0, 1, n) + 20
    # deliberately miscalibrated raw quantiles (too narrow)
    lo_raw = np.full(n, 20 - 0.5)
    hi_raw = np.full(n, 20 + 0.5)
    cqr = CQR(ALPHA).calibrate(y_cal, lo_raw, hi_raw)
    lo, hi = cqr.interval(lo_raw, hi_raw)
    cov = np.mean((y_te >= lo) & (y_te <= hi))
    assert abs(cov - (1 - ALPHA)) < 0.03


def test_lower_bound_clipped_nonnegative():
    sc = SplitConformal(ALPHA)
    sc.qhat_ = 100.0
    lo, hi = sc.interval(np.array([1.0]))
    assert lo[0] == 0.0
