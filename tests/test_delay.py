"""Tests for the presorted radius lookup and the label-arrival (delay) model."""
import heapq
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import (ACITrace, MondrianACI, _conformal_quantile, _sorted_quantile,
                           feedback_schedule)
from src.evaluation import delay_coverage_bound

ALPHA = 0.1


def _reference_run_trace(scores_sorted, alpha, gamma, y, pred, pools=None):
    """Sequential ACI with a re-sorting quantile lookup at every step."""
    alpha_t = alpha
    n = len(y)
    lo, hi, alphas = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        pool = scores_sorted if pools is None else pools[t]
        if alpha_t <= 0.0:
            r = np.inf
        elif alpha_t >= 1.0:
            r = 0.0
        else:
            r = _conformal_quantile(pool, alpha_t)
        lo[t] = max(pred[t] - r, 0.0) if np.isfinite(r) else 0.0
        hi[t] = pred[t] + r
        alphas[t] = alpha_t
        err = 1.0 if (y[t] < lo[t] or y[t] > hi[t]) else 0.0
        alpha_t = alpha_t + gamma * (alpha - err)
    return lo, hi, alphas


def _reference_schedule(issue_ts, arrival):
    """Heap form of the delay model: (label, issue index) pairs in application order."""
    heap, applied = [], []
    for j in range(len(issue_ts)):
        while heap and heap[0][0] <= issue_ts[j]:
            _, i = heapq.heappop(heap)
            applied.append((i, j))
        heapq.heappush(heap, (arrival[j], j))
    return applied


def _irregular_trace(rng, n_rows=400, frac_inst=0.8, horizon=1):
    """Row timestamps with duplicates and gaps, instance rows and their label rows."""
    steps = rng.choice([0, 1, 1, 1, 1, 2, 3, 7], size=n_rows)
    tau = np.cumsum(steps).astype(float)
    tau -= tau[0]
    rows = np.flatnonzero(rng.random(n_rows - horizon) < frac_inst)
    return tau[rows], tau[rows + horizon]


def _shifted_data(rng, n_cal=300, n=600):
    y_cal = rng.normal(0, 1, n_cal) + 10
    pred_cal = np.full(n_cal, 10.0)
    scale = np.where(np.arange(n) < n // 2, 4.0, 0.2)
    y = np.abs(rng.normal(0, 1, n) * scale + 10)
    pred = np.full(n, 10.0) + rng.normal(0, 0.05, n)
    return y_cal, pred_cal, y, pred


def test_sorted_quantile_matches_conformal_quantile():
    rng = np.random.default_rng(0)
    for n in (0, 1, 2, 5, 49, 53, 200, 1001):
        scores = np.round(np.abs(rng.normal(0, 3, n)), 1)  # with ties
        srt = np.sort(scores)
        alphas = [1e-12, 0.001, 0.05, 0.1, 0.33, 0.5, 0.9, 0.999999]
        if n > 0:
            alphas += [1 / (n + 1), 1 / (n + 1) + 1e-15, 1 / (n + 1) - 1e-15]
        alphas += list(rng.uniform(0, 1, 50))
        for a in alphas:
            q_ref = _conformal_quantile(scores, a)
            q_new = _sorted_quantile(srt, a)
            assert q_new == q_ref or (np.isinf(q_new) and np.isinf(q_ref))


@pytest.mark.parametrize("gamma", [0.05, 0.5])
def test_run_trace_matches_resorting_reference(gamma):
    rng = np.random.default_rng(1)
    y_cal, pred_cal, y, pred = _shifted_data(rng)
    y[350:450] = pred[350:450]  # a covered stretch drives alpha_t to 1 and beyond
    aci = ACITrace(ALPHA, gamma=gamma).calibrate(y_cal, pred_cal)
    lo, hi, al = aci.run_trace(y, pred)
    lo_r, hi_r, al_r = _reference_run_trace(aci.cal_scores_, ALPHA, gamma, y, pred)
    assert np.array_equal(lo, lo_r) and np.array_equal(hi, hi_r) and np.array_equal(al, al_r)
    if gamma == 0.5:
        assert (al <= 0).any() and (al >= 1).any()

    groups = rng.choice(["a", "b", "c"], size=len(y))
    g_cal = rng.choice(["a", "b", "thin"], size=len(y_cal), p=[0.5, 0.45, 0.05])
    maci = MondrianACI(ALPHA, gamma=gamma).calibrate(y_cal, pred_cal, g_cal)
    lo, hi, al = maci.run_trace(y, pred, groups)
    pools = [maci.group_scores_.get(g, maci.pooled_scores_) for g in groups]
    lo_r, hi_r, al_r = _reference_run_trace(None, ALPHA, gamma, y, pred, pools)
    assert np.array_equal(lo, lo_r) and np.array_equal(hi, hi_r) and np.array_equal(al, al_r)


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_delay_zero_horizon_one_equals_run_trace(seed):
    rng = np.random.default_rng(seed)
    issue, label = _irregular_trace(rng)
    assert (np.diff(issue) == 0).any() and (np.diff(issue) > 1).any()
    n = len(issue)
    y_cal, pred_cal, y, pred = _shifted_data(rng, n=n)
    for gamma in (0.05, 0.5):
        aci = ACITrace(ALPHA, gamma=gamma).calibrate(y_cal, pred_cal)
        ref = aci.run_trace(y, pred)
        lo, hi, al, st = aci.run_delayed(y, pred, issue, label, 0.0)
        assert np.array_equal(lo, ref[0]) and np.array_equal(hi, ref[1])
        assert np.array_equal(al, ref[2])
        assert st["max_outstanding"] == 0 and st["n_applied"] == n - 1

        groups = rng.choice(["a", "b"], size=n)
        maci = MondrianACI(ALPHA, gamma=gamma).calibrate(
            y_cal, pred_cal, rng.choice(["a", "b"], size=len(y_cal)))
        ref = maci.run_trace(y, pred, groups)
        lo, hi, al, st = maci.run_delayed(y, pred, groups, issue, label, 0.0)
        assert np.array_equal(lo, ref[0]) and np.array_equal(hi, ref[1])
        assert np.array_equal(al, ref[2])
        assert st["max_outstanding"] == 0 and st["n_applied"] == n - 1


def test_schedule_arrival_order_and_ties():
    issue = np.array([0.0, 1.0, 2.0, 3.0, 8.0])
    label = np.array([6.0, 3.0, 3.0, 2.5, 9.0])
    order, step = feedback_schedule(issue, label, 0.0)
    assert order.tolist() == [1, 2, 3, 0, 4]
    assert step.tolist() == [3, 3, 4, 4, 5]


@pytest.mark.parametrize("delay", [0.0, 1.0, 2.0, 5.0, 10.0, 37.5])
def test_schedule_matches_heap_reference(delay):
    rng = np.random.default_rng(5)
    issue, label = _irregular_trace(rng, horizon=3)
    label = label + rng.choice([0.0, 0.0, 4.0, -2.0], size=len(label))
    order, step = feedback_schedule(issue, label, delay)
    applied = [(int(i), int(j)) for i, j in zip(order, step) if j < len(issue)]
    assert applied == _reference_schedule(issue, label + delay)


def test_schedule_rejects_unsorted_issue_times():
    with pytest.raises(ValueError):
        feedback_schedule(np.array([0.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0]))


def test_labels_after_last_issue_are_not_applied():
    rng = np.random.default_rng(6)
    n, delay, gamma = 10, 5.0, 0.05
    issue = np.arange(n, dtype=float)
    label = issue + 1.0
    y_cal = rng.normal(0, 1, 200) + 5
    aci = ACITrace(ALPHA, gamma=gamma).calibrate(y_cal, np.full(200, 5.0))
    y = rng.normal(0, 3, n) + 5
    pred = np.full(n, 5.0)
    lo, hi, al, st = aci.run_delayed(y, pred, issue, label, delay)
    err = ((y < lo) | (y > hi)).astype(float)
    # label i arrives at i + 6, so it is applied before issue i + 6 (labels 0..3 only)
    expected = np.full(n, ALPHA)
    for j in range(n):
        for i in range(max(0, j - 5)):
            expected[j] += gamma * (ALPHA - err[i])
    np.testing.assert_allclose(al, expected, rtol=0, atol=1e-15)
    assert st["n_applied"] == 4
    assert st["max_outstanding"] == 5


def test_delay_longer_than_trace():
    rng = np.random.default_rng(7)
    n = 50
    issue = np.arange(n, dtype=float)
    y_cal = rng.normal(0, 1, 100) + 5
    aci = ACITrace(ALPHA).calibrate(y_cal, np.full(100, 5.0))
    lo, hi, al, st = aci.run_delayed(rng.normal(0, 5, n) + 5, np.full(n, 5.0),
                                     issue, issue + 1.0, 1e6)
    assert np.all(al == ALPHA)
    assert st["n_applied"] == 0 and st["max_outstanding"] == n - 1
    assert delay_coverage_bound(ALPHA, 0.05, st["max_outstanding"], st["n_applied"]) == np.inf


@pytest.mark.parametrize("d", [0, 1, 2, 5, 10])
def test_outstanding_and_applied_counts_regular_trace(d):
    n = 300
    issue = np.arange(n, dtype=float)
    rng = np.random.default_rng(8)
    aci = ACITrace(ALPHA).calibrate(rng.normal(0, 1, 100), np.zeros(100))
    _, _, _, st = aci.run_delayed(rng.normal(0, 2, n), np.zeros(n), issue, issue + 1.0, d)
    assert st["max_outstanding"] == d
    assert st["n_applied"] == n - 1 - d


def test_alpha_boundary_counts():
    rng = np.random.default_rng(9)
    n = 400
    y_cal = rng.normal(0, 1, 100) + 5
    pred_cal = np.full(100, 5.0)
    pred = np.full(n, 5.0)
    y = pred.copy()                       # every interval covers, so alpha_t climbs past 1
    y[250:] = 5.0 + rng.normal(0, 50, n - 250)  # then mostly misses, so alpha_t falls below 0
    issue = np.arange(n, dtype=float)
    aci = ACITrace(ALPHA, gamma=0.5).calibrate(y_cal, pred_cal)
    lo, hi, al, st = aci.run_delayed(y, pred, issue, issue + 1.0, 2.0)
    covered = (y >= lo) & (y <= hi)
    assert st["n_alpha_le0"] == int((al <= 0).sum()) > 0
    assert st["n_alpha_ge1"] == int((al >= 1).sum()) > 0
    assert st["n_alpha_ge1_covered"] == int(((al >= 1) & covered).sum()) > 0
    assert np.all(hi[al >= 1] == pred[al >= 1])
    assert np.all(np.isinf(hi[al <= 0]))


def test_bound_formula():
    assert delay_coverage_bound(0.1, 0.05, 0, 100) == pytest.approx(0.19)
    assert delay_coverage_bound(0.1, 0.05, 4, 96) == pytest.approx((0.9 + 0.25) / (0.05 * 96))
    out = delay_coverage_bound(0.1, 0.05, np.array([0, 2]), np.array([0, 10]))
    assert np.isinf(out[0]) and out[1] == pytest.approx((0.9 + 0.15) / 0.5)


@pytest.mark.parametrize("seed", range(10, 16))
def test_bound_holds_with_recorded_counts(seed):
    """Mean error over applied labels stays within the bound evaluated at the recorded
    maximum outstanding count when no zero-radius interval covers."""
    rng = np.random.default_rng(seed)
    issue, label = _irregular_trace(rng, n_rows=800, horizon=int(rng.integers(1, 6)))
    n = len(issue)
    y_cal, pred_cal, y, pred = _shifted_data(rng, n=n)
    gamma = float(rng.choice([0.05, 0.2, 0.5]))
    delay = float(rng.choice([0.0, 1.0, 2.0, 5.0, 10.0]))
    groups = rng.choice(["a", "b", "c"], size=n)
    maci = MondrianACI(ALPHA, gamma=gamma).calibrate(
        y_cal, pred_cal, rng.choice(["a", "b", "c"], size=len(y_cal)))
    lo, hi, al, st = maci.run_delayed(y, pred, groups, issue, label, delay)
    assert st["n_alpha_ge1_covered"] == 0
    err = ((y < lo) | (y > hi)).astype(float)
    order, step = feedback_schedule(issue, label, delay)
    applied = order[step < n]
    assert len(applied) == st["n_applied"]
    dev = abs(err[applied].mean() - ALPHA)
    assert dev <= delay_coverage_bound(ALPHA, gamma, st["max_outstanding"], st["n_applied"])
