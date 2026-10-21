"""Edge cases of the label-arrival model, checked against a literal step-by-step reference."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import (ACITrace, MondrianACI, MondrianConformal, _conformal_quantile,
                           feedback_schedule)
from src.evaluation import delay_coverage_bound, trace_interval_metrics, winkler_score
from src.experiments import fit_cell, run_online, trace_positions
from src.features import build_features, instance_timestamps

ALPHA = 0.1


def _literal_run(pools, alpha, gamma, y, pred, issue_ts, label_ts, delay):
    """Delay model applied literally: before issue j, apply every earlier unapplied label
    with label_ts[i] + delay <= issue_ts[j], in order of arrival with ties in issue order."""
    n = len(y)
    arrival = [float(label_ts[i]) + float(delay) for i in range(n)]
    applied = [False] * n
    errs = [0.0] * n
    alpha_t = alpha
    lo, hi, al = np.empty(n), np.empty(n), np.empty(n)
    max_out = n_app = n_le0 = n_ge1 = n_ge1_cov = 0
    for j in range(n):
        ready = [i for i in range(j) if not applied[i] and arrival[i] <= issue_ts[j]]
        for i in sorted(ready, key=lambda i: (arrival[i], i)):
            alpha_t = alpha_t + gamma * (alpha - errs[i])
            applied[i] = True
            n_app += 1
        max_out = max(max_out, sum(not applied[i] for i in range(j)))
        if alpha_t <= 0.0:
            r = np.inf
        elif alpha_t >= 1.0:
            r = 0.0
        else:
            r = _conformal_quantile(pools[j], alpha_t)
        lo[j] = max(pred[j] - r, 0.0) if np.isfinite(r) else 0.0
        hi[j] = pred[j] + r
        al[j] = alpha_t
        errs[j] = 1.0 if (y[j] < lo[j] or y[j] > hi[j]) else 0.0
        if alpha_t <= 0.0:
            n_le0 += 1
        elif alpha_t >= 1.0:
            n_ge1 += 1
            n_ge1_cov += errs[j] == 0.0
    stats = {"max_outstanding": max_out, "n_applied": n_app, "n_alpha_le0": n_le0,
             "n_alpha_ge1": n_ge1, "n_alpha_ge1_covered": int(n_ge1_cov)}
    return lo, hi, al, stats


def _assert_same(res, ref):
    for k in range(3):
        assert np.array_equal(res[k], ref[k])
    assert res[3] == ref[3]


def _scores(rng, n=200):
    y_cal = np.abs(rng.normal(10, 3, n))
    return y_cal, np.full(n, 10.0)


def _case(rng, kind, n=120, horizon=1):
    """Row timestamps of one trace for a named timing pattern; returns (issue, label)."""
    rows = n + horizon
    if kind == "duplicates":
        steps = rng.choice([0, 0, 0, 1, 1, 2], size=rows)
    elif kind == "all_equal":
        steps = np.zeros(rows, dtype=int)
    elif kind == "long_gaps":
        steps = rng.choice([1, 1, 1, 40, 250], size=rows)
    else:
        steps = np.ones(rows, dtype=int)
    tau = np.cumsum(steps).astype(float)
    tau -= tau[0]
    return tau[:n], tau[horizon:horizon + n]


@pytest.mark.parametrize("kind", ["duplicates", "all_equal", "long_gaps", "regular"])
@pytest.mark.parametrize("horizon", [1, 5])
@pytest.mark.parametrize("delay", [0.0, 1.0, 2.0, 5.0, 10.0])
def test_aci_matches_literal_reference(kind, horizon, delay):
    rng = np.random.default_rng(zlib.crc32(f"{kind}-{horizon}-{delay}".encode()))
    issue, label = _case(rng, kind, horizon=horizon)
    n = len(issue)
    y_cal, pred_cal = _scores(rng)
    pred = np.full(n, 10.0) + rng.normal(0, 1, n)
    y = np.abs(pred + rng.normal(0, 1, n) * np.where(np.arange(n) < n // 2, 6.0, 0.3))
    for gamma in (0.05, 0.6):
        aci = ACITrace(ALPHA, gamma=gamma).calibrate(y_cal, pred_cal)
        res = aci.run_delayed(y, pred, issue, label, delay)
        ref = _literal_run([aci.cal_scores_] * n, ALPHA, gamma, y, pred, issue, label, delay)
        _assert_same(res, ref)


def test_zero_second_label_gap_is_applied_before_next_issue():
    """A label whose row shares the issue timestamp arrives at the issue time and is
    applied before the next issue, which also has that timestamp."""
    issue = np.zeros(6)
    label = np.zeros(6)
    rng = np.random.default_rng(0)
    y_cal, pred_cal = _scores(rng)
    y = np.array([10.0, 50.0, 50.0, 10.0, 50.0, 10.0])
    pred = np.full(6, 10.0)
    aci = ACITrace(ALPHA, gamma=0.05).calibrate(y_cal, pred_cal)
    lo, hi, al, st = aci.run_delayed(y, pred, issue, label, 0.0)
    ref = aci.run_trace(y, pred)
    assert np.array_equal(al, ref[2]) and np.array_equal(lo, ref[0])
    assert st["max_outstanding"] == 0 and st["n_applied"] == 5


def test_same_timestamp_issues_do_not_see_one_second_labels():
    issue = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    label = issue + 1.0
    rng = np.random.default_rng(1)
    y_cal, pred_cal = _scores(rng)
    y = np.array([50.0, 50.0, 10.0, 10.0, 10.0])
    pred = np.full(5, 10.0)
    gamma = 0.05
    aci = ACITrace(ALPHA, gamma=gamma).calibrate(y_cal, pred_cal)
    lo, hi, al, st = aci.run_delayed(y, pred, issue, label, 0.0)
    assert al[0] == al[1] == al[2] == ALPHA
    # labels 0, 1, 2 arrive at 1.0 and are applied before issue 3 in issue order
    expected = ALPHA
    for e in (1.0, 1.0, 0.0):
        expected = expected + gamma * (ALPHA - e)
    assert al[3] == expected and al[4] == expected
    assert st["max_outstanding"] == 2 and st["n_applied"] == 3


def test_single_instance_trace():
    rng = np.random.default_rng(2)
    y_cal, pred_cal = _scores(rng)
    aci = ACITrace(ALPHA).calibrate(y_cal, pred_cal)
    lo, hi, al, st = aci.run_delayed(np.array([3.0]), np.array([10.0]), np.array([42.0]),
                                     np.array([43.0]), 0.0)
    assert al.tolist() == [ALPHA]
    q = _conformal_quantile(aci.cal_scores_, ALPHA)
    assert lo[0] == max(10.0 - q, 0.0) and hi[0] == 10.0 + q
    assert st == {"max_outstanding": 0, "n_applied": 0, "n_alpha_le0": 0, "n_alpha_ge1": 0,
                  "n_alpha_ge1_covered": 0}
    assert delay_coverage_bound(ALPHA, 0.05, 0, 0) == np.inf


def test_all_labels_after_end_of_trace():
    n = 30
    issue = np.arange(n, dtype=float)
    rng = np.random.default_rng(3)
    y_cal, pred_cal = _scores(rng)
    maci = MondrianACI(ALPHA).calibrate(y_cal, pred_cal, rng.choice(["a", "b"], size=200))
    groups = rng.choice(["a", "b"], size=n)
    lo, hi, al, st = maci.run_delayed(np.full(n, 99.0), np.full(n, 10.0), groups, issue,
                                      issue + 1.0, float(n))
    assert np.all(al == ALPHA)
    assert st["n_applied"] == 0 and st["max_outstanding"] == n - 1
    order, step = feedback_schedule(issue, issue + 1.0, float(n))
    assert np.all(step == n) and sorted(order.tolist()) == list(range(n))


def test_schedule_rejects_nan_and_length_mismatch():
    with pytest.raises(ValueError):
        feedback_schedule(np.array([0.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        feedback_schedule(np.array([0.0, 1.0]), np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        feedback_schedule(np.array([0.0, 1.0]), np.array([1.0]))


def test_alpha_exactly_zero_and_one():
    """alpha = 0.5, gamma = 1 moves the level in steps of 0.5, so it hits 0 and 1 exactly:
    alpha_t = 0 gives [0, inf) and alpha_t = 1 a zero radius, and both are counted."""
    alpha, gamma = 0.5, 1.0
    rng = np.random.default_rng(4)
    y_cal, pred_cal = _scores(rng)
    pred = np.full(8, 10.0)
    # covered, covered (level 1.5 after two updates) ... misses bring it to 0 and below
    y = np.array([10.0, 10.0, 10.0, 99.0, 99.0, 99.0, 99.0, 10.0])
    issue = np.arange(8, dtype=float)
    aci = ACITrace(alpha, gamma=gamma).calibrate(y_cal, pred_cal)
    lo, hi, al, st = aci.run_delayed(y, pred, issue, issue + 1.0, 0.0)
    assert al.tolist() == [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.0]
    assert np.all(hi[al >= 1.0] == pred[al >= 1.0]) and np.all(lo[al >= 1.0] == 10.0)
    assert lo[7] == 0.0 and np.isinf(hi[7])
    assert st["n_alpha_ge1"] == 5 and st["n_alpha_le0"] == 1
    assert st["n_alpha_ge1_covered"] == 2
    ref = _literal_run([aci.cal_scores_] * 8, alpha, gamma, y, pred, issue, issue + 1.0, 0.0)
    _assert_same((lo, hi, al, st), ref)


def test_alpha_crossing_zero_and_one_under_delay():
    rng = np.random.default_rng(5)
    n = 300
    y_cal, pred_cal = _scores(rng)
    pred = np.full(n, 10.0)
    y = pred.copy()
    y[120:220] = 10.0 + rng.normal(0, 40, 100)
    y = np.abs(y)
    issue, label = _case(rng, "duplicates", n=n, horizon=2)
    aci = ACITrace(ALPHA, gamma=0.4).calibrate(y_cal, pred_cal)
    res = aci.run_delayed(y, pred, issue, label, 5.0)
    al = res[2]
    assert (al <= 0).any() and (al >= 1).any()
    _assert_same(res, _literal_run([aci.cal_scores_] * n, ALPHA, 0.4, y, pred, issue, label, 5.0))


def test_mondrian_thin_and_unseen_groups():
    """Groups with fewer than 50 calibration scores and groups absent from calibration use
    the pooled scores, offline and online, with and without delay."""
    rng = np.random.default_rng(6)
    n_cal = 49 + 50 + 300
    g_cal = np.array(["thin"] * 49 + ["edge"] * 50 + ["big"] * 300)
    y_cal = np.abs(rng.normal(10, np.where(g_cal == "big", 1.0, 8.0)))
    pred_cal = np.full(n_cal, 10.0)
    maci = MondrianACI(ALPHA, gamma=0.3).calibrate(y_cal, pred_cal, g_cal)
    assert set(maci.group_scores_) == {"edge", "big"}
    mc = MondrianConformal(ALPHA).calibrate(y_cal, pred_cal, g_cal)
    assert mc.fallback_groups_ == {"thin"}
    lo_off, hi_off = mc.interval(np.array([10.0, 10.0, 10.0]), np.array(["thin", "unseen", "edge"]))
    assert hi_off[0] == hi_off[1] == 10.0 + mc.pooled_qhat_
    assert hi_off[2] == 10.0 + mc.group_qhat_["edge"]

    n = 200
    groups = rng.choice(["thin", "edge", "big", "unseen"], size=n)
    pred = np.full(n, 10.0)
    y = np.abs(pred + rng.normal(0, 6, n))
    issue, label = _case(rng, "duplicates", n=n, horizon=1)
    pools = [maci.group_scores_.get(g, maci.pooled_scores_) for g in groups]
    assert all(pools[i] is maci.pooled_scores_ for i in np.flatnonzero(np.isin(groups, ["thin", "unseen"])))
    for d in (0.0, 2.0, 10.0):
        res = maci.run_delayed(y, pred, groups, issue, label, d)
        _assert_same(res, _literal_run(pools, ALPHA, 0.3, y, pred, issue, label, d))
    ref = maci.run_trace(y, pred, groups)
    res = maci.run_delayed(y, pred, groups, issue, label, 0.0)
    for k in range(3):
        assert np.array_equal(res[k], ref[k])


def test_trace_metrics_single_instance_and_zero_radius_interval():
    """One-instance traces, an all-infinite trace and a zero-radius interval with a negative
    prediction ([0, pred] with pred < 0, the empty set with width 0) follow Eq. (7) on
    finite intervals only."""
    sid = ["a", "b", "b", "c"]
    y = np.array([2.0, 0.0, 3.0, 1.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([-0.5, np.inf, np.inf, 4.0])
    m = trace_interval_metrics(sid, y, lo, hi, ALPHA).set_index("trace_id")
    a = m.loc["a"]
    assert (a.n, a.n_cov, a.n_inf, a.n_finite, a.n_inverted) == (1, 0, 0, 1, 1)
    assert a.width_finite_median == 0.0
    assert a.winkler_finite_mean == pytest.approx(20 * 2.5)
    assert a.winkler_finite_mean == winkler_score(y[:1], lo[:1], hi[:1], ALPHA,
                                                  inverted_empty=True)[0]
    b = m.loc["b"]
    assert (b.n, b.n_cov, b.n_inf, b.n_finite, b.n_y0) == (2, 2, 2, 0, 1)
    assert np.isnan(b.winkler_finite_mean) and np.isnan(b.coverage_finite)
    c = m.loc["c"]
    assert c.winkler_finite_mean == 4.0 and c.coverage_finite == 1.0
    assert b.n_inverted == 0 and c.n_inverted == 0


def _raw_trace(rng, sid, mobility, n=140):
    steps = rng.choice([0, 1, 1, 1, 1, 2, 3, 30], size=n)
    steps[0] = 0
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(steps), unit="s")
    speed = {"static": 0.0, "car": 40.0}[mobility]
    return pd.DataFrame({
        "timestamp": ts, "Speed": np.abs(rng.normal(speed, 8, n)),
        "RSRP": rng.normal(-100, 5, n), "RSRQ": rng.normal(-12, 2, n),
        "SNR": rng.normal(10, 3, n), "CQI": rng.integers(1, 15, n).astype(float),
        "RSSI": rng.normal(-85, 5, n),
        "dl_mbps": np.clip(rng.gamma(2, 5) + np.cumsum(rng.normal(0, 1.5, n)), 0, None),
        "ul_mbps": rng.gamma(2, 1, n), "state": rng.choice(["D", "D", "D", "I"], size=n),
        "is_5g_mode": 0.0, "trace_id": sid, "tech": "4G", "mobility": mobility,
        "operator": "A", "app": "unknown",
    })


@pytest.mark.parametrize("horizon", [1, 5])
def test_run_online_matches_literal_reference_per_trace(horizon):
    """run_online on a fitted cell: each trace starts at the nominal level, traces do not
    share labels, and instance rows with gaps (rows not in state D) follow the model."""
    rng = np.random.default_rng(horizon)
    raw = pd.concat([_raw_trace(rng, f"4G/{m}/s{i}", m) for i in range(8)
                     for m in ("static", "car")], ignore_index=True)
    f = build_features(raw, horizon=horizon)
    feat = pd.concat([f, instance_timestamps(f, raw, horizon)], axis=1)
    fit = fit_cell(feat, "cross_mobility_4g", 0, ALPHA, "gbm", 0.05)
    assert (np.diff(fit.test.t_index) > 1).any()
    for d in (0.0, 1.0, 5.0, 10.0):
        out, stats = run_online(fit, d)
        st = stats.set_index(["method", "trace_id"])
        for sid, idx in trace_positions(fit.test):
            issue = fit.test.ts.to_numpy()[idx]
            label = fit.test.ts_label.to_numpy()[idx]
            y, pred, groups = fit.y[idx], fit.pred[idx], fit.groups[idx]
            pools = {"aci": [fit.aci.cal_scores_] * len(idx),
                     "mondrian_aci": [fit.mondrian_aci.group_scores_.get(
                         g, fit.mondrian_aci.pooled_scores_) for g in groups]}
            for m in ("aci", "mondrian_aci"):
                ref = _literal_run(pools[m], ALPHA, 0.05, y, pred, issue, label, d)
                for k in range(3):
                    assert np.array_equal(out[m][k][idx], ref[k])
                assert st.loc[(m, sid)].to_dict() == ref[3]
