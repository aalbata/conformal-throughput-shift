"""Hand-computed checks of the per-trace and per-group interval metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (group_interval_metrics, trace_interval_metrics,
                            trace_reproduction_metrics, winkler_score)

INF = np.inf


def _example():
    sid = ["A", "A", "A", "A", "B", "B", "C", "C", "C"]
    y = [1.0, 2.0, 0.0, 5.0, 3.0, 4.0, 0.0, 6.0, 2.0]
    lo = [0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.5, 5.0, 1.0]
    hi = [2.0, 3.0, INF, 4.0, INF, INF, 1.5, 7.0, 3.0]
    return sid, y, lo, hi


def test_trace_metrics_hand_computed():
    m = trace_interval_metrics(*_example(), alpha=0.1).set_index("trace_id")
    assert list(m.index) == ["A", "B", "C"]

    a = m.loc["A"]
    # covered: yes, no (2 < 2.5), yes (infinite), no (5 > 4)
    assert (a.n, a.n_cov, a.n_inf, a.n_finite, a.n_cov_finite) == (4, 2, 1, 3, 1)
    assert a.coverage == 0.5 and a.frac_inf == 0.25
    assert a.abs_dev == pytest.approx(0.4)
    assert a.coverage_finite == pytest.approx(1 / 3)
    # finite widths 2, 0.5, 4
    assert a.width_finite_median == pytest.approx(2.0)
    assert a.width_finite_mean == pytest.approx(6.5 / 3)
    # interval scores 2, 0.5 + 20 * 0.5, 4 + 20 * 1
    assert a.winkler_finite_mean == pytest.approx((2.0 + 10.5 + 24.0) / 3)
    assert (a.n_y0, a.n_pos, a.n_cov_pos) == (1, 3, 1)
    assert a.coverage_pos == pytest.approx(1 / 3)

    b = m.loc["B"]
    assert (b.n, b.n_cov, b.n_inf, b.n_finite) == (2, 2, 2, 0)
    assert b.coverage == 1.0 and b.frac_inf == 1.0
    assert np.isnan(b.coverage_finite) and np.isnan(b.width_finite_median)
    assert np.isnan(b.width_finite_mean) and np.isnan(b.winkler_finite_mean)
    assert (b.n_y0, b.n_pos, b.n_cov_pos) == (0, 2, 2)
    assert (m.n_inverted == 0).all()

    c = m.loc["C"]
    # y = 0 below lo = 0.5 (miss); 6 in [5, 7]; 2 in [1, 3]
    assert (c.n, c.n_cov, c.n_inf, c.n_finite, c.n_cov_finite) == (3, 2, 0, 3, 2)
    assert c.coverage_finite == pytest.approx(2 / 3)
    assert c.width_finite_median == pytest.approx(2.0)
    assert c.width_finite_mean == pytest.approx(5.0 / 3)
    assert c.winkler_finite_mean == pytest.approx((1.0 + 20 * 0.5 + 2.0 + 2.0) / 3)
    assert (c.n_y0, c.n_pos, c.n_cov_pos) == (1, 2, 2)
    assert np.isnan(m.loc["B"].coverage_finite)


def test_trace_interval_metrics_match_trace_metrics():
    """coverage and frac_inf equal the means used by evaluation.trace_metrics, and the mean
    finite width equals its mean_width once inverted intervals have width 0."""
    from src.evaluation import trace_metrics
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 997
    sid = rng.choice(["s1", "s2", "s3"], size=n)
    y = np.abs(rng.normal(5, 3, n))
    lo = np.clip(y - rng.normal(1, 1, n), 0, None)
    hi = np.where(rng.random(n) < 0.1, np.inf, y + rng.normal(0.5, 1, n))
    assert (hi < lo).any()
    ours = trace_interval_metrics(sid, y, lo, hi, 0.1).set_index("trace_id")
    ref = trace_metrics(pd.DataFrame({"trace_id": sid, "y": y, "lo": lo, "hi": hi}),
                        0.1).set_index("trace_id")
    ref_w = trace_metrics(pd.DataFrame({"trace_id": sid, "y": y, "lo": lo,
                                        "hi": np.maximum(hi, lo)}), 0.1).set_index("trace_id")
    for s in ref.index:
        assert ours.loc[s, "n"] == ref.loc[s, "n"]
        assert ours.loc[s, "coverage"] == ref.loc[s, "coverage"]
        assert ours.loc[s, "frac_inf"] == ref.loc[s, "frac_inf"]
        assert ours.loc[s, "width_finite_mean"] == pytest.approx(ref_w.loc[s, "mean_width"])
        assert ours.loc[s, "n_inverted"] == np.sum((np.asarray(sid) == s) & (hi < lo))
    rep = trace_reproduction_metrics(sid, y, lo, hi).set_index("trace_id")
    assert list(rep.columns) == ["n", "coverage", "frac_inf"]
    for s in ref.index:
        assert rep.loc[s, "n"] == ref.loc[s, "n"]
        assert rep.loc[s, "coverage"] == ref.loc[s, "coverage"]
        assert rep.loc[s, "frac_inf"] == ref.loc[s, "frac_inf"]


def test_inverted_intervals_are_empty_sets():
    """A finite interval with hi < lo is counted in n_inverted and has width
    max(hi - lo, 0) = 0 in the width statistics and in the width term of the interval
    score; coverage and frac_inf are those of trace_reproduction_metrics."""
    sid = ["A", "A", "A", "B", "B"]
    y = [1.0, 0.5, 3.0, 2.0, 2.0]
    lo = [0.0, 0.8, 0.0, 0.0, 1.0]
    hi = [2.0, 0.3, -0.4, INF, 3.0]
    m = trace_interval_metrics(sid, y, lo, hi, alpha=0.1).set_index("trace_id")
    a = m.loc["A"]
    # widths 2, max(-0.5, 0) = 0 and max(-0.4, 0) = 0; only the first interval covers
    assert (a.n, a.n_cov, a.n_inf, a.n_finite) == (3, 1, 0, 3)
    assert (a.n_cov_finite, a.n_inverted) == (1, 2)
    assert a.width_finite_median == 0.0
    assert a.width_finite_mean == pytest.approx(2.0 / 3)
    # interval scores 2, 0 + 20 * ((0.8 - 0.5) + (0.5 - 0.3)) and 0 + 20 * (3.0 + 0.4)
    assert a.winkler_finite_mean == pytest.approx((2.0 + 10.0 + 68.0) / 3)
    b = m.loc["B"]
    assert (b.n, b.n_cov, b.n_inf, b.n_finite, b.n_inverted) == (2, 2, 1, 1, 0)
    assert b.width_finite_mean == 2.0 and b.winkler_finite_mean == 2.0
    rep = trace_reproduction_metrics(sid, y, lo, hi).set_index("trace_id")
    assert m[["n", "coverage", "frac_inf"]].equals(rep[["n", "coverage", "frac_inf"]])

    inverted = ([0.5], [0.8], [0.3], 0.1)
    assert winkler_score(*inverted, inverted_empty=True)[0] == pytest.approx(10.0)
    assert winkler_score(*inverted)[0] == pytest.approx(9.5)
    ordered = ([1.0, 3.0], [0.0, 1.0], [2.0, 2.0], 0.1)
    assert np.array_equal(winkler_score(*ordered, inverted_empty=True), winkler_score(*ordered))


def test_group_metrics_hand_computed():
    sid, y, lo, hi = _example()
    groups = ["g2", "g1", "g1", "g2", "g1", "g3", "g3", "g2", "g1"]
    g = group_interval_metrics(groups, y, lo, hi).set_index("group")
    assert list(g.index) == ["g1", "g2", "g3"]
    # g1: rows 1 (miss), 2 (cover), 4 (cover), 8 (cover)
    assert (g.loc["g1", "n"], g.loc["g1", "n_cov"]) == (4, 3)
    # g2: rows 0 (cover), 3 (miss), 7 (cover)
    assert (g.loc["g2", "n"], g.loc["g2", "n_cov"]) == (3, 2)
    # g3: rows 5 (cover, infinite), 6 (miss)
    assert (g.loc["g3", "n"], g.loc["g3", "n_cov"]) == (2, 1)
    assert g.loc["g1", "coverage"] == 0.75
