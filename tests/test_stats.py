"""Paired Wilcoxon signed-rank test of src/stats.py on inputs that are equal in exact
arithmetic but carry floating-point summation noise."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import paired_wilcoxon  # noqa: E402

ALPHA = 0.1
N_SEEDS = 5


def seed_mean_absdev(counts, n: int) -> float:
    """|mean over seeds of counts[s] / n - (1 - ALPHA)|, summed in the order of counts."""
    total = 0.0
    for k in counts:
        total += k / n
    return abs(total / len(counts) - (1 - ALPHA))


def paired_traces(permute: bool) -> tuple[np.ndarray, np.ndarray]:
    """Absolute coverage deviations of two methods a and b on 24 paired traces with
    N_SEEDS seeds each. Traces 16-19 have the same seed counts for both methods (zero
    difference in exact arithmetic) and traces 20-23 repeat traces 0-3 (|d| tied in
    exact arithmetic). With permute=True the seed order of method b on traces 16-19 and
    of both methods on traces 20-23 is reversed, which changes only the floating-point
    summation order; with permute=False equal values are computed in the same order and
    agree bit for bit."""
    rng = np.random.default_rng(22)
    traces = []
    for _ in range(16):
        n = int(rng.integers(30, 200))
        traces.append((n, rng.binomial(n, 0.90, N_SEEDS).tolist(),
                       rng.binomial(n, 0.89, N_SEEDS).tolist()))
    for _ in range(4):
        n = int(rng.integers(30, 200))
        c = rng.binomial(n, 0.90, N_SEEDS).tolist()
        traces.append((n, c, c[::-1] if permute else c))
    for n, ca, cb in traces[:4]:
        traces.append((n, ca[::-1], cb[::-1]) if permute else (n, ca, cb))
    a = np.array([seed_mean_absdev(ca, n) for n, ca, _ in traces])
    b = np.array([seed_mean_absdev(cb, n) for n, _, cb in traces])
    return a, b


def unrounded(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    """Number of non-zero differences, p-value and statistic of the Wilcoxon test on the
    unrounded inputs (scipy forms the differences a - b)."""
    nz = (a - b) != 0
    res = sps.wilcoxon(a[nz], b[nz], zero_method="wilcox", method="auto")
    return int(nz.sum()), float(res.pvalue), float(res.statistic)


def test_paired_wilcoxon_is_invariant_to_summation_order():
    a0, b0 = paired_traces(permute=False)
    a1, b1 = paired_traces(permute=True)
    assert np.abs(a1 - a0).max() < 1e-15 and np.abs(b1 - b0).max() < 1e-15
    d0, d1 = a0 - b0, a1 - b1
    # exact zeros and ties with one summation order, noise with the other
    assert (d0[16:20] == 0).all() and (d1[16:20] != 0).any()
    assert (np.abs(d0[20:]) == np.abs(d0[:4])).all() and (d0[:4] != 0).all()
    assert (np.abs(d1[20:]) != np.abs(d1[:4])).any()
    # the test on unrounded inputs depends on the noise
    assert unrounded(a1, b1) != unrounded(a0, b0)
    assert unrounded(a1, b1)[0] > unrounded(a0, b0)[0]
    # the rounded test does not, and on bit-consistent inputs it equals the unrounded test
    res0, res1 = paired_wilcoxon(a0, b0), paired_wilcoxon(a1, b1)
    assert res1 == res0
    assert (res0["p"], res0["stat"]) == pytest.approx(unrounded(a0, b0)[1:], rel=1e-12)
    assert res0["n"] == 24


def test_paired_wilcoxon_noise_only_differences_are_zero():
    rng = np.random.default_rng(1)
    counts = [rng.integers(25, 31, size=N_SEEDS).tolist() for _ in range(12)]
    a = np.array([seed_mean_absdev(c, 30) for c in counts])
    b = np.array([seed_mean_absdev(c[::-1], 30) for c in counts])
    assert (a != b).any()
    assert paired_wilcoxon(a, b) == {"n": 12, "p": 1.0, "stat": 0.0,
                                     "rank_biserial": 0.0, "median_diff": 0.0}


def test_paired_wilcoxon_keeps_small_genuine_differences_and_drops_nan():
    a = np.linspace(0.01, 0.2, 10)
    b = a + 1e-9
    res = paired_wilcoxon(np.append(a, np.nan), np.append(b, 0.5))
    # ten tied negative differences: deterministic permutation p-value 2 / 2**10
    assert res["n"] == 10 and res["stat"] == 0.0 and res["rank_biserial"] == -1.0
    assert res["p"] == pytest.approx(2 / 2 ** 10)
    assert res["median_diff"] == pytest.approx(-1e-9, rel=1e-9)
