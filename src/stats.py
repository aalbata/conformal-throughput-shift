"""Trace-level statistical inference: paired tests, effect sizes, bootstrap CIs, BH."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 10000, level: float = 0.95,
                      seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of x. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.mean(x)), float(lo), float(hi)


DIFF_DECIMALS = 12  # decimal places of the paired inputs and differences in paired_wilcoxon


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank on paired trace metrics a vs b,
    with matched-pairs rank-biserial correlation as effect size.

    Both inputs and their differences d = a - b are rounded to DIFF_DECIMALS decimal
    places, so that values equal in exact arithmetic but computed in a different
    summation order (for example seed means of ratios of integers) give exactly zero
    differences and exactly tied |d|. Pairs with d = 0 are dropped (Wilcoxon zero
    method); the test runs on the rounded non-zero differences (one-sample form), and
    the rank-biserial correlation and the median difference (over all pairs) use the
    same rounded differences."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = np.round(a[mask], DIFF_DECIMALS)
    b = np.round(b[mask], DIFF_DECIMALS)
    d = np.round(a - b, DIFF_DECIMALS)
    nz = d != 0
    if nz.sum() == 0:
        return {"n": int(len(d)), "p": 1.0, "stat": 0.0, "rank_biserial": 0.0,
                "median_diff": 0.0}
    dd = d[nz]
    res = sps.wilcoxon(dd, zero_method="wilcox", method="auto")
    ranks = sps.rankdata(np.abs(dd))
    w_pos = ranks[dd > 0].sum()
    w_neg = ranks[dd < 0].sum()
    rb = float((w_pos - w_neg) / (w_pos + w_neg))
    return {"n": int(len(d)), "p": float(res.pvalue), "stat": float(res.statistic),
            "rank_biserial": rb, "median_diff": float(np.median(d))}


def bh_correct(pvals: list[float], alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject_flags, adjusted_p)."""
    rej, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return rej.tolist(), p_adj.tolist()
