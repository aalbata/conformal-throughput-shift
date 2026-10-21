"""Per-trace interval-quality metrics. Trace = independent replication unit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def winkler_score(y, lo, hi, alpha: float, inverted_empty: bool = False) -> np.ndarray:
    """Interval (Winkler) score; lower is better.

    With inverted_empty, an interval with hi < lo is the empty set and the width term
    is max(hi - lo, 0); the penalty terms are the same in both cases.
    """
    y, lo, hi = map(np.asarray, (y, lo, hi))
    width = hi - lo
    if inverted_empty:
        width = np.maximum(width, 0.0)
    below = np.where(y < lo, lo - y, 0.0)
    above = np.where(y > hi, y - hi, 0.0)
    return width + (2.0 / alpha) * (below + above)


def trace_metrics(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """df columns: trace_id, y, lo, hi (+ optional group cols carried through).

    Returns one row per trace with coverage, widths, and Winkler score.
    """
    rows = []
    for sid, g in df.groupby("trace_id", sort=False):
        y, lo, hi = g["y"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy()
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        finite = np.isfinite(width)
        # canonical ACI can emit infinite (always-covering) intervals under severe
        # shift; report their fraction and compute width stats on finite intervals
        rows.append({
            "trace_id": sid,
            "n": len(g),
            "coverage": float(covered.mean()),
            "frac_inf": float((~finite).mean()),
            "mean_width": float(np.mean(width[finite])) if finite.any() else np.inf,
            "median_width": float(np.median(width)),  # inf-robust up to 50% inf
            "winkler": float(np.median(winkler_score(y, lo, hi, alpha))),
        })
    return pd.DataFrame(rows)


def point_metrics(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
    }


def rate_selection_metrics(y, lo, pred) -> dict:
    """Downstream conservative rate selection.

    Selecting rate = lower interval bound vs rate = point prediction.
    overshoot = fraction of seconds where the selected rate exceeds realised
    throughput (stall-risk proxy); utilisation = selected/realised capped at 1.
    """
    y, lo, pred = map(np.asarray, (y, lo, pred))
    ok = y > 0
    y, lo, pred = y[ok], lo[ok], pred[ok]
    if len(y) == 0:
        return {"overshoot_lb": np.nan, "util_lb": np.nan,
                "overshoot_point": np.nan, "util_point": np.nan}
    return {
        "overshoot_lb": float(np.mean(lo > y)),
        "util_lb": float(np.mean(np.minimum(lo, y) / y)),
        "overshoot_point": float(np.mean(pred > y)),
        "util_point": float(np.mean(np.minimum(pred, y) / y)),
    }


def trace_interval_metrics(trace_id, y, lo, hi, alpha: float) -> pd.DataFrame:
    """Per-trace coverage, finite-interval and zero-target metrics of one interval method.

    One row per trace (order of first appearance) with counts n, n_cov, n_inf, n_finite,
    n_cov_finite, n_inverted (finite intervals with hi < lo), n_y0 (targets equal to 0),
    n_pos and n_cov_pos (targets > 0), the ratios coverage, frac_inf, coverage_finite and
    coverage_pos, abs_dev = |coverage - (1 - alpha)|, the median and mean width of the
    finite intervals, and winkler_finite_mean, the mean interval score over the finite
    intervals. An inverted interval is the empty set: its width is max(hi - lo, 0) = 0 in
    the width statistics and in the width term of the interval score. Ratios and finite
    statistics are NaN when their denominator is 0.
    """
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    finite = np.isfinite(hi - lo)
    inverted = finite & (hi < lo)
    width = np.maximum(hi - lo, 0.0)
    covered = (y >= lo) & (y <= hi)
    positive = y > 0
    wink = winkler_score(y, lo, hi, alpha, inverted_empty=True)
    df = pd.DataFrame({
        "trace_id": np.asarray(trace_id),
        "cov": covered.astype(np.int64),
        "inf": (~finite).astype(np.int64),
        "fin": finite.astype(np.int64),
        "cov_fin": (covered & finite).astype(np.int64),
        "inv": inverted.astype(np.int64),
        "y0": (y == 0).astype(np.int64),
        "pos": positive.astype(np.int64),
        "cov_pos": (covered & positive).astype(np.int64),
        "w_fin": np.where(finite, width, np.nan),
        "wk_fin": np.where(finite, wink, np.nan),
    })
    g = df.groupby("trace_id", sort=False)
    out = pd.DataFrame({
        "n": g["cov"].size(),
        "n_cov": g["cov"].sum(),
        "n_inf": g["inf"].sum(),
        "n_finite": g["fin"].sum(),
        "n_cov_finite": g["cov_fin"].sum(),
        "n_inverted": g["inv"].sum(),
        "n_y0": g["y0"].sum(),
        "n_pos": g["pos"].sum(),
        "n_cov_pos": g["cov_pos"].sum(),
        "width_finite_median": g["w_fin"].median(),
        "width_finite_mean": g["w_fin"].mean(),
        "winkler_finite_mean": g["wk_fin"].mean(),
    })
    out.index.name = "trace_id"
    out = out.reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["coverage"] = out["n_cov"] / out["n"]
        out["frac_inf"] = out["n_inf"] / out["n"]
        out["coverage_finite"] = np.where(out["n_finite"] > 0,
                                          out["n_cov_finite"] / out["n_finite"], np.nan)
        out["coverage_pos"] = np.where(out["n_pos"] > 0,
                                       out["n_cov_pos"] / out["n_pos"], np.nan)
    out["abs_dev"] = (out["coverage"] - (1.0 - alpha)).abs()
    cols = ["trace_id", "n", "n_cov", "coverage", "abs_dev", "n_inf", "frac_inf",
            "n_finite", "n_cov_finite", "coverage_finite", "n_inverted", "width_finite_median",
            "width_finite_mean", "winkler_finite_mean", "n_y0", "n_pos", "n_cov_pos",
            "coverage_pos"]
    return out[cols]


def group_interval_metrics(groups, y, lo, hi) -> pd.DataFrame:
    """Instance count n, covered count n_cov and coverage per group, sorted by group."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    covered = ((y >= lo) & (y <= hi)).astype(np.int64)
    df = pd.DataFrame({"group": np.asarray(groups).astype(str), "cov": covered})
    g = df.groupby("group", sort=True)["cov"]
    out = pd.DataFrame({"n": g.size(), "n_cov": g.sum()})
    out.index.name = "group"
    out = out.reset_index()
    out["coverage"] = out["n_cov"] / out["n"]
    return out


def delay_coverage_bound(alpha_1: float, gamma: float, max_outstanding, n_applied):
    """Deterministic long-run bound (max(a1, 1 - a1) + (Dmax + 1) gamma) / (gamma M).

    Dmax is the largest number of outstanding labels at any issue time and M the
    number of applied labels; the bound bounds |mean error over applied labels -
    alpha| and is inf when M = 0. Accepts scalars or arrays.
    """
    d = np.asarray(max_outstanding, dtype=float)
    m = np.asarray(n_applied, dtype=float)
    num = max(alpha_1, 1.0 - alpha_1) + (d + 1.0) * gamma
    with np.errstate(divide="ignore"):
        out = np.where(m > 0, num / (gamma * np.where(m > 0, m, 1.0)), np.inf)
    return float(out) if out.ndim == 0 else out


def trace_reproduction_metrics(trace_id, y, lo, hi) -> pd.DataFrame:
    """n, coverage and frac_inf per trace, computed as in trace_metrics."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    df = pd.DataFrame({"trace_id": np.asarray(trace_id),
                       "cov": ((y >= lo) & (y <= hi)).astype(np.int64),
                       "inf": (~np.isfinite(hi - lo)).astype(np.int64)})
    g = df.groupby("trace_id", sort=False)
    out = pd.DataFrame({"n": g["cov"].size(), "n_cov": g["cov"].sum(), "n_inf": g["inf"].sum()})
    out.index.name = "trace_id"
    out = out.reset_index()
    out["coverage"] = out["n_cov"] / out["n"]
    out["frac_inf"] = out["n_inf"] / out["n"]
    return out[["trace_id", "n", "coverage", "frac_inf"]]
