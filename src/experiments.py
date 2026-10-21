"""Experiment orchestration: regimes x methods x seeds, trace-level splits,
per-cell result files with resume, deterministic seeding.

A 'regime' defines source (train+calibration) and target (test) trace pools:
  in_domain_4g / in_domain_5g : random trace split within one technology
  cross_mobility_4g           : low-mobility source -> vehicular target (4G)
  cross_mobility_5g           : static source -> driving target (5G)
  cross_tech_4to5 / 5to4      : whole-technology transfer
  cross_app_5g                : Download source -> streaming target (5G)

Groups for Mondrian methods are runtime-observable speed bins (km/h):
  [0,1) static-like, [1,15) pedestrian-like, [15,50) urban vehicular, [50,inf) fast.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import ACITrace, CQR, MondrianACI, MondrianConformal, SplitConformal
from .data_loading import read_cell_rows, read_feature_table
from .evaluation import point_metrics, rate_selection_metrics, trace_metrics
from .features import FEATURE_COLS, apply_impute, train_median_impute
from .models import GaussianIntervals, GBMPoint, GBMQuantiles, PersistenceModel, RidgeModel

SPEED_BINS = [0.0, 1.0, 15.0, 50.0, np.inf]
SPEED_LABELS = ["speed_0", "speed_low", "speed_med", "speed_high"]
METHODS = ["gaussian", "quantile_raw", "split", "cqr", "mondrian", "aci", "mondrian_aci"]
ONLINE_METHODS = ["aci", "mondrian_aci"]


def speed_groups(X: pd.DataFrame) -> np.ndarray:
    s = pd.to_numeric(X["Speed"], errors="coerce")
    g = pd.cut(s, bins=SPEED_BINS, labels=SPEED_LABELS, right=False, include_lowest=True)
    return g.astype(str).fillna("unknown").to_numpy()


def regime_pools(feat: pd.DataFrame, regime: str):
    """Return (source_df, target_df_or_None). target None => random in-domain split."""
    if regime == "in_domain_4g":
        return feat[feat.tech == "4G"], None
    if regime == "in_domain_5g":
        return feat[feat.tech == "5G"], None
    if regime == "cross_mobility_4g":
        f = feat[feat.tech == "4G"]
        return f[f.mobility.isin(["static", "pedestrian"])], f[f.mobility.isin(["car", "bus", "train"])]
    if regime == "cross_mobility_5g":
        f = feat[feat.tech == "5G"]
        return f[f.mobility == "static"], f[f.mobility == "driving"]
    if regime == "cross_tech_4to5":
        return feat[feat.tech == "4G"], feat[feat.tech == "5G"]
    if regime == "cross_tech_5to4":
        return feat[feat.tech == "5G"], feat[feat.tech == "4G"]
    if regime == "cross_app_5g":
        f = feat[feat.tech == "5G"]
        return f[f.app == "Download"], f[f.app.isin(["Netflix", "Amazon_Prime"])]
    raise ValueError(f"unknown regime {regime}")


def split_traces(source: pd.DataFrame, target: pd.DataFrame | None, seed: int,
                 frac_train=0.6, frac_cal=0.2):
    """Trace-level split. In-domain: train/cal/test from source. Cross-domain:
    train/cal from source (80/20), test = all target traces."""
    rng = np.random.default_rng(seed)
    src_traces = np.array(sorted(source.trace_id.unique()))
    rng.shuffle(src_traces)
    if target is None:
        n = len(src_traces)
        n_tr = int(round(frac_train * n))
        n_ca = int(round(frac_cal * n))
        tr = set(src_traces[:n_tr])
        ca = set(src_traces[n_tr:n_tr + n_ca])
        te = set(src_traces[n_tr + n_ca:])
        return (source[source.trace_id.isin(tr)],
                source[source.trace_id.isin(ca)],
                source[source.trace_id.isin(te)])
    n = len(src_traces)
    n_tr = int(round(0.8 * n))
    tr = set(src_traces[:n_tr])
    ca = set(src_traces[n_tr:])
    return (source[source.trace_id.isin(tr)],
            source[source.trace_id.isin(ca)],
            target)


@dataclass
class CellFit:
    """Fitted state of one (regime, seed) cell, shared by run_cell and cell_instances.

    test holds the test instances with index 0..n-1 in feature-table order; y, pred
    and groups are aligned with it; offline maps the five methods without feedback to
    their (lo, hi); aci and mondrian_aci are the calibrated online methods.
    """
    test: pd.DataFrame
    y: np.ndarray
    pred: np.ndarray
    groups: np.ndarray
    offline: dict
    aci: ACITrace
    mondrian_aci: MondrianACI


def fit_cell(feat: pd.DataFrame, regime: str, seed: int, alpha: float,
             base_model: str = "gbm", gamma: float = 0.05) -> CellFit:
    """Split, fit the base model(s) and calibrate every interval method of one cell."""
    source, target = regime_pools(feat, regime)
    train, cal, test = split_traces(source, target, seed)
    med = train_median_impute(train)
    Xtr, ytr = apply_impute(train, med), train["y"].to_numpy()
    Xca, yca = apply_impute(cal, med), cal["y"].to_numpy()
    Xte, yte = apply_impute(test, med), test["y"].to_numpy()
    g_ca, g_te = speed_groups(cal), speed_groups(test)

    if base_model == "gbm":
        point = GBMPoint(seed).fit(Xtr, ytr)
    elif base_model == "ridge":
        point = RidgeModel(seed).fit(Xtr, ytr)
    elif base_model == "persistence":
        point = PersistenceModel(FEATURE_COLS.index("dl_last")).fit(Xtr, ytr)
    else:
        raise ValueError(base_model)
    pred_tr = point.predict(Xtr)
    pred_ca = point.predict(Xca)
    pred_te = point.predict(Xte)

    qm = GBMQuantiles(alpha, seed).fit(Xtr, ytr)
    qlo_ca, qhi_ca = qm.predict(Xca)
    qlo_te, qhi_te = qm.predict(Xte)

    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gauss = GaussianIntervals(alpha).fit_sigma(ytr, pred_tr)
    intervals["gaussian"] = gauss.interval(pred_te)
    intervals["quantile_raw"] = (np.clip(qlo_te, 0, None), qhi_te)
    intervals["split"] = SplitConformal(alpha).calibrate(yca, pred_ca).interval(pred_te)
    intervals["cqr"] = CQR(alpha).calibrate(yca, qlo_ca, qhi_ca).interval(
        np.clip(qlo_te, 0, None), qhi_te)
    intervals["mondrian"] = MondrianConformal(alpha).calibrate(
        yca, pred_ca, g_ca).interval(pred_te, g_te)

    aci = ACITrace(alpha, gamma=gamma).calibrate(yca, pred_ca)
    maci = MondrianACI(alpha, gamma=gamma).calibrate(yca, pred_ca, g_ca)
    return CellFit(test=test.reset_index(drop=True), y=yte, pred=pred_te, groups=g_te,
                   offline=intervals, aci=aci, mondrian_aci=maci)


def trace_positions(test: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """(trace_id, positions in issue order) per test trace, in order of appearance."""
    return [(sid, g.sort_values("t_index").index.to_numpy())
            for sid, g in test.groupby("trace_id", sort=False)]


def run_online(fit: CellFit, delay_s: float | None = None) -> tuple[dict, pd.DataFrame]:
    """Intervals of aci and mondrian_aci for every test trace of a fitted cell.

    delay_s None gives immediate feedback in row order (run_trace, as in run_cell);
    delay_s >= 0 gives the delay model, with the label of each instance arriving at
    ts_label + delay_s and intervals issued at ts (columns of the feature table).
    Returns ({method: (lo, hi, alpha_at_issue)}, per-trace statistics); the statistics
    frame (method, trace_id, max_outstanding, n_applied, n_alpha_le0, n_alpha_ge1,
    n_alpha_ge1_covered) is empty for immediate feedback.
    """
    test, y, pred, groups = fit.test, fit.y, fit.pred, fit.groups
    if delay_s is not None:
        missing = [c for c in ("ts", "ts_label") if c not in test.columns]
        if missing:
            raise ValueError(f"delay model needs feature columns {missing}")
        issue = test["ts"].to_numpy(dtype=float)
        label = test["ts_label"].to_numpy(dtype=float)
    n = len(test)
    out = {m: (np.empty(n), np.empty(n), np.empty(n)) for m in ONLINE_METHODS}
    stats = []
    for sid, idx in trace_positions(test):
        if delay_s is None:
            res = {"aci": fit.aci.run_trace(y[idx], pred[idx]),
                   "mondrian_aci": fit.mondrian_aci.run_trace(y[idx], pred[idx],
                                                              groups[idx])}
        else:
            ra = fit.aci.run_delayed(y[idx], pred[idx], issue[idx], label[idx], delay_s)
            rm = fit.mondrian_aci.run_delayed(y[idx], pred[idx], groups[idx], issue[idx],
                                              label[idx], delay_s)
            res = {"aci": ra[:3], "mondrian_aci": rm[:3]}
            stats.append({"method": "aci", "trace_id": sid, **ra[3]})
            stats.append({"method": "mondrian_aci", "trace_id": sid, **rm[3]})
        for m in ONLINE_METHODS:
            for k in range(3):
                out[m][k][idx] = res[m][k]
    cols = ["method", "trace_id", "max_outstanding", "n_applied", "n_alpha_le0",
            "n_alpha_ge1", "n_alpha_ge1_covered"]
    return out, pd.DataFrame(stats, columns=cols)


def cell_instances(feat: pd.DataFrame, regime: str, seed: int, alpha: float = 0.10,
                   base_model: str = "gbm", gamma: float = 0.05,
                   delay_s: float = 0.0) -> pd.DataFrame:
    """Per-instance intervals of every method for the test traces of one cell.

    Uses fit_cell (the splits, fits and calibration of run_cell) and the delay model
    for aci and mondrian_aci with reporting delay delay_s; feat needs columns ts and
    ts_label. One row per test instance, traces in order of appearance and instances
    in issue order, with columns trace_id, t_index, ts, ts_label, y, pred, group,
    lo_<m> and hi_<m> for every method m in METHODS, alpha_aci and alpha_mondrian_aci
    (the level at issue).
    """
    missing = [c for c in ("ts", "ts_label") if c not in feat.columns]
    if missing:
        raise ValueError(f"feature table lacks columns {missing}")
    fit = fit_cell(feat, regime, seed, alpha, base_model, gamma)
    online, _ = run_online(fit, delay_s)
    order = np.concatenate([idx for _, idx in trace_positions(fit.test)])
    t = fit.test
    out = pd.DataFrame({
        "trace_id": t["trace_id"].to_numpy()[order],
        "t_index": t["t_index"].to_numpy()[order],
        "ts": t["ts"].to_numpy(dtype=float)[order],
        "ts_label": t["ts_label"].to_numpy(dtype=float)[order],
        "y": fit.y[order],
        "pred": fit.pred[order],
        "group": fit.groups[order],
    })
    for m in METHODS:
        lo, hi = fit.offline[m] if m in fit.offline else online[m][:2]
        out[f"lo_{m}"] = np.asarray(lo, dtype=float)[order]
        out[f"hi_{m}"] = np.asarray(hi, dtype=float)[order]
    for m in ONLINE_METHODS:
        out[f"alpha_{m}"] = online[m][2][order]
    return out


@lru_cache(maxsize=2)
def read_features(path: str) -> pd.DataFrame:
    """Feature table (data_loading.read_feature_table) read once per process (callers
    must not modify it)."""
    return read_feature_table(path)


def run_cell(feat: pd.DataFrame, regime: str, seed: int, alpha: float,
             base_model: str = "gbm", gamma: float = 0.05) -> pd.DataFrame:
    """One (regime, seed) cell: fit base model(s), build intervals for every method,
    return per-trace metric rows."""
    fit = fit_cell(feat, regime, seed, alpha, base_model, gamma)
    test, yte, pred_te, g_te = fit.test, fit.y, fit.pred, fit.groups
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = dict(fit.offline)
    aci, maci = fit.aci, fit.mondrian_aci
    lo_a = np.empty(len(test)); hi_a = np.empty(len(test))
    lo_m = np.empty(len(test)); hi_m = np.empty(len(test))
    for sid, g in test.groupby("trace_id", sort=False):
        idx = g.sort_values("t_index").index.to_numpy()
        la, ha, _ = aci.run_trace(yte[idx], pred_te[idx])
        lm, hm, _ = maci.run_trace(yte[idx], pred_te[idx], g_te[idx])
        lo_a[idx], hi_a[idx] = la, ha
        lo_m[idx], hi_m[idx] = lm, hm
    intervals["aci"] = (lo_a, hi_a)
    intervals["mondrian_aci"] = (lo_m, hi_m)

    pm = point_metrics(yte, pred_te)
    rows = []
    for method, (lo, hi) in intervals.items():
        df = pd.DataFrame({"trace_id": test["trace_id"].to_numpy(), "y": yte,
                           "lo": lo, "hi": hi, "group": g_te,
                           "mobility": test["mobility"].to_numpy()})
        marginal_cov = float(((df.y >= df.lo) & (df.y <= df.hi)).mean())
        sm = trace_metrics(df, alpha)
        per_group = {}
        for gname, gg in df.groupby("group"):
            cov = float(((gg.y >= gg.lo) & (gg.y <= gg.hi)).mean())
            per_group[gname] = cov
        for _, r in sm.iterrows():
            trace_mask = df.trace_id == r.trace_id
            ds = rate_selection_metrics(df.y[trace_mask], df.lo[trace_mask],
                                        pred_te[trace_mask.to_numpy()])
            rows.append({"regime": regime, "seed": seed, "alpha": alpha,
                         "gamma": gamma,
                         "base_model": base_model, "method": method,
                         **r.to_dict(), **ds,
                         "marginal_coverage": marginal_cov,
                         "rmse_point": pm["rmse"], "mae_point": pm["mae"],
                         "group_coverage_json": json.dumps(per_group)})
    return pd.DataFrame(rows)


def run_experiment(feat_path: Path, out_dir: Path, regimes: list[str], seeds: list[int],
                   alpha: float, base_models: list[str], n_jobs: int = 1,
                   gamma: float = 0.05, log=print) -> pd.DataFrame:
    """Run all (regime, base_model, seed) cells with per-cell result files and resume."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    feat = read_feature_table(feat_path, compression="infer")
    cells = [(r, m, s) for r in regimes for m in base_models for s in seeds]
    pending = [(r, m, s) for (r, m, s) in cells
               if not (cells_dir / f"{r}__{m}__{s}.csv").exists()]
    log(f"{len(cells)} cells total, {len(pending)} to run ({len(cells)-len(pending)} cached)")

    def one(r, m, s):
        t0 = time.time()
        df = run_cell(feat, r, s, alpha, m, gamma=gamma)
        df.to_csv(cells_dir / f"{r}__{m}__{s}.csv", index=False)
        return r, m, s, time.time() - t0, len(df)

    if n_jobs > 1 and pending:
        from joblib import Parallel, delayed
        for r, m, s, dt, nrow in Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(one)(r, m, s) for (r, m, s) in pending):
            log(f"  done {r}/{m}/seed{s}: {nrow} rows in {dt:.1f}s")
    else:
        for (r, m, s) in pending:
            _, _, _, dt, nrow = one(r, m, s)
            log(f"  done {r}/{m}/seed{s}: {nrow} rows in {dt:.1f}s")

    parts = [read_cell_rows(cells_dir / f"{r}__{m}__{s}.csv") for (r, m, s) in cells
             if (cells_dir / f"{r}__{m}__{s}.csv").exists()]
    allrows = pd.concat(parts, ignore_index=True)
    allrows.to_csv(out_dir / "rows.csv.gz", index=False, compression="gzip")
    return allrows
