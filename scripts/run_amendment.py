"""Runner for protocol amendment 1, sections 1.3 to 1.7 (protocol/amendments.md).

Tags (alpha = 0.10, gamma = 0.05, gradient boosting):
  main  7 regimes x seeds 0-19, h = 1, data/processed/features_h1_timed.csv.gz.
        Per cell: reproduction criterion of 1.3 against the locked main rows for all
        7 methods (aci and mondrian_aci under the delay model with D = 0, which must
        also equal the immediate-feedback run bit for bit); then the per-trace and
        per-speed-group metrics of 1.5 and 1.6 for every method and the delay grid
        D in {0, 1, 2, 5, 10} s of 1.4 for aci and mondrian_aci, with the delay
        statistics and bound of 1.4 (positive-target coverage only for D = 0).
  h5    in_domain_5g and cross_tech_4to5 x seeds 0-4, h = 5,
        data/processed/features_h5_timed.csv.gz. Immediate-feedback reproduction check
        against the locked abl_h5 rows; then aci and mondrian_aci under the delay model
        with D = 0 (labels arrive at the time of row t + 5) and the metrics of 1.5
        and 1.4 for every method.
  ts    7 regimes x seeds 0-4 on the timestamp-based features of 1.7,
        data/processed/features_timebase_h1.csv.gz, delay model with D = 0; per-trace
        counts for marginal coverage and infinite-interval fraction only.

In the per-trace metrics of main and h5, n_inverted counts the finite intervals with
hi < lo; such an interval is the empty set and enters the finite width statistics and
the width term of the Winkler score as max(hi - lo, 0) = 0.

A cell that fails the reproduction criterion gets a failure record and no metrics, and
execution stops: the remaining cells are cancelled, and a later run of the tag is refused
while an unresolved failure record exists, unless --retry-failed is given.
Outputs go to results/amendment1/: per-cell files in cells/ (a cell with a .done.json
file is skipped on a later run), timing_log.csv, and, once every cell of a tag is
done, <tag>_traces.csv.gz, <tag>_groups.csv.gz and run_config_<tag>.json.
--reproduce-only computes and compares n, coverage and frac_inf only and writes only
reproduce_only/reproduction_log.csv and timing_log.csv.

Usage:
  python scripts/run_amendment.py --tag main [--regimes R ...] [--seeds S ...] [--n-jobs 6]
  python scripts/run_amendment.py --tag main --reproduce-only --seeds 0 1
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.data_loading import read_cell_rows  # noqa: E402
from src.evaluation import (delay_coverage_bound, group_interval_metrics,  # noqa: E402
                            trace_interval_metrics, trace_reproduction_metrics)
from src.experiments import (ONLINE_METHODS, fit_cell, read_features,  # noqa: E402
                             run_online)

PROTO_DIR = ROOT / "protocol"
PROTO = json.loads((PROTO_DIR / "protocol.json").read_text(encoding="utf-8"))
ALPHA = PROTO["alpha_primary"]
GAMMA = PROTO["gamma_selected"]
BASE_MODEL = PROTO["base_model_primary"]
TOL = 1e-12
DELAY_GRID = [0.0, 1.0, 2.0, 5.0, 10.0]

PROC = ROOT / "data" / "processed"
LOCKED = ROOT / "results" / "locked" / "locked_rows.csv.gz"
OUT = ROOT / "results" / "amendment1"
CELLS = OUT / "cells"
REPRO_DIR = OUT / "reproduce_only"
TIMING_LOG = OUT / "timing_log.csv"

# per-trace columns written for each tag: 1.5 (and 1.4) quantities, 1.6 zero-target
# counts and positive-target coverage (main configuration only), 1.7 coverage
COLS_15 = ["n", "n_cov", "coverage", "abs_dev", "n_inf", "frac_inf", "n_finite",
           "n_cov_finite", "coverage_finite", "n_inverted", "width_finite_median",
           "width_finite_mean", "winkler_finite_mean"]
COLS_16 = ["n_y0", "n_pos", "n_cov_pos", "coverage_pos"]
POSITIVE_COLS = ["n_pos", "n_cov_pos", "coverage_pos"]
COLS_17 = ["n", "n_cov", "coverage", "n_inf", "frac_inf"]
TAGS = {
    "main": {"features": PROC / "features_h1_timed.csv.gz", "horizon": 1,
             "regimes": PROTO["regimes"], "seeds": PROTO["locked_seeds"],
             "locked_tag": "main", "delays": DELAY_GRID, "groups": True,
             "trace_cols": COLS_15 + COLS_16, "delay_stats": True},
    "h5": {"features": PROC / "features_h5_timed.csv.gz", "horizon": 5,
           "regimes": ["in_domain_5g", "cross_tech_4to5"], "seeds": PROTO["ablation_seeds"],
           "locked_tag": "abl_h5", "delays": [0.0], "groups": True,
           "trace_cols": COLS_15, "delay_stats": True},
    "ts": {"features": PROC / "features_timebase_h1.csv.gz", "horizon": 1,
           "regimes": PROTO["regimes"], "seeds": PROTO["ablation_seeds"],
           "locked_tag": None, "delays": [0.0], "groups": False,
           "trace_cols": COLS_17, "delay_stats": False},
}
# larger test sets first, so that the parallel pool finishes evenly
REGIME_ORDER = ["cross_tech_5to4", "cross_tech_4to5", "cross_mobility_4g", "in_domain_4g",
                "cross_app_5g", "cross_mobility_5g", "in_domain_5g"]
ID_COLS = ["tag", "regime", "seed", "base_model", "alpha", "gamma", "horizon", "method",
           "feedback", "delay_s"]
STAT_COLS = ["max_outstanding", "n_applied", "n_alpha_le0", "n_alpha_ge1",
             "n_alpha_ge1_covered"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_name(tag: str, regime: str, seed: int) -> str:
    return f"{tag}__{regime}__s{seed}"


def compare_with_locked(new: pd.DataFrame, locked: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Reproduction criterion of 1.3 for one cell: every (method, trace) of either side
    is present in both, n is equal and coverage and frac_inf agree within TOL."""
    keys = ["method", "trace_id"]
    m = new.merge(locked[keys + ["n", "coverage", "frac_inf"]], on=keys, how="outer",
                  suffixes=("_new", "_locked"), indicator=True)
    both = m["_merge"] == "both"
    d_cov = (m["coverage_new"] - m["coverage_locked"]).abs()
    d_inf = (m["frac_inf_new"] - m["frac_inf_locked"]).abs()
    ok = both & (m["n_new"] == m["n_locked"]) & (d_cov <= TOL) & (d_inf <= TOL)
    summary = {
        "n_pairs": int(len(m)), "n_matched": int(ok.sum()),
        "n_unpaired": int((~both).sum()),
        "n_methods": int(m["method"].nunique()), "n_traces": int(m["trace_id"].nunique()),
        "max_abs_diff_coverage": float(d_cov[both].max()) if both.any() else None,
        "max_abs_diff_frac_inf": float(d_inf[both].max()) if both.any() else None,
        "passed": bool(ok.all() and len(m) > 0),
    }
    bad = m.loc[~ok].drop(columns="_merge").head(50)
    return summary, bad


def reproduction_frame(fit, intervals: dict) -> pd.DataFrame:
    sid = fit.test["trace_id"].to_numpy()
    parts = []
    for method, (lo, hi) in intervals.items():
        r = trace_reproduction_metrics(sid, fit.y, lo, hi)
        r.insert(0, "method", method)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def metric_tables(tag: str, regime: str, seed: int, fit,
                  variants: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trace and per-speed-group metric tables of one cell for a list of
    (method, feedback, delay_s, lo, hi, stats) variants, restricted to the columns
    of the tag (TAGS[tag]["trace_cols"], delay statistics and bound if
    TAGS[tag]["delay_stats"], speed groups if TAGS[tag]["groups"])."""
    cfg = TAGS[tag]
    horizon, with_groups = cfg["horizon"], cfg["groups"]
    sid = fit.test["trace_id"].to_numpy()
    trace_parts, group_parts = [], []
    for method, feedback, delay_s, lo, hi, stats in variants:
        ids = {"tag": tag, "regime": regime, "seed": seed, "base_model": BASE_MODEL,
               "alpha": ALPHA, "gamma": GAMMA, "horizon": horizon, "method": method,
               "feedback": feedback, "delay_s": delay_s}
        tm = trace_interval_metrics(sid, fit.y, lo, hi, ALPHA)
        tm = tm[["trace_id"] + cfg["trace_cols"]].copy()
        if feedback == "delayed" and delay_s > 0 and POSITIVE_COLS[0] in tm.columns:
            tm[POSITIVE_COLS] = np.nan
        if cfg["delay_stats"] and stats is not None:
            tm = tm.merge(stats[["trace_id"] + STAT_COLS], on="trace_id", how="left",
                          validate="one_to_one")
            if tm[STAT_COLS].isna().any().any():
                raise RuntimeError(f"missing delay statistics for {method}")
            tm["bound"] = delay_coverage_bound(ALPHA, GAMMA, tm["max_outstanding"].to_numpy(),
                                               tm["n_applied"].to_numpy())
        elif cfg["delay_stats"]:
            for c in STAT_COLS + ["bound"]:
                tm[c] = np.nan
        for k, v in reversed(list(ids.items())):
            tm.insert(0, k, v)
        trace_parts.append(tm)
        if with_groups:
            gm = group_interval_metrics(fit.groups, fit.y, lo, hi)
            for k, v in reversed(list(ids.items())):
                gm.insert(0, k, v)
            group_parts.append(gm)
    traces = pd.concat(trace_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    return traces, groups


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def run_one(tag: str, regime: str, seed: int, locked: pd.DataFrame | None,
            reproduce_only: bool) -> dict:
    """Execute one cell; returns a summary for the timing and reproduction logs."""
    cfg = TAGS[tag]
    name = cell_name(tag, regime, seed)
    t_start = time.perf_counter()
    res = {"tag": tag, "regime": regime, "seed": seed, "pid": os.getpid(),
           "mode": "reproduce_only" if reproduce_only else "full",
           "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        feat = read_features(str(cfg["features"]))
        fit = fit_cell(feat, regime, seed, ALPHA, BASE_MODEL, GAMMA)
        t_fit = time.perf_counter()
        res.update(n_test=int(len(fit.test)), n_traces=int(fit.test.trace_id.nunique()))

        online0 = stats0 = None
        if cfg["locked_tag"] is not None:
            immediate, _ = run_online(fit, None)
            if tag == "main":
                online0, stats0 = run_online(fit, 0.0)
                same = all(np.array_equal(immediate[m][k], online0[m][k])
                           for m in ONLINE_METHODS for k in range(3))
                checked = {**fit.offline, **{m: online0[m][:2] for m in ONLINE_METHODS}}
            else:
                same = None
                checked = {**fit.offline, **{m: immediate[m][:2] for m in ONLINE_METHODS}}
            summary, bad = compare_with_locked(reproduction_frame(fit, checked), locked)
            res.update({f"repro_{k}": v for k, v in summary.items()})
            res["d0_equals_immediate"] = same
            passed = summary["passed"] and same is not False
            res["repro_passed"] = passed
            if not passed and not reproduce_only:
                res.update(status="failed", seconds_fit=t_fit - t_start,
                           seconds_total=time.perf_counter() - t_start)
                write_json(CELLS / f"{name}.failed.json",
                           {**res, "mismatches": bad.to_dict(orient="records")})
                return res
        t_repro = time.perf_counter()
        res["seconds_fit"] = t_fit - t_start
        res["seconds_reproduction"] = t_repro - t_fit
        if reproduce_only:
            res["status"] = "ok" if res.get("repro_passed", False) else "failed"
            res["seconds_total"] = time.perf_counter() - t_start
            return res

        variants = [(m, "none", np.nan, lo, hi, None) for m, (lo, hi) in fit.offline.items()]
        for d in cfg["delays"]:
            if d == 0.0 and online0 is not None:
                online, stats = online0, stats0
            else:
                online, stats = run_online(fit, d)
            for m in ONLINE_METHODS:
                variants.append((m, "delayed", d, online[m][0], online[m][1],
                                 stats[stats.method == m]))
        t_online = time.perf_counter()
        traces, groups = metric_tables(tag, regime, seed, fit, variants)
        gz = {"method": "gzip", "mtime": 0}
        traces.to_csv(CELLS / f"{name}.traces.csv.gz", index=False, compression=gz)
        if cfg["groups"]:
            groups.to_csv(CELLS / f"{name}.groups.csv.gz", index=False, compression=gz)
        t_end = time.perf_counter()
        res.update(seconds_online=t_online - t_repro, seconds_metrics=t_end - t_online,
                   seconds_total=t_end - t_start, n_trace_rows=int(len(traces)),
                   n_group_rows=int(len(groups)), status="ok")
        write_json(CELLS / f"{name}.done.json",
                   {**res, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return res
    except Exception as e:  # noqa: BLE001
        res.update(status="error", error=repr(e), seconds_total=time.perf_counter() - t_start)
        if not reproduce_only:
            write_json(CELLS / f"{name}.error.json", res)
        return res


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


TIMING_COLS = ["finished_utc", "mode", "tag", "regime", "seed", "status", "seconds_total",
               "seconds_fit", "seconds_reproduction", "seconds_online", "seconds_metrics",
               "n_test", "n_traces", "pid", "started_utc"]
REPRO_COLS = ["run_utc", "tag", "regime", "seed", "status", "repro_passed", "repro_n_pairs",
              "repro_n_matched", "repro_n_unpaired", "repro_n_methods", "repro_n_traces",
              "repro_max_abs_diff_coverage", "repro_max_abs_diff_frac_inf",
              "d0_equals_immediate", "seconds_total", "error"]


def collect(tag: str) -> dict | None:
    """Concatenate the per-cell files of a tag once every cell of its design is done."""
    cfg = TAGS[tag]
    names = [cell_name(tag, r, s) for r in cfg["regimes"] for s in cfg["seeds"]]
    if not all((CELLS / f"{n}.done.json").exists() for n in names):
        n_done = sum((CELLS / f"{n}.done.json").exists() for n in names)
        print(f"{tag}: {n_done}/{len(names)} cells done; combined files not written")
        return None
    gz = {"method": "gzip", "mtime": 0}
    traces = pd.concat([pd.read_csv(CELLS / f"{n}.traces.csv.gz") for n in names],
                       ignore_index=True)
    traces.to_csv(OUT / f"{tag}_traces.csv.gz", index=False, compression=gz)
    files = {f"{tag}_traces.csv.gz": sha256(OUT / f"{tag}_traces.csv.gz")}
    if cfg["groups"]:
        groups = pd.concat([pd.read_csv(CELLS / f"{n}.groups.csv.gz") for n in names],
                           ignore_index=True)
        groups.to_csv(OUT / f"{tag}_groups.csv.gz", index=False, compression=gz)
        files[f"{tag}_groups.csv.gz"] = sha256(OUT / f"{tag}_groups.csv.gz")
    print(f"{tag}: combined {len(names)} cells")
    return files


def run_config(tag: str, files: dict | None) -> dict:
    import joblib
    import scipy
    import sklearn
    cfg = TAGS[tag]
    return {
        "tag": tag, "amendment": "protocol/amendments.md, amendment 1",
        "amendment_sha256": sha256(PROTO_DIR / "amendments.md"),
        "protocol_sha256": sha256(PROTO_DIR / "protocol.json"),
        "features": cfg["features"].name, "features_sha256": sha256(cfg["features"]),
        "locked_rows_sha256": sha256(LOCKED) if cfg["locked_tag"] else None,
        "regimes": cfg["regimes"], "seeds": cfg["seeds"], "horizon": cfg["horizon"],
        "delays_s": cfg["delays"], "alpha": ALPHA, "gamma": GAMMA, "base_model": BASE_MODEL,
        "reproduction_tolerance": TOL, "outputs": files,
        "environment": {"machine": platform.node(), "processor": platform.processor(),
                        "platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                        "joblib": joblib.__version__,
                        "thread_env": {v: os.environ.get(v) for v in (
                            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS")}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, choices=sorted(TAGS))
    ap.add_argument("--regimes", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--n-jobs", type=int, default=PROTO["compute_budget"]["n_jobs_max"])
    ap.add_argument("--reproduce-only", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    digest = sha256(PROTO_DIR / "protocol.json")
    frozen = json.loads((PROTO_DIR / "protocol_hashes.json").read_text())["protocol.json"]
    assert digest == frozen, "protocol.json differs from its frozen hash"
    cfg = TAGS[args.tag]
    if args.reproduce_only and cfg["locked_tag"] is None:
        ap.error(f"tag {args.tag} has no locked rows to reproduce")
    regimes = args.regimes or cfg["regimes"]
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    unknown = (set(regimes) - set(cfg["regimes"])) | (set(seeds) - set(cfg["seeds"]))
    if unknown:
        ap.error(f"not in the design of tag {args.tag}: {sorted(map(str, unknown))}")
    if not cfg["features"].exists():
        ap.error(f"{cfg['features']} is missing; run scripts/prepare_timestamps.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.reproduce_only:
        CELLS.mkdir(exist_ok=True)
        unresolved = [p.name for p in sorted(CELLS.glob(f"{args.tag}__*.failed.json"))
                      if not p.with_name(p.name.replace(".failed.json", ".done.json")).exists()]
        if unresolved and not args.retry_failed:
            ap.error(f"reproduction criterion failed ({', '.join(unresolved)}); execution "
                     "stays stopped until the cause is identified, then use --retry-failed")
    cells = sorted(((r, s) for r in regimes for s in seeds),
                   key=lambda c: (REGIME_ORDER.index(c[0]), c[1]))
    if not args.reproduce_only:
        def pending(r, s):
            n = cell_name(args.tag, r, s)
            if (CELLS / f"{n}.done.json").exists():
                return False
            return args.retry_failed or not (CELLS / f"{n}.failed.json").exists()
        cells = [c for c in cells if pending(*c)]

    locked = None
    if cfg["locked_tag"] is not None:
        cols = ["tag", "regime", "seed", "alpha", "gamma", "base_model", "method",
                "trace_id", "n", "coverage", "frac_inf"]
        lk = read_cell_rows(LOCKED, usecols=cols)
        locked = lk[(lk.tag == cfg["locked_tag"]) & (lk.base_model == BASE_MODEL)
                    & np.isclose(lk.alpha, ALPHA) & np.isclose(lk.gamma, GAMMA)]
    mode = "reproduce-only" if args.reproduce_only else "full"
    print(f"tag {args.tag} ({mode}): {len(cells)} cells to run, n_jobs={args.n_jobs}",
          flush=True)

    def locked_slice(r, s):
        if locked is None:
            return None
        return locked[(locked.regime == r) & (locked.seed == s)].reset_index(drop=True)

    t0 = time.perf_counter()
    results = []
    stopped = None
    jobs = (delayed(run_one)(args.tag, r, s, locked_slice(r, s), args.reproduce_only)
            for r, s in cells)
    outputs = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs)
    for res in outputs:
        res["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results.append(res)
        append_csv(TIMING_LOG, [res], TIMING_COLS)
        line = (f"  {res['status']:6s} {res['tag']}/{res['regime']}/s{res['seed']}: "
                f"{res.get('seconds_total', float('nan')):.1f}s")
        if "repro_passed" in res:
            line += (f"  reproduction {res['repro_n_matched']}/{res['repro_n_pairs']}"
                     f" (method, trace) pairs matched, d0_equals_immediate="
                     f"{res.get('d0_equals_immediate')}")
        if res["status"] == "error":
            line += f"  {res['error']}"
        print(line, flush=True)
        if res["status"] == "failed" and not args.reproduce_only:
            stopped = res
            break
    if stopped is not None:
        outputs.close()
    wall = time.perf_counter() - t0
    print(f"wall-clock {wall / 60:.1f} min; status counts "
          f"{pd.Series([r['status'] for r in results]).value_counts().to_dict()}")

    if stopped is not None:
        name = cell_name(stopped["tag"], stopped["regime"], stopped["seed"])
        print(f"reproduction criterion failed in {name} (cells/{name}.failed.json); "
              "execution stopped and the remaining cells were cancelled", flush=True)
        sys.exit(1)
    if args.reproduce_only:
        REPRO_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_csv(REPRO_DIR / "reproduction_log.csv",
                   [{**r, "run_utc": stamp} for r in results], REPRO_COLS)
        return
    files = collect(args.tag)
    if files:
        write_json(OUT / f"run_config_{args.tag}.json", run_config(args.tag, files))


if __name__ == "__main__":
    main()
