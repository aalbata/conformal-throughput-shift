"""Adaptive-streaming simulation of protocol amendment 1.8.

For every streaming cell (six regimes x seeds 0-19; gradient boosting, alpha = 0.10,
gamma = 0.05, h = 1, D = 0) the per-instance predictions and intervals come from
src.experiments.cell_instances. Every eligible test trace (all 4G traces and the 5G
Download traces) is replayed as a capacity trace with each variant in src.has.VARIANTS,
and the per-trace metrics of a cell are written to
results/amendment1/has/cells/<regime>__s<seed>.csv. Cells already written are skipped.

Usage: python scripts/run_has.py [--regimes R ...] [--seeds S ...] [--n-jobs 6]
Requires the raw traces in data/raw/, the timestamped feature table from
scripts/prepare_timestamps.py and the ladder from scripts/fetch_ladder.py.

Alignment: the instance at row t of a trace is issued once row t has been observed and
predicts the throughput of row t + 1. Simulation second s therefore carries the
capacity of row s + 1 and the estimates of the instance at row s. Playback starts at
the row of the first instance of the trace and ends with its last row. Missing
throughput values in a trace are replayed as zero capacity and counted per trace.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.has import (BUFFER_S, FLOOR_MBPS, HORIZON, LATENCY_S, METRIC_KEYS,  # noqa: E402
                     VARIANTS, WINDOW, Ladder, load_ladder, simulate)

REGIMES = ["in_domain_4g", "in_domain_5g", "cross_mobility_4g", "cross_mobility_5g",
           "cross_tech_4to5", "cross_tech_5to4"]
SEEDS = list(range(20))
ALPHA = 0.1
GAMMA = 0.05
BASE_MODEL = "gbm"
DELAY_S = 0.0
ESTIMATE_COLUMNS = {"point": "pred", "split": "lo_split", "aci": "lo_aci",
                    "mondrian_aci": "lo_mondrian_aci"}
REQUIRED_COLUMNS = ["trace_id", "t_index", "y", *ESTIMATE_COLUMNS.values()]
OUTPUT_COLUMNS = ["regime", "seed", "trace_id", "tech", "mobility", "variant", "rule",
                  "estimator", "trace_rows", "start_second", "capacity_missing",
                  *METRIC_KEYS]

OUT = ROOT / "results" / "amendment1" / "has"
FEATURES = ROOT / "data" / "processed" / "features_h1_timed.csv.gz"
LADDER_CSV = ROOT / "data" / "ladder" / "segmentByteCostinColumns-bbb-x264-4sec-main.csv"
LADDER_MANIFEST = ROOT / "data" / "ladder" / "ladder_manifest.json"


def eligible(tech: str, app: str) -> bool:
    """Traces recorded during bulk downloads: all 4G traces and 5G Download traces."""
    return tech == "4G" or app == "Download"


def trace_table(raw: pd.DataFrame) -> dict:
    """Per eligible trace: all rows of dl_mbps in load order, tech and mobility."""
    traces = {}
    for sid, g in raw.groupby("trace_id", sort=False):
        tech, app = g["tech"].iloc[0], g["app"].iloc[0]
        if eligible(tech, app):
            traces[sid] = {"dl": g["dl_mbps"].to_numpy(dtype=float), "tech": tech,
                           "mobility": g["mobility"].iloc[0]}
    return traces


def check_instances(inst: pd.DataFrame) -> None:
    """Contract checks on the output of cell_instances."""
    missing = [c for c in REQUIRED_COLUMNS if c not in inst.columns]
    if missing:
        raise ValueError(f"instance table lacks columns {missing}")
    for sid, g in inst.groupby("trace_id", sort=False):
        t = g["t_index"].to_numpy()
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise ValueError(f"{sid}: instances not in increasing t_index order")


def trace_inputs(g: pd.DataFrame, dl: np.ndarray):
    """Capacity, start second and per-second estimates of one trace.

    g holds the instances of the trace in issue order and dl the throughput of every
    row. Returns (capacity, start_second, estimates, n_missing) where capacity[s] is
    dl[s + 1] with missing values set to 0, estimates[name][s] is the estimate of the
    instance at row s (NaN for rows without an instance) and n_missing counts the
    missing capacity values from start_second onward.
    """
    t = g["t_index"].to_numpy(dtype=np.int64)
    n_rows = len(dl)
    if t[0] < 0 or t[-1] + 1 >= n_rows:
        raise ValueError(f"{g['trace_id'].iloc[0]}: t_index outside the trace")
    if not np.allclose(g["y"].to_numpy(dtype=float), dl[t + 1], rtol=0.0, atol=1e-9):
        raise ValueError(f"{g['trace_id'].iloc[0]}: y differs from the next row's throughput")
    capacity = dl[1:].copy()
    start = int(t[0])
    n_missing = int(np.isnan(capacity[start:]).sum())
    capacity[np.isnan(capacity)] = 0.0
    estimates = {}
    for name, col in ESTIMATE_COLUMNS.items():
        vals = g[col].to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"{g['trace_id'].iloc[0]}: non-finite values in {col}")
        e = np.full(n_rows - 1, np.nan)
        e[t] = vals
        estimates[name] = e
    return capacity, start, estimates, n_missing


def has_cell(inst: pd.DataFrame, traces: dict, ladder: Ladder, regime: str,
             seed: int) -> pd.DataFrame:
    """Simulate every variant on every eligible test trace of one cell."""
    check_instances(inst)
    rows = []
    for sid, g in inst.groupby("trace_id", sort=False):
        if sid not in traces:
            continue
        tr = traces[sid]
        capacity, start, estimates, n_missing = trace_inputs(g, tr["dl"])
        for rule, estimator in VARIANTS:
            m, _ = simulate(capacity, start, estimates, ladder.bitrates_kbps,
                            ladder.segment_bits, rule, estimator, seg_s=ladder.segment_s)
            rows.append({"regime": regime, "seed": seed, "trace_id": sid,
                         "tech": tr["tech"], "mobility": tr["mobility"],
                         "variant": f"{rule}_{estimator}", "rule": rule,
                         "estimator": estimator, "trace_rows": len(tr["dl"]),
                         "start_second": start, "capacity_missing": n_missing,
                         **{k: m[k] for k in METRIC_KEYS}})
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def cell_path(out_dir: Path, regime: str, seed: int) -> Path:
    return out_dir / "cells" / f"{regime}__s{seed}.csv"


def write_combined(out_dir: Path, cells: list) -> tuple[Path, int]:
    """Concatenate the cell files of `cells` into has_rows.csv.gz; returns (path, rows).

    Floats are read back exactly and the gzip header carries no time stamp, so equal
    cell files give an identical combined file.
    """
    rows = pd.concat([pd.read_csv(cell_path(out_dir, r, s), float_precision="round_trip")
                      for r, s in cells], ignore_index=True)
    path = out_dir / "has_rows.csv.gz"
    rows.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path, len(rows)


def _instances_from_pipeline(feat: pd.DataFrame, regime: str, seed: int) -> pd.DataFrame:
    from src.experiments import cell_instances
    return cell_instances(feat, regime, seed, ALPHA, BASE_MODEL, GAMMA, DELAY_S)


def _job(regime: str, seed: int, out_dir: Path, traces: dict, ladder: Ladder,
         instance_fn, instance_arg):
    t0 = time.time()
    inst = instance_fn(instance_arg, regime, seed)
    df = has_cell(inst, traces, ladder, regime, seed)
    path = cell_path(out_dir, regime, seed)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return regime, seed, df["trace_id"].nunique() if len(df) else 0, len(df), time.time() - t0


def run_cells(cells: list, out_dir: Path, traces: dict, ladder: Ladder, instance_fn,
              instance_arg, n_jobs: int = 6, log=print) -> list:
    """Run the (regime, seed) cells that have no output file yet.

    instance_fn(instance_arg, regime, seed) returns the instance table of a cell.
    """
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    pending = [(r, s) for r, s in cells if not cell_path(out_dir, r, s).exists()]
    log(f"{len(cells)} cells, {len(pending)} to run ({len(cells) - len(pending)} done), "
        f"n_jobs={n_jobs}")
    if n_jobs > 1 and len(pending) > 1:
        from joblib import Parallel, delayed
        stream = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_job)(r, s, out_dir, traces, ladder, instance_fn, instance_arg)
            for r, s in pending)
    else:
        stream = (_job(r, s, out_dir, traces, ladder, instance_fn, instance_arg)
                  for r, s in pending)
    results = []
    for r, s, n_tr, n_rows, dt in stream:
        log(f"  {r}/seed{s}: {n_tr} traces, {n_rows} rows in {dt:.1f}s")
        results.append((r, s, n_tr, n_rows, dt))
    return results


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_config(ladder: Ladder, ladder_sha: str, n_jobs: int) -> dict:
    """Design, player parameters and provenance of a launch."""
    import joblib
    import sklearn
    return {
        "design": {"regimes": REGIMES, "seeds": SEEDS, "alpha": ALPHA, "gamma": GAMMA,
                   "base_model": BASE_MODEL, "delay_s": DELAY_S, "horizon_rows": 1,
                   "variants": [f"{r}_{e}" for r, e in VARIANTS],
                   "estimate_columns": ESTIMATE_COLUMNS},
        "player": {"buffer_s": BUFFER_S, "segment_s": ladder.segment_s,
                   "latency_s": LATENCY_S, "floor_mbps": FLOOR_MBPS,
                   "mpc_horizon": HORIZON, "hm_window": WINDOW,
                   "mu": float(ladder.bitrates_kbps[-1]) / 1000.0,
                   "ladder_kbps": [int(x) for x in ladder.bitrates_kbps],
                   "n_video_segments": int(ladder.segment_bits.shape[0])},
        "provenance": {"ladder_csv_sha256": ladder_sha,
                       "has_py_sha256": sha256_file(ROOT / "src" / "has.py"),
                       "run_has_py_sha256": sha256_file(Path(__file__).resolve())},
        "environment": {"host": platform.node(), "python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
                        "n_jobs": n_jobs},
    }


def load_features(path: Path) -> pd.DataFrame:
    """Feature table of h = 1 with the timestamp columns ts and ts_label."""
    if not path.exists():
        raise SystemExit(f"{path} missing: run scripts/prepare_timestamps.py first")
    from src.data_loading import read_feature_table
    feat = read_feature_table(path)
    missing = [c for c in ("ts", "ts_label") if c not in feat.columns]
    if missing:
        raise SystemExit(f"{path} lacks columns {missing}")
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--features", type=Path, default=FEATURES)
    args = ap.parse_args()
    if not set(args.seeds) <= set(SEEDS):
        raise SystemExit(f"seeds must be within {SEEDS[0]}-{SEEDS[-1]}")

    if not LADDER_CSV.exists() or not LADDER_MANIFEST.exists():
        raise SystemExit("ladder missing: run scripts/fetch_ladder.py first")
    ladder_sha = sha256_file(LADDER_CSV)
    expected = json.loads(LADDER_MANIFEST.read_text(encoding="utf-8"))["ladder_csv"]["sha256"]
    if ladder_sha != expected:
        raise SystemExit(f"{LADDER_CSV}: sha256 differs from {LADDER_MANIFEST}")
    ladder = load_ladder(LADDER_CSV)

    OUT.mkdir(parents=True, exist_ok=True)
    config = run_config(ladder, ladder_sha, args.n_jobs)
    config_path = OUT / "run_config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("design", "player", "provenance"):
            if stored[key] != config[key]:
                raise SystemExit(f"{config_path}: '{key}' differs from this launch; "
                                 "outputs would mix configurations")
    else:
        config["first_launch_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        config_path.write_text(json.dumps(config, indent=1), encoding="utf-8")

    from src.data_loading import load_all
    t0 = time.time()
    traces = trace_table(load_all(ROOT / "data" / "raw"))
    feat = load_features(args.features)
    print(f"{len(traces)} eligible traces; features {len(feat)} rows "
          f"({time.time() - t0:.0f}s)")

    cells = [(r, s) for r in args.regimes for s in args.seeds]
    t1 = time.time()
    run_cells(cells, OUT, traces, ladder, _instances_from_pipeline, feat, args.n_jobs)
    print(f"cells finished in {(time.time() - t1) / 60:.1f} min")

    all_cells = [(r, s) for r in REGIMES for s in SEEDS]
    missing = [c for c in all_cells if not cell_path(OUT, *c).exists()]
    if missing:
        print(f"{len(missing)} of {len(all_cells)} cells not yet run; combined file not written")
        return
    path, n_rows = write_combined(OUT, all_cells)
    print(f"wrote {path} with {n_rows} rows")


if __name__ == "__main__":
    main()
