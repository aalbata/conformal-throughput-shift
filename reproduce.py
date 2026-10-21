"""Reproduce the summaries, tables and figures of the article.

Fast path (default): from the shipped result files only. Checks the shipped inputs against
SHA256SUMS, regenerates results/summary/ (scripts/analyze_locked.py,
scripts/analyze_amendment.py --stored-audit, scripts/make_tables.py) and the shipped files
of figures/ (scripts/make_figures.py), and compares every regenerated file with its SHA-256
in SHA256SUMS, printing PASS or FAIL per file. Three files of results/summary/ need the
processed data and are inputs of the fast path: audit.json (data audit and ladder
bitrates) and pool_sizes.csv and pool_sizes.json (calibration pool sizes).

Full path (--full): from the public datasets, in a working folder (default full_run/):
downloads and verifies the datasets (scripts/download_data.py), rebuilds the feature
tables (scripts/prepare_data.py, scripts/prepare_timestamps.py) and the pool sizes
(scripts/pool_sizes.py), reruns the pilot (scripts/run_pilot.py), the locked runs
(scripts/run_locked.py), the runs of protocol amendment 1 (scripts/run_amendment.py, tags
main, h5 and ts), the ladder (scripts/fetch_ladder.py) and the streaming simulation
(scripts/run_has.py), and recomputes audit.json (scripts/analyze_amendment.py). The
regenerated result files are compared with the shipped ones: the per-trace row files row by
row (the identifier column is compared under its name trace_id, and the gzip header is
ignored), audit.json, pool_sizes.csv, pool_sizes.json and pilot_meta.json byte by byte.
The fast path then runs in this folder. The run scripts skip finished cells, so an
interrupted full path resumes when the same command is run again.

Usage:
  python reproduce.py
  python reproduce.py --full [--workdir DIR] [--n-jobs N] [--offline]
Exit code 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMS = ROOT / "SHA256SUMS"
STORED = ("results/summary/audit.json", "results/summary/pool_sizes.csv",
          "results/summary/pool_sizes.json")
FAST_STEPS = [
    ("locked analysis", ["scripts/analyze_locked.py"]),
    ("amendment analysis", ["scripts/analyze_amendment.py", "--stored-audit"]),
    ("tables and number macros", ["scripts/make_tables.py"]),
]
ROW_FILES = ["results/pilot/pilot_rows.csv.gz", "results/locked/locked_rows.csv.gz",
             "results/amendment1/main_traces.csv.gz", "results/amendment1/main_groups.csv.gz",
             "results/amendment1/h5_traces.csv.gz", "results/amendment1/h5_groups.csv.gz",
             "results/amendment1/ts_traces.csv.gz", "results/amendment1/has/has_rows.csv.gz"]
EXACT_FILES = ["results/pilot/pilot_meta.json", *STORED]
DATA_ARCHIVES = ["data/raw/LTE_Dataset.zip", "data/raw/5G-production-dataset.zip",
                 "data/external/4K_segment_distribution.tar"]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
REL_TOL = 1e-12


class StepFailed(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> dict:
    """{relative path: SHA-256} from a file of '<sha256>  <relative path>' lines."""
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split(None, 1)
            sums[rel.strip()] = digest.lower()
    return sums


def regenerated(rel: str) -> bool:
    """True for the files that the fast path writes."""
    return rel.startswith("figures/") or (rel.startswith("results/summary/")
                                          and rel not in STORED)


def child_env(threads: str | None = None) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    if threads is not None:
        env.update({v: threads for v in THREAD_VARS})
    return env


def run_step(name: str, args: list, cwd: Path, env: dict | None = None) -> float:
    """Run python <args> in cwd; returns the wall-clock seconds."""
    print(f"\n== {name}: python {' '.join(args)}", flush=True)
    t0 = time.perf_counter()
    code = subprocess.run([sys.executable, *args], cwd=cwd, env=env or child_env()).returncode
    dt = time.perf_counter() - t0
    if code != 0:
        raise StepFailed(f"{name} stopped with exit code {code} after {dt / 60:.1f} min")
    print(f"== {name}: done in {dt / 60:.1f} min", flush=True)
    return dt


def fast_path(root: Path = ROOT) -> bool:
    """Regenerate results/summary/ and figures/ and compare them with SHA256SUMS."""
    sums = read_sums(root / "SHA256SUMS")
    inputs = sorted(r for r in sums if not regenerated(r))
    targets = sorted(r for r in sums if regenerated(r))
    print(f"\nFast path in {root}\n{len(inputs)} input files and {len(targets)} regenerated "
          "files listed in SHA256SUMS")
    bad_inputs = [r for r in inputs if not (root / r).is_file() or sha256(root / r) != sums[r]]
    for rel in bad_inputs:
        print(f"FAIL  input {rel} is missing or differs from SHA256SUMS")
    print(f"inputs: {len(inputs) - len(bad_inputs)} of {len(inputs)} match SHA256SUMS")

    backup = Path(tempfile.mkdtemp(prefix="shipped_outputs_"))
    for rel in targets:
        p = root / rel
        if p.is_file():
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, backup / rel)
            p.unlink()
    step_error = None
    t0 = time.perf_counter()
    try:
        for name, args in FAST_STEPS:
            run_step(name, args, root)
        with tempfile.TemporaryDirectory(prefix="figures_") as tmp:
            run_step("figures", ["scripts/make_figures.py", "--out", tmp], root)
            for rel in targets:
                src = Path(tmp) / rel[len("figures/"):] if rel.startswith("figures/") else None
                if src is not None and src.is_file():
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, root / rel)
    except StepFailed as e:
        step_error = str(e)
        print(f"\n{step_error}", flush=True)
    minutes = (time.perf_counter() - t0) / 60

    print("\nRegenerated files against SHA256SUMS:")
    n_fail = 0
    for rel in targets:
        p = root / rel
        if p.is_file():
            ok = sha256(p) == sums[rel]
            note = "" if ok else "  (differs; shipped copy in " + str(backup / rel) + ")"
        else:
            ok = False
            note = "  (not regenerated; shipped copy restored)"
            if (backup / rel).is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup / rel, p)
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {rel}{note}")
    if n_fail == 0:
        shutil.rmtree(backup, ignore_errors=True)
    passed = n_fail == 0 and not bad_inputs and step_error is None
    print(f"\nFast path: {len(targets) - n_fail} of {len(targets)} regenerated files identical "
          f"to SHA256SUMS; {len(inputs) - len(bad_inputs)} of {len(inputs)} inputs intact; "
          f"{minutes:.1f} min. {'PASS' if passed else 'FAIL'}")
    return passed


def _id_name(col: str) -> str:
    return "trace_id" if col.endswith("_id") else col


def compare_rows(new: Path, shipped: Path) -> tuple[bool, str]:
    """Compare two gzip CSV row files row by row (gzip header and the name of the
    identifier column ignored); values are compared when the text differs."""
    a = gzip.decompress(new.read_bytes()).decode("utf-8").splitlines()
    b = gzip.decompress(shipped.read_bytes()).decode("utf-8").splitlines()
    ha = [_id_name(c) for c in a[0].split(",")] if a else []
    hb = [_id_name(c) for c in b[0].split(",")] if b else []
    if ha != hb:
        return False, f"columns differ: {ha} versus {hb}"
    if len(a) != len(b):
        return False, f"{len(a) - 1} rows versus {len(b) - 1} shipped"
    if a[1:] == b[1:]:
        return True, f"{len(a) - 1} rows, identical"
    import numpy as np
    import pandas as pd
    da = pd.read_csv(new, float_precision="round_trip", dtype=str)
    db = pd.read_csv(shipped, float_precision="round_trip", dtype=str)
    da.columns, db.columns = ha, hb
    worst = 0.0
    for col in ha:
        x, y = da[col], db[col]
        if x.equals(y):
            continue
        xf, yf = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        text_rows = xf.isna() & x.notna() | yf.isna() & y.notna()
        if text_rows.any() and not x[text_rows].equals(y[text_rows]):
            return False, f"column {col} differs"
        xv, yv = xf.to_numpy(dtype=float), yf.to_numpy(dtype=float)
        if not np.array_equal(np.isnan(xv), np.isnan(yv)):
            return False, f"column {col}: missing values differ"
        m = ~np.isnan(xv) & (xv != yv)
        if m.any():
            rel = np.abs(xv[m] - yv[m]) / np.maximum(np.abs(xv[m]), np.abs(yv[m]))
            worst = max(worst, float(rel.max()))
    if worst > REL_TOL:
        return False, f"largest relative difference {worst:.3g}"
    return True, f"{len(a) - 1} rows, equal values (largest relative difference {worst:.3g})"


def full_path(work: Path, n_jobs: int, offline: bool) -> bool:
    """Rerun everything from the public datasets in work, compare, then the fast path."""
    work.mkdir(parents=True, exist_ok=True)
    print(f"Full path in {work}")
    for d in ("src", "scripts", "protocol"):
        shutil.copytree(ROOT / d, work / d, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (work / "data" / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "data" / "raw" / "download_manifest.json",
                 work / "data" / "raw" / "download_manifest.json")
    (work / "audit").mkdir(exist_ok=True)
    if offline:
        missing = [a for a in DATA_ARCHIVES if not (work / a).is_file()]
        if missing:
            print(f"--offline needs {', '.join(str(work / a) for a in missing)}")
            return False
    jobs = ["--n-jobs", str(n_jobs)]
    steps = [
        ("datasets", ["scripts/download_data.py"] + (["--offline"] if offline else []), None),
        ("features h = 1", ["scripts/prepare_data.py", "--horizon", "1"], None),
        ("features h = 5", ["scripts/prepare_data.py", "--horizon", "5"], None),
        ("timestamped features", ["scripts/prepare_timestamps.py"], None),
        ("pool sizes", ["scripts/pool_sizes.py"], None),
        ("pilot", ["scripts/run_pilot.py"], child_env(threads="4")),
        ("locked runs", ["scripts/run_locked.py"], None),
        ("amendment runs, main", ["scripts/run_amendment.py", "--tag", "main", *jobs], None),
        ("amendment runs, h5", ["scripts/run_amendment.py", "--tag", "h5", *jobs], None),
        ("amendment runs, ts", ["scripts/run_amendment.py", "--tag", "ts", *jobs], None),
        ("video ladder", ["scripts/fetch_ladder.py"], None),
        ("streaming simulation", ["scripts/run_has.py", *jobs], None),
        ("data audit", ["scripts/analyze_amendment.py"], None),
    ]
    t0 = time.perf_counter()
    times = []
    try:
        for name, args, env in steps:
            times.append((name, run_step(name, args, work, env)))
    except StepFailed as e:
        print(f"\n{e}; run the same command again to resume")
        return False
    print("\nStep times (min): " + ", ".join(f"{n} {s / 60:.1f}" for n, s in times)
          + f"; total {(time.perf_counter() - t0) / 60:.1f}")

    print("\nRegenerated result files against the shipped files:")
    ok_all = True
    for rel in ROW_FILES + EXACT_FILES:
        new, shipped = work / rel, ROOT / rel
        if not new.is_file():
            ok, msg = False, "not written"
        elif rel in ROW_FILES:
            ok, msg = compare_rows(new, shipped)
        else:
            ok = new.read_bytes() == shipped.read_bytes()
            msg = "identical" if ok else "differs"
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {rel}: {msg}")
    fast_ok = fast_path(ROOT)
    print(f"\nFull path: {'PASS' if ok_all and fast_ok else 'FAIL'}")
    return ok_all and fast_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true",
                    help="rerun everything from the public datasets, then the fast path")
    ap.add_argument("--workdir", type=Path, default=Path("full_run"),
                    help="working folder of the full path (default full_run, relative to "
                         "this folder)")
    ap.add_argument("--n-jobs", type=int, default=6,
                    help="worker processes of run_amendment.py and run_has.py (default 6)")
    ap.add_argument("--offline", action="store_true",
                    help="full path without downloads: the two dataset archives must be in "
                         "<workdir>/data/raw/ and 4K_segment_distribution.tar in "
                         "<workdir>/data/external/")
    args = ap.parse_args()
    print(f"Python {platform.python_version()} on {platform.platform()}")
    if args.full:
        work = args.workdir if args.workdir.is_absolute() else ROOT / args.workdir
        ok = full_path(work, args.n_jobs, args.offline)
    else:
        ok = fast_path(ROOT)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
