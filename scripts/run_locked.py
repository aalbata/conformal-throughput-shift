"""Locked (confirmatory) experiment runner. Executes exactly the pre-registered design
in protocol/protocol.json. Refuses to run if gamma_selected is null or the protocol
hash file is missing. Per-cell result files in results/locked/cells/ allow resume;
nothing is deleted or overwritten.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.data_loading import read_cell_rows, read_feature_table  # noqa: E402
from src.experiments import run_cell  # noqa: E402

PROTO = json.loads((ROOT / "protocol" / "protocol.json").read_text(encoding="utf-8"))
HASH_FILE = ROOT / "protocol" / "protocol_hashes.json"
assert PROTO["gamma_selected"] is not None, "protocol gamma not filled - freeze first"
assert HASH_FILE.exists(), "protocol not hashed - freeze first"

# verify the protocol on disk matches the frozen hash
digest = hashlib.sha256((ROOT / "protocol" / "protocol.json").read_bytes()).hexdigest()
frozen = json.loads(HASH_FILE.read_text())["protocol.json"]
assert digest == frozen, "protocol.json changed after freeze!"

GAMMA = PROTO["gamma_selected"]
ALPHA = PROTO["alpha_primary"]
OUT = ROOT / "results" / "locked"
CELLS = OUT / "cells"
CELLS.mkdir(parents=True, exist_ok=True)
FAILED_LEDGER = ROOT / "audit" / "failed_runs.md"

feat_h1 = ROOT / "data" / "processed" / "features_h1.csv.gz"
feat_h5 = ROOT / "data" / "processed" / "features_h5.csv.gz"

# ---- cell list per protocol ----
cells = []  # (tag, feat_path, regime, base_model, seed, alpha, gamma)
for r in PROTO["regimes"]:
    for s in PROTO["locked_seeds"]:
        cells.append(("main", feat_h1, r, "gbm", s, ALPHA, GAMMA))
for bm in PROTO["base_model_ablations"]:
    for r in ["in_domain_4g", "cross_mobility_5g", "cross_tech_4to5"]:
        for s in PROTO["ablation_seeds"]:
            cells.append(("abl_base", feat_h1, r, bm, s, ALPHA, GAMMA))
for r in PROTO["regimes"]:
    for s in PROTO["ablation_seeds"]:
        cells.append(("abl_alpha05", feat_h1, r, "gbm", s, 0.05, GAMMA))
for g in [x for x in PROTO["gamma_grid"] if x != GAMMA]:
    for r in PROTO["shifted_regimes"]:
        for s in PROTO["ablation_seeds"]:
            cells.append(("abl_gamma", feat_h1, r, "gbm", s, ALPHA, g))
for r in ["in_domain_5g", "cross_tech_4to5"]:
    for s in PROTO["ablation_seeds"]:
        cells.append(("abl_h5", feat_h5, r, "gbm", s, ALPHA, GAMMA))


def cell_path(tag, regime, bm, seed, alpha, gamma):
    return CELLS / f"{tag}__{regime}__{bm}__s{seed}__a{alpha}__g{gamma}.csv"


def one(tag, fp, regime, bm, seed, alpha, gamma):
    out = cell_path(tag, regime, bm, seed, alpha, gamma)
    if out.exists():
        return ("cached", tag, regime, bm, seed, 0.0)
    try:
        t0 = time.time()
        feat = read_feature_table(fp)
        df = run_cell(feat, regime, seed, alpha, bm, gamma=gamma)
        df.insert(0, "tag", tag)
        df.to_csv(out, index=False)
        return ("ok", tag, regime, bm, seed, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        with open(FAILED_LEDGER, "a", encoding="utf-8") as f:
            f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} LOCKED FAIL "
                    f"{tag}/{regime}/{bm}/s{seed}/a{alpha}/g{gamma}: {e!r}\n")
        return ("fail", tag, regime, bm, seed, 0.0)


if __name__ == "__main__":
    n_jobs = min(PROTO["compute_budget"]["n_jobs_max"], max(1, os.cpu_count() - 2))
    pending = [c for c in cells if not cell_path(c[0], c[2], c[3], c[4], c[5], c[6]).exists()]
    print(f"{len(cells)} cells total, {len(pending)} to run, n_jobs={n_jobs}")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one)(*c) for c in pending)
    n_fail = sum(1 for r in results if r[0] == "fail")
    print(f"done in {(time.time()-t0)/60:.1f} min; {n_fail} failures (see ledger)")
    parts = []
    for c in cells:
        p = cell_path(c[0], c[2], c[3], c[4], c[5], c[6])
        if p.exists():
            parts.append(read_cell_rows(p))
    allr = pd.concat(parts, ignore_index=True)
    allr.to_csv(OUT / "locked_rows.csv.gz", index=False, compression="gzip")
    print(f"wrote {OUT/'locked_rows.csv.gz'} with {len(allr)} rows")
