"""Exploratory pilot (NOT confirmatory). Seeds 100-102, disjoint from locked seeds.

Purposes (per protocol): verify the task is nontrivial, baselines are fair,
metrics behave, runtime/memory fit the budget, and select the ACI step size
gamma from {0.01, 0.02, 0.05} on pilot data only.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import pandas as pd  # noqa: E402

from src.data_loading import read_cell_rows, read_feature_table  # noqa: E402
from src.experiments import run_cell  # noqa: E402

OUT = ROOT / "results" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)

feat = read_feature_table(ROOT / "data" / "processed" / "features_h1.csv.gz")

PILOT_SEEDS = [100, 101, 102]
GAMMAS = [0.01, 0.02, 0.05]
REGIMES = ["in_domain_4g", "in_domain_5g", "cross_mobility_5g", "cross_tech_4to5"]

rows = []
t_start = time.time()
for regime in REGIMES:
    for seed in PILOT_SEEDS:
        for gamma in GAMMAS:
            cell_csv = OUT / f"pilot_{regime}_{seed}_{gamma}.csv"
            if cell_csv.exists():
                rows.append(read_cell_rows(cell_csv))
                continue
            t0 = time.time()
            df = run_cell(feat, regime, seed=seed, alpha=0.1, base_model="gbm",
                          gamma=gamma)
            df.to_csv(cell_csv, index=False)
            rows.append(df)
            print(f"{regime} seed{seed} gamma{gamma}: {time.time()-t0:.0f}s")

# baseline sanity: persistence vs gbm point accuracy (one cell each)
for bm in ["persistence", "ridge"]:
    cell_csv = OUT / f"pilot_base_{bm}.csv"
    if not cell_csv.exists():
        df = run_cell(feat, "in_domain_4g", seed=100, alpha=0.1, base_model=bm)
        df.to_csv(cell_csv, index=False)
        print(f"base_model={bm} done")
    rows.append(read_cell_rows(cell_csv))

allr = pd.concat(rows, ignore_index=True)
allr.to_csv(OUT / "pilot_rows.csv.gz", index=False, compression="gzip")

summary = (allr.groupby(["regime", "base_model", "gamma", "method"])
           [["coverage", "marginal_coverage", "median_width", "frac_inf",
             "winkler", "rmse_point"]].mean().round(4))
summary.to_csv(OUT / "pilot_summary.csv")
print(summary.to_string())
print(f"total pilot wall-clock: {(time.time()-t_start)/60:.1f} min")
(OUT / "pilot_meta.json").write_text(json.dumps(
    {"pilot_seeds": PILOT_SEEDS, "gammas": GAMMAS, "regimes": REGIMES,
     "date": "2026-08-28",
     "note": "exploratory only; never reported as confirmatory"}, indent=1))
