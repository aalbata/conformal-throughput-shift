"""Calibration score-pool sizes of the speed groups used by Mondrian and Mondrian-ACI.

For every regime and seed of the main configuration, the calibration traces are those of
src.experiments.split_traces; the number of calibration instances in each speed bin is the
size of that group's score pool. Groups with fewer than MIN_GROUP_N scores use the pooled
distribution (src.conformal.MondrianConformal and MondrianACI).

Output: results/summary/pool_sizes.csv (one row per regime, seed and speed group) and
results/summary/pool_sizes.json (per regime and group: minimum, median and maximum pool size
over seeds and the number of seeds in which the pooled fallback is used).

Usage: python scripts/pool_sizes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import SPEED_LABELS, read_features, regime_pools, speed_groups, split_traces  # noqa: E402

FEATURES = ROOT / "data" / "processed" / "features_h1.csv.gz"
OUT = ROOT / "results" / "summary"
REGIMES = ["in_domain_4g", "in_domain_5g", "cross_mobility_4g", "cross_mobility_5g",
           "cross_tech_4to5", "cross_tech_5to4", "cross_app_5g"]
SEEDS = range(20)
MIN_GROUP_N = 50


def pool_sizes(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        source, target = regime_pools(feat, regime)
        for seed in SEEDS:
            _, cal, _ = split_traces(source, target, seed)
            groups = pd.Series(speed_groups(cal))
            counts = groups.value_counts()
            for g in SPEED_LABELS:
                n = int(counts.get(g, 0))
                rows.append({"regime": regime, "seed": seed, "group": g, "n_pool": n,
                             "n_pooled": int(len(cal)), "fallback": n < MIN_GROUP_N})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for (regime, g), d in df.groupby(["regime", "group"], sort=False):
        out[f"{regime}.{g}"] = {
            "n_pool_min": int(d.n_pool.min()),
            "n_pool_median": float(np.median(d.n_pool)),
            "n_pool_max": int(d.n_pool.max()),
            "n_seeds_fallback": int(d.fallback.sum()),
            "n_pooled_min": int(d.n_pooled.min()),
        }
    return out


def main() -> None:
    feat = read_features(str(FEATURES))
    df = pool_sizes(feat)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pool_sizes.csv", index=False)
    with open(OUT / "pool_sizes.json", "w", encoding="utf-8") as f:
        json.dump(summarise(df), f, indent=1, sort_keys=True)
    print(f"wrote {len(df)} rows")


if __name__ == "__main__":
    main()
