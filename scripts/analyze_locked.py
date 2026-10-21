"""Confirmatory analysis of locked results, exactly per protocol/protocol.json section 7.

Reads results/locked/locked_rows.csv.gz (read only), writes to results/summary/locked/:
  summary_main.csv          per (regime, method) aggregates with CIs
  tests_confirmatory.csv    5 pre-registered tests (BH family 1)
  tests_secondary.csv       15 secondary tests (BH family 2)
  group_coverage.csv        per-speed-group coverage per regime/method
  downstream.csv            rate-selection metrics
  ablations_*.csv           ablation aggregates
  analysis_provenance.json  hashes + row counts feeding every table
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loading import read_cell_rows  # noqa: E402
from src.stats import bh_correct, bootstrap_ci_mean, paired_wilcoxon  # noqa: E402

PROTO = json.loads((ROOT / "protocol" / "protocol.json").read_text(encoding="utf-8"))
ALPHA = PROTO["alpha_primary"]
TAB = ROOT / "results" / "summary" / "locked"
TAB.mkdir(parents=True, exist_ok=True)
LOCKED = ROOT / "results" / "locked" / "locked_rows.csv.gz"


def protocol_value(suffix: str):
    """Value of the one protocol.json key that ends in suffix."""
    (value,) = (v for k, v in PROTO.items() if k.endswith(suffix))
    return value


MIN_TEST_INSTANCES = protocol_value("_min_test_instances")  # per trace

rows = read_cell_rows(LOCKED)
rows = rows[rows.n >= MIN_TEST_INSTANCES]

main = rows[(rows.tag == "main")]


def seed_avg_trace(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average metric over seeds per (regime, method, trace): trace = paired unit."""
    return (df.groupby(["regime", "method", "trace_id"])[metric]
            .mean().reset_index())


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (regime, method), g in df.groupby(["regime", "method"]):
        # marginal coverage: one value per seed -> bootstrap over seeds
        mseed = g.groupby("seed")["marginal_coverage"].first()
        m_mean, m_lo, m_hi = bootstrap_ci_mean(mseed.to_numpy())
        per_trace = g.groupby("trace_id")["coverage"].mean()
        t_mean, t_lo, t_hi = bootstrap_ci_mean(per_trace.to_numpy())
        dev = (g.groupby("trace_id")["coverage"].mean() - (1 - ALPHA)).abs()
        d_mean, d_lo, d_hi = bootstrap_ci_mean(dev.to_numpy())
        w = g.groupby("trace_id")["median_width"].mean()
        w_med = float(np.median(w))
        out.append({
            "regime": regime, "method": method,
            "n_traces": int(g.trace_id.nunique()), "n_seeds": int(g.seed.nunique()),
            "marginal_coverage": round(m_mean, 4), "marg_lo": round(m_lo, 4),
            "marg_hi": round(m_hi, 4),
            "trace_coverage": round(t_mean, 4), "trace_lo": round(t_lo, 4),
            "trace_hi": round(t_hi, 4),
            "abs_dev": round(d_mean, 4), "dev_lo": round(d_lo, 4), "dev_hi": round(d_hi, 4),
            "median_width": round(w_med, 3),
            "frac_inf": round(float(g.frac_inf.mean()), 5),
            "winkler_median": round(float(np.median(g.winkler)), 3),
            "rmse_point": round(float(g.rmse_point.mean()), 3),
        })
    return pd.DataFrame(out)


summary = summarize(main)
summary.to_csv(TAB / "summary_main.csv", index=False)

# ---- confirmatory family: mondrian_aci vs split, per shifted regime ----
dev = seed_avg_trace(main, "coverage")
dev["absdev"] = (dev.coverage - (1 - ALPHA)).abs()


def run_family(comparisons):
    recs = []
    for regime, m1, m2 in comparisons:
        a = dev[(dev.regime == regime) & (dev.method == m1)].set_index("trace_id")["absdev"]
        b = dev[(dev.regime == regime) & (dev.method == m2)].set_index("trace_id")["absdev"]
        common = a.index.intersection(b.index)
        res = paired_wilcoxon(a.loc[common].to_numpy(), b.loc[common].to_numpy())
        recs.append({"regime": regime, "method_a": m1, "method_b": m2, **res})
    df = pd.DataFrame(recs)
    rej, p_adj = bh_correct(df["p"].tolist(), PROTO["statistical_alpha"])
    df["p_bh"] = p_adj
    df["significant_bh"] = rej
    return df


conf = run_family([(r, "mondrian_aci", "split") for r in PROTO["shifted_regimes"]])
conf.to_csv(TAB / "tests_confirmatory.csv", index=False)

sec = run_family([(r, "mondrian_aci", m) for r in PROTO["shifted_regimes"]
                  for m in ("aci", "mondrian", "cqr")])
sec.to_csv(TAB / "tests_secondary.csv", index=False)

# ---- per-speed-group coverage (pooled over seeds within regime/method) ----
grecs = []
for (regime, method), g in main.groupby(["regime", "method"]):
    agg: dict[str, list] = {}
    for js, n in zip(g.group_coverage_json, g.n):
        for k, v in json.loads(js).items():
            agg.setdefault(k, []).append(v)
    for k, vals in agg.items():
        grecs.append({"regime": regime, "method": method, "group": k,
                      "coverage_mean": round(float(np.mean(vals)), 4),
                      "n_cells": len(vals)})
pd.DataFrame(grecs).to_csv(TAB / "group_coverage.csv", index=False)

# ---- downstream rate selection ----
ds = []
for (regime, method), g in main.groupby(["regime", "method"]):
    per_trace = g.groupby("trace_id")[["overshoot_lb", "util_lb",
                                       "overshoot_point", "util_point"]].mean()
    for col in per_trace.columns:
        m, lo, hi = bootstrap_ci_mean(per_trace[col].to_numpy())
        ds.append({"regime": regime, "method": method, "metric": col,
                   "mean": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
pd.DataFrame(ds).to_csv(TAB / "downstream.csv", index=False)

# ---- ablations ----
for tag in ("abl_base", "abl_alpha05", "abl_gamma", "abl_h5"):
    sub = rows[rows.tag == tag]
    if len(sub):
        cols = ["regime", "base_model", "alpha", "gamma", "method"]
        agg = (sub.groupby(cols)[["coverage", "marginal_coverage", "median_width",
                                  "frac_inf"]].mean().round(4).reset_index())
        agg.to_csv(TAB / f"ablations_{tag}.csv", index=False)

prov = {
    "locked_rows_sha256": hashlib.sha256(LOCKED.read_bytes()).hexdigest(),
    "n_rows_locked": int(len(rows)),
    "n_rows_main": int(len(main)),
    "excluded_short_traces_rule": MIN_TEST_INSTANCES,
    "tables": sorted(p.name for p in TAB.glob("*.csv")),
}
(TAB / "analysis_provenance.json").write_text(
    json.dumps(prov, indent=1), encoding="utf-8")
print(json.dumps(prov, indent=1))
print("\nSummary (main):")
print(summary.to_string(index=False))
print("\nConfirmatory tests:")
print(conf.to_string(index=False))
