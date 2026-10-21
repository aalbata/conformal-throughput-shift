"""Summaries of protocol amendment 1 (protocol/amendments.md, sections 1.3 to 1.8) and
full-precision summaries of the locked confirmatory outputs.

Inputs (read only):
  results/amendment1/main_traces.csv.gz, main_groups.csv.gz   main configuration, delay grid
  results/amendment1/h5_traces.csv.gz                          5-s horizon under the delay model
  results/amendment1/ts_traces.csv.gz                          timestamp-based features (1.7)
  results/amendment1/has/has_rows.csv.gz                       streaming simulation (1.8)
  results/amendment1/timing_log.csv                            per-cell compute times
  results/amendment1/batch_timeline.txt                        batch timeline, one
                                                               "<UTC time>  <message>" line
                                                               per event (--timeline PATH
                                                               names another file)
  results/locked/locked_rows.csv.gz                            locked confirmatory rows
  data/processed/features_h1_timed.csv.gz, features_h5_timed.csv.gz, data_audit_h1.json,
  timestamps_audit.json; data/ladder/ (segment sizes of the video ladder). When these data
  files exist, the data audit and the ladder bitrates are computed from them and written to
  audit.json; without them, or with --stored-audit, they are read from
  results/summary/audit.json.

Outputs in results/summary/ (full precision; NaN is an empty CSV field and null in JSON):
  audit.json             data audit and ladder bitrates (written when computed from the data)
  main_methods.csv       7 methods x 7 regimes, main configuration (h = 1, D = 0, seeds 0-19)
  main_groups.csv        speed-group coverage of the main configuration
  delay_grid.csv         aci and mondrian_aci for D in {0, 1, 2, 5, 10} s (and h = 5, D = 0)
  delay_outstanding.csv  distribution of K (maximum number of outstanding labels)
  delay_bound.csv        check of Proposition 1 (ii), per (tag, method, delay, regime)
  horizon5.csv           5-s horizon, delay model (D = 0) beside immediate feedback (locked)
  timestamp.csv          timestamp-based features beside the row-based locked rows, seeds 0-4
  audit_has_eligible.csv streaming-eligible test traces per regime and seed
  has_metrics.csv        streaming metrics per regime, variant and metric
  has_tests.csv          the 12 pre-specified Wilcoxon tests on per-trace QoE
  runtime.csv            compute time per tag (timing log) and per batch (timeline)
  locked_main.csv, locked_tests.csv, locked_groups.csv, locked_downstream.csv,
  locked_ablations.csv   locked confirmatory and ablation summaries at full precision
  summary.json           every scalar of the files above and of the data audit, by name
  digest.md              the summaries at 3 to 4 significant digits

Aggregation (amendments.md 1.4 to 1.8 and the manuscript methodology):
  marginal quantities pool instances within a seed and average the seed values; their 95%
  percentile bootstrap CI resamples the seed values (ordered by seed). Trace quantities
  average each trace over the seeds in which it was tested and then summarise across traces;
  their CI resamples traces (ordered by trace_id). The absolute coverage deviation Delta_s
  is computed from the seed-averaged trace coverage. The infinite-interval fraction is given
  at instance level, as the mean over trace-seed rows (the definition of the locked analysis,
  scripts/analyze_locked.py) and as the mean of seed-averaged per-trace fractions. Bootstrap:
  src.stats.bootstrap_ci_mean (10,000 resamples, seed 0). Immediate feedback at h = 5 comes
  from the locked abl_h5 rows and the row-based reference of the timestamp analysis from the
  locked main rows of seeds 0-4; their finite-interval counts follow exactly from the stored
  coverage and infinite-interval fraction (counts_from_locked).

Usage: python scripts/analyze_amendment.py [--timeline PATH] [--stored-audit] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loading import read_cell_rows, read_feature_table  # noqa: E402
from src.has import VARIANTS, load_ladder  # noqa: E402
from src.stats import bh_correct, bootstrap_ci_mean, paired_wilcoxon  # noqa: E402

PROTO = json.loads((ROOT / "protocol" / "protocol.json").read_text(encoding="utf-8"))
ALPHA = PROTO["alpha_primary"]
GAMMA = PROTO["gamma_selected"]
REGIMES = PROTO["regimes"]
SHIFTED = PROTO["shifted_regimes"]
METHODS = PROTO["methods"]
ONLINE = ["aci", "mondrian_aci"]
DELAYS = [0.0, 1.0, 2.0, 5.0, 10.0]
H5_REGIMES = ["in_domain_5g", "cross_tech_4to5"]
H5_METHODS = ["split", "aci", "mondrian_aci"]
GROUPS = ["speed_0", "speed_low", "speed_med", "speed_high"]
GROUP_LABELS = {"speed_0": "[0,1)", "speed_low": "[1,15)", "speed_med": "[15,50)",
                "speed_high": "[50,inf)"}

HAS_REGIMES = ["in_domain_4g", "in_domain_5g", "cross_mobility_4g", "cross_mobility_5g",
               "cross_tech_4to5", "cross_tech_5to4"]
HAS_VARIANTS = [f"{r}_{e}" for r, e in VARIANTS]
HAS_METRICS = ["avg_bitrate_mbps", "rebuffer_s", "rebuffer_ratio", "stalls", "startup_s",
               "switches_per_min", "mean_abs_switch_mbps", "qoe_per_segment"]
HAS_TEST_REGIMES = ["cross_mobility_4g", "cross_mobility_5g", "cross_tech_4to5",
                    "cross_tech_5to4"]
HAS_TEST_VARIANT = "mpc_mondrian_aci"
HAS_TEST_COMPARATORS = ["mpc_aci", "mpc_split", "mpc_robust"]
HAS_SEEDS = list(range(20))

AMEND = ROOT / "results" / "amendment1"
LOCKED = ROOT / "results" / "locked" / "locked_rows.csv.gz"
PROC = ROOT / "data" / "processed"
LADDER_CSV = ROOT / "data" / "ladder" / "segmentByteCostinColumns-bbb-x264-4sec-main.csv"
OUT = ROOT / "results" / "summary"
TIMELINE = AMEND / "batch_timeline.txt"
AUDIT_JSON = OUT / "audit.json"
AUDIT_INPUTS = (PROC / "features_h1_timed.csv.gz", PROC / "features_h5_timed.csv.gz", LADDER_CSV)


def protocol_value(suffix: str):
    """Value of the one protocol.json key that ends in suffix."""
    (value,) = (v for k, v in PROTO.items() if k.endswith(suffix))
    return value


MIN_TEST_INSTANCES = protocol_value("_min_test_instances")
NAN = float("nan")


# ---------------------------------------------------------------- aggregation primitives

def mean_ci(values) -> tuple[float, float, float]:
    """(mean, lo, hi) of the non-NaN values with the percentile bootstrap of src.stats;
    NaN triple when no value is left. The order of values determines the resamples."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return NAN, NAN, NAN
    return bootstrap_ci_mean(v)


def seed_ratio(df: pd.DataFrame, num: str, den: str) -> pd.Series:
    """sum(num) / sum(den) within each seed, indexed by seed in ascending order
    (NaN for a seed whose denominator is 0)."""
    g = df.groupby("seed")[[num, den]].sum().sort_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (g[num] / g[den].where(g[den] > 0)).astype(float)


def trace_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-trace mean of col over the seeds in which the trace was tested (NaN skipped),
    indexed by trace_id in ascending order."""
    return df.groupby("trace_id")[col].mean().sort_index()


def add_ci(rec: dict, name: str, triple) -> None:
    rec[name], rec[f"{name}_ci_lo"], rec[f"{name}_ci_hi"] = triple


def interval_summary(df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Coverage, infinite-interval, finite-interval, width, Winkler and zero-target summary
    of one (regime, method, variant) from its per-trace rows over seeds.

    df needs seed, trace_id, n, n_cov, n_inf, n_finite, n_cov_finite; width_finite_median,
    width_finite_mean, winkler_finite_mean, n_inverted, n_y0, n_pos and n_cov_pos are used
    when present.
    """
    rec: dict = {"n_seeds": int(df["seed"].nunique()), "n_traces": int(df["trace_id"].nunique()),
                 "n_trace_rows": int(len(df)), "n_instances": int(df["n"].sum())}
    target = 1.0 - alpha
    add_ci(rec, "marginal_coverage", mean_ci(seed_ratio(df, "n_cov", "n")))
    cov = df["n_cov"] / df["n"]
    tc = trace_mean(df.assign(_c=cov), "_c")
    add_ci(rec, "trace_coverage", mean_ci(tc))
    add_ci(rec, "abs_dev", mean_ci((tc - target).abs()))
    add_ci(rec, "frac_inf_instance", mean_ci(seed_ratio(df, "n_inf", "n")))
    finf = df["n_inf"] / df["n"]
    rec["frac_inf_trace_rows"] = float(finf.mean())
    add_ci(rec, "frac_inf_trace", mean_ci(trace_mean(df.assign(_f=finf), "_f")))
    add_ci(rec, "finite_coverage", mean_ci(seed_ratio(df, "n_cov_finite", "n_finite")))
    with np.errstate(invalid="ignore", divide="ignore"):
        cf = (df["n_cov_finite"] / df["n_finite"].where(df["n_finite"] > 0)).astype(float)
    add_ci(rec, "finite_coverage_trace", mean_ci(trace_mean(df.assign(_cf=cf), "_cf")))
    rec["n_trace_rows_without_finite"] = int((df["n_finite"] == 0).sum())
    for col, name in (("width_finite_median", "width_finite_median"),
                      ("width_finite_mean", "width_finite_mean_median")):
        if col in df.columns and df[col].notna().any():
            rec[name] = float(np.nanmedian(trace_mean(df, col)))
        else:
            rec[name] = NAN
    if "winkler_finite_mean" in df.columns and df["winkler_finite_mean"].notna().any():
        wt = trace_mean(df, "winkler_finite_mean")
        rec["winkler_median"] = float(np.nanmedian(wt))
        add_ci(rec, "winkler_mean", mean_ci(wt))
    else:
        rec["winkler_median"] = NAN
        add_ci(rec, "winkler_mean", (NAN, NAN, NAN))
    rec["n_inverted"] = int(df["n_inverted"].sum()) if "n_inverted" in df.columns else -1
    if {"n_pos", "n_cov_pos"} <= set(df.columns) and df["n_pos"].notna().all():
        add_ci(rec, "coverage_pos", mean_ci(seed_ratio(df, "n_cov_pos", "n_pos")))
    else:
        add_ci(rec, "coverage_pos", (NAN, NAN, NAN))
    if "n_y0" in df.columns and df["n_y0"].notna().all():
        rec["zero_share"] = float(seed_ratio(df, "n_y0", "n").mean())
    else:
        rec["zero_share"] = NAN
    return rec


def counts_from_locked(rows: pd.DataFrame) -> pd.DataFrame:
    """Locked per-trace rows with the counts of the amendment tables added.

    n_cov = n * coverage and n_inf = n * frac_inf (integral to rounding error). Every
    infinite interval is [0, inf) and every target is non-negative, so infinite intervals
    cover and the finite intervals hold n_cov - n_inf covered targets out of n - n_inf.
    """
    out = rows.copy()
    n_cov = out["n"] * out["coverage"]
    n_inf = out["n"] * out["frac_inf"]
    for name, x in (("n_cov", n_cov), ("n_inf", n_inf)):
        r = np.round(x)
        if np.abs(x - r).max() > 1e-6:
            raise ValueError(f"{name} is not integral in the locked rows")
        out[name] = r.astype(np.int64)
    return with_finite_counts(out)


def with_finite_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """rows with n_finite = n - n_inf and n_cov_finite = n_cov - n_inf added when absent
    (every infinite interval is [0, inf) and covers its non-negative target)."""
    if {"n_finite", "n_cov_finite"} <= set(rows.columns):
        return rows
    out = rows.copy()
    out["n_finite"] = out["n"] - out["n_inf"]
    out["n_cov_finite"] = out["n_cov"] - out["n_inf"]
    if (out["n_cov_finite"] < 0).any():
        raise ValueError("more infinite intervals than covered targets")
    return out


def select_d0(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows of the configuration used for all seven methods: offline methods (no feedback)
    and the online methods under the delay model with D = 0."""
    keep = (rows["feedback"] == "none") | ((rows["feedback"] == "delayed")
                                           & (rows["delay_s"] == 0.0))
    return rows[keep]


def summarise(rows: pd.DataFrame, keys: list[str], alpha: float = ALPHA) -> pd.DataFrame:
    """interval_summary for every combination of keys present in rows."""
    out = []
    for k, g in rows.groupby(keys, sort=False, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        out.append({**dict(zip(keys, k)), **interval_summary(g, alpha)})
    return pd.DataFrame(out)


def order_frame(df: pd.DataFrame, orders: dict) -> pd.DataFrame:
    """Sort df by the listed columns in the given category orders."""
    df = df.copy()
    sort_cols = []
    for col, order in orders.items():
        key = f"_o_{col}"
        df[key] = df[col].map({v: i for i, v in enumerate(order)})
        sort_cols.append(key)
    return df.sort_values(sort_cols, kind="stable").drop(columns=sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------- speed groups

def group_summary(groups: pd.DataFrame) -> pd.DataFrame:
    """Per (regime, method, group): coverage averaged over seeds with a CI over the seed
    values, and the per-seed test counts."""
    out = []
    for (regime, method, group), g in groups.groupby(["regime", "method", "group"], sort=False):
        per_seed = g.groupby("seed")[["n", "n_cov"]].sum().sort_index()
        rec = {"regime": regime, "method": method, "group": group,
               "n_seeds": int(len(per_seed)),
               "n_test_min": int(per_seed["n"].min()), "n_test_max": int(per_seed["n"].max()),
               "n_test_mean": float(per_seed["n"].mean()), "n_test_total": int(per_seed["n"].sum())}
        add_ci(rec, "coverage", mean_ci(per_seed["n_cov"] / per_seed["n"]))
        out.append(rec)
    return pd.DataFrame(out)


def worst_groups(gs: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Per (regime, method): the group with the lowest seed-averaged coverage, its CI, and
    the largest absolute deviation of a group coverage from 1 - alpha."""
    out = []
    for (regime, method), g in gs.groupby(["regime", "method"], sort=False):
        w = g.loc[g["coverage"].idxmin()]
        out.append({"regime": regime, "method": method, "worst_group": w["group"],
                    "worst_group_coverage": float(w["coverage"]),
                    "worst_group_coverage_ci_lo": float(w["coverage_ci_lo"]),
                    "worst_group_coverage_ci_hi": float(w["coverage_ci_hi"]),
                    "max_group_abs_dev": float((g["coverage"] - (1.0 - alpha)).abs().max()),
                    "min_group_n_test": int(g["n_test_min"].min())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- delayed feedback

def bound_values(k, m, n, alpha: float = ALPHA, gamma: float = GAMMA, alpha_1: float | None = None):
    """Right sides of Proposition 1 (ii) over the applied labels and of its all-interval
    form, for K = k outstanding labels, M = m applied labels and N = n intervals."""
    a1 = alpha if alpha_1 is None else alpha_1
    k, m, n = (np.asarray(x, dtype=float) for x in (k, m, n))
    num = max(a1, 1.0 - a1) + (k + 1.0) * gamma
    with np.errstate(divide="ignore"):
        applied = np.where(m > 0, num / (gamma * np.where(m > 0, m, 1.0)), np.inf)
    all_form = num / (gamma * n) + (k + 1.0) * max(alpha, 1.0 - alpha) / n
    return applied, all_form


def bound_check(df: pd.DataFrame, alpha: float = ALPHA, gamma: float = GAMMA) -> pd.DataFrame:
    """Per trace-seed row: Proposition 1 (ii) on the applied labels and its all-interval form.

    The miss count over all N intervals is E = N - n_cov. The M applied labels miss
    between max(0, E - (N - M)) and min(E, M) times, because the N - M labels that are not
    applied can account for at most N - M misses. The deviation over the applied labels is
    therefore bracketed by dev_applied_min and dev_applied_max; ratio_applied_max <= 1
    proves the bound for the row and ratio_applied_min > 1 proves a violation. The bound
    presumes that no interval issued with alpha_t >= 1 covered (bound_applies).
    """
    k, m, n = df["max_outstanding"], df["n_applied"], df["n"]
    applied, all_form = bound_values(k, m, n, alpha, gamma)
    e_all = n - df["n_cov"]
    e_lo = np.maximum(0, e_all - (n - m))
    e_hi = np.minimum(e_all, m)
    mm = m.where(m > 0)
    r_lo, r_hi = e_lo / mm, e_hi / mm
    dev_max = np.maximum((r_lo - alpha).abs(), (r_hi - alpha).abs())
    inside = (r_lo <= alpha) & (r_hi >= alpha)
    dev_min = np.where(inside, 0.0, np.minimum((r_lo - alpha).abs(), (r_hi - alpha).abs()))
    dev_all = (e_all / n - alpha).abs()
    out = df[[c for c in ("tag", "regime", "seed", "method", "delay_s", "trace_id")
              if c in df.columns]].copy()
    out["K"] = k.to_numpy()
    out["M"] = m.to_numpy()
    out["N"] = n.to_numpy()
    out["unapplied"] = (n - m).to_numpy()
    out["unapplied_within_K_plus_1"] = ((n - m) <= k + 1).to_numpy()
    out["bound_applied"] = applied
    out["bound_all"] = all_form
    out["bound_stored_abs_diff"] = (np.abs(applied - df["bound"].to_numpy())
                                    if "bound" in df.columns else NAN)
    out["dev_applied_min"] = dev_min
    out["dev_applied_max"] = dev_max.to_numpy()
    out["ratio_applied_min"] = dev_min / applied
    out["ratio_applied_max"] = dev_max.to_numpy() / applied
    out["dev_all"] = dev_all.to_numpy()
    out["ratio_all"] = dev_all.to_numpy() / all_form
    out["bound_applies"] = ((df["n_alpha_ge1_covered"] == 0) & (m > 0)).to_numpy()
    return out


def bound_summary(chk: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Summary of bound_check rows and of the delay statistics of one group of rows."""
    k = chk["K"]
    return {
        "n_trace_rows": int(len(chk)),
        "K_min": int(k.min()), "K_median": float(k.median()), "K_mean": float(k.mean()),
        "K_max": int(k.max()),
        "M_min": int(chk["M"].min()), "M_median": float(chk["M"].median()),
        "M_max": int(chk["M"].max()),
        "unapplied_max": int(chk["unapplied"].max()),
        "n_rows_unapplied_above_K_plus_1": int((~chk["unapplied_within_K_plus_1"]).sum()),
        "bound_applied_min": float(chk["bound_applied"].min()),
        "bound_applied_median": float(chk["bound_applied"].median()),
        "bound_applied_max": float(chk["bound_applied"].max()),
        "bound_all_min": float(chk["bound_all"].min()),
        "bound_all_median": float(chk["bound_all"].median()),
        "bound_all_max": float(chk["bound_all"].max()),
        "bound_stored_max_abs_diff": float(np.nanmax(chk["bound_stored_abs_diff"]))
        if np.isfinite(chk["bound_stored_abs_diff"]).any() else NAN,
        "n_alpha_le0": int(rows["n_alpha_le0"].sum()),
        "n_alpha_ge1": int(rows["n_alpha_ge1"].sum()),
        "n_alpha_ge1_covered": int(rows["n_alpha_ge1_covered"].sum()),
        "n_rows_alpha_ge1": int((rows["n_alpha_ge1"] > 0).sum()),
        "n_rows_bound_not_applicable": int((~chk["bound_applies"]).sum()),
        "max_ratio_applied_upper": float(chk["ratio_applied_max"].max()),
        "max_ratio_applied_lower": float(chk["ratio_applied_min"].max()),
        "max_ratio_all": float(chk["ratio_all"].max()),
        "n_rows_applied_proven": int((chk["ratio_applied_max"] <= 1.0).sum()),
        "n_rows_applied_violated": int((chk["ratio_applied_min"] > 1.0).sum()),
        "n_rows_applied_unresolved": int(((chk["ratio_applied_max"] > 1.0)
                                          & (chk["ratio_applied_min"] <= 1.0)).sum()),
        "n_rows_all_violated": int((chk["ratio_all"] > 1.0).sum()),
    }


def outstanding_counts(chk: pd.DataFrame) -> pd.DataFrame:
    """Number and share of trace-seed rows per value of K, per (tag, method, delay, regime)
    and pooled over regimes (regime 'all')."""
    keys = ["tag", "method", "delay_s"]
    parts = []
    for scope, frame in (("regime", chk), ("all", chk.assign(regime="all"))):
        c = frame.groupby(keys + ["regime", "K"]).size().rename("n_trace_rows").reset_index()
        c["share"] = c["n_trace_rows"] / c.groupby(keys + ["regime"])["n_trace_rows"].transform("sum")
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- streaming simulation

def has_trace_means(has: pd.DataFrame) -> pd.DataFrame:
    """Per (regime, variant, trace): metrics averaged over the seeds in which the trace was
    tested (NaN skipped) and the number of those seeds."""
    g = has.groupby(["regime", "variant", "trace_id"], sort=True)
    out = g[HAS_METRICS].mean()
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def has_summary(tm: pd.DataFrame) -> pd.DataFrame:
    """Mean with 95% CI over traces, median, and counts, per (regime, variant, metric)."""
    out = []
    for (regime, variant), g in tm.groupby(["regime", "variant"], sort=False):
        g = g.sort_values("trace_id")
        for metric in HAS_METRICS:
            v = g[metric].to_numpy(dtype=float)
            rec = {"regime": regime, "variant": variant, "metric": metric,
                   "n_traces": int(np.isfinite(v).sum()),
                   "n_traces_undefined": int((~np.isfinite(v)).sum())}
            add_ci(rec, "mean", mean_ci(v))
            rec["median"] = float(np.nanmedian(v)) if np.isfinite(v).any() else NAN
            out.append(rec)
    return pd.DataFrame(out)


def has_tests(tm: pd.DataFrame, metric: str = "qoe_per_segment") -> pd.DataFrame:
    """The pre-specified family: within MPC, the Mondrian-ACI bound against the ACI bound,
    the split CP bound and RobustMPC in the four shifted regimes; paired two-sided Wilcoxon
    signed-rank tests on per-trace metric, BH-corrected as one family. Positive median
    difference and rank-biserial favour Mondrian-ACI (larger QoE)."""
    recs = []
    for regime in HAS_TEST_REGIMES:
        a = tm[(tm.regime == regime) & (tm.variant == HAS_TEST_VARIANT)].set_index("trace_id")[metric]
        for comp in HAS_TEST_COMPARATORS:
            b = tm[(tm.regime == regime) & (tm.variant == comp)].set_index("trace_id")[metric]
            common = a.index.intersection(b.index).sort_values()
            res = paired_wilcoxon(a.loc[common].to_numpy(), b.loc[common].to_numpy())
            recs.append({"regime": regime, "variant_a": HAS_TEST_VARIANT, "variant_b": comp,
                         "metric": metric, "mean_a": float(a.loc[common].mean()),
                         "mean_b": float(b.loc[common].mean()), **res})
    df = pd.DataFrame(recs)
    rej, p_adj = bh_correct(df["p"].tolist(), PROTO["statistical_alpha"])
    df["p_bh"] = p_adj
    df["significant_bh"] = rej
    return df


def has_eligibility(has: pd.DataFrame) -> pd.DataFrame:
    """Streaming-eligible test traces per regime and seed (0 for a seed without any)."""
    c = has.groupby(["regime", "seed"]).agg(
        n_eligible=("trace_id", "nunique"),
        n_eligible_4g=("trace_id", lambda s: s[has.loc[s.index, "tech"] == "4G"].nunique()),
        n_eligible_5g=("trace_id", lambda s: s[has.loc[s.index, "tech"] == "5G"].nunique()))
    idx = pd.MultiIndex.from_product([HAS_REGIMES, HAS_SEEDS], names=["regime", "seed"])
    return c.reindex(idx, fill_value=0).reset_index()


def ladder_bitrates(path: Path = LADDER_CSV) -> dict:
    """Nominal representation bitrates of the ladder and their measured mean bitrates
    (mean segment size over the segment duration), in kbit/s."""
    lad = load_ladder(path)
    measured = lad.segment_bits.mean(axis=0) / lad.segment_s / 1000.0
    return {"nominal_kbps": [float(x) for x in lad.bitrates_kbps],
            "measured_mean_kbps": [float(x) for x in measured],
            "n_video_segments": int(lad.segment_bits.shape[0]),
            "segment_s": float(lad.segment_s)}


# ---------------------------------------------------------------- data audit

def data_audit(feat1: pd.DataFrame, feat5: pd.DataFrame | None = None) -> dict:
    """Trace and instance counts, the time between prediction row and target row, and the
    zero-target shares, from the timestamped feature tables."""
    a: dict = {}
    tr = feat1.groupby("trace_id")[["tech", "app", "mobility"]].first()
    a["n_traces"] = int(len(tr))
    a["n_traces_4g"] = int((tr.tech == "4G").sum())
    a["n_traces_5g"] = int((tr.tech == "5G").sum())
    for app in sorted(tr.loc[tr.tech == "5G", "app"].unique()):
        a[f"n_traces_5g_{app.lower()}"] = int(((tr.tech == "5G") & (tr.app == app)).sum())
    for (tech, mob), n in tr.groupby(["tech", "mobility"]).size().items():
        a[f"n_traces_{tech.lower()}_{mob}"] = int(n)
    a["n_traces_streaming_eligible"] = int(((tr.tech == "4G") | (tr.app == "Download")).sum())
    a["n_instances"] = int(len(feat1))
    a["n_instances_4g"] = int((feat1.tech == "4G").sum())
    a["n_instances_5g"] = int((feat1.tech == "5G").sum())
    dt = (feat1["ts_label"] - feat1["ts"]).to_numpy(dtype=float)
    a["h1_target_dt_integral"] = bool(np.all(dt == np.round(dt)))
    a["h1_share_target_exactly_1s"] = float(np.mean(dt == 1.0))
    a["h1_share_target_0s"] = float(np.mean(dt == 0.0))
    a["h1_share_target_ge_2s"] = float(np.mean(dt >= 2.0))
    a["h1_n_target_negative_dt"] = int(np.sum(dt < 0))
    a["h1_n_target_exactly_1s"] = int(np.sum(dt == 1.0))
    a["h1_n_target_0s"] = int(np.sum(dt == 0.0))
    a["h1_n_target_ge_2s"] = int(np.sum(dt >= 2.0))
    a["h1_target_dt_max_s"] = float(dt.max())
    y = feat1["y"].to_numpy(dtype=float)
    a["zero_share_all"] = float(np.mean(y == 0))
    a["zero_share_4g"] = float(np.mean(y[(feat1.tech == "4G").to_numpy()] == 0))
    a["zero_share_5g"] = float(np.mean(y[(feat1.tech == "5G").to_numpy()] == 0))
    app_test = ((feat1.tech == "5G") & feat1.app.isin(["Netflix", "Amazon_Prime"])).to_numpy()
    a["n_instances_app_shift_test"] = int(app_test.sum())
    a["zero_share_app_shift_test"] = float(np.mean(y[app_test] == 0))
    a["y_min_mbps"] = float(y.min())
    if feat5 is not None:
        dt5 = (feat5["ts_label"] - feat5["ts"]).to_numpy(dtype=float)
        a["n_instances_h5"] = int(len(feat5))
        a["h5_share_target_exactly_5s"] = float(np.mean(dt5 == 5.0))
        a["h5_share_target_below_5s"] = float(np.mean(dt5 < 5.0))
        a["h5_share_target_above_5s"] = float(np.mean(dt5 > 5.0))
    return a


def compute_audit(proc: Path = PROC, ladder_csv: Path = LADDER_CSV) -> dict:
    """{'data_audit': ..., 'ladder': ...} from the data files: data_audit of the timestamped
    feature tables, the raw-row count of data_audit_h1.json and timestamps_audit.json and
    the instance count of the timestamp-based table (when those audit files exist), and
    ladder_bitrates of the ladder file."""
    feat1 = read_feature_table(proc / "features_h1_timed.csv.gz")
    feat5 = read_feature_table(proc / "features_h5_timed.csv.gz",
                               usecols=["trace_id", "ts", "ts_label"])
    audit = data_audit(feat1, feat5)
    for name in ("data_audit_h1.json", "timestamps_audit.json"):
        p = proc / name
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            if "n_raw_rows" in j:
                audit["n_raw_rows"] = int(j["n_raw_rows"])
            if "timebase" in j:
                audit["n_instances_timestamp_features"] = int(j["timebase"]["n_instances"])
    return {"data_audit": audit, "ladder": ladder_bitrates(ladder_csv)}


def write_audit(store: dict, path: Path) -> None:
    """Write the output of compute_audit as JSON; floats keep their exact values and the
    keys keep their order."""
    Path(path).write_text(json.dumps(store, indent=1) + "\n", encoding="utf-8")


def read_audit(path: Path) -> dict:
    """Read a file written by write_audit."""
    store = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(store) != {"data_audit", "ladder"}:
        raise ValueError(f"{path}: expected the keys data_audit and ladder")
    return store


# ---------------------------------------------------------------- runtime

def busy_minutes(starts, ends) -> float:
    """Length in minutes of the union of the intervals [start, end]."""
    iv = sorted(zip(starts, ends))
    total, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += (cur_e - cur_s).total_seconds()
    return total / 60.0


def timing_summary(timing: pd.DataFrame) -> pd.DataFrame:
    """Per (mode, tag): cells, worker seconds, the span from the first cell start to the
    last cell finish, and the busy time (union of the cell intervals), which differs from
    the span when the cells were run in separate launches."""
    out = []
    for (mode, tag), g in timing.groupby(["mode", "tag"], sort=True):
        starts = pd.to_datetime(g["started_utc"])
        ends = pd.to_datetime(g["finished_utc"])
        start, end = starts.min(), ends.max()
        out.append({"source": "timing_log", "mode": mode, "tag": tag, "n_cells": int(len(g)),
                    "n_ok": int((g["status"] == "ok").sum()),
                    "n_workers": int(g["pid"].nunique()),
                    "worker_seconds_total": float(g["seconds_total"].sum()),
                    "worker_seconds_mean": float(g["seconds_total"].mean()),
                    "worker_seconds_max": float(g["seconds_total"].max()),
                    "first_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "last_finish_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "wall_minutes": (end - start).total_seconds() / 60.0,
                    "busy_minutes": busy_minutes(starts.tolist(), ends.tolist())})
    return pd.DataFrame(out)


def read_text_any(path: Path) -> str:
    """Text of a file in UTF-16 (with byte-order mark) or UTF-8."""
    raw = Path(path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def parse_timeline(text: str) -> pd.DataFrame:
    """Steps of a batch timeline. Each line is '<UTC time>  <message>'; a message ending in
    ' start' opens a step that ends at the time of the next line. A last row 'total' spans
    the first to the last line."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        stamp, _, msg = line.partition(" ")
        events.append((pd.Timestamp(stamp), msg.strip()))
    out = []
    for i, (t, msg) in enumerate(events):
        if msg.endswith(" start") and i + 1 < len(events):
            t_end = events[i + 1][0]
            out.append({"source": "timeline", "tag": msg[:-len(" start")],
                        "first_start_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "last_finish_utc": t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "wall_minutes": (t_end - t).total_seconds() / 60.0})
    if events:
        out.append({"source": "timeline", "tag": "total",
                    "first_start_utc": events[0][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "last_finish_utc": events[-1][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "wall_minutes": (events[-1][0] - events[0][0]).total_seconds() / 60.0})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- locked outputs

def locked_main_summary(main: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """scripts/analyze_locked.py summary_main at full precision, with the instance-level
    infinite fraction and the finite-interval coverage of amendment 1.5 added."""
    out = []
    for (regime, method), g in main.groupby(["regime", "method"]):
        rec = {"regime": regime, "method": method, "n_traces": int(g.trace_id.nunique()),
               "n_seeds": int(g.seed.nunique())}
        add_ci(rec, "marginal_coverage",
               mean_ci(g.groupby("seed")["marginal_coverage"].first().sort_index()))
        per_trace = g.groupby("trace_id")["coverage"].mean()
        add_ci(rec, "trace_coverage", mean_ci(per_trace))
        add_ci(rec, "abs_dev", mean_ci((per_trace - (1 - alpha)).abs()))
        rec["median_width"] = float(np.median(g.groupby("trace_id")["median_width"].mean()))
        rec["frac_inf_trace_rows"] = float(g.frac_inf.mean())
        rec["winkler_median_rows"] = float(np.median(g.winkler))
        rec["rmse_point"] = float(g.rmse_point.mean())
        c = counts_from_locked(g)
        rec["frac_inf_instance"] = float(seed_ratio(c, "n_inf", "n").mean())
        rec["finite_coverage"] = float(seed_ratio(c, "n_cov_finite", "n_finite").mean())
        rec["marginal_coverage_from_counts"] = float(seed_ratio(c, "n_cov", "n").mean())
        out.append(rec)
    return pd.DataFrame(out)


def locked_tests(main: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Confirmatory and secondary families of scripts/analyze_locked.py at full precision."""
    dev = main.groupby(["regime", "method", "trace_id"])["coverage"].mean().reset_index()
    dev["absdev"] = (dev.coverage - (1 - alpha)).abs()

    def family(name, comparisons):
        recs = []
        for regime, m1, m2 in comparisons:
            a = dev[(dev.regime == regime) & (dev.method == m1)].set_index("trace_id")["absdev"]
            b = dev[(dev.regime == regime) & (dev.method == m2)].set_index("trace_id")["absdev"]
            common = a.index.intersection(b.index)
            res = paired_wilcoxon(a.loc[common].to_numpy(), b.loc[common].to_numpy())
            recs.append({"family": name, "regime": regime, "method_a": m1, "method_b": m2, **res})
        df = pd.DataFrame(recs)
        rej, p_adj = bh_correct(df["p"].tolist(), PROTO["statistical_alpha"])
        df["p_bh"] = p_adj
        df["significant_bh"] = rej
        return df

    conf = family("confirmatory", [(r, "mondrian_aci", "split") for r in SHIFTED])
    sec = family("secondary", [(r, "mondrian_aci", m) for r in SHIFTED
                               for m in ("aci", "mondrian", "cqr")])
    return pd.concat([conf, sec], ignore_index=True)


def locked_groups(main: pd.DataFrame) -> pd.DataFrame:
    """Speed-group coverage of the locked main rows: mean over seeds of the stored
    per-seed group coverage, with a CI over the seed values, and the mean over trace-seed
    rows used by scripts/analyze_locked.py."""
    out = []
    for (regime, method), g in main.groupby(["regime", "method"]):
        per_seed = g.groupby("seed")["group_coverage_json"].first().sort_index()
        vals: dict[str, list] = {}
        for js in per_seed:
            for k, v in json.loads(js).items():
                vals.setdefault(k, []).append(v)
        rows_vals: dict[str, list] = {}
        for js in g["group_coverage_json"]:
            for k, v in json.loads(js).items():
                rows_vals.setdefault(k, []).append(v)
        for k in sorted(vals):
            rec = {"regime": regime, "method": method, "group": k, "n_seeds": len(vals[k])}
            add_ci(rec, "coverage", mean_ci(vals[k]))
            rec["coverage_rows_mean"] = float(np.mean(rows_vals[k]))
            out.append(rec)
    return pd.DataFrame(out)


def locked_downstream(main: pd.DataFrame) -> pd.DataFrame:
    """Rate-selection proxy of scripts/analyze_locked.py at full precision."""
    out = []
    cols = ["overshoot_lb", "util_lb", "overshoot_point", "util_point"]
    for (regime, method), g in main.groupby(["regime", "method"]):
        per_trace = g.groupby("trace_id")[cols].mean()
        for col in cols:
            rec = {"regime": regime, "method": method, "metric": col}
            add_ci(rec, "mean", mean_ci(per_trace[col]))
            out.append(rec)
    return pd.DataFrame(out)


def locked_ablations(rows: pd.DataFrame) -> pd.DataFrame:
    """Ablation summaries of scripts/analyze_locked.py (means over trace-seed rows) at full
    precision, with the finite-interval coverage of amendment 1.5 computed exactly from the
    stored per-trace coverage and infinite-interval fractions."""
    out = []
    cols = ["tag", "regime", "base_model", "alpha", "gamma", "method"]
    for key, g in rows[rows.tag != "main"].groupby(cols):
        rec = dict(zip(cols, key))
        for c in ("coverage", "marginal_coverage", "median_width", "frac_inf"):
            rec[f"{c}_rows_mean"] = float(g[c].mean())
        s = interval_summary(counts_from_locked(g), alpha=float(rec["alpha"]))
        for c in ("n_seeds", "n_traces", "marginal_coverage", "marginal_coverage_ci_lo",
                  "marginal_coverage_ci_hi", "trace_coverage", "abs_dev", "frac_inf_instance",
                  "frac_inf_trace_rows", "frac_inf_trace", "finite_coverage",
                  "finite_coverage_trace"):
            rec[c] = s[c]
        rec["target"] = 1.0 - float(rec["alpha"])
        out.append(rec)
    return pd.DataFrame(out)


def locked_derived(lm: pd.DataFrame, lt: pd.DataFrame, lg: pd.DataFrame,
                   ld: pd.DataFrame, la: pd.DataFrame) -> dict:
    """Scalars derived from the locked summaries that the text quotes as ranges or ratios."""
    d: dict = {}

    def cell(df, regime, method, col, **kw):
        sel = (df.regime == regime) & (df.method == method)
        for k, v in kw.items():
            sel &= df[k] == v
        (v,) = df.loc[sel, col]
        return float(v)

    d["width_ratio_split_over_mondrian_aci_cross_app_5g"] = (
        cell(lm, "cross_app_5g", "split", "median_width")
        / cell(lm, "cross_app_5g", "mondrian_aci", "median_width"))
    d["util_lb_ratio_mondrian_aci_over_split_cross_app_5g"] = (
        cell(ld, "cross_app_5g", "mondrian_aci", "mean", metric="util_lb")
        / cell(ld, "cross_app_5g", "split", "mean", metric="util_lb"))
    conf = lt[lt.family == "confirmatory"]
    d["confirmatory_p_bh_max"] = float(conf.p_bh.max())
    d["confirmatory_rank_biserial_min"] = float(conf.rank_biserial.min())
    d["confirmatory_rank_biserial_max"] = float(conf.rank_biserial.max())
    sec = lt[lt.family == "secondary"]
    mc = sec[sec.method_b.isin(["mondrian", "cqr"])]
    d["secondary_mondrian_cqr_abs_rank_biserial_min"] = float(mc.rank_biserial.abs().min())
    sa = sec[(sec.method_b == "aci") & sec.significant_bh.astype(bool) & (sec.median_diff < 0)]
    d["secondary_aci_n_significant_reductions"] = int(len(sa))
    d["secondary_aci_significant_reduction_min"] = float(-sa.median_diff.max())
    d["secondary_aci_significant_reduction_max"] = float(-sa.median_diff.min())
    shifted = lm[lm.regime.isin(SHIFTED)]
    for m in ("mondrian_aci", "aci"):
        s = shifted[shifted.method == m]
        d[f"{m}_abs_dev_max_shifted"] = float(s.abs_dev.max())
        d[f"{m}_frac_inf_trace_rows_min_shifted"] = float(s.frac_inf_trace_rows.min())
        d[f"{m}_frac_inf_trace_rows_max_shifted"] = float(s.frac_inf_trace_rows.max())
        a = lm[lm.method == m]
        d[f"{m}_marginal_coverage_min"] = float(a.marginal_coverage.min())
        d[f"{m}_marginal_coverage_max"] = float(a.marginal_coverage.max())
    ov = ld[(ld.method == "mondrian_aci") & (ld.metric == "overshoot_lb")]
    d["mondrian_aci_overshoot_lb_min"] = float(ov["mean"].min())
    d["mondrian_aci_overshoot_lb_max"] = float(ov["mean"].max())
    op = ld[(ld.method == "mondrian_aci") & (ld.metric == "overshoot_point")
            & ld.regime.isin(SHIFTED)]
    d["overshoot_point_min_shifted"] = float(op["mean"].min())
    d["overshoot_point_max_shifted"] = float(op["mean"].max())
    g45 = lg[(lg.regime == "cross_tech_4to5") & (lg.method == "mondrian_aci")]
    d["mondrian_aci_group_coverage_min_cross_tech_4to5"] = float(g45.coverage_rows_mean.min())
    d["mondrian_aci_group_coverage_max_cross_tech_4to5"] = float(g45.coverage_rows_mean.max())
    m_abl = la[la.method == "mondrian_aci"]
    d["ablation_mondrian_aci_max_abs_marginal_minus_target"] = float(
        (m_abl.marginal_coverage_rows_mean - m_abl.target).abs().max())
    s45 = la[(la.method == "split") & (la.regime == "cross_tech_4to5")
             & la.tag.isin(["abl_base", "abl_h5"])]
    d["ablation_split_cross_tech_4to5_base_h5_min"] = float(s45.marginal_coverage_rows_mean.min())
    d["ablation_split_cross_tech_4to5_base_h5_max"] = float(s45.marginal_coverage_rows_mean.max())
    a05 = la[(la.tag == "abl_alpha05") & (la.method == "mondrian_aci")]
    d["ablation_alpha05_mondrian_aci_frac_inf_rows_max"] = float(a05.frac_inf_rows_mean.max())
    return d


# ---------------------------------------------------------------- flattening and output

def json_value(v):
    """JSON-safe scalar: NaN, NA and infinities become None, numpy types become Python
    types."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    return str(v)


def id_token(col: str, v) -> str:
    """Key component for an identifying column value."""
    if col == "delay_s":
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "Dnone"
        return f"D{float(v):g}"
    if col in ("alpha", "gamma"):
        return f"{col}{float(v):g}"
    if col == "seed":
        return f"s{int(v)}"
    return str(v)


def flatten(df: pd.DataFrame, prefix: str, ids: list[str]) -> dict:
    """{prefix.id1.id2...column: value} for every non-identifying column of every row."""
    out = {}
    vals = [c for c in df.columns if c not in ids]
    for _, r in df.iterrows():
        base = ".".join([prefix] + [id_token(c, r[c]) for c in ids])
        for c in vals:
            out[f"{base}.{c}"] = json_value(r[c])
    return out


def range_keys(df: pd.DataFrame, prefix: str, by: list[str], cols: list[str],
               subsets: dict) -> dict:
    """Minimum and maximum of cols over the regimes of each subset, per combination of by."""
    out = {}
    for name, regimes in subsets.items():
        sub = df[df.regime.isin(regimes)]
        for k, g in sub.groupby(by, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            base = ".".join([prefix] + [id_token(c, v) for c, v in zip(by, k)] + [name])
            for c in cols:
                v = g[c].astype(float)
                if v.notna().any():
                    out[f"{base}.{c}_min"] = json_value(v.min())
                    out[f"{base}.{c}_max"] = json_value(v.max())
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- digest

def f3(x, nd: int = 4) -> str:
    """Proportion with nd decimals ('-' for missing)."""
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fg(x, sig: int = 4) -> str:
    """Number with sig significant digits ('-' for missing)."""
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.{sig}g}"


def fpct(x, sig: int = 3) -> str:
    """Proportion as a percentage with sig significant digits."""
    if x is None or not np.isfinite(x):
        return "-"
    return f"{100 * x:.{sig}g}%"


def fci(x, lo, hi, nd: int = 4) -> str:
    return f"{f3(x, nd)} [{f3(lo, nd)}, {f3(hi, nd)}]"


def fp(p) -> str:
    return "-" if p is None or not np.isfinite(p) else f"{p:.2g}"


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines + [""]


def digest(s: dict) -> str:
    """Markdown digest of all summaries (dict of DataFrames and scalars from main())."""
    L = ["# Digest of the amendment 1 and locked summaries", "",
         "Values at 3 to 4 significant digits; full precision in the CSV files and summary.json "
         "of this folder. Coverage targets 0.90 (alpha = 0.10) unless stated. CIs are 95% "
         "percentile bootstrap intervals (10,000 resamples, seed 0) over seed values for "
         "marginal and group quantities and over traces for trace quantities. 'inf inst' is the "
         "instance-level infinite-interval fraction (pooled within seed, mean over seeds); "
         "'inf rows' the mean over trace-seed rows (definition of the locked analysis); "
         "'inf trace' the mean of seed-averaged per-trace fractions. Widths and Winkler scores "
         "in Mbit/s over finite intervals.", ""]
    mm = s["main_methods"]
    L += ["## A. Main configuration (h = 1, D = 0, seeds 0-19)", "",
          "### A1. Coverage and infinite intervals", ""]
    rows = []
    for _, r in mm.iterrows():
        rows.append([r.regime, r.method, fci(r.marginal_coverage, r.marginal_coverage_ci_lo,
                                             r.marginal_coverage_ci_hi),
                     f3(r.trace_coverage), f3(r.abs_dev), fpct(r.frac_inf_instance),
                     fpct(r.frac_inf_trace_rows), fpct(r.frac_inf_trace),
                     fci(r.coverage_pos, r.coverage_pos_ci_lo, r.coverage_pos_ci_hi)])
    L += md_table(["regime", "method", "marginal cov [CI]", "trace cov", "mean Delta_s",
                   "inf inst", "inf rows", "inf trace", "cov y>0 [CI]"], rows)
    L += ["### A2. Finite intervals, widths, Winkler score, worst speed group", ""]
    rows = []
    for _, r in mm.iterrows():
        rows.append([r.regime, r.method,
                     fci(r.finite_coverage, r.finite_coverage_ci_lo, r.finite_coverage_ci_hi),
                     f3(r.finite_coverage_trace), fg(r.width_finite_median),
                     fg(r.width_finite_mean_median), fg(r.winkler_median), fg(r.winkler_mean),
                     str(int(r.n_inverted)),
                     f"{GROUP_LABELS.get(r.worst_group, r.worst_group)} "
                     f"{fci(r.worst_group_coverage, r.worst_group_coverage_ci_lo, r.worst_group_coverage_ci_hi)}"])
    L += md_table(["regime", "method", "finite cov [CI]", "finite cov trace",
                   "finite width (median of medians)", "finite width (median of means)",
                   "Winkler median", "Winkler mean", "inverted", "worst group cov [CI]"], rows)
    L += ["### A3. Speed-group coverage (mean over seeds [CI]; minimum test instances per seed)", ""]
    gs = s["main_groups"]
    rows = []
    for (regime, method), g in gs.groupby(["regime", "method"], sort=False):
        g = g.set_index("group")
        cells = [f"{fci(g.loc[k, 'coverage'], g.loc[k, 'coverage_ci_lo'], g.loc[k, 'coverage_ci_hi'], 3)}"
                 f" (n>={int(g.loc[k, 'n_test_min'])})" if k in g.index else "-" for k in GROUPS]
        rows.append([regime, method] + cells)
    L += md_table(["regime", "method"] + [f"{GROUP_LABELS[k]} km/h" for k in GROUPS], rows)
    L += ["### A4. Share of zero targets in the test sets (pooled within seed, mean over seeds)", ""]
    zs = mm[mm.method == "split"]
    L += md_table(["regime", "zero share"], [[r.regime, fpct(r.zero_share)] for _, r in zs.iterrows()])

    dg = s["delay_grid"]
    L += ["## B. Feedback delay (aci and mondrian_aci, h = 1, seeds 0-19)", "",
          "### B1. Coverage and informativeness by delay D", ""]
    rows = []
    for _, r in dg[dg.tag == "main"].iterrows():
        rows.append([r.method, r.regime, f"{r.delay_s:g}",
                     fci(r.marginal_coverage, r.marginal_coverage_ci_lo, r.marginal_coverage_ci_hi),
                     f3(r.trace_coverage), f3(r.abs_dev), fpct(r.frac_inf_instance),
                     fpct(r.frac_inf_trace_rows), fpct(r.frac_inf_trace), f3(r.finite_coverage),
                     f3(r.finite_coverage_trace), fg(r.width_finite_median), fg(r.winkler_median)])
    L += md_table(["method", "regime", "D (s)", "marginal cov [CI]", "trace cov", "mean Delta_s",
                   "inf inst", "inf rows", "inf trace", "finite cov", "finite cov trace",
                   "finite width", "Winkler median"], rows)
    L += ["### B2. Outstanding labels K, applied labels M, bound of Proposition 1 (ii)", "",
          "ratio = observed deviation / bound. The applied-label deviation is bracketed from N, M "
          "and the miss count (the N - M unapplied labels are not stored individually); "
          "'ratio applied <=' is the largest upper end of the bracket, so a value <= 1 proves "
          "the bound for every trace-seed row.", ""]
    rows = []
    for _, r in dg.iterrows():
        rows.append([r.tag, r.method, r.regime, f"{r.delay_s:g}",
                     f"{int(r.K_min)}/{fg(r.K_median, 3)}/{int(r.K_max)}",
                     f"{int(r.M_min)}/{fg(r.M_median, 4)}", fg(r.bound_applied_median, 3),
                     fg(r.bound_applied_max, 3), f"{int(r.n_alpha_ge1)}/{int(r.n_alpha_ge1_covered)}",
                     str(int(r.n_alpha_le0)), f3(r.max_ratio_applied_upper, 3),
                     f3(r.max_ratio_all, 3),
                     str(int(r.n_rows_applied_violated + r.n_rows_applied_unresolved
                             + r.n_rows_all_violated))])
    L += md_table(["tag", "method", "regime", "D", "K min/med/max", "M min/med",
                   "bound median", "bound max", "alpha>=1 issued/covered", "alpha<=0 issued",
                   "ratio applied <=", "ratio all", "rows not proven"], rows)
    oc = s["delay_outstanding"]
    oc = oc[oc.regime == "all"]
    L += ["### B3. Distribution of K over trace-seed rows (all regimes)", ""]
    rows = []
    for (tag, method, d), g in oc.groupby(["tag", "method", "delay_s"], sort=False):
        rows.append([tag, method, f"{d:g}",
                     ", ".join(f"{int(k)}: {fpct(sh)}" for k, sh in zip(g.K, g.share))])
    L += md_table(["tag", "method", "D", "K: share of rows"], rows)

    h5 = s["horizon5"]
    L += ["## C. 5-s horizon (seeds 0-4): delay model (D = 0) versus immediate feedback", ""]
    rows = []
    for _, r in h5.iterrows():
        rows.append([r.regime, r.method, r.feedback,
                     fci(r.marginal_coverage, r.marginal_coverage_ci_lo, r.marginal_coverage_ci_hi),
                     f3(r.trace_coverage), f3(r.abs_dev), fpct(r.frac_inf_instance),
                     fpct(r.frac_inf_trace_rows), f3(r.finite_coverage),
                     f3(r.finite_coverage_trace), fg(r.width_finite_median),
                     fg(r.winkler_median)])
    L += md_table(["regime", "method", "feedback", "marginal cov [CI]", "trace cov",
                   "mean Delta_s", "inf inst", "inf rows", "finite cov", "finite cov trace",
                   "finite width", "Winkler median"], rows)

    ts = s["timestamp"]
    L += ["## D. Timestamp-based features versus row-based (seeds 0-4)", ""]
    rows = []
    for _, r in ts.iterrows():
        rows.append([r.regime, r.method, f3(r.ts_marginal_coverage), f3(r.row_marginal_coverage),
                     f"{r.ts_marginal_coverage - r.row_marginal_coverage:+.4f}",
                     fpct(r.ts_frac_inf_instance), fpct(r.row_frac_inf_instance),
                     fpct(r.ts_frac_inf_trace_rows), fpct(r.row_frac_inf_trace_rows)])
    L += md_table(["regime", "method", "cov (timestamps)", "cov (rows)", "difference",
                   "inf inst (timestamps)", "inf inst (rows)", "inf rows (timestamps)",
                   "inf rows (rows)"], rows)

    a = s["audit"]
    L += ["## E. Data audit", ""]
    L += [f"- {k}: {json.dumps(json_value(v)) if not isinstance(v, list) else v}"
          for k, v in a.items()]
    L += ["", "Streaming-eligible test traces per seed (min-max over seeds; distinct over seeds):", ""]
    el = s["eligibility"]
    rows = []
    for regime, g in el.groupby("regime", sort=False):
        rows.append([regime, f"{int(g.n_eligible.min())}-{int(g.n_eligible.max())}",
                     str(int((g.n_eligible == 0).sum())),
                     str(int(s["has_distinct"].get(regime, 0)))])
    L += md_table(["regime", "eligible per seed", "seeds without eligible trace",
                   "distinct traces"], rows)

    hm = s["has_metrics"]
    L += ["## F. Adaptive-streaming simulation (per-trace seed averages; mean [CI] over traces; "
          "median)", ""]
    short = {"avg_bitrate_mbps": "bitrate (Mbit/s)", "rebuffer_s": "rebuffer (s)",
             "rebuffer_ratio": "rebuffer ratio", "stalls": "stalls",
             "startup_s": "startup (s)", "switches_per_min": "switches/min",
             "mean_abs_switch_mbps": "abs change/switch (Mbit/s)",
             "qoe_per_segment": "QoE/segment"}
    for regime in HAS_REGIMES:
        sub = hm[hm.regime == regime]
        n_tr = int(sub[sub.metric == "avg_bitrate_mbps"].n_traces.max())
        L += [f"### F. {regime} ({n_tr} traces)", ""]
        rows = []
        for variant in HAS_VARIANTS:
            v = sub[sub.variant == variant].set_index("metric")
            rows.append([variant] + [f"{fg(v.loc[m, 'mean'], 4)} [{fg(v.loc[m, 'mean_ci_lo'], 3)}, "
                                     f"{fg(v.loc[m, 'mean_ci_hi'], 3)}]; {fg(v.loc[m, 'median'], 3)}"
                                     for m in HAS_METRICS])
        L += md_table(["variant"] + [short[m] for m in HAS_METRICS], rows)
    ht = s["has_tests"]
    L += ["### F. Pre-specified tests on per-trace QoE per segment (MPC family, BH over 12)", ""]
    rows = [[r.regime, f"{r.variant_a} vs {r.variant_b}", str(int(r.n)), fg(r.mean_a, 4),
             fg(r.mean_b, 4), f"{r.median_diff:+.4g}", f"{r.rank_biserial:+.3f}", fp(r.p),
             fp(r.p_bh)] for _, r in ht.iterrows()]
    L += md_table(["regime", "comparison", "n", "mean QoE a", "mean QoE b", "median diff a-b",
                   "rank-biserial", "p", "p BH"], rows)
    lb = s["ladder"]
    L += ["Ladder: decisions, average bitrate and QoE use the nominal bitrates "
          f"{', '.join(f'{x:g}' for x in lb['nominal_kbps'])} kbit/s (QoE weights mu = mu_s = "
          f"{lb['nominal_kbps'][-1] / 1000:g}); download times use the per-segment sizes, whose "
          f"mean bitrates are {', '.join(f'{x:.1f}' for x in lb['measured_mean_kbps'])} kbit/s "
          f"({lb['n_video_segments']} segments of {lb['segment_s']:g} s).", ""]

    rt = s["runtime"]
    L += ["## G. Runtime", ""]
    rows = []
    for _, r in rt.iterrows():
        rows.append([r.source, str(r.get("mode", "") if isinstance(r.get("mode", ""), str) else ""),
                     r.tag, "-" if pd.isna(r.get("n_cells")) else str(int(r.n_cells)),
                     fg(r.get("worker_seconds_total"), 4), r.first_start_utc, r.last_finish_utc,
                     fg(r.wall_minutes, 3), fg(r.get("busy_minutes"), 3)])
    L += md_table(["source", "mode", "tag/step", "cells", "worker s", "start (UTC)",
                   "end (UTC)", "span min", "busy min"], rows)

    L += ["## L. Locked confirmatory outputs at full precision", "",
          "### L1. Main: marginal coverage [CI over seeds], median width including infinite "
          "widths, infinite fraction (mean over trace-seed rows)", ""]
    lm = s["locked_main"]
    rows = [[r.regime, r.method, fci(r.marginal_coverage, r.marginal_coverage_ci_lo,
                                     r.marginal_coverage_ci_hi, 5),
             f3(r.trace_coverage, 5), f3(r.abs_dev, 5), fg(r.median_width, 5),
             fpct(r.frac_inf_trace_rows, 4)] for _, r in lm.iterrows()]
    L += md_table(["regime", "method", "marginal cov [CI]", "trace cov", "mean Delta_s",
                   "median width", "inf rows"], rows)
    L += ["### L2. Confirmatory and secondary tests on Delta_s", ""]
    rows = [[r.family, r.regime, f"{r.method_a} vs {r.method_b}", str(int(r.n)),
             f"{r.median_diff:+.4g}", f"{r.rank_biserial:+.4f}", fp(r.p), fp(r.p_bh)]
            for _, r in s["locked_tests"].iterrows()]
    L += md_table(["family", "regime", "comparison", "n", "median diff", "rank-biserial", "p",
                   "p BH"], rows)
    L += ["### L3. Ablations (seeds 0-4): marginal coverage and infinite fraction as in the "
          "locked analysis (means over trace-seed rows), finite-interval coverage (1.5)", ""]
    la = s["locked_ablations"]
    rows = [[r.tag, r.regime, r.base_model, f"{r.alpha:g}", f"{r.gamma:g}", r.method,
             f3(r.marginal_coverage_rows_mean, 5), fpct(r.frac_inf_rows_mean, 4),
             fpct(r.frac_inf_instance, 4), f3(r.finite_coverage, 4), f3(r.finite_coverage_trace, 4)]
            for _, r in la.iterrows()]
    L += md_table(["tag", "regime", "base", "alpha", "gamma", "method", "marginal cov",
                   "inf rows", "inf inst", "finite cov", "finite cov trace"], rows)
    L += ["### L4. Rate-selection proxy (mean over traces of seed averages [CI])", ""]
    ld = s["locked_downstream"]
    rows = []
    for (regime, method), g in ld.groupby(["regime", "method"], sort=False):
        if method not in ("split", "aci", "mondrian_aci"):
            continue
        g = g.set_index("metric")
        rows.append([regime, method] + [fci(g.loc[c, "mean"], g.loc[c, "mean_ci_lo"],
                                            g.loc[c, "mean_ci_hi"], 4)
                                        for c in ("overshoot_lb", "util_lb", "overshoot_point")])
    L += md_table(["regime", "method", "overshoot (lower bound)", "utilisation (lower bound)",
                   "overshoot (point)"], rows)
    L += ["### L5. Derived locked scalars", ""]
    L += [f"- {k}: {fg(v, 5)}" for k, v in s["locked_derived"].items()]
    L += ["", "## Consistency checks", ""]
    L += [f"- {k}: {json.dumps(json_value(v))}" for k, v in s["checks"].items()]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeline", type=Path, default=None,
                    help="batch timeline file ('<UTC time>  <message>' per line); default "
                         "results/amendment1/batch_timeline.txt when that file exists")
    ap.add_argument("--stored-audit", action="store_true",
                    help="read the data audit and the ladder bitrates from "
                         "results/summary/audit.json even when the data files exist")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline = args.timeline
    if timeline is None and TIMELINE.exists():
        timeline = TIMELINE
    present = [p.exists() for p in AUDIT_INPUTS]
    if not args.stored_audit and any(present) and not all(present):
        missing = [str(p.relative_to(ROOT)) for p, ok in zip(AUDIT_INPUTS, present) if not ok]
        raise SystemExit(f"missing {', '.join(missing)}: run scripts/prepare_timestamps.py "
                         "and scripts/fetch_ladder.py, or use --stored-audit")

    exact = {"float_precision": "round_trip"}
    main_rows = pd.read_csv(AMEND / "main_traces.csv.gz", **exact)
    main_groups = pd.read_csv(AMEND / "main_groups.csv.gz", **exact)
    h5_rows = pd.read_csv(AMEND / "h5_traces.csv.gz", **exact)
    ts_rows = pd.read_csv(AMEND / "ts_traces.csv.gz", **exact)
    has_rows = pd.read_csv(AMEND / "has" / "has_rows.csv.gz", **exact)
    timing = pd.read_csv(AMEND / "timing_log.csv")
    locked = read_cell_rows(LOCKED)
    n_locked_short = int((locked.n < MIN_TEST_INSTANCES).sum())
    locked = locked[locked.n >= MIN_TEST_INSTANCES]
    checks: dict = {"locked_rows_sha256": sha256(LOCKED),
                    "locked_rows_below_min_test_instances": n_locked_short}
    for name, df in (("main", main_rows), ("h5", h5_rows), ("ts", ts_rows)):
        checks[f"{name}_rows_below_min_test_instances"] = int((df.n < MIN_TEST_INSTANCES).sum())
        df.drop(df.index[df.n < MIN_TEST_INSTANCES], inplace=True)
    for f in ("main_traces.csv.gz", "main_groups.csv.gz", "h5_traces.csv.gz", "ts_traces.csv.gz"):
        checks[f"{f}_sha256"] = sha256(AMEND / f)
    checks["has_rows.csv.gz_sha256"] = sha256(AMEND / "has" / "has_rows.csv.gz")

    # agreement of the main configuration with the locked main rows (criterion of 1.3)
    d0 = select_d0(main_rows)
    lk_main = locked[locked.tag == "main"]
    mg = d0.merge(lk_main[["regime", "seed", "method", "trace_id", "n", "coverage", "frac_inf"]],
                  on=["regime", "seed", "method", "trace_id"], how="outer",
                  suffixes=("", "_locked"), indicator=True)
    checks["main_vs_locked_unpaired_rows"] = int((mg["_merge"] != "both").sum())
    checks["main_vs_locked_n_mismatch"] = int((mg.n != mg.n_locked).sum())
    checks["main_vs_locked_max_abs_diff_coverage"] = float((mg.coverage - mg.coverage_locked).abs().max())
    checks["main_vs_locked_max_abs_diff_frac_inf"] = float((mg.frac_inf - mg.frac_inf_locked).abs().max())

    # A. main configuration, all methods
    mm = summarise(d0, ["regime", "method"])
    gs = group_summary(select_d0(main_groups))
    gs = order_frame(gs, {"regime": REGIMES, "method": METHODS, "group": GROUPS})
    wg = worst_groups(gs)
    mm = mm.merge(wg, on=["regime", "method"], how="left", validate="one_to_one")
    mm = order_frame(mm, {"regime": REGIMES, "method": METHODS})
    zero = mm.groupby("regime")["zero_share"].agg(["min", "max"])
    checks["zero_share_equal_across_methods"] = bool(((zero["max"] - zero["min"]).abs() < 1e-15).all())
    gn = gs.groupby(["regime", "group"])[["n_test_min", "n_test_max", "n_test_total"]].nunique()
    checks["group_counts_equal_across_methods"] = bool((gn == 1).all().all())
    checks["group_n_test_min_overall"] = int(gs.n_test_min.min())

    # B. delay grid (and the h = 5 delayed run for the bound check)
    delayed = pd.concat([main_rows[main_rows.feedback == "delayed"],
                         h5_rows[h5_rows.feedback == "delayed"]], ignore_index=True)
    dg_parts, chk_parts = [], []
    for (tag, method, d, regime), g in delayed.groupby(["tag", "method", "delay_s", "regime"],
                                                       sort=False):
        chk = bound_check(g)
        chk_parts.append(chk)
        dg_parts.append({"tag": tag, "method": method, "delay_s": d, "regime": regime,
                         **interval_summary(g), **bound_summary(chk, g)})
    dg = order_frame(pd.DataFrame(dg_parts), {"tag": ["main", "h5"], "method": ONLINE,
                                              "delay_s": DELAYS, "regime": REGIMES})
    chk_all = pd.concat(chk_parts, ignore_index=True)
    oc = outstanding_counts(chk_all)
    oc = order_frame(oc, {"tag": ["main", "h5"], "method": ONLINE, "delay_s": DELAYS,
                          "regime": REGIMES + ["all"]})
    bound_cols = ["tag", "method", "delay_s", "regime", "n_trace_rows", "K_min", "K_median",
                  "K_mean", "K_max", "M_min", "M_median", "M_max", "unapplied_max",
                  "n_rows_unapplied_above_K_plus_1", "bound_applied_min", "bound_applied_median",
                  "bound_applied_max", "bound_all_min", "bound_all_median", "bound_all_max",
                  "bound_stored_max_abs_diff", "n_alpha_le0", "n_alpha_ge1",
                  "n_alpha_ge1_covered", "n_rows_alpha_ge1", "n_rows_bound_not_applicable",
                  "max_ratio_applied_upper", "max_ratio_applied_lower", "max_ratio_all",
                  "n_rows_applied_proven", "n_rows_applied_violated",
                  "n_rows_applied_unresolved", "n_rows_all_violated"]
    bound_tab = dg[bound_cols]
    checks["delay_bound_rows_checked"] = int(len(chk_all))
    checks["delay_bound_rows_proven_applied"] = int((chk_all.ratio_applied_max <= 1).sum())
    checks["delay_bound_rows_violated_applied"] = int((chk_all.ratio_applied_min > 1).sum())
    checks["delay_bound_rows_violated_all_form"] = int((chk_all.ratio_all > 1).sum())
    checks["delay_bound_rows_not_applicable"] = int((~chk_all.bound_applies).sum())
    checks["delay_rows_unapplied_above_K_plus_1"] = int((~chk_all.unapplied_within_K_plus_1).sum())
    checks["delay_bound_stored_max_abs_diff"] = float(np.nanmax(chk_all.bound_stored_abs_diff))

    # C. 5-s horizon: delay model (h5 tag) beside immediate feedback (locked abl_h5)
    h5_parts = []
    imm = counts_from_locked(locked[locked.tag == "abl_h5"])
    for regime in H5_REGIMES:
        for method in H5_METHODS:
            dl = h5_rows[(h5_rows.regime == regime) & (h5_rows.method == method)]
            label = "delayed_D0" if method in ONLINE else "none"
            h5_parts.append({"regime": regime, "method": method, "feedback": label,
                             **interval_summary(dl)})
            im = imm[(imm.regime == regime) & (imm.method == method)]
            h5_parts.append({"regime": regime, "method": method,
                             "feedback": "immediate" if method in ONLINE else "none_locked",
                             **interval_summary(im)})
    h5 = pd.DataFrame(h5_parts)
    lk_h5 = locked[(locked.tag == "abl_h5") & ~locked.method.isin(ONLINE)]
    off = h5_rows[h5_rows.feedback == "none"].merge(
        lk_h5[["regime", "seed", "method", "trace_id", "n", "coverage", "frac_inf"]],
        on=["regime", "seed", "method", "trace_id"], how="outer", suffixes=("", "_locked"),
        indicator=True)
    checks["h5_offline_vs_locked_unpaired_rows"] = int((off["_merge"] != "both").sum())
    checks["h5_offline_vs_locked_max_abs_diff_coverage"] = float(
        (off.coverage - off.coverage_locked).abs().max())

    # D. timestamp sensitivity
    ts_d0 = with_finite_counts(select_d0(ts_rows))
    row5 = counts_from_locked(lk_main[lk_main.seed.isin(PROTO["ablation_seeds"])])
    ts_parts = []
    for regime in REGIMES:
        for method in METHODS:
            a = interval_summary(ts_d0[(ts_d0.regime == regime) & (ts_d0.method == method)])
            b = interval_summary(row5[(row5.regime == regime) & (row5.method == method)])
            rec = {"regime": regime, "method": method}
            for prefix, src in (("ts", a), ("row", b)):
                for c in ("n_seeds", "n_traces", "n_instances", "marginal_coverage",
                          "marginal_coverage_ci_lo", "marginal_coverage_ci_hi", "trace_coverage",
                          "frac_inf_instance", "frac_inf_trace_rows", "frac_inf_trace"):
                    rec[f"{prefix}_{c}"] = src[c]
            ts_parts.append(rec)
    ts = pd.DataFrame(ts_parts)

    # E. data audit (and the ladder bitrates of F)
    if args.stored_audit or not all(present):
        store = read_audit(AUDIT_JSON)
        print(f"data audit and ladder bitrates read from {AUDIT_JSON}")
    else:
        store = compute_audit()
        write_audit(store, out_dir / "audit.json")
        print(f"data audit and ladder bitrates computed from the data files and written to "
              f"{out_dir / 'audit.json'}")
    audit = dict(store["data_audit"])
    lad = store["ladder"]
    app = d0[(d0.regime == "cross_app_5g") & (d0.method == "split")]
    audit["zero_share_app_shift_test_from_runs"] = float(seed_ratio(app, "n_y0", "n").mean())
    eligibility = has_eligibility(has_rows)
    has_distinct = has_rows.groupby("regime")["trace_id"].nunique().to_dict()
    for regime, n in has_distinct.items():
        audit[f"n_streaming_traces_distinct_{regime}"] = int(n)
    for regime, g in eligibility.groupby("regime", sort=False):
        audit[f"n_streaming_eligible_per_seed_min_{regime}"] = int(g.n_eligible.min())
        audit[f"n_streaming_eligible_per_seed_max_{regime}"] = int(g.n_eligible.max())
        audit[f"n_seeds_without_streaming_trace_{regime}"] = int((g.n_eligible == 0).sum())

    # F. streaming simulation
    tm = has_trace_means(has_rows)
    hm = order_frame(has_summary(tm), {"regime": HAS_REGIMES, "variant": HAS_VARIANTS,
                                       "metric": HAS_METRICS})
    ht = has_tests(tm)

    # G. runtime
    rt = timing_summary(timing)
    if timeline is not None:
        rt = pd.concat([rt, parse_timeline(read_text_any(timeline))], ignore_index=True)
    for c in ("n_cells", "n_ok", "n_workers"):
        rt[c] = rt[c].astype("Int64")

    # locked confirmatory outputs at full precision
    lm = order_frame(locked_main_summary(lk_main), {"regime": REGIMES, "method": METHODS})
    lt = locked_tests(lk_main)
    lg = order_frame(locked_groups(lk_main), {"regime": REGIMES, "method": METHODS,
                                               "group": GROUPS})
    ld = order_frame(locked_downstream(lk_main), {"regime": REGIMES, "method": METHODS})
    la = locked_ablations(locked)
    lder = locked_derived(lm, lt, lg, ld, la)
    checks["locked_marginal_coverage_counts_max_abs_diff"] = float(
        (lm.marginal_coverage - lm.marginal_coverage_from_counts).abs().max())
    checks["locked_groups_rows_vs_seed_mean_max_abs_diff"] = float(
        (lg.coverage - lg.coverage_rows_mean).abs().max())
    cmp = mm.merge(lm, on=["regime", "method"], suffixes=("", "_locked"))
    for c in ("marginal_coverage", "marginal_coverage_ci_lo", "marginal_coverage_ci_hi",
              "trace_coverage", "abs_dev", "frac_inf_trace_rows", "frac_inf_instance",
              "finite_coverage"):
        checks[f"main_vs_locked_summary_max_abs_diff_{c}"] = float(
            (cmp[c] - cmp[f"{c}_locked"]).abs().max())

    # write
    tables = {"main_methods.csv": mm, "main_groups.csv": gs, "delay_grid.csv": dg,
              "delay_outstanding.csv": oc, "delay_bound.csv": bound_tab, "horizon5.csv": h5,
              "timestamp.csv": ts, "audit_has_eligible.csv": eligibility,
              "has_metrics.csv": hm, "has_tests.csv": ht, "runtime.csv": rt,
              "locked_main.csv": lm, "locked_tests.csv": lt, "locked_groups.csv": lg,
              "locked_downstream.csv": ld, "locked_ablations.csv": la}
    for name, df in tables.items():
        write_csv(df, out_dir / name)

    summary: dict = {}
    summary.update(flatten(mm, "main", ["regime", "method"]))
    summary.update(range_keys(mm, "main.range", ["method"],
                              ["marginal_coverage", "trace_coverage", "abs_dev",
                               "frac_inf_instance", "frac_inf_trace_rows", "frac_inf_trace",
                               "finite_coverage", "finite_coverage_trace", "width_finite_median",
                               "winkler_median", "coverage_pos", "worst_group_coverage",
                               "zero_share"],
                              {"all_regimes": REGIMES, "shifted_regimes": SHIFTED}))
    summary.update(flatten(gs, "main_group", ["regime", "method", "group"]))
    dg_main = dg.drop(columns=[c for c in bound_cols if c not in ("tag", "method", "delay_s",
                                                                   "regime")])
    summary.update(flatten(dg_main, "delay", ["tag", "method", "delay_s", "regime"]))
    summary.update(range_keys(dg_main[dg_main.tag == "main"], "delay.range",
                              ["method", "delay_s"],
                              ["marginal_coverage", "trace_coverage", "abs_dev",
                               "frac_inf_instance", "frac_inf_trace_rows", "frac_inf_trace",
                               "finite_coverage", "finite_coverage_trace", "width_finite_median",
                               "winkler_median"],
                              {"all_regimes": REGIMES, "shifted_regimes": SHIFTED}))
    summary.update(flatten(bound_tab, "delay_bound", ["tag", "method", "delay_s", "regime"]))
    for (tag, method, d), g in chk_all.groupby(["tag", "method", "delay_s"], sort=False):
        base = f"delay_bound.{tag}.{method}.{id_token('delay_s', d)}.all_regimes"
        summary[f"{base}.K_max"] = int(g.K.max())
        summary[f"{base}.max_ratio_applied_upper"] = json_value(g.ratio_applied_max.max())
        summary[f"{base}.max_ratio_applied_lower"] = json_value(g.ratio_applied_min.max())
        summary[f"{base}.max_ratio_all"] = json_value(g.ratio_all.max())
    summary.update(flatten(oc, "delay_K", ["tag", "method", "delay_s", "regime", "K"]))
    summary.update(flatten(h5, "h5", ["regime", "method", "feedback"]))
    summary.update(flatten(ts, "ts", ["regime", "method"]))
    summary.update({f"audit.{k}": json_value(v) for k, v in audit.items()})
    summary.update(flatten(eligibility, "audit.has_eligible", ["regime", "seed"]))
    summary.update(flatten(hm, "has", ["regime", "variant", "metric"]))
    summary.update(flatten(ht, "has_test", ["regime", "variant_a", "variant_b"]))
    for i, (nom, meas) in enumerate(zip(lad["nominal_kbps"], lad["measured_mean_kbps"])):
        summary[f"has_ladder.rep{i + 1}.nominal_kbps"] = nom
        summary[f"has_ladder.rep{i + 1}.measured_mean_kbps"] = meas
    summary["has_ladder.n_video_segments"] = lad["n_video_segments"]
    summary["has_ladder.segment_s"] = lad["segment_s"]
    summary["has_ladder.qoe_mu"] = lad["nominal_kbps"][-1] / 1000.0
    summary["has_ladder.bitrate_basis"] = ("nominal bitrates for decisions, average bitrate and "
                                           "QoE; per-segment sizes for download times")
    rt_ids = rt.assign(mode=rt.get("mode", pd.Series(dtype=str)).fillna(""))
    rt_ids["key"] = [f"{s}.{m}.{t}".replace("..", ".").replace(" ", "_")
                     for s, m, t in zip(rt_ids.source, rt_ids["mode"], rt_ids.tag)]
    summary.update(flatten(rt_ids.drop(columns=["source", "mode", "tag"]), "runtime", ["key"]))
    summary.update(flatten(lm, "locked.main", ["regime", "method"]))
    lt_ids = lt.copy()
    summary.update(flatten(lt_ids, "locked.test", ["family", "regime", "method_a", "method_b"]))
    summary.update(flatten(lg, "locked.group", ["regime", "method", "group"]))
    summary.update(flatten(ld, "locked.downstream", ["regime", "method", "metric"]))
    summary.update(flatten(la, "locked.ablation", ["tag", "regime", "base_model", "alpha",
                                                   "gamma", "method"]))
    summary.update({f"locked.derived.{k}": json_value(v) for k, v in lder.items()})
    summary.update({f"check.{k}": json_value(v) for k, v in checks.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    dig = digest({"main_methods": mm, "main_groups": gs, "delay_grid": dg,
                  "delay_outstanding": oc, "horizon5": h5, "timestamp": ts, "audit": audit,
                  "eligibility": eligibility, "has_distinct": has_distinct, "has_metrics": hm,
                  "has_tests": ht, "ladder": lad, "runtime": rt, "locked_main": lm,
                  "locked_tests": lt, "locked_ablations": la, "locked_downstream": ld,
                  "locked_derived": lder, "checks": checks})
    (out_dir / "digest.md").write_text(dig, encoding="utf-8", newline="\n")
    print(f"wrote {len(tables)} tables, summary.json ({len(summary)} keys) and digest.md "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
