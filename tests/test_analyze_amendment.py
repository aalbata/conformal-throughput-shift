"""Aggregation rules of scripts/analyze_amendment.py on small synthetic inputs."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import delay_coverage_bound  # noqa: E402
from src.stats import bh_correct, bootstrap_ci_mean, paired_wilcoxon  # noqa: E402


@pytest.fixture(scope="module")
def aa():
    spec = importlib.util.spec_from_file_location("analyze_amendment",
                                                  ROOT / "scripts" / "analyze_amendment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    """Two seeds, three traces; trace a in both seeds, b only in seed 0, c only in seed 1."""
    return pd.DataFrame({
        "seed": [0, 0, 1, 1],
        "trace_id": ["a", "b", "a", "c"],
        "n": [10, 30, 10, 20],
        "n_cov": [9, 24, 10, 16],
        "n_inf": [2, 0, 5, 4],
        "n_finite": [8, 30, 5, 16],
        "n_cov_finite": [7, 24, 5, 12],
        "width_finite_median": [2.0, 10.0, 4.0, 6.0],
        "width_finite_mean": [3.0, 12.0, 5.0, 7.0],
        "winkler_finite_mean": [10.0, 30.0, 20.0, 40.0],
        "n_inverted": [1, 0, 2, 0],
        "n_y0": [1, 3, 0, 2],
        "n_pos": [9, 27, 10, 18],
        "n_cov_pos": [8, 21, 10, 14],
    })


def test_interval_summary_marginal_and_trace_aggregation(aa):
    s = aa.interval_summary(_rows(), alpha=0.1)
    assert s["n_seeds"] == 2 and s["n_traces"] == 3 and s["n_instances"] == 70
    seeds = np.array([33 / 40, 26 / 30])
    assert s["marginal_coverage"] == pytest.approx(seeds.mean(), abs=1e-15)
    assert (s["marginal_coverage_ci_lo"], s["marginal_coverage_ci_hi"]) == \
        pytest.approx(bootstrap_ci_mean(seeds)[1:], abs=1e-15)
    # trace coverage: a = (0.9 + 1.0) / 2, b = 0.8, c = 0.8
    assert s["trace_coverage"] == pytest.approx((0.95 + 0.8 + 0.8) / 3)
    # Delta_s from seed-averaged coverage (0.05, 0.1, 0.1), not the row mean 0.075
    assert s["abs_dev"] == pytest.approx((0.05 + 0.1 + 0.1) / 3)
    assert s["frac_inf_instance"] == pytest.approx((2 / 40 + 9 / 30) / 2)
    assert s["frac_inf_trace_rows"] == pytest.approx((0.2 + 0.0 + 0.5 + 0.2) / 4)
    assert s["frac_inf_trace"] == pytest.approx((0.35 + 0.0 + 0.2) / 3)
    assert s["finite_coverage"] == pytest.approx((31 / 38 + 17 / 21) / 2)
    assert s["finite_coverage_trace"] == pytest.approx(((7 / 8 + 1.0) / 2 + 0.8 + 0.75) / 3)
    assert s["width_finite_median"] == pytest.approx(6.0)        # median of 3, 10, 6
    assert s["width_finite_mean_median"] == pytest.approx(7.0)   # median of 4, 12, 7
    assert s["winkler_median"] == pytest.approx(30.0)            # median of 15, 30, 40
    assert s["winkler_mean"] == pytest.approx((15 + 30 + 40) / 3)
    assert s["n_inverted"] == 3
    assert s["coverage_pos"] == pytest.approx((29 / 36 + 24 / 28) / 2)
    assert s["zero_share"] == pytest.approx((4 / 40 + 2 / 30) / 2)
    assert s["n_trace_rows_without_finite"] == 0


def test_interval_summary_bootstrap_order_is_by_seed(aa):
    rows = _rows()
    shuffled = rows.iloc[[3, 1, 2, 0]].reset_index(drop=True)
    a, b = aa.interval_summary(rows), aa.interval_summary(shuffled)
    for k in ("marginal_coverage_ci_lo", "trace_coverage_ci_hi", "abs_dev_ci_lo"):
        assert a[k] == b[k]


def test_interval_summary_without_optional_columns(aa):
    rows = _rows()[["seed", "trace_id", "n", "n_cov", "n_inf"]]
    s = aa.interval_summary(aa.with_finite_counts(rows))
    assert s["finite_coverage"] == pytest.approx((31 / 38 + 17 / 21) / 2)
    assert np.isnan(s["width_finite_median"]) and np.isnan(s["winkler_median"])
    assert np.isnan(s["coverage_pos"]) and np.isnan(s["zero_share"])
    assert s["n_inverted"] == -1


def test_counts_from_locked_is_exact_and_rejects_non_integral(aa):
    locked = pd.DataFrame({"seed": [0, 0], "trace_id": ["a", "b"], "n": [7, 9],
                           "coverage": [6 / 7, 1.0], "frac_inf": [2 / 7, 1 / 9]})
    c = aa.counts_from_locked(locked)
    assert c["n_cov"].tolist() == [6, 9] and c["n_inf"].tolist() == [2, 1]
    assert c["n_finite"].tolist() == [5, 8] and c["n_cov_finite"].tolist() == [4, 8]
    bad = locked.assign(coverage=[0.5, 1.0])
    with pytest.raises(ValueError):
        aa.counts_from_locked(bad)


def test_select_d0(aa):
    rows = pd.DataFrame({"feedback": ["none", "delayed", "delayed", "delayed"],
                         "delay_s": [np.nan, 0.0, 1.0, 10.0]})
    assert aa.select_d0(rows).index.tolist() == [0, 1]


def test_group_summary_and_worst_group(aa):
    g = pd.DataFrame({"regime": "r", "method": "m",
                      "group": ["g1", "g2"] * 3, "seed": [2, 2, 0, 0, 1, 1],
                      "n": [100, 50, 120, 40, 110, 60],
                      "n_cov": [90, 40, 108, 30, 99, 54]})
    gs = aa.group_summary(g)
    g1 = gs[gs.group == "g1"].iloc[0]
    g2 = gs[gs.group == "g2"].iloc[0]
    assert g1.coverage == pytest.approx(0.9)
    by_seed = np.array([30 / 40, 54 / 60, 40 / 50])            # seeds 0, 1, 2
    assert g2.coverage == pytest.approx(by_seed.mean())
    assert (g2.coverage_ci_lo, g2.coverage_ci_hi) == pytest.approx(bootstrap_ci_mean(by_seed)[1:])
    assert (g2.n_test_min, g2.n_test_max, g2.n_test_total) == (40, 60, 150)
    w = aa.worst_groups(gs).iloc[0]
    assert w.worst_group == "g2" and w.worst_group_coverage == pytest.approx(by_seed.mean())
    assert w.max_group_abs_dev == pytest.approx(abs(by_seed.mean() - 0.9))
    assert w.min_group_n_test == 40


def test_bound_values_match_manuscript_constants_and_src(aa):
    k, m, n = np.array([0, 4]), np.array([100, 100]), np.array([100, 100])
    applied, all_form = aa.bound_values(k, m, n, alpha=0.1, gamma=0.05)
    assert applied == pytest.approx([19 / 100, 23 / 100])
    assert all_form == pytest.approx([19.9 / 100, 27.5 / 100])
    assert applied == pytest.approx(delay_coverage_bound(0.1, 0.05, k, m))
    assert np.isinf(aa.bound_values(0, 0, 5)[0])


def _delayed_rows():
    return pd.DataFrame({
        "tag": "main", "regime": "r", "seed": 0, "method": "aci", "delay_s": 0.0,
        "trace_id": ["ok", "violated", "unresolved", "ge1"],
        "n": [100, 100, 1000, 100], "n_cov": [90, 50, 870, 90],
        "max_outstanding": [0, 0, 9, 0], "n_applied": [99, 99, 990, 99],
        "n_alpha_le0": [0, 0, 0, 0], "n_alpha_ge1": [0, 0, 0, 2],
        "n_alpha_ge1_covered": [0, 0, 0, 1],
    })


def test_bound_check_brackets_the_applied_label_deviation(aa):
    rows = _delayed_rows()
    rows["bound"] = delay_coverage_bound(0.1, 0.05, rows.max_outstanding, rows.n_applied)
    chk = aa.bound_check(rows).set_index("trace_id")
    # ok: 10 misses, 1 unapplied label -> applied misses in [9, 10] of 99
    assert chk.loc["ok", "dev_applied_min"] == 0.0
    assert chk.loc["ok", "dev_applied_max"] == pytest.approx(0.1 - 9 / 99)
    assert chk.loc["ok", "ratio_applied_max"] == pytest.approx((0.1 - 9 / 99) / (19 / 99))
    assert chk.loc["ok", "ratio_all"] == pytest.approx(0.0)
    # violated: applied misses in [49, 50] of 99, far above alpha + 19/99
    assert chk.loc["violated", "ratio_applied_min"] > 1
    # unresolved: applied misses in [120, 130] of 990 straddle the bound 1.4 / 49.5
    b = 1.4 / (0.05 * 990)
    assert chk.loc["unresolved", "bound_applied"] == pytest.approx(b)
    assert chk.loc["unresolved", "dev_applied_min"] == pytest.approx(120 / 990 - 0.1)
    assert chk.loc["unresolved", "dev_applied_max"] == pytest.approx(130 / 990 - 0.1)
    assert chk.loc["unresolved", "ratio_applied_min"] <= 1 < chk.loc["unresolved", "ratio_applied_max"]
    assert not chk.loc["ge1", "bound_applies"] and chk.loc["ok", "bound_applies"]
    assert chk["unapplied_within_K_plus_1"].all()
    assert chk["bound_stored_abs_diff"].max() < 1e-15
    s = aa.bound_summary(chk.reset_index(), rows)
    assert (s["n_rows_applied_proven"], s["n_rows_applied_violated"],
            s["n_rows_applied_unresolved"]) == (2, 1, 1)
    assert s["n_rows_bound_not_applicable"] == 1 and s["n_alpha_ge1_covered"] == 1
    assert s["K_max"] == 9 and s["unapplied_max"] == 10


def test_outstanding_counts_shares(aa):
    chk = pd.DataFrame({"tag": "main", "method": "aci", "delay_s": 1.0,
                        "regime": ["r1", "r1", "r2", "r2"], "K": [1, 2, 2, 2]})
    oc = aa.outstanding_counts(chk)
    pooled = oc[oc.regime == "all"].set_index("K")
    assert pooled.loc[2, "n_trace_rows"] == 3 and pooled.loc[1, "share"] == pytest.approx(0.25)
    assert oc.groupby("regime")["share"].sum().tolist() == pytest.approx([1.0, 1.0, 1.0])


def _has_rows(aa, rng):
    parts = []
    for regime in aa.HAS_TEST_REGIMES:
        for seed in (0, 1):
            for t in range(6):
                for variant in aa.HAS_VARIANTS:
                    rec = {"regime": regime, "seed": seed, "trace_id": f"t{t}",
                           "tech": "4G" if t < 4 else "5G", "variant": variant}
                    rec.update({m: float(rng.normal(5, 2)) for m in aa.HAS_METRICS})
                    parts.append(rec)
    return pd.DataFrame(parts)


def test_has_trace_means_summary_and_tests(aa):
    rng = np.random.default_rng(3)
    has = _has_rows(aa, rng)
    sel = (has.regime == "cross_mobility_4g") & (has.variant == "rate_split") & (has.trace_id == "t0")
    has.loc[sel & (has.seed == 0), "mean_abs_switch_mbps"] = np.nan
    has.loc[(has.regime == "cross_mobility_4g") & (has.variant == "rate_split")
            & (has.trace_id == "t1"), "mean_abs_switch_mbps"] = np.nan
    tm = aa.has_trace_means(has)
    t0 = tm[(tm.regime == "cross_mobility_4g") & (tm.variant == "rate_split") & (tm.trace_id == "t0")]
    expect = has.loc[sel & (has.seed == 1), "mean_abs_switch_mbps"].iloc[0]
    assert t0["mean_abs_switch_mbps"].iloc[0] == pytest.approx(expect)
    assert t0["n_seeds"].iloc[0] == 2
    hs = aa.has_summary(tm)
    row = hs[(hs.regime == "cross_mobility_4g") & (hs.variant == "rate_split")
             & (hs.metric == "mean_abs_switch_mbps")].iloc[0]
    assert (row.n_traces, row.n_traces_undefined) == (5, 1)
    q = tm[(tm.regime == "cross_tech_4to5") & (tm.variant == "mpc_split")].sort_values("trace_id")
    r = hs[(hs.regime == "cross_tech_4to5") & (hs.variant == "mpc_split")
           & (hs.metric == "qoe_per_segment")].iloc[0]
    assert (r["mean"], r.mean_ci_lo, r.mean_ci_hi) == pytest.approx(
        bootstrap_ci_mean(q["qoe_per_segment"].to_numpy()))
    assert r["median"] == pytest.approx(np.median(q["qoe_per_segment"]))

    ht = aa.has_tests(tm)
    assert len(ht) == 12
    ps = []
    for regime in aa.HAS_TEST_REGIMES:
        a = tm[(tm.regime == regime) & (tm.variant == "mpc_mondrian_aci")].set_index("trace_id")
        for comp in aa.HAS_TEST_COMPARATORS:
            b = tm[(tm.regime == regime) & (tm.variant == comp)].set_index("trace_id")
            res = paired_wilcoxon(a["qoe_per_segment"].to_numpy(), b["qoe_per_segment"].to_numpy())
            got = ht[(ht.regime == regime) & (ht.variant_b == comp)].iloc[0]
            assert got.p == pytest.approx(res["p"]) and got.n == res["n"] == 6
            assert got.rank_biserial == pytest.approx(res["rank_biserial"])
            ps.append(res["p"])
    assert ht.p_bh.tolist() == pytest.approx(bh_correct(ps, 0.05)[1])


def test_has_eligibility_fills_missing_seeds(aa):
    has = pd.DataFrame({"regime": ["in_domain_5g"] * 3, "seed": [0, 0, 2],
                        "trace_id": ["x", "y", "x"], "tech": ["5G", "5G", "5G"]})
    el = aa.has_eligibility(has)
    sub = el[el.regime == "in_domain_5g"].set_index("seed")
    assert sub.loc[0, "n_eligible"] == 2 and sub.loc[1, "n_eligible"] == 0
    assert sub.loc[2, "n_eligible_5g"] == 1 and sub.loc[2, "n_eligible_4g"] == 0
    assert len(el) == len(aa.HAS_REGIMES) * len(aa.HAS_SEEDS)


def test_ladder_bitrates_from_segment_sizes(aa, tmp_path):
    reps = [39537, 14817, 4282, 1815, 752, 237]
    lines = ["bbb, seg_Dur, " + ", ".join(map(str, reps)),
             "x264, 6, " + ", ".join(["100"] * 6),
             ", , " + ", ".join(["50"] * 6),
             "0, 4, " + ", ".join(["1"] * 6)]
    sizes = np.array([[1000 * (i + 1) * (j + 1) for j in range(6)] for i in range(3)])
    for i, row in enumerate(sizes, start=1):
        lines.append(f"{i}, 4, " + ", ".join(map(str, row)))
    p = tmp_path / "ladder.csv"
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    lb = aa.ladder_bitrates(p)
    assert lb["nominal_kbps"] == sorted(float(r) for r in reps)
    # columns in file order map to descending nominal bitrates
    expect = sizes.mean(axis=0)[::-1] * 8 / 4 / 1000
    assert lb["measured_mean_kbps"] == pytest.approx(expect.tolist())
    assert lb["n_video_segments"] == 3 and lb["segment_s"] == 4.0


def test_data_audit_shares(aa):
    feat = pd.DataFrame({
        "trace_id": ["a", "a", "b", "b", "c", "d"],
        "tech": ["4G", "4G", "5G", "5G", "5G", "5G"],
        "app": ["unknown", "unknown", "Download", "Download", "Netflix", "Amazon_Prime"],
        "mobility": ["car", "car", "static", "static", "driving", "driving"],
        "y": [1.0, 0.0, 2.0, 0.0, 0.0, 3.0],
        "ts": [0.0, 1.0, 5.0, 6.0, 0.0, 2.0],
        "ts_label": [1.0, 3.0, 6.0, 6.0, 1.0, 3.0],
    })
    a = aa.data_audit(feat)
    assert (a["n_traces"], a["n_traces_4g"], a["n_traces_5g"]) == (4, 1, 3)
    assert a["n_traces_5g_download"] == 1 and a["n_traces_streaming_eligible"] == 2
    assert a["h1_share_target_exactly_1s"] == pytest.approx(4 / 6)
    assert a["h1_share_target_0s"] == pytest.approx(1 / 6)
    assert a["h1_share_target_ge_2s"] == pytest.approx(1 / 6)
    assert a["zero_share_4g"] == pytest.approx(0.5) and a["zero_share_5g"] == pytest.approx(0.5)
    assert a["zero_share_app_shift_test"] == pytest.approx(0.5)
    assert a["n_traces_5g_driving"] == 2


def test_audit_store_round_trip_is_exact(aa, tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    feat = pd.DataFrame({
        "trace_id": ["a", "a", "b", "c"], "tech": ["4G", "4G", "5G", "5G"],
        "app": ["unknown", "unknown", "Download", "Netflix"],
        "mobility": ["car", "car", "static", "driving"],
        "y": [0.1, 0.0, 2.0 / 3.0, 3.0], "ts": [0.0, 1.0, 5.0, 0.0],
        "ts_label": [1.0, 3.0, 6.0, 1.0]})
    feat.to_csv(proc / "features_h1_timed.csv.gz", index=False)
    feat.assign(ts_label=feat.ts + 5.0).to_csv(proc / "features_h5_timed.csv.gz", index=False)
    (proc / "data_audit_h1.json").write_text(json.dumps({"n_raw_rows": 10}), encoding="utf-8")
    (proc / "timestamps_audit.json").write_text(
        json.dumps({"n_raw_rows": 10, "timebase": {"n_instances": 3}}), encoding="utf-8")
    reps = [237, 752, 1815, 4282, 14817, 39537]
    lines = ["bbb, seg_Dur, " + ", ".join(map(str, reps)), "x264, 6, " + ", ".join(["1"] * 6),
             ", , " + ", ".join(["1"] * 6), "0, 4, " + ", ".join(["1"] * 6),
             "1, 4, " + ", ".join(str(1000 * r // 7) for r in reps)]
    ladder = tmp_path / "ladder.csv"
    ladder.write_text("\n".join(lines) + "\n", encoding="ascii")
    store = aa.compute_audit(proc, ladder)
    assert store["data_audit"]["n_raw_rows"] == 10
    assert store["data_audit"]["n_instances_timestamp_features"] == 3
    assert store["data_audit"]["n_instances_h5"] == 4
    path = tmp_path / "audit.json"
    aa.write_audit(store, path)
    back = aa.read_audit(path)
    assert back == store
    for part in ("data_audit", "ladder"):
        assert list(back[part]) == list(store[part])
        for k, v in store[part].items():
            assert type(back[part][k]) is type(v)
    with pytest.raises(ValueError):
        path.write_text(json.dumps({"data_audit": {}}), encoding="utf-8")
        aa.read_audit(path)


def test_timeline_utf16_and_timing_summary(aa, tmp_path):
    text = ("2026-10-21T10:00:00Z  batch1 main start\n"
            "2026-10-21T10:20:00Z  batch1 main done\n"
            "2026-10-21T10:20:00Z  batch2 h5 start\n"
            "2026-10-21T10:22:30Z  batch2 done\n")
    p = tmp_path / "timeline.txt"
    p.write_bytes(text.encode("utf-16"))
    tl = aa.parse_timeline(aa.read_text_any(p)).set_index("tag")
    assert tl.loc["batch1 main", "wall_minutes"] == pytest.approx(20.0)
    assert tl.loc["batch2 h5", "wall_minutes"] == pytest.approx(2.5)
    assert tl.loc["total", "wall_minutes"] == pytest.approx(22.5)
    timing = pd.DataFrame({
        "mode": "full", "tag": "main", "status": "ok", "pid": [1, 2, 1],
        "seconds_total": [60.0, 30.0, 60.0],
        "started_utc": ["2026-10-21T10:00:00+00:00", "2026-10-21T10:00:30+00:00",
                        "2026-10-21T11:00:00+00:00"],
        "finished_utc": ["2026-10-21T10:01:00+00:00", "2026-10-21T10:01:30+00:00",
                         "2026-10-21T11:01:00+00:00"]})
    ts = aa.timing_summary(timing).iloc[0]
    assert ts.n_cells == 3 and ts.n_workers == 2 and ts.worker_seconds_total == 150.0
    assert ts.wall_minutes == pytest.approx(61.0) and ts.busy_minutes == pytest.approx(2.5)


def test_flatten_json_values_and_ranges(aa):
    df = pd.DataFrame({"method": ["aci", "aci"], "delay_s": [0.0, 10.0],
                       "regime": ["r1", "r2"], "x": [0.5, np.nan], "k": [3, 4],
                       "g": ["speed_0", "speed_low"]})
    flat = aa.flatten(df, "delay", ["method", "delay_s", "regime"])
    assert flat["delay.aci.D0.r1.x"] == 0.5 and flat["delay.aci.D10.r2.x"] is None
    assert flat["delay.aci.D0.r1.k"] == 3 and isinstance(flat["delay.aci.D0.r1.k"], int)
    assert flat["delay.aci.D10.r2.g"] == "speed_low"
    json.dumps(flat, allow_nan=False)
    assert aa.json_value(np.inf) is None and aa.json_value(pd.NA) is None
    assert aa.id_token("delay_s", np.nan) == "Dnone" and aa.id_token("alpha", 0.05) == "alpha0.05"
    rk = aa.range_keys(df.assign(x=[0.5, 0.7]), "main.range", ["method"], ["x"],
                       {"all": ["r1", "r2"], "one": ["r2"]})
    assert rk["main.range.aci.all.x_min"] == 0.5 and rk["main.range.aci.all.x_max"] == 0.7
    assert rk["main.range.aci.one.x_min"] == 0.7


def test_locked_main_summary_definitions(aa):
    rows = pd.DataFrame({
        "regime": "r", "method": "m", "seed": [0, 0, 1, 1],
        "trace_id": ["a", "b", "a", "c"], "n": [10, 30, 10, 20],
        "coverage": [0.9, 0.8, 1.0, 0.8], "frac_inf": [0.2, 0.0, 0.5, 0.2],
        "marginal_coverage": [33 / 40, 33 / 40, 26 / 30, 26 / 30],
        "median_width": [2.0, 10.0, 4.0, 6.0], "winkler": [1.0, 2.0, 3.0, 4.0],
        "rmse_point": [5.0, 5.0, 7.0, 7.0]})
    s = aa.locked_main_summary(rows).iloc[0]
    assert s.marginal_coverage == pytest.approx((33 / 40 + 26 / 30) / 2)
    assert s.marginal_coverage == pytest.approx(s.marginal_coverage_from_counts)
    assert s.median_width == pytest.approx(6.0)
    assert s.frac_inf_trace_rows == pytest.approx(0.225)
    assert s.frac_inf_instance == pytest.approx((2 / 40 + 9 / 30) / 2)
    assert s.finite_coverage == pytest.approx((31 / 38 + 17 / 21) / 2)
    assert s.winkler_median_rows == pytest.approx(2.5) and s.rmse_point == pytest.approx(6.0)
