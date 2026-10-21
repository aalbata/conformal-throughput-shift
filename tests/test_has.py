"""Streaming simulator tests: hand-computed outcomes on small synthetic traces."""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.has import (VARIANTS, Ladder, download_finish, harmonic_mean,  # noqa: E402
                     load_ladder, mpc_choice, rate_choice, simulate)


def run(capacity, rule="rate", estimator="hm", ladder=(1000, 2000), bits=((4e6, 8e6),),
        start=0, estimates=None, **kw):
    kw.setdefault("latency_s", 0.0)
    return simulate(np.asarray(capacity, dtype=float), start, estimates, ladder,
                    np.asarray(bits, dtype=float), rule, estimator, **kw)


def test_constant_capacity_startup_waiting_and_qoe():
    # 4 Mbit/s; segments 4 Mbit (1 Mbit/s) or 8 Mbit (2 Mbit/s); 12-s buffer, so the
    # player waits once the buffer exceeds 8 s.
    m, log = run([4.0] * 40, buffer_s=12.0)
    assert log["rep"].tolist() == [0] + [1] * 11
    assert log["request_s"][:5].tolist() == pytest.approx([0, 1, 3, 5, 9])
    assert log["finish_s"][:5].tolist() == pytest.approx([1, 3, 5, 7, 11])
    assert log["wait_s"].tolist() == pytest.approx([0, 0, 0] + [2] * 8 + [1])
    assert log["throughput_mbps"].tolist() == pytest.approx([4.0] * 12)
    assert m["n_segments"] == 12
    assert m["startup_s"] == pytest.approx(1.0)
    assert m["rebuffer_s"] == 0.0 and m["stalls"] == 0 and m["discarded"] == 0
    assert m["play_s"] == pytest.approx(39.0)
    assert m["wait_s"] == pytest.approx(17.0)
    assert m["avg_bitrate_mbps"] == pytest.approx(23 / 12)
    assert m["switches"] == 1
    assert m["switches_per_min"] == pytest.approx(60 / 39)
    assert m["mean_abs_switch_mbps"] == pytest.approx(1.0)
    # mu = mu_s = highest bitrate = 2 Mbit/s: 23 - 1 - 2 * 0 - 2 * 1
    assert m["qoe_total"] == pytest.approx(20.0)
    assert m["qoe_per_segment"] == pytest.approx(20 / 12)
    assert m["duration_s"] == 40.0


def test_capacity_drop_forces_stall_and_final_download_is_discarded():
    cap = [8.0] * 3 + [0.5] * 27
    m, log = run(cap, buffer_s=30.0)
    assert log["rep"].tolist() == [0, 1, 1, 1, 1, 0]
    assert log["finish_s"][:5].tolist() == pytest.approx([0.5, 1.5, 2.5, 11.0, 27.0])
    assert np.isinf(log["finish_s"][5])
    # harmonic mean of 8, 8, 8, 8/8.5 and of those plus 0.5
    assert log["estimate_mbps"][4] == pytest.approx(4 / (3 / 8 + 8.5 / 8))
    assert log["estimate_mbps"][5] == pytest.approx(5 / (3 / 8 + 8.5 / 8 + 2))
    assert log["rebuffer_s"][:5].tolist() == pytest.approx([0, 0, 0, 0, 10.5])
    assert m["n_segments"] == 5
    assert m["stalls"] == 1
    assert m["rebuffer_s"] == pytest.approx(10.5)
    assert m["rebuffer_final_s"] == 0.0
    assert m["discarded"] == 1
    assert m["startup_s"] == pytest.approx(0.5)
    assert m["play_s"] == pytest.approx(19.0)
    assert m["rebuffer_ratio"] == pytest.approx(10.5 / 29.5)
    assert m["qoe_total"] == pytest.approx(9 - 1 - 2 * 10.5 - 2 * 0.5)
    assert m["qoe_per_segment"] == pytest.approx(-14 / 5)


def test_download_finish_with_zero_capacity_seconds():
    cap = [4.0, 0.0, 0.0, 4.0]
    cum = np.concatenate(([0.0], np.cumsum(cap))).tolist()
    assert download_finish(cum, cap, 0.0, 6.0) == pytest.approx(3.5)
    assert download_finish(cum, cap, 1.5, 2.0) == pytest.approx(3.5)
    assert download_finish(cum, cap, 0.5, 2.0) == pytest.approx(1.0)
    assert download_finish(cum, cap, 3.0, 4.0) == pytest.approx(4.0)
    assert download_finish(cum, cap, 0.0, 9.0) == np.inf
    assert download_finish(cum, cap, 4.0, 1.0) == np.inf
    m, log = run([4.0, 0.0, 0.0, 4.0, 4.0], ladder=(1000,), bits=((6e6,),))
    assert log["finish_s"][0] == pytest.approx(3.5)
    assert log["throughput_mbps"][0] == pytest.approx(6 / 3.5)
    assert m["startup_s"] == pytest.approx(3.5)


def test_stall_running_into_the_end_of_the_trace():
    cap = [8.0, 8.0] + [0.0] * 18
    m, log = run(cap, ladder=(1000,), bits=((4e6,),), buffer_s=30.0)
    assert log["finish_s"][:4].tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert np.isinf(log["finish_s"][4])
    assert m["n_segments"] == 4
    assert m["rebuffer_final_s"] == pytest.approx(3.5)
    assert m["rebuffer_s"] == pytest.approx(3.5)
    assert m["stalls"] == 1
    assert m["play_s"] == pytest.approx(16.0)
    assert m["rebuffer_ratio"] == pytest.approx(3.5 / 19.5)
    # buffer runs out exactly at the end of the trace: no stall
    m, log = run([4.0, 0.0, 0.0, 0.0, 0.0], ladder=(1000,), bits=((4e6,),))
    assert np.isinf(log["finish_s"][1])
    assert m["stalls"] == 0 and m["rebuffer_s"] == 0.0
    assert m["play_s"] == pytest.approx(4.0)


def test_latency_and_start_second():
    m, log = run([2.0] * 10, ladder=(1000,), bits=((1e6,),), start=3, latency_s=0.08)
    assert log["request_s"][0] == 3.0
    assert m["startup_s"] == pytest.approx(0.58)
    assert log["throughput_mbps"][0] == pytest.approx(2.0)
    assert m["duration_s"] == 7.0


def test_external_estimates_switch_counting_forward_fill_and_floor():
    # 8-s buffer: after the first segment every request waits for 4 s of space, so
    # requests fall in seconds 0, 0, 4, 8, 12, 16.
    est = np.full(20, np.nan)
    est[0], est[4], est[12], est[16] = 5.0, 1.5, 2.5, -0.3
    m, log = run([100.0] * 20, estimator="point", ladder=(1000, 2000, 4000),
                 bits=((1e6, 1e6, 1e6),), buffer_s=8.0, estimates={"point": est})
    assert log["request_s"].tolist() == pytest.approx([0, 0.01, 4.01, 8.01, 12.01, 16.01])
    assert log["estimate_mbps"][1:].tolist() == pytest.approx([5.0, 1.5, 1.5, 2.5, 0.01])
    assert log["rep"].tolist() == [0, 2, 0, 0, 1, 0]
    assert m["switches"] == 4
    assert m["mean_abs_switch_mbps"] == pytest.approx(8 / 4)
    assert m["avg_bitrate_mbps"] == pytest.approx(10 / 6)


def test_no_estimate_before_first_instance_gives_lowest():
    est = np.full(30, np.nan)
    est[10] = 50.0
    _, log = run([100.0] * 30, estimator="point", ladder=(1000, 2000, 4000),
                 bits=((1e6, 1e6, 1e6),), buffer_s=8.0, estimates={"point": est})
    early = log["request_s"] < 10
    assert np.all(log["rep"][early] == 0)
    assert np.all(log["rep"][~early] == 2)


@pytest.mark.parametrize("rule", ["rate", "mpc"])
def test_zero_lower_bound_is_floored(rule):
    est = np.zeros(30)
    m, log = run([5.0] * 30, rule=rule, estimator="aci", ladder=(1000, 2000, 4000),
                 bits=((4e6, 8e6, 16e6),), estimates={"aci": est})
    assert np.all(log["estimate_mbps"][1:] == 0.01)
    assert np.all(log["rep"] == 0)


def test_segment_sizes_cycle():
    _, log = run([2.0] * 20, ladder=(1000,), bits=((2e6,), (6e6,)))
    assert log["mbit"][:5].tolist() == pytest.approx([2, 6, 2, 6, 2])
    assert log["finish_s"][:5].tolist() == pytest.approx([1, 4, 5, 8, 9])


def test_harmonic_mean_and_rate_choice():
    assert harmonic_mean([1.0, 2.0, 4.0]) == pytest.approx(3 / 1.75)
    assert harmonic_mean([3.0]) == 3.0
    ladder = [1.0, 2.0, 4.0]
    assert rate_choice(0.5, ladder) == 0
    assert rate_choice(2.0, ladder) == 1
    assert rate_choice(3.99, ladder) == 1
    assert rate_choice(40.0, ladder) == 2


def test_robust_estimate_arithmetic():
    # segment throughputs 4, 2, 4, 8, 8, 4, 4, ...
    cap = [4.0, 2.0, 2.0, 4.0, 8.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    _, log = run(cap, estimator="robust", ladder=(1000,), bits=((4e6,),))
    assert log["throughput_mbps"][:7].tolist() == pytest.approx([4, 2, 4, 8, 8, 4, 4])
    # error of the previous harmonic mean on the segment it was made for, maximum of
    # the last five errors
    expected = [4.0, (8 / 3) / 2, 3 / 2, (32 / 9) / 2, 4 / 2, 4 / 2, 5 / (1 + 5 / 8)]
    assert log["estimate_mbps"][1:8].tolist() == pytest.approx(expected)


def test_mpc_hand_computed_optimum():
    sizes = np.array([[4.0, 8.0], [4.0, 8.0]])
    ladder = [1.0, 2.0]
    kw = dict(seg_s=4.0, latency_s=0.0, buffer_s=30.0)
    # estimate 4: no sequence rebuffers; (2, 2) maximises 4 - |2 - 1|
    assert mpc_choice(4.0, 2.0, 0, sizes, ladder, 2.0, **kw) == 1
    # estimate 2: rep 1 first rebuffers 2 s; (1, 1) and (1, 2) tie at 2, first wins
    assert mpc_choice(2.0, 2.0, 0, sizes, ladder, 2.0, **kw) == 0
    # rate rule would take rep 1 at estimate 2
    assert rate_choice(2.0, ladder) == 1
    # smoothing: horizon 1 ties 1 - 0 against 4 - 3; horizon 2 prefers (4, 4)
    big = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert mpc_choice(100.0, 10.0, 0, big[:1], [1.0, 4.0], 4.0, **kw) == 0
    assert mpc_choice(100.0, 10.0, 0, big, [1.0, 4.0], 4.0, **kw) == 1


def test_mpc_planned_buffer_is_capped():
    # Buffer 26 s of 30 s: a near-instant first download cannot raise the planned
    # buffer above 26 s, so both first choices rebuffer 4 s on the 30-s second
    # download and (2, 2) wins with 4 - 2 * 4. Without the cap, a low first segment
    # would avoid the rebuffering and win.
    sizes = np.array([[1e-6, 4.0], [30.0, 30.0]])
    got = mpc_choice(1.0, 26.0, 1, sizes, [1.0, 2.0], 2.0, seg_s=4.0, latency_s=0.0,
                     buffer_s=30.0)
    assert got == 1


def test_download_equal_to_buffer_is_not_a_stall():
    # second segment takes 4 s with exactly 4 s in the buffer
    m, log = run([4.0] * 10, ladder=(1000,), bits=((4e6,), (16e6,)))
    assert log["finish_s"][:3].tolist() == pytest.approx([1.0, 5.0, 6.0])
    assert log["buffer_s"][1] == pytest.approx(4.0)
    assert m["stalls"] == 0 and m["rebuffer_s"] == 0.0


def _mpc_reference(est, buf, last, sizes, ladder, mu, seg_s, latency_s, buffer_s):
    """First representations of all sequences within 1e-9 (1 + |best|) of the best."""
    h, n = sizes.shape
    objs = {}
    for combo in itertools.product(range(n), repeat=h):
        b, reb, obj, prev = buf, 0.0, 0.0, last
        for j, q in enumerate(combo):
            d = latency_s + sizes[j, q] / est
            reb += max(d - b, 0.0)
            b = min(max(b - d, 0.0) + seg_s, buffer_s - seg_s)
            obj += ladder[q] - abs(ladder[q] - ladder[prev])
            prev = q
        objs[combo] = obj - mu * reb
    best = max(objs.values())
    return {c[0] for c, v in objs.items() if v >= best - 1e-9 * (1.0 + abs(best))}


def test_mpc_matches_exhaustive_reference():
    # ties go to the lexicographically first sequence, whose first element is the
    # smallest first element among the tied sequences
    rng = np.random.default_rng(0)
    ladder = [0.237, 0.752, 1.815, 4.282, 14.817, 39.537]
    for _ in range(40):
        sizes = rng.uniform(0.3, 1.7, (5, 6)) * np.array(ladder) * 4.0
        est = float(rng.uniform(0.2, 60.0))
        buf = float(rng.uniform(0.0, 26.0))
        last = int(rng.integers(0, 6))
        got = mpc_choice(est, buf, last, sizes, ladder, 39.537, 4.0, 0.08, 30.0)
        ref = _mpc_reference(est, buf, last, sizes, ladder, 39.537, 4.0, 0.08, 30.0)
        assert got == min(ref)


@pytest.mark.parametrize("estimator", ["hm", "robust", "point"])
def test_mpc_decisions_use_upcoming_segment_sizes(estimator):
    cap, ladder, bits, est = _random_case(2)
    _, log = simulate(cap, 15, est, ladder, bits, "mpc", estimator)
    sizes = bits / 1e6
    n_video = len(sizes)
    checked = 0
    for k in range(1, len(log["rep"])):
        if np.isnan(log["estimate_mbps"][k]):
            continue
        idx = [(k + j) % n_video for j in range(5)]
        rep = mpc_choice(log["estimate_mbps"][k], log["buffer_s"][k], log["rep"][k - 1],
                         sizes[idx], ladder / 1000.0, 39.537)
        assert rep == log["rep"][k]
        assert log["mbit"][k] == pytest.approx(sizes[k % n_video, rep])
        checked += 1
    assert checked > 50


def _random_case(seed, n=600):
    rng = np.random.default_rng(seed)
    cap = rng.gamma(1.5, 6.0, n)
    cap[rng.random(n) < 0.05] = 0.0
    ladder = np.array([237, 752, 1815, 4282, 14817, 39537], dtype=float)
    bits = rng.uniform(0.5, 1.5, (40, 6)) * ladder * 1000 * 4.0
    pred = cap * rng.lognormal(0.0, 0.3, n)
    est = {"point": pred, "split": np.clip(pred - 5.0, 0, None),
           "aci": np.clip(pred - 3.0, 0, None), "mondrian_aci": np.clip(pred - 4.0, 0, None)}
    for v in est.values():
        v[rng.random(n) < 0.1] = np.nan
    return cap, ladder, bits, est


@pytest.mark.parametrize("rule,estimator", VARIANTS)
def test_accounting_identities(rule, estimator):
    cap, ladder, bits, est = _random_case(1)
    m, log = simulate(cap, 15, est, ladder, bits, rule, estimator)
    done = np.isfinite(log["finish_s"])
    assert m["n_segments"] == int(done.sum())
    assert m["rebuffer_s"] == pytest.approx(log["rebuffer_s"].sum())
    assert m["stalls"] == int((log["rebuffer_s"] > 0).sum())
    assert m["wait_s"] == pytest.approx(log["wait_s"].sum())
    assert m["startup_s"] + m["play_s"] + m["rebuffer_s"] == pytest.approx(m["duration_s"])
    content = m["n_segments"] * 4.0 - m["play_s"]
    assert -1e-9 <= content <= 30.0 + 1e-9
    rates = ladder[log["rep"][done]] / 1000.0
    mu = 39.537
    qoe = rates.sum() - np.abs(np.diff(rates)).sum() - mu * (m["rebuffer_s"] + m["startup_s"])
    assert m["qoe_total"] == pytest.approx(qoe)
    assert np.all(log["buffer_s"] <= 26.0 + 1e-9)
    assert np.all(np.diff(log["request_s"]) > 0)


def test_load_ladder(tmp_path):
    text = (" bbb,  seg_Dur ,    4000,     2000,     1000\n"
            "x264,        3 ,     1920,     1280,      640\n"
            "    ,          ,     1080,      720,      360\n"
            "   0,          ,      820,      820,      820\n"
            "   1,    4.000 ,  2000000,  1000000,   500000\n"
            "   2,    4.000 ,  2100000,  1100000,   400000\n"
            "   3,    4.000 ,  1900000,   900000,   600000")
    p = tmp_path / "costs.csv"
    p.write_text(text, encoding="ascii")
    lad = load_ladder(p, (1000, 4000))
    assert lad.bitrates_kbps.tolist() == [1000.0, 4000.0]
    assert lad.segment_s == 4.0
    assert lad.segment_bits.tolist() == [[4e6, 16e6], [3.2e6, 16.8e6], [4.8e6, 15.2e6]]
    with pytest.raises(ValueError):
        load_ladder(p, (1000, 3000))


def _load_run_has():
    spec = importlib.util.spec_from_file_location("run_has", ROOT / "scripts" / "run_has.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_traces():
    rng = np.random.default_rng(3)
    traces = {}
    for sid, tech, n in [("4G/car/A_x", "4G", 300), ("5G/Download/Driving/B_y", "5G", 250)]:
        dl = rng.gamma(1.5, 6.0, n)
        dl[[40, 41]] = np.nan
        traces[sid] = {"dl": dl, "tech": tech, "mobility": "car"}
    return traces


def _stub_instances(traces, regime, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for sid, tr in list(traces.items()) + [("5G/Netflix/Driving/C_z", {"dl": np.ones(200)})]:
        dl = tr["dl"]
        t = np.array([i for i in range(10, len(dl) - 1)
                      if np.isfinite(dl[i + 1]) and rng.random() < 0.9])
        pred = np.nan_to_num(dl[t]) + rng.normal(0, 2, len(t))
        parts.append(pd.DataFrame({
            "trace_id": sid, "t_index": t, "ts": t.astype(float),
            "ts_label": t + 1.0, "y": dl[t + 1], "pred": pred, "group": "speed_med",
            "lo_split": np.clip(pred - 6, 0, None), "hi_split": pred + 6,
            "lo_aci": np.clip(pred - 4, 0, None), "hi_aci": pred + 4,
            "lo_mondrian_aci": np.clip(pred - 5, 0, None), "hi_mondrian_aci": pred + 5}))
    return pd.concat(parts, ignore_index=True)


def test_run_has_eligible_traces():
    rh = _load_run_has()
    raw = pd.DataFrame({
        "trace_id": ["a"] * 3 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2,
        "tech": ["4G"] * 3 + ["5G"] * 6,
        "app": ["unknown"] * 3 + ["Download"] * 2 + ["Netflix"] * 2 + ["Amazon_Prime"] * 2,
        "mobility": ["train"] * 3 + ["driving"] * 6,
        "dl_mbps": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
    traces = rh.trace_table(raw)
    assert sorted(traces) == ["a", "b"]
    assert traces["a"]["dl"].tolist() == [1.0, 2.0, 3.0]
    assert traces["b"]["tech"] == "5G" and traces["a"]["mobility"] == "train"


def test_run_has_alignment_and_cell_output():
    rh = _load_run_has()
    traces = _stub_traces()
    inst = _stub_instances(traces, "cross_tech_4to5", 0)
    lad = Ladder(
        np.array([237.0, 752.0, 1815.0, 4282.0, 14817.0, 39537.0]),
        np.random.default_rng(5).uniform(0.5, 1.5, (30, 6))
        * np.array([237, 752, 1815, 4282, 14817, 39537]) * 4000.0, 4.0)

    sid = "4G/car/A_x"
    g = inst[inst.trace_id == sid]
    capacity, start, est, n_missing = rh.trace_inputs(g, traces[sid]["dl"])
    dl = traces[sid]["dl"]
    assert len(capacity) == len(dl) - 1
    assert np.array_equal(capacity[np.isfinite(dl[1:])], dl[1:][np.isfinite(dl[1:])])
    assert np.all(capacity[np.isnan(dl[1:])] == 0.0)
    assert n_missing == 2
    assert start == g.t_index.iloc[0]
    t = g.t_index.to_numpy()
    assert np.array_equal(est["point"][t], g.pred.to_numpy())
    assert np.array_equal(est["aci"][t], g.lo_aci.to_numpy())
    others = np.setdiff1d(np.arange(len(capacity)), t)
    assert np.all(np.isnan(est["split"][others]))

    df = rh.has_cell(inst, traces, lad, "cross_tech_4to5", 0)
    assert list(df.columns) == rh.OUTPUT_COLUMNS
    assert set(df.trace_id) == set(traces)
    assert len(df) == 2 * len(VARIANTS)
    row = df[(df.trace_id == sid) & (df.variant == "mpc_mondrian_aci")].iloc[0]
    m, _ = simulate(capacity, start, est, lad.bitrates_kbps, lad.segment_bits, "mpc",
                    "mondrian_aci")
    assert row.qoe_total == pytest.approx(m["qoe_total"])
    assert row.n_segments == m["n_segments"]

    bad = inst.copy()
    bad.loc[bad.index[5], "y"] += 1.0
    with pytest.raises(ValueError):
        rh.has_cell(bad, traces, lad, "cross_tech_4to5", 0)
    shuffled = inst.iloc[::-1]
    with pytest.raises(ValueError):
        rh.has_cell(shuffled, traces, lad, "cross_tech_4to5", 0)


def test_run_has_cells_resume_and_empty_cell(tmp_path):
    rh = _load_run_has()
    traces = _stub_traces()
    lad = Ladder(
        np.array([237.0, 752.0, 1815.0, 4282.0, 14817.0, 39537.0]),
        np.full((10, 6), 1.0) * np.array([237, 752, 1815, 4282, 14817, 39537]) * 4000.0, 4.0)
    calls = []

    def stub(arg, regime, seed):
        calls.append((regime, seed))
        if regime == "in_domain_5g":
            return _stub_instances({}, regime, seed)
        return _stub_instances(arg, regime, seed)

    cells = [("cross_tech_4to5", 0), ("cross_tech_4to5", 1), ("in_domain_5g", 0)]
    rh.run_cells(cells, tmp_path, traces, lad, stub, traces, n_jobs=1, log=lambda s: None)
    assert len(calls) == 3
    for r, s in cells:
        assert rh.cell_path(tmp_path, r, s).exists()
    empty = pd.read_csv(rh.cell_path(tmp_path, "in_domain_5g", 0))
    assert len(empty) == 0 and list(empty.columns) == rh.OUTPUT_COLUMNS
    full = pd.read_csv(rh.cell_path(tmp_path, "cross_tech_4to5", 1))
    assert len(full) == 2 * len(VARIANTS)
    rh.run_cells(cells, tmp_path, traces, lad, stub, traces, n_jobs=1, log=lambda s: None)
    assert len(calls) == 3

    # combined file: exact floats, no gzip time stamp, identical on a second write
    path, n_rows = rh.write_combined(tmp_path, cells)
    data = path.read_bytes()
    assert data[:2] == b"\x1f\x8b" and data[4:8] == b"\x00\x00\x00\x00"
    assert n_rows == 2 * 2 * len(VARIANTS)
    parts = [pd.read_csv(rh.cell_path(tmp_path, r, s), float_precision="round_trip")
             for r, s in cells]
    back = pd.read_csv(path, float_precision="round_trip")
    pd.testing.assert_frame_equal(back, pd.concat(parts[:2], ignore_index=True),
                                  check_exact=True)
    assert rh.write_combined(tmp_path, cells)[0].read_bytes() == data
