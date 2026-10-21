"""Streaming simulator: hand-derived outcomes with the default 80-ms request latency.

Each expected value below follows from the player model of protocol amendment 1.8
by hand: 80 ms of latency before every transfer, fluid transfer over the per-second
capacity, a 30-s buffer (unless a test sets another) with a wait until one segment of
space is free, the first segment at the lowest representation and mu = mu_s = the
highest bitrate in Mbit/s.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.has import LADDER_KBPS, mpc_choice, simulate  # noqa: E402


def sim(capacity, ladder, bits, rule="rate", estimator="hm", estimates=None, **kw):
    return simulate(np.asarray(capacity, dtype=float), 0, estimates, ladder,
                    np.asarray(bits, dtype=float), rule, estimator, **kw)


def test_stall_caused_by_capacity_drop_and_by_latency():
    # 20 Mbit/s for 1 s, then 1 Mbit/s; rungs 1 and 4 Mbit/s with 4 and 16 Mbit segments.
    # seg 0: 0.08 + 4/20 = 0.28 s. seg 1 (hm 20 -> 4 Mbit/s): 12.8 Mbit in [0.36, 1),
    # 3.2 Mbit at 1 Mbit/s, done at 4.2; 3.92 s <= 4 s of buffer. seg 2 (hm
    # 2/(1/20 + 3.84/16) = 6.90 -> 4 Mbit/s): 16 s at 1 Mbit/s, done at 20.28, stall
    # 16.08 - 4.08 = 12. segs 3 and 4 (hm 2.33 and 1.75 -> 1 Mbit/s): 4.08 s each with
    # 4 s of buffer, a 0.08-s stall each. seg 5 at 28.44 cannot finish by 30 s.
    cap = [20.0] + [1.0] * 29
    m, log = sim(cap, (1000, 4000), [(4e6, 16e6)])
    assert log["rep"].tolist() == [0, 1, 1, 0, 0, 0]
    assert log["request_s"].tolist() == pytest.approx([0, 0.28, 4.2, 20.28, 24.36, 28.44])
    assert log["finish_s"][:5].tolist() == pytest.approx([0.28, 4.2, 20.28, 24.36, 28.44])
    assert np.isinf(log["finish_s"][5])
    assert log["buffer_s"].tolist() == pytest.approx([0, 4, 4.08, 4, 4, 4])
    assert log["rebuffer_s"].tolist() == pytest.approx([0, 0, 12.0, 0.08, 0.08, 0])
    assert log["estimate_mbps"][1:].tolist() == pytest.approx(
        [20.0, 2 / 0.29, 3 / 1.29, 4 / 2.29, 5 / 3.29])
    assert log["throughput_mbps"][:5].tolist() == pytest.approx([20, 16 / 3.84, 1, 1, 1])
    assert m["n_segments"] == 5 and m["discarded"] == 1
    assert m["stalls"] == 3
    assert m["rebuffer_s"] == pytest.approx(12.16)
    assert m["rebuffer_final_s"] == 0.0
    assert m["startup_s"] == pytest.approx(0.28)
    assert m["play_s"] == pytest.approx(30 - 0.28 - 12.16)
    assert m["rebuffer_ratio"] == pytest.approx(12.16 / 29.72)
    assert m["avg_bitrate_mbps"] == pytest.approx(2.2)
    assert m["switches"] == 2
    assert m["switches_per_min"] == pytest.approx(2 / (17.56 / 60))
    assert m["mean_abs_switch_mbps"] == pytest.approx(3.0)
    # 11 - 6 - 4 * 12.16 - 4 * 0.28
    assert m["qoe_total"] == pytest.approx(-44.76)
    assert m["qoe_per_segment"] == pytest.approx(-44.76 / 5)


def test_buffer_full_wait_with_latency():
    # 10 Mbit/s, 4-Mbit segments: each download takes 0.08 + 0.4 = 0.48 s. Buffer 12 s,
    # so the player waits whenever more than 8 s are buffered after a download.
    m, log = sim([10.0] * 20, (1000,), [(4e6,)], buffer_s=12.0)
    assert log["request_s"].tolist() == pytest.approx([0, 0.48, 0.96, 4.48, 8.48, 12.48, 16.48])
    assert log["finish_s"].tolist() == pytest.approx([0.48, 0.96, 1.44, 4.96, 8.96, 12.96, 16.96])
    assert log["buffer_s"].tolist() == pytest.approx([0, 4, 7.52, 8, 8, 8, 8])
    # the last wait is cut at the end of the trace
    assert log["wait_s"].tolist() == pytest.approx([0, 0, 3.04, 3.52, 3.52, 3.52, 3.04])
    assert m["n_segments"] == 7 and m["discarded"] == 0 and m["stalls"] == 0
    assert m["wait_s"] == pytest.approx(16.64)
    assert m["startup_s"] == pytest.approx(0.48)
    assert m["play_s"] == pytest.approx(19.52)
    assert m["switches"] == 0 and m["switches_per_min"] == 0.0
    assert np.isnan(m["mean_abs_switch_mbps"])
    assert m["qoe_total"] == pytest.approx(7 - 0.48)


def test_zero_capacity_seconds_with_latency():
    # The latency of the first request falls in a zero-capacity second; 5 Mbit arrive in
    # second 2, none in second 3 and the last 1 Mbit in 0.2 s of second 4. Later
    # segments take 0.08 + 6/5 = 1.28 s; the sixth request at 9.32 s gets only 3 Mbit.
    cap = [0.0, 0.0, 5.0, 0.0] + [5.0] * 6
    m, log = sim(cap, (1500,), [(6e6,)])
    assert log["finish_s"].tolist()[:5] == pytest.approx([4.2, 5.48, 6.76, 8.04, 9.32])
    assert log["throughput_mbps"][0] == pytest.approx(6 / 4.12)
    assert log["throughput_mbps"][1:5].tolist() == pytest.approx([5.0] * 4)
    assert np.isinf(log["finish_s"][5])
    assert m["startup_s"] == pytest.approx(4.2)
    assert m["n_segments"] == 5 and m["stalls"] == 0 and m["discarded"] == 1
    assert m["play_s"] == pytest.approx(5.8)
    assert m["qoe_total"] == pytest.approx(5 * 1.5 - 1.5 * 4.2)


def test_stall_during_an_outage():
    # seg 1 starts at 0.66: 2.72 Mbit in [0.66, 1), nothing in seconds 1-6, 1.28 Mbit in
    # 0.16 s of second 7; 6.58 s with 4 s of buffer is a 2.58-s stall.
    cap = [8.0] + [0.0] * 6 + [8.0]
    m, log = sim(cap, (1000,), [(4e6,)])
    assert log["finish_s"][:3].tolist() == pytest.approx([0.58, 7.16, 7.74])
    assert log["rebuffer_s"].tolist() == pytest.approx([0, 2.58, 0, 0])
    assert log["throughput_mbps"][1] == pytest.approx(4 / 6.5)
    assert np.isinf(log["finish_s"][3])
    assert m["stalls"] == 1 and m["rebuffer_s"] == pytest.approx(2.58)
    assert m["play_s"] == pytest.approx(8 - 0.58 - 2.58)
    assert m["rebuffer_ratio"] == pytest.approx(2.58 / 7.42)
    assert m["qoe_total"] == pytest.approx(3 - 2.58 - 0.58)


def test_stall_into_the_end_of_the_trace_with_latency():
    # three 0.58-s downloads leave 10.84 s of buffer at 1.74 s; the fourth download
    # never finishes, the buffer runs out at 12.58 s and the stall lasts to 14 s.
    cap = [8.0, 8.0] + [0.0] * 12
    m, log = sim(cap, (1000,), [(4e6,)])
    assert log["finish_s"][:3].tolist() == pytest.approx([0.58, 1.16, 1.74])
    assert np.isinf(log["finish_s"][3])
    assert log["buffer_s"][3] == pytest.approx(10.84)
    assert m["rebuffer_s"] == pytest.approx(1.42)
    assert m["rebuffer_final_s"] == pytest.approx(1.42)
    assert m["stalls"] == 1 and m["discarded"] == 1
    assert m["play_s"] == pytest.approx(12.0)
    assert m["qoe_per_segment"] == pytest.approx((3 - 1.42 - 0.58) / 3)


@pytest.mark.parametrize("rule,last_reps,avg", [("rate", [1, 1], 1.2), ("mpc", [2, 2], 1.6)])
def test_infinite_interval_bound_is_floored(rule, last_reps, avg):
    # 50 Mbit/s; rungs 1, 2, 4 Mbit/s with constant-bitrate segments. Lower bound: no
    # instance before second 2 (lowest rung), 0 (infinite interval) in seconds 2-5,
    # 3 Mbit/s from second 6. Rate rule at 3: 2 Mbit/s. MPC at 3 with 26 s of buffer:
    # 16-Mbit downloads take 5.41 s and never rebuffer, so the top rung wins.
    est = np.full(12, np.nan)
    est[2:6] = 0.0
    est[6:] = 3.0
    m, log = sim([50.0] * 12, (1000, 2000, 4000), [(4e6, 8e6, 16e6)], rule=rule,
                 estimator="aci", estimates={"aci": est})
    assert log["request_s"].tolist() == pytest.approx(
        [0, 0.16, 0.32, 0.48, 0.64, 0.80, 0.96, 2.16, 6.16, 10.16])
    assert np.all(np.isnan(log["estimate_mbps"][:7]))
    assert log["estimate_mbps"][7:].tolist() == pytest.approx([0.01, 3.0, 3.0])
    assert log["rep"].tolist() == [0] * 8 + last_reps
    assert log["buffer_s"][7:].tolist() == pytest.approx([26.0, 26.0, 26.0])
    assert m["stalls"] == 0 and m["discarded"] == 0
    assert m["avg_bitrate_mbps"] == pytest.approx(avg)


def test_mpc_toy_decision_depends_on_latency():
    # Rungs 1 and 2 Mbit/s, segments 4 and 8 Mbit, estimate 2 Mbit/s, 4 s of buffer,
    # previous rung 1 Mbit/s, horizon 2. With 80 ms latency the downloads take 2.08 s
    # and 4.08 s: (1, 1) 2, (1, 2) 2, (2, 1) 1 - 0.08 mu, (2, 2) 3 - 0.16 mu.
    sizes = np.array([[4.0, 8.0], [4.0, 8.0]])
    assert mpc_choice(2.0, 4.0, 0, sizes, [1.0, 2.0], 2.0) == 1     # 2.68 > 2
    assert mpc_choice(2.0, 4.0, 0, sizes, [1.0, 2.0], 10.0) == 0    # 1.4 < 2
    # without latency (2, 2) does not rebuffer and is worth 3
    assert mpc_choice(2.0, 4.0, 0, sizes, [1.0, 2.0], 10.0, latency_s=0.0) == 1


def test_mpc_ties_go_to_the_lexicographically_first_sequence():
    # Ladder of the study, constant-bitrate segments, estimate 0.5 Mbit/s, 8 s of buffer,
    # previous rung 752 kbit/s. (0.237, 0.237, 0.752, 0.752, 0.752) and
    # (0.752, 0.237, 0.237, 0.752, 0.752) both avoid rebuffering and are worth
    # 2.730 - 1.030 = 1.700; the exhaustive search below finds no other sequence of
    # that value. The lexicographically first optimum starts with the lowest rung.
    ladder = [k / 1000.0 for k in LADDER_KBPS]
    sizes = np.tile(np.array(ladder) * 4.0, (5, 1))
    assert mpc_choice(0.5, 8.0, 1, sizes, ladder, ladder[-1]) == 0
    values = {}
    for combo in itertools.product(range(6), repeat=5):
        b, reb, util, prev = 8.0, 0.0, 0.0, 1
        for j, q in enumerate(combo):
            d = 0.08 + sizes[j, q] / 0.5
            reb += max(d - b, 0.0)
            b = min(max(b - d, 0.0) + 4.0, 26.0)
            util += ladder[q] - abs(ladder[q] - ladder[prev])
            prev = q
        values[combo] = util - ladder[-1] * reb
    best = max(values.values())
    assert best == pytest.approx(1.7)
    optima = sorted(c for c, v in values.items() if v >= best - 1e-9)
    assert optima == [(0, 0, 1, 1, 1), (1, 0, 0, 1, 1)]


def test_trace_shorter_than_one_segment():
    m, log = sim([1.0] * 4, (1000,), [(4e6,)])   # 0.08 + 4 s does not fit in 4 s
    assert m["n_segments"] == 0 and m["discarded"] == 1 and m["duration_s"] == 4.0
    assert np.isnan(m["qoe_per_segment"]) and np.isnan(m["startup_s"])
    assert np.isinf(log["finish_s"]).tolist() == [True]
    m, _ = sim([1.0] * 4, (1000,), [(4e6,)], latency_s=0.0)   # ends exactly at 4 s
    assert m["n_segments"] == 1 and m["discarded"] == 0 and m["play_s"] == 0.0
    assert m["startup_s"] == pytest.approx(4.0) and np.isnan(m["rebuffer_ratio"])
