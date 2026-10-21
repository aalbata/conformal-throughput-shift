"""Trace-driven adaptive-streaming simulation (protocol amendment 1.8).

A player downloads fixed-duration video segments over a per-second capacity trace
and plays them from a finite buffer. A download starts with a fixed request
latency during which no data arrive; the transfer then follows the capacity trace
as a fluid (the time to finish is found by integrating capacity over time).
Bitrate rules:
  rate : highest representation whose nominal bitrate does not exceed the estimate
  mpc  : model predictive control (Yin et al., SIGCOMM 2015) with exhaustive search
         over the next `horizon` segments, the estimate held constant over the
         horizon and the first decision executed
Estimators:
  hm     : harmonic mean of the last five segment throughputs
  robust : RobustMPC lower bound, hm / (1 + maximum relative error of the last
           five harmonic-mean estimates)
  other  : a key of `estimates`, per-second external estimates (point predictions
           or interval lower bounds)

Time convention: second s of the simulation is the interval [s, s + 1) and has
capacity capacity[s]. The external estimate estimates[name][s] is available to a
request issued during second s; NaN marks a second without a new estimate, in
which case the most recent earlier estimate is used.

Units: time in s, segment sizes in Mbit inside the simulator, rates in Mbit/s.
"""
from __future__ import annotations

import csv
import itertools
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

LADDER_KBPS = (237, 752, 1815, 4282, 14817, 39537)
SEGMENT_S = 4.0
BUFFER_S = 30.0
LATENCY_S = 0.08
FLOOR_MBPS = 0.01
HORIZON = 5
WINDOW = 5
RULES = ("rate", "mpc")
INTERNAL_ESTIMATORS = ("hm", "robust")
VARIANTS = (("rate", "hm"), ("rate", "point"), ("rate", "split"), ("rate", "aci"),
            ("rate", "mondrian_aci"),
            ("mpc", "hm"), ("mpc", "robust"), ("mpc", "point"), ("mpc", "split"),
            ("mpc", "aci"), ("mpc", "mondrian_aci"))
METRIC_KEYS = ("n_segments", "avg_bitrate_mbps", "rebuffer_s", "rebuffer_final_s",
               "rebuffer_ratio", "stalls", "startup_s", "play_s", "switches",
               "switches_per_min", "mean_abs_switch_mbps", "qoe_total", "qoe_per_segment",
               "wait_s", "discarded", "duration_s")


class Ladder(NamedTuple):
    """Representations in ascending bitrate order and per-segment sizes."""
    bitrates_kbps: np.ndarray   # shape (n_reps,)
    segment_bits: np.ndarray    # shape (n_segments, n_reps)
    segment_s: float


def load_ladder(path: str | Path, reps_kbps=LADDER_KBPS) -> Ladder:
    """Read a UCC 4K DASH 'segmentByteCostinColumns' CSV (Quinlan and Sreenan, 2018).

    Layout: row 1 holds the video name, 'seg_Dur' and the nominal bitrate (kbit/s)
    of every representation; row 2 the codec, the number of representations and the
    widths; row 3 the heights; then one row per segment with its index, duration (s)
    and size in bytes per representation. Index 0 is the initialisation segment and
    is skipped. Returns the representations in reps_kbps, sorted by bitrate, with
    sizes in bits.
    """
    with open(path, newline="", encoding="ascii") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    rows = [r for r in rows if any(r)]
    head = rows[0]
    if len(head) < 3 or head[1] != "seg_Dur":
        raise ValueError(f"{path}: first row is not a segment byte-cost header")
    bitrates = [int(x) for x in head[2:]]
    if int(rows[1][1]) != len(bitrates):
        raise ValueError(f"{path}: representation count does not match the header")
    index, durations, sizes = [], [], []
    for r in rows[3:]:
        if len(r) != len(head):
            raise ValueError(f"{path}: row of length {len(r)}, expected {len(head)}")
        i = int(r[0])
        if i == 0:
            continue
        index.append(i)
        durations.append(float(r[1]))
        sizes.append([int(x) for x in r[2:]])
    if index != list(range(1, len(index) + 1)):
        raise ValueError(f"{path}: segment indices are not 1..n")
    if len(set(durations)) != 1:
        raise ValueError(f"{path}: segments of unequal duration {sorted(set(durations))}")
    reps = sorted(int(x) for x in reps_kbps)
    cols = []
    for rep in reps:
        if bitrates.count(rep) != 1:
            raise ValueError(f"{path}: representation {rep} kbit/s not found exactly once")
        cols.append(bitrates.index(rep))
    seg_bytes = np.asarray(sizes, dtype=np.int64)[:, cols]
    if np.any(seg_bytes <= 0):
        raise ValueError(f"{path}: non-positive segment size")
    return Ladder(np.asarray(reps, dtype=float), seg_bytes.astype(float) * 8.0,
                  durations[0])


def harmonic_mean(values) -> float:
    """Harmonic mean of positive values."""
    values = list(values)
    return len(values) / sum(1.0 / v for v in values)


def rate_choice(estimate: float, ladder_mbps) -> int:
    """Index of the highest bitrate not above estimate (0 if none)."""
    return max(bisect_right(list(ladder_mbps), estimate) - 1, 0)


def download_finish(cum, capacity, t0: float, mbit: float) -> float:
    """Time at which mbit Mbit are delivered when the transfer starts at t0.

    cum[k] is the capacity integrated over [0, k) (cum[0] = 0) and capacity[k] the
    rate during second k. Returns inf when the trace ends before the transfer does.
    """
    n = len(capacity)
    if t0 >= n:
        return np.inf
    k0 = int(t0)
    target = cum[k0] + capacity[k0] * (t0 - k0) + mbit
    if target > cum[n]:
        return np.inf
    k = bisect_left(cum, target, k0 + 1)
    return (k - 1) + (target - cum[k - 1]) / capacity[k - 1]


@lru_cache(maxsize=16)
def _mpc_tables(ladder_mbps: tuple, horizon: int) -> np.ndarray:
    """Throughput-independent part of the MPC objective for every sequence.

    Row l, column c: sum of bitrates of sequence c minus the absolute bitrate changes
    along it, including the change from representation l before the sequence.
    Sequences are ordered as itertools.product(range(n_reps), repeat=horizon).
    """
    r = np.asarray(ladder_mbps, dtype=float)
    combos = np.array(list(itertools.product(range(len(r)), repeat=horizon)))
    rc = r[combos]
    inner = rc.sum(axis=1) - np.abs(np.diff(rc, axis=1)).sum(axis=1)
    first = np.abs(rc[:, 0][None, :] - r[:, None])
    return inner[None, :] - first


def mpc_choice(estimate: float, buffer_level: float, last_rep: int, sizes_mbit: np.ndarray,
               ladder_mbps, mu: float, seg_s: float = SEGMENT_S,
               latency_s: float = LATENCY_S, buffer_s: float = BUFFER_S) -> int:
    """First representation of the sequence that maximises the linear QoE.

    sizes_mbit has shape (horizon, n_reps): sizes of the next segments. Planned
    download time of a segment is latency_s + size / estimate. The planned buffer
    follows B <- min(max(B - d, 0) + seg_s, buffer_s - seg_s), the cap reflecting the
    wait for one segment of free space; rebuffering accrues as max(d - B, 0).
    Objective: sum of bitrates minus absolute bitrate changes (from last_rep) minus
    mu times planned rebuffering. Objectives within 1e-9 * (1 + |maximum|) of the
    maximum are ties (sequences of equal value can differ by rounding), and ties go
    to the first sequence in lexicographic order of representation indices.
    """
    horizon, n_reps = sizes_mbit.shape
    base = _mpc_tables(tuple(float(x) for x in ladder_mbps), horizon)
    cap = buffer_s - seg_s
    b = np.array([float(buffer_level)])
    reb = np.zeros(1)
    for j in range(horizon):
        d = latency_s + sizes_mbit[j] / estimate
        short = d[None, :] - b[:, None]
        reb = (reb[:, None] + np.maximum(short, 0.0)).ravel()
        b = np.minimum(np.maximum(-short, 0.0) + seg_s, cap).ravel()
    obj = base[last_rep] - mu * reb
    top = obj.max()
    best = int(np.argmax(obj >= top - 1e-9 * (1.0 + abs(top))))
    return best // n_reps ** (horizon - 1)


def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Replace NaN by the last earlier non-NaN value (NaN if there is none)."""
    idx = np.where(np.isnan(values), -1, np.arange(len(values)))
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)


def _empty_metrics(duration: float) -> dict:
    """Metrics of a trace whose first segment does not arrive before the trace ends."""
    m = {k: np.nan for k in METRIC_KEYS}
    m.update(n_segments=0, stalls=0, switches=0, rebuffer_s=0.0, rebuffer_final_s=0.0,
             wait_s=0.0, discarded=1, duration_s=duration)
    return m


def simulate(capacity_mbps, start_second: int, estimates: dict | None, ladder_kbps,
             segment_bits, rule: str, estimator: str, buffer_s: float = BUFFER_S,
             seg_s: float = SEGMENT_S, latency_s: float = LATENCY_S,
             floor_mbps: float = FLOOR_MBPS, mu: float | None = None,
             horizon: int = HORIZON) -> tuple[dict, dict]:
    """Play one trace from start_second to the end of the capacity trace.

    The first segment is fetched at the lowest representation and playback starts
    when it has arrived. Later requests follow `rule` with the estimate of
    `estimator`, floored at floor_mbps; an external estimator without any estimate
    at or before the request second gives the lowest representation. After each
    download the player waits until the buffer has one segment of free space.
    Segment sizes repeat cyclically. A download still incomplete at the end of the
    trace is discarded; rebuffering until the end of the trace still counts.
    mu weights rebuffering and startup time in the QoE (default: the highest
    bitrate in Mbit/s).

    Returns (metrics, log). metrics:
      n_segments          segments downloaded before the end of the trace
      avg_bitrate_mbps    mean nominal bitrate of those segments
      rebuffer_s          stall time after startup, up to the end of the trace
      rebuffer_final_s    part of rebuffer_s during the discarded final download
      rebuffer_ratio      rebuffer_s / (play_s + rebuffer_s)
      stalls              number of stalls after startup
      startup_s           time from start_second to the arrival of the first segment
      play_s              playback time from startup to the end of the trace
      switches            changes of representation between consecutive segments
      switches_per_min    switches per minute of playback
      mean_abs_switch_mbps  mean absolute bitrate change per switch
      qoe_total           sum of bitrates - sum of absolute bitrate changes
                          - mu * rebuffer_s - mu * startup_s (Mbit/s units)
      qoe_per_segment     qoe_total / n_segments
      wait_s              time spent waiting for buffer space
      discarded           1 if a final download was cut by the end of the trace
      duration_s          length of the playback window, len(capacity) - start_second
    log: per request arrays request_s, estimate_mbps (NaN for the first segment),
    rep, mbit, finish_s (inf if discarded), throughput_mbps, buffer_s (at request),
    rebuffer_s (during the download), wait_s (after the download).
    """
    cap = np.asarray(capacity_mbps, dtype=float)
    if cap.ndim != 1 or len(cap) == 0:
        raise ValueError("capacity must be a non-empty 1-D array")
    if not np.all(np.isfinite(cap)) or np.any(cap < 0):
        raise ValueError("capacity must be finite and non-negative")
    n = len(cap)
    start = int(start_second)
    if not 0 <= start < n:
        raise ValueError("start_second outside the capacity trace")
    ladder = np.asarray(ladder_kbps, dtype=float) / 1000.0
    if ladder.ndim != 1 or np.any(ladder <= 0) or np.any(np.diff(ladder) <= 0):
        raise ValueError("ladder must be positive and strictly increasing")
    sizes = np.asarray(segment_bits, dtype=float) / 1e6
    if sizes.ndim != 2 or sizes.shape[1] != len(ladder) or np.any(sizes <= 0):
        raise ValueError("segment_bits must have shape (n_segments, n_reps), positive")
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule}")
    if buffer_s < seg_s:
        raise ValueError("buffer must hold at least one segment")
    mu = float(ladder[-1]) if mu is None else float(mu)

    ext = None
    if estimator not in INTERNAL_ESTIMATORS:
        if estimates is None or estimator not in estimates:
            raise ValueError(f"no estimates supplied for {estimator}")
        e = np.asarray(estimates[estimator], dtype=float)
        if e.shape != cap.shape:
            raise ValueError("estimates must be aligned with capacity")
        if np.any(np.isinf(e)):
            raise ValueError("estimates must not be infinite")
        ext = _forward_fill(e).tolist()

    cum = np.concatenate(([0.0], np.cumsum(cap))).tolist()
    capl = cap.tolist()
    ladder_l = ladder.tolist()
    sizes_l = sizes.tolist()
    n_video = len(sizes_l)
    t_end = float(n)
    wait_cap = buffer_s - seg_s

    t = float(start)
    buf = 0.0
    playing = False
    k = 0
    last_rep = 0
    startup = np.nan
    t_play = np.nan
    rebuf_total = 0.0
    rebuf_final = 0.0
    stalls = 0
    wait_total = 0.0
    discarded = 0
    tputs: list[float] = []
    hm_hist: list[float] = []
    err_hist: list[float] = []
    reps: list[int] = []
    log = {"request_s": [], "estimate_mbps": [], "rep": [], "mbit": [], "finish_s": [],
           "throughput_mbps": [], "buffer_s": [], "rebuffer_s": [], "wait_s": []}

    while t < t_end:
        if k == 0:
            est = np.nan
            rep = 0
        else:
            if estimator == "hm":
                est = harmonic_mean(tputs[-WINDOW:])
            elif estimator == "robust":
                h = harmonic_mean(tputs[-WINDOW:])
                if hm_hist:
                    err_hist.append(abs(hm_hist[-1] - tputs[-1]) / tputs[-1])
                max_err = max(err_hist[-WINDOW:]) if err_hist else 0.0
                hm_hist.append(h)
                est = h / (1.0 + max_err)
            else:
                est = ext[int(t)]
            if est != est:
                rep = 0
            else:
                est = max(est, floor_mbps)
                if rule == "rate":
                    rep = max(bisect_right(ladder_l, est) - 1, 0)
                else:
                    idx = [(k + j) % n_video for j in range(horizon)]
                    rep = mpc_choice(est, buf, last_rep, sizes[idx], ladder_l, mu,
                                     seg_s, latency_s, buffer_s)
        mbit = sizes_l[k % n_video][rep]
        finish = download_finish(cum, capl, t + latency_s, mbit)
        buf_before = buf
        if finish == np.inf:
            dt = t_end - t
            if playing:
                if dt > buf:
                    rebuf_final = dt - buf
                    rebuf_total += rebuf_final
                    stalls += 1
                    buf = 0.0
                else:
                    buf -= dt
            discarded = 1
            for key, v in (("request_s", t), ("estimate_mbps", est), ("rep", rep),
                           ("mbit", mbit), ("finish_s", np.inf), ("throughput_mbps", np.nan),
                           ("buffer_s", buf_before), ("rebuffer_s", rebuf_final),
                           ("wait_s", 0.0)):
                log[key].append(v)
            t = t_end
            break
        dt = finish - t
        reb = 0.0
        if playing:
            if dt > buf:
                reb = dt - buf
                rebuf_total += reb
                stalls += 1
                buf = 0.0
            else:
                buf -= dt
        else:
            startup = finish - start
            t_play = finish
            playing = True
        buf += seg_s
        tput = mbit / (finish - t - latency_s)
        tputs.append(tput)
        reps.append(rep)
        last_rep = rep
        req = t
        t = finish
        k += 1
        wait = buf - wait_cap
        if wait > 0.0:
            wait = min(wait, t_end - t)
            buf -= wait
            t += wait
            wait_total += wait
        else:
            wait = 0.0
        for key, v in (("request_s", req), ("estimate_mbps", est), ("rep", rep),
                       ("mbit", mbit), ("finish_s", finish), ("throughput_mbps", tput),
                       ("buffer_s", buf_before), ("rebuffer_s", reb), ("wait_s", wait)):
            log[key].append(v)

    log = {key: np.asarray(v, dtype=int if key == "rep" else float) for key, v in log.items()}
    duration = t_end - start
    n_seg = len(reps)
    if n_seg == 0:
        return _empty_metrics(duration), log
    rates = ladder[np.asarray(reps)]
    changes = np.abs(np.diff(rates))
    switches = int(np.count_nonzero(changes))
    change_sum = float(changes.sum())
    window = t_end - t_play
    play = window - rebuf_total
    qoe = float(rates.sum()) - change_sum - mu * rebuf_total - mu * startup
    metrics = {
        "n_segments": n_seg,
        "avg_bitrate_mbps": float(rates.mean()),
        "rebuffer_s": rebuf_total,
        "rebuffer_final_s": rebuf_final,
        "rebuffer_ratio": rebuf_total / window if window > 0 else np.nan,
        "stalls": stalls,
        "startup_s": startup,
        "play_s": play,
        "switches": switches,
        "switches_per_min": switches / (play / 60.0) if play > 0 else np.nan,
        "mean_abs_switch_mbps": change_sum / switches if switches else np.nan,
        "qoe_total": qoe,
        "qoe_per_segment": qoe / n_seg,
        "wait_s": wait_total,
        "discarded": discarded,
        "duration_s": duration,
    }
    return metrics, log
