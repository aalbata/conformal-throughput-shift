"""Tests for instance timestamps, the CSV column append and the timestamp-based features."""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import append_csv_columns, discover_traces, load_all, load_trace
from src.features import (FEATURE_COLS, RADIO_FEATURES, build_features,
                          build_features_timestamp, instance_timestamps)


def _irregular_times(rng, n):
    steps = rng.choice([0, 1, 1, 1, 1, 1, 2, 4, 15], size=n)
    steps[0] = 0
    return pd.Timestamp("2020-01-01 10:00:00") + pd.to_timedelta(np.cumsum(steps), unit="s")


def _write_raw_tree(root: Path, rng) -> None:
    """Small raw tree in the 4G dataset layout, rows shuffled within each file."""
    for mob, k in (("car", 3), ("static", 2)):
        d = root / "4G" / "Dataset" / mob
        d.mkdir(parents=True)
        for j in range(k):
            n = int(rng.integers(90, 140))
            ts = _irregular_times(rng, n)
            df = pd.DataFrame({
                "Timestamp": ts.strftime("%Y.%m.%d_%H.%M.%S"),
                "Speed": rng.integers(0, 80, n), "RSRP": rng.integers(-110, -80, n),
                "RSRQ": rng.integers(-15, -5, n), "SNR": rng.integers(0, 20, n),
                "CQI": rng.integers(1, 15, n), "RSSI": rng.integers(-90, -60, n),
                "DL_bitrate": rng.permutation(np.arange(n) * 7 + 1000 * j + 3),
                "UL_bitrate": rng.integers(0, 500, n),
                "State": rng.choice(["D", "D", "D", "I"], size=n),
                "NetworkMode": "LTE",
            })
            df = df.iloc[rng.permutation(n)]
            df.to_csv(d / f"A_trace{j}.csv", index=False)


@pytest.mark.parametrize("horizon", [1, 5])
def test_instance_timestamps_join_through_loader(tmp_path, horizon):
    rng = np.random.default_rng(horizon)
    _write_raw_tree(tmp_path, rng)
    raw = load_all(tmp_path, min_rows=60)
    feat = build_features(raw, horizon=horizon)
    ts = instance_timestamps(feat, raw, horizon)
    assert list(ts.index) == list(feat.index)

    traces = {m.trace_id: load_trace(m) for m in discover_traces(tmp_path)}
    for i in rng.choice(len(feat), size=60, replace=False):
        r = feat.iloc[i]
        s = traces[r.trace_id]
        t0 = s.timestamp.iloc[0]
        assert ts.ts.iloc[i] == (s.timestamp.iloc[r.t_index] - t0).total_seconds()
        assert ts.ts_label.iloc[i] == (s.timestamp.iloc[r.t_index + horizon] - t0).total_seconds()
        assert r.dl_last == s.dl_mbps.iloc[r.t_index]
        assert r.y == s.dl_mbps.iloc[r.t_index + horizon]
    assert (ts.ts_label >= ts.ts).all()

    shifted = feat.copy()
    shifted["t_index"] = shifted["t_index"] - 1
    with pytest.raises(ValueError):
        instance_timestamps(shifted, raw, horizon)
    wrong_y = feat.copy()
    wrong_y.loc[wrong_y.index[5], "y"] += 1.0
    with pytest.raises(ValueError):
        instance_timestamps(wrong_y, raw, horizon)


def test_append_csv_columns_keeps_existing_text(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.integers(0, 9, 50),
                       "s": [f"4G/car/A_{i}" for i in range(50)]})
    df.loc[3, "a"] = np.nan
    src = tmp_path / "x.csv.gz"
    df.to_csv(src, index=False, lineterminator="\r\n")
    ts = np.arange(50, dtype=float) * 1.5
    dst1, dst2 = tmp_path / "y.csv.gz", tmp_path / "z.csv.gz"
    append_csv_columns(src, dst1, {"ts": ts, "ts_label": ts + 1})
    append_csv_columns(src, dst2, {"ts": ts, "ts_label": ts + 1})
    assert dst1.read_bytes() == dst2.read_bytes()

    old_lines = gzip.open(src, "rb").read().split(b"\r\n")
    new_lines = gzip.open(dst1, "rb").read().split(b"\r\n")
    assert len(old_lines) == len(new_lines)
    assert all(n.startswith(o + b",") for o, n in zip(old_lines[:-1], new_lines[:-1]))
    back = pd.read_csv(dst1)
    pd.testing.assert_frame_equal(back[["a", "b", "s"]], pd.read_csv(src), check_exact=True)
    assert np.array_equal(back["ts"].to_numpy(), ts)
    with pytest.raises(ValueError):
        append_csv_columns(src, tmp_path / "w.csv.gz", {"ts": ts[:-1]})


def _timed_trace(rng, n=160, sid="s"):
    ts = _irregular_times(rng, n)
    return pd.DataFrame({
        "timestamp": ts, "Speed": rng.uniform(0, 60, n), "RSRP": rng.normal(-100, 5, n),
        "RSRQ": rng.normal(-12, 2, n), "SNR": rng.normal(10, 3, n),
        "CQI": rng.integers(1, 15, n).astype(float), "RSSI": rng.normal(-85, 5, n),
        "dl_mbps": rng.gamma(2, 5, n), "ul_mbps": rng.gamma(2, 1, n),
        "state": rng.choice(["D", "D", "D", "I"], size=n), "is_5g_mode": 0.0,
        "trace_id": sid, "tech": "4G", "mobility": "car", "operator": "A", "app": "unknown",
    })


def test_timestamp_features_definition():
    rng = np.random.default_rng(1)
    df = pd.concat([_timed_trace(rng, sid="s1"), _timed_trace(rng, sid="s2")],
                   ignore_index=True)
    feat = build_features_timestamp(df, horizon=1.0, window_s=10.0)
    assert {"ts", "ts_label", "y", "t_index", *FEATURE_COLS} <= set(feat.columns)
    n_checked = 0
    for sid, g in df.groupby("trace_id", sort=False):
        g = g.reset_index(drop=True)
        tau = (g.timestamp - g.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        f = feat[feat.trace_id == sid].set_index("t_index")
        for t in range(len(g)):
            target_rows = np.flatnonzero(tau == tau[t] + 1.0)
            eligible = g.state[t] == "D" and tau[t] >= 10.0 and len(target_rows) > 0
            assert (t in f.index) == eligible
            if not eligible:
                continue
            r = f.loc[t]
            assert r.ts == tau[t] and r.ts_label == tau[t] + 1.0
            assert r.y == g.dl_mbps[target_rows[-1]]
            w = g.dl_mbps[(tau > tau[t] - 10.0) & (tau <= tau[t])].to_numpy()
            assert r.dl_mean_w == pytest.approx(w.mean(), rel=1e-12)
            assert r.dl_min_w == w.min() and r.dl_max_w == w.max()
            if len(w) > 1:
                assert r.dl_std_w == pytest.approx(w.std(ddof=1), rel=1e-12)
            else:
                assert np.isnan(r.dl_std_w)
            assert r.dl_last == g.dl_mbps[t]
            for c in RADIO_FEATURES:
                assert r[c] == g[c][t]
            n_checked += 1
    assert n_checked > 100
    dup = df.groupby("trace_id").timestamp.apply(lambda s: s.duplicated().any())
    assert dup.all()


def test_timestamp_features_no_future_leakage():
    rng = np.random.default_rng(2)
    df = _timed_trace(rng, n=200)
    feat_a = build_features_timestamp(df)
    tau = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds()
    t_star = float(tau.iloc[120])
    df2 = df.copy()
    future = tau > t_star
    df2.loc[future, "dl_mbps"] += 1000.0
    for c in RADIO_FEATURES:
        df2.loc[future, c] = df2.loc[future, c] + 7.0
    feat_b = build_features_timestamp(df2)
    a = feat_a[feat_a.ts <= t_star].reset_index(drop=True)
    b = feat_b[feat_b.ts <= t_star].reset_index(drop=True)
    assert len(a) > 50
    pd.testing.assert_frame_equal(a[FEATURE_COLS + ["t_index", "ts"]],
                                  b[FEATURE_COLS + ["t_index", "ts"]], check_exact=True)
    known = a.ts_label <= t_star
    assert np.array_equal(a.y[known].to_numpy(), b.y[known].to_numpy())
    assert (a.y[~known].to_numpy() != b.y[~known].to_numpy()).all()
