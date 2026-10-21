"""Data-pipeline tests: leakage prevention, loader integrity, split disjointness."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import read_cell_rows, read_feature_table
from src.features import FEATURE_COLS, build_features
from src.experiments import regime_pools, speed_groups, split_traces


def _synthetic_trace(sid: str, n: int = 100, tech="4G", mobility="car") -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(sid)) % (2**32))
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="s"),
        "Speed": rng.uniform(0, 60, n), "RSRP": rng.normal(-100, 5, n),
        "RSRQ": rng.normal(-12, 2, n), "SNR": rng.normal(10, 3, n),
        "CQI": rng.integers(1, 15, n).astype(float), "RSSI": rng.normal(-85, 5, n),
        "dl_mbps": rng.gamma(2, 5, n), "ul_mbps": rng.gamma(2, 1, n),
        "state": "D", "is_5g_mode": 0.0, "trace_id": sid,
        "tech": tech, "mobility": mobility, "operator": "A", "app": "unknown",
    })


def test_no_future_leakage():
    """Poison the future: features at time t must not change when y_{t+1:} changes."""
    df = _synthetic_trace("s1")
    feat_a = build_features(df, horizon=1)
    df2 = df.copy()
    t_star = 60
    df2.loc[df2.index > t_star, "dl_mbps"] += 1000.0  # perturb only future throughput
    feat_b = build_features(df2, horizon=1)
    row_a = feat_a[feat_a.t_index == t_star - 10][FEATURE_COLS[:6]]
    row_b = feat_b[feat_b.t_index == t_star - 10][FEATURE_COLS[:6]]
    pd.testing.assert_frame_equal(row_a.reset_index(drop=True), row_b.reset_index(drop=True))


def test_target_is_future_value():
    df = _synthetic_trace("s2")
    feat = build_features(df, horizon=1)
    r = feat[feat.t_index == 50].iloc[0]
    assert np.isclose(r["y"], df.loc[51, "dl_mbps"])
    assert np.isclose(r["dl_last"], df.loc[50, "dl_mbps"])


def test_split_traces_disjoint():
    frames = [_synthetic_trace(f"s{i}") for i in range(20)]
    feat = build_features(pd.concat(frames, ignore_index=True), horizon=1)
    tr, ca, te = split_traces(feat, None, seed=0)
    a = set(tr.trace_id) & set(ca.trace_id)
    b = set(tr.trace_id) & set(te.trace_id)
    c = set(ca.trace_id) & set(te.trace_id)
    assert not a and not b and not c
    assert set(tr.trace_id) | set(ca.trace_id) | set(te.trace_id) == set(feat.trace_id)


def test_split_deterministic_per_seed():
    frames = [_synthetic_trace(f"s{i}") for i in range(12)]
    feat = build_features(pd.concat(frames, ignore_index=True), horizon=1)
    tr1, _, _ = split_traces(feat, None, seed=7)
    tr2, _, _ = split_traces(feat, None, seed=7)
    assert set(tr1.trace_id) == set(tr2.trace_id)


def test_speed_groups_bins():
    X = pd.DataFrame({"Speed": [0.0, 0.5, 5.0, 20.0, 80.0, np.nan]})
    g = speed_groups(X)
    assert list(g[:5]) == ["speed_0", "speed_0", "speed_low", "speed_med", "speed_high"]
    assert g[5] in ("unknown", "nan")


def test_regime_pools_disjoint_cross():
    frames = ([_synthetic_trace(f"a{i}", tech="4G", mobility="static") for i in range(3)]
              + [_synthetic_trace(f"b{i}", tech="4G", mobility="car") for i in range(3)])
    feat = build_features(pd.concat(frames, ignore_index=True), horizon=1)
    src, tgt = regime_pools(feat, "cross_mobility_4g")
    assert not (set(src.trace_id) & set(tgt.trace_id))


def test_readers_name_the_identifier_column_trace_id(tmp_path):
    """Both readers return the single column ending in _id as trace_id, also with usecols,
    leave a trace_id column as it is and reject tables without exactly one such column."""
    df = pd.DataFrame({"tag": ["main", "main"], "unit_id": ["4G/car/A_1", "5G/x/B_2"],
                       "n": [40, 31], "coverage": [0.9, 0.85]})
    stored = tmp_path / "stored.csv.gz"
    df.to_csv(stored, index=False)
    for read in (read_feature_table, read_cell_rows):
        back = read(stored)
        assert list(back.columns) == ["tag", "trace_id", "n", "coverage"]
        assert back.trace_id.tolist() == ["4G/car/A_1", "5G/x/B_2"]
        part = read(stored, usecols=["trace_id", "coverage"])
        assert list(part.columns) == ["trace_id", "coverage"]
        assert part.coverage.tolist() == [0.9, 0.85]
    renamed = tmp_path / "renamed.csv"
    df.rename(columns={"unit_id": "trace_id"}).to_csv(renamed, index=False)
    assert read_cell_rows(renamed).equals(read_cell_rows(stored))
    for cols in ({"unit_id": "a_id", "tag": "b_id"}, {"unit_id": "unit"}):
        bad = tmp_path / "bad.csv"
        df.rename(columns=cols).to_csv(bad, index=False)
        with pytest.raises(ValueError):
            read_feature_table(bad)
