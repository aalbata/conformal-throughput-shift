"""Contract of experiments.cell_instances and its agreement with run_cell."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import (METHODS, ONLINE_METHODS, cell_instances, fit_cell, regime_pools,
                             run_cell, run_online, split_traces)
from src.features import build_features, instance_timestamps

ALPHA, GAMMA = 0.1, 0.05


def _raw_trace(rng, sid, tech, mobility, n=130):
    steps = rng.choice([0, 1, 1, 1, 1, 2, 3], size=n)
    steps[0] = 0
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(steps), unit="s")
    speed = {"static": 0.0, "car": 40.0}[mobility]
    level = rng.gamma(2, 5)
    return pd.DataFrame({
        "timestamp": ts, "Speed": np.abs(rng.normal(speed, 8, n)),
        "RSRP": rng.normal(-100, 5, n), "RSRQ": rng.normal(-12, 2, n),
        "SNR": rng.normal(10, 3, n), "CQI": rng.integers(1, 15, n).astype(float),
        "RSSI": rng.normal(-85, 5, n),
        "dl_mbps": np.clip(level + np.cumsum(rng.normal(0, 1.5, n)), 0, None),
        "ul_mbps": rng.gamma(2, 1, n), "state": "D",
        "is_5g_mode": 1.0 if tech == "5G" else 0.0, "trace_id": sid, "tech": tech,
        "mobility": mobility, "operator": "A", "app": "unknown",
    })


@pytest.fixture(scope="module")
def feat():
    rng = np.random.default_rng(0)
    frames = [_raw_trace(rng, f"4G/{m}/s{i}", "4G", m)
              for i in range(8) for m in ("static", "car")]
    raw = pd.concat(frames, ignore_index=True)
    f = build_features(raw, horizon=1)
    return pd.concat([f, instance_timestamps(f, raw, 1)], axis=1)


def test_cell_instances_contract(feat):
    out = cell_instances(feat, "in_domain_4g", seed=0)
    expected = (["trace_id", "t_index", "ts", "ts_label", "y", "pred", "group"]
                + [f"{p}_{m}" for m in METHODS for p in ("lo", "hi")]
                + [f"alpha_{m}" for m in ONLINE_METHODS])
    assert sorted(out.columns) == sorted(expected)
    for m in METHODS:
        assert {f"lo_{m}", f"hi_{m}"} <= set(out.columns)

    source, target = regime_pools(feat, "in_domain_4g")
    _, _, test = split_traces(source, target, 0)
    assert len(out) == len(test)
    assert list(dict.fromkeys(out.trace_id)) == list(dict.fromkeys(test.trace_id))
    for _, g in out.groupby("trace_id", sort=False):
        assert (np.diff(g.t_index.to_numpy()) > 0).all()
        assert (np.diff(g.ts.to_numpy()) >= 0).all()
        assert g.alpha_aci.iloc[0] == ALPHA and g.alpha_mondrian_aci.iloc[0] == ALPHA
    assert (out.ts_label >= out.ts).all()
    for m in METHODS:
        assert (out[f"lo_{m}"] >= 0).all()


def test_cell_instances_matches_run_cell(feat):
    out = cell_instances(feat, "in_domain_4g", seed=1, alpha=ALPHA, gamma=GAMMA)
    rows = run_cell(feat, "in_domain_4g", 1, ALPHA, "gbm", gamma=GAMMA)
    for m in METHODS:
        cov = ((out.y >= out[f"lo_{m}"]) & (out.y <= out[f"hi_{m}"]))
        per = cov.groupby(out.trace_id, sort=False).agg(["size", "mean"])
        ref = rows[rows.method == m].set_index("trace_id")
        assert list(per.index) == list(ref.index)
        assert (per["size"].to_numpy() == ref["n"].to_numpy()).all()
        assert np.array_equal(per["mean"].to_numpy(), ref["coverage"].to_numpy())


def test_delay_zero_equals_immediate_and_delay_changes_only_online(feat):
    fit = fit_cell(feat, "in_domain_4g", 2, ALPHA, "gbm", GAMMA)
    imm, st_imm = run_online(fit, None)
    d0, st0 = run_online(fit, 0.0)
    assert st_imm.empty and len(st0) == 2 * fit.test.trace_id.nunique()
    for m in ONLINE_METHODS:
        for k in range(3):
            assert np.array_equal(imm[m][k], d0[m][k])
    assert (st0.max_outstanding == 0).all()

    a = cell_instances(feat, "in_domain_4g", seed=2, delay_s=0.0)
    b = cell_instances(feat, "in_domain_4g", seed=2, delay_s=5.0)
    for m in METHODS:
        if m not in ONLINE_METHODS:
            assert np.array_equal(a[f"lo_{m}"], b[f"lo_{m}"])
            assert np.array_equal(a[f"hi_{m}"], b[f"hi_{m}"])
    assert not np.array_equal(a.alpha_aci, b.alpha_aci)


def test_cell_instances_requires_timestamps(feat):
    with pytest.raises(ValueError):
        cell_instances(feat.drop(columns=["ts", "ts_label"]), "in_domain_4g", seed=0)
