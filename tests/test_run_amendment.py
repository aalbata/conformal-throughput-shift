"""Reproduction criterion, per-tag output columns and stop rule of scripts/run_amendment.py."""
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
from src.experiments import METHODS, fit_cell, run_cell, run_online  # noqa: E402
from src.features import build_features, instance_timestamps  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_amendment",
                                                  ROOT / "scripts" / "run_amendment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ra():
    return _load_runner()


def _raw_trace(rng, sid, mobility, n=130):
    steps = rng.choice([0, 1, 1, 1, 1, 2, 3], size=n)
    steps[0] = 0
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(steps), unit="s")
    speed = {"static": 0.0, "car": 40.0}[mobility]
    return pd.DataFrame({
        "timestamp": ts, "Speed": np.abs(rng.normal(speed, 8, n)),
        "RSRP": rng.normal(-100, 5, n), "RSRQ": rng.normal(-12, 2, n),
        "SNR": rng.normal(10, 3, n), "CQI": rng.integers(1, 15, n).astype(float),
        "RSSI": rng.normal(-85, 5, n),
        "dl_mbps": np.clip(rng.gamma(2, 5) + np.cumsum(rng.normal(0, 1.5, n)), 0, None),
        "ul_mbps": rng.gamma(2, 1, n), "state": "D", "is_5g_mode": 0.0,
        "trace_id": sid, "tech": "4G", "mobility": mobility, "operator": "A",
        "app": "unknown",
    })


@pytest.fixture(scope="module")
def feat():
    rng = np.random.default_rng(11)
    raw = pd.concat([_raw_trace(rng, f"4G/{m}/s{i}", m) for i in range(8)
                     for m in ("static", "car")], ignore_index=True)
    f = build_features(raw, horizon=1)
    return pd.concat([f, instance_timestamps(f, raw, 1)], axis=1)


def _locked_like(feat, cells, tag="main"):
    parts = []
    for regime, seed in cells:
        rows = run_cell(feat, regime, seed, 0.1, "gbm", gamma=0.05)
        rows.insert(0, "tag", tag)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def test_compare_with_locked_tolerance_and_pairing(ra):
    new = pd.DataFrame({"method": ["aci", "aci", "split"], "trace_id": ["a", "b", "a"],
                        "n": [10, 20, 10], "coverage": [0.9, 0.85, 0.7],
                        "frac_inf": [0.1, 0.0, 0.0]})
    ok, _ = ra.compare_with_locked(new, new.copy())
    assert ok["passed"] and ok["n_matched"] == 3

    at_tol = new.copy()
    at_tol.loc[0, "coverage"] += 1e-12 * 0.99
    at_tol.loc[1, "frac_inf"] += 1e-12 * 0.99
    assert ra.compare_with_locked(new, at_tol)[0]["passed"]

    for col, delta in (("coverage", 2e-12), ("frac_inf", 2e-12), ("n", 1)):
        bad = new.copy()
        bad.loc[2, col] += delta
        s, mism = ra.compare_with_locked(new, bad)
        assert not s["passed"] and s["n_matched"] == 2 and len(mism) == 1

    s, _ = ra.compare_with_locked(new, new.iloc[:2])
    assert not s["passed"] and s["n_unpaired"] == 1
    extra = pd.concat([new, new.iloc[[0]].assign(trace_id="c")], ignore_index=True)
    assert not ra.compare_with_locked(new, extra)[0]["passed"]
    nan = new.copy()
    nan.loc[0, "coverage"] = np.nan
    assert not ra.compare_with_locked(new, nan)[0]["passed"]
    assert not ra.compare_with_locked(new.iloc[:0], new.iloc[:0])[0]["passed"]


def test_metric_tables_columns_per_tag(ra, feat):
    fit = fit_cell(feat, "cross_mobility_4g", 0, 0.1, "gbm", 0.05)
    variants = [(m, "none", np.nan, lo, hi, None) for m, (lo, hi) in fit.offline.items()]
    for d in (0.0, 5.0):
        online, stats = run_online(fit, d)
        for m in ra.ONLINE_METHODS:
            variants.append((m, "delayed", d, online[m][0], online[m][1],
                             stats[stats.method == m]))
    n_traces = fit.test.trace_id.nunique()

    traces, groups = ra.metric_tables("main", "cross_mobility_4g", 0, fit, variants)
    assert list(traces.columns) == (ra.ID_COLS + ["trace_id"] + ra.COLS_15 + ra.COLS_16
                                    + ra.STAT_COLS + ["bound"])
    assert len(traces) == n_traces * len(variants)
    d5 = traces[traces.delay_s == 5.0]
    assert d5[ra.POSITIVE_COLS].isna().all().all() and d5["n_y0"].notna().all()
    d0 = traces[traces.delay_s == 0.0]
    assert d0[ra.POSITIVE_COLS].notna().all().all()
    offline = traces[traces.feedback == "none"]
    assert offline[ra.STAT_COLS + ["bound"]].isna().all().all()
    online = traces[traces.feedback == "delayed"]
    assert np.allclose(online["bound"], delay_coverage_bound(
        0.1, 0.05, online["max_outstanding"].to_numpy(), online["n_applied"].to_numpy()))
    assert (d0["max_outstanding"] == 0).all()
    assert set(groups.columns) == set(ra.ID_COLS + ["group", "n", "n_cov", "coverage"])
    assert groups.groupby(["method", "delay_s"], dropna=False)["n"].sum().eq(len(fit.test)).all()

    t5, _ = ra.metric_tables("h5", "cross_mobility_4g", 0, fit, variants)
    assert list(t5.columns) == (ra.ID_COLS + ["trace_id"] + ra.COLS_15 + ra.STAT_COLS
                                + ["bound"])
    tts, gts = ra.metric_tables("ts", "cross_mobility_4g", 0, fit, variants)
    assert list(tts.columns) == ra.ID_COLS + ["trace_id"] + ra.COLS_17
    assert gts.empty


def _patch_runner(ra, monkeypatch, tmp_path, feat, cells):
    proc = tmp_path / "processed"
    proc.mkdir()
    feat.to_csv(proc / "features_h1_timed.csv.gz", index=False)
    out = tmp_path / "amendment1"
    monkeypatch.setattr(ra, "LOCKED", tmp_path / "locked_rows.csv.gz")
    monkeypatch.setattr(ra, "OUT", out)
    monkeypatch.setattr(ra, "CELLS", out / "cells")
    monkeypatch.setattr(ra, "REPRO_DIR", out / "reproduce_only")
    monkeypatch.setattr(ra, "TIMING_LOG", out / "timing_log.csv")
    cfg = dict(ra.TAGS["main"], features=proc / "features_h1_timed.csv.gz",
               regimes=sorted({r for r, _ in cells}), seeds=sorted({s for _, s in cells}))
    monkeypatch.setitem(ra.TAGS, "main", cfg)


def _run_main(ra, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_amendment.py", "--tag", "main", "--n-jobs", "1",
                                      *args])
    ra.main()


def test_execution_stops_when_reproduction_fails(ra, feat, tmp_path, monkeypatch):
    cells = [("cross_mobility_4g", 0), ("cross_mobility_4g", 1), ("in_domain_4g", 0),
             ("in_domain_4g", 1)]
    _patch_runner(ra, monkeypatch, tmp_path, feat, cells)
    locked = _locked_like(feat, cells)
    corrupt = locked.copy()
    first = corrupt.index[(corrupt.regime == "cross_mobility_4g") & (corrupt.seed == 0)][0]
    corrupt.loc[first, "coverage"] += 1e-9
    corrupt.to_csv(ra.LOCKED, index=False)

    with pytest.raises(SystemExit) as stop:
        _run_main(ra, monkeypatch)
    assert stop.value.code == 1
    assert (ra.CELLS / "main__cross_mobility_4g__s0.failed.json").exists()
    assert not list(ra.CELLS.glob("*.done.json"))
    assert not list(ra.CELLS.glob("*.traces.csv.gz"))
    assert not (ra.OUT / "main_traces.csv.gz").exists()

    with pytest.raises(SystemExit) as refused:
        _run_main(ra, monkeypatch)
    assert refused.value.code == 2
    assert not list(ra.CELLS.glob("*.done.json"))

    locked.to_csv(ra.LOCKED, index=False)
    _run_main(ra, monkeypatch, "--retry-failed")
    assert len(list(ra.CELLS.glob("*.done.json"))) == len(cells)
    traces = pd.read_csv(ra.OUT / "main_traces.csv.gz")
    assert sorted(traces.delay_s.dropna().unique()) == ra.DELAY_GRID
    assert set(traces.method) == set(METHODS)
    cfg = json.loads((ra.OUT / "run_config_main.json").read_text())
    assert cfg["reproduction_tolerance"] == 1e-12


def test_reproduce_only_does_not_stop_or_write_metrics(ra, feat, tmp_path, monkeypatch):
    cells = [("cross_mobility_4g", 0), ("in_domain_4g", 0)]
    _patch_runner(ra, monkeypatch, tmp_path, feat, cells)
    locked = _locked_like(feat, cells)
    locked.loc[locked.index[(locked.regime == "cross_mobility_4g")][0], "n"] += 1
    locked.to_csv(ra.LOCKED, index=False)
    _run_main(ra, monkeypatch, "--reproduce-only")
    log = pd.read_csv(ra.REPRO_DIR / "reproduction_log.csv")
    assert sorted(log.regime) == ["cross_mobility_4g", "in_domain_4g"]
    assert log.set_index("regime").loc["cross_mobility_4g", "repro_passed"] == False  # noqa: E712
    assert log.set_index("regime").loc["in_domain_4g", "repro_passed"] == True  # noqa: E712
    assert not ra.CELLS.exists()
