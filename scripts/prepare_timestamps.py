"""Attach instance timestamps to the processed feature tables and build the
timestamp-based feature table (protocol amendment 1, sections 1.4 and 1.7).

Usage: python scripts/prepare_timestamps.py
Reads data/raw (through src/data_loading.py, as prepare_data.py does) and
data/processed/features_h{1,5}.csv.gz, which are left unchanged. Writes
  data/processed/features_h{1,5}_timed.csv.gz  existing table plus columns ts, ts_label
                                              (seconds since trace start of row t and
                                              of target row t + h)
  data/processed/features_timebase_h1.csv.gz   features and target on timestamps (1.7)
  data/processed/timestamps_audit.json         join checks, sizes and SHA-256 digests
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loading import append_csv_columns, load_all, read_feature_table  # noqa: E402
from src.features import build_features_timestamp, instance_timestamps  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach(raw: pd.DataFrame, horizon: int) -> dict:
    """Write features_h{horizon}_timed.csv.gz and verify it against the source table."""
    src = PROC / f"features_h{horizon}.csv.gz"
    dst = PROC / f"features_h{horizon}_timed.csv.gz"
    digest_before = sha256(src)
    exact = read_feature_table(src, float_precision="round_trip")
    ts = instance_timestamps(exact, raw, horizon)
    append_csv_columns(src, dst, {"ts": ts["ts"].to_numpy(), "ts_label": ts["ts_label"].to_numpy()})

    old = read_feature_table(src)
    new = read_feature_table(dst)
    pd.testing.assert_frame_equal(new[list(old.columns)], old, check_exact=True)
    assert list(new.columns) == list(old.columns) + ["ts", "ts_label"]
    new_exact = read_feature_table(dst, float_precision="round_trip")
    assert np.array_equal(new_exact["ts"].to_numpy(), ts["ts"].to_numpy())
    assert np.array_equal(new_exact["ts_label"].to_numpy(), ts["ts_label"].to_numpy())
    assert sha256(src) == digest_before
    return {
        "source": src.name, "source_sha256": digest_before,
        "output": dst.name, "output_sha256": sha256(dst),
        "n_instances": int(len(old)),
        "check_dl_last_equals_raw_row_t": "passed for every instance",
        "check_y_equals_raw_row_t_plus_h": "passed for every instance",
        "check_existing_columns_unchanged": "passed (exact comparison of the parsed tables)",
        "ts_nondecreasing_within_trace": bool(
            (new.groupby("trace_id", sort=False)["ts"].diff().fillna(0) >= 0).all()),
    }


def main():
    raw = load_all(RAW)
    audit = {"n_raw_rows": int(len(raw)), "n_traces": int(raw.trace_id.nunique()),
             "timed": [attach(raw, h) for h in (1, 5)]}

    tb = build_features_timestamp(raw, horizon=1)
    path = PROC / "features_timebase_h1.csv.gz"
    tb.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    back = read_feature_table(path)
    assert list(back.columns) == list(tb.columns) and len(back) == len(tb)
    audit["timebase"] = {"output": path.name, "output_sha256": sha256(path),
                         "horizon_s": 1, "window_s": 10, "n_instances": int(len(tb)),
                         "n_traces": int(tb.trace_id.nunique())}
    (PROC / "timestamps_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
