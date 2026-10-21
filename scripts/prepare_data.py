"""Build the processed feature table from raw traces (deterministic, no seeds).

Usage: python scripts/prepare_data.py [--horizon 1]
Writes data/processed/features_h{H}.csv.gz and a data-audit JSON.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loading import load_all  # noqa: E402
from src.features import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(raw)
    feat = build_features(df, horizon=args.horizon)

    audit = {
        "n_raw_rows": int(len(df)),
        "n_traces_loaded": int(df.trace_id.nunique()),
        "dropped_short_traces": df.attrs.get("dropped_traces", []),
        "n_feature_rows": int(len(feat)),
        "n_feature_traces": int(feat.trace_id.nunique()),
        "rows_by_tech": feat.groupby("tech").size().to_dict(),
        "traces_by_tech_mobility": {
            f"{t}/{m}": int(n) for (t, m), n in
            feat.groupby(["tech", "mobility"]).trace_id.nunique().items()},
        "target_stats_mbps": {
            "mean": float(feat.y.mean()), "median": float(feat.y.median()),
            "p95": float(feat.y.quantile(0.95)), "max": float(feat.y.max()),
            "frac_zero": float((feat.y == 0).mean())},
        "horizon": args.horizon,
    }
    path = out / f"features_h{args.horizon}.csv.gz"
    feat.to_csv(path, index=False, compression="gzip")
    (out / f"data_audit_h{args.horizon}.json").write_text(
        json.dumps(audit, indent=1), encoding="utf-8")
    print(json.dumps(audit, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
