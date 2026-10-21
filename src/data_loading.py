"""Load and clean the UCC MISL 4G (MMSys'18) and 5G (MMSys'20) trace datasets.

Each CSV is one measurement trace at 1 Hz. Trace metadata is encoded in the
directory layout and filename prefix (operator). Missing values are '-'.
DL_bitrate / UL_bitrate are in kbit/s (G-NetTrack Pro convention); we convert to Mbit/s.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLS = ["Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate"]

# mobility labels normalised across the two datasets
MOBILITY_MAP = {
    "bus": "bus", "car": "car", "pedestrian": "pedestrian", "static": "static",
    "train": "train", "Driving": "driving", "Static": "static",
}


@dataclass(frozen=True)
class TraceMeta:
    trace_id: str
    tech: str        # '4G' | '5G'
    mobility: str    # normalised label
    operator: str    # 'A' | 'B' | 'unknown'
    app: str         # 5G: 'Amazon_Prime' | 'Netflix' | 'Download'; 4G: 'unknown'
    path: str


def discover_traces(raw_root: Path) -> list[TraceMeta]:
    """Enumerate all trace CSVs with metadata derived from the folder layout."""
    traces: list[TraceMeta] = []
    four_g = raw_root / "4G" / "Dataset"
    if four_g.exists():
        for mob_dir in sorted(four_g.iterdir()):
            if not mob_dir.is_dir():
                continue
            for f in sorted(mob_dir.glob("*.csv")):
                op = f.name.split("_")[0]
                traces.append(TraceMeta(
                    trace_id=f"4G/{mob_dir.name}/{f.stem}",
                    tech="4G", mobility=MOBILITY_MAP.get(mob_dir.name, mob_dir.name),
                    operator=op if op in ("A", "B") else "unknown",
                    app="unknown", path=str(f)))
    five_g = raw_root / "5G" / "5G-production-dataset"
    if five_g.exists():
        for f in sorted(five_g.rglob("*.csv")):
            if "__MACOSX" in str(f):
                continue
            rel = f.relative_to(five_g).parts
            app = rel[0]
            mobility = MOBILITY_MAP.get(rel[1], rel[1]) if len(rel) > 1 else "unknown"
            op = f.name.split("_")[0]
            traces.append(TraceMeta(
                trace_id="5G/" + "/".join(rel)[:-4],
                tech="5G", mobility=mobility,
                operator=op if op in ("A", "B") else "unknown",
                app=app, path=str(f)))
    return traces


def load_trace(meta: TraceMeta) -> pd.DataFrame:
    """Load one trace CSV, coerce numerics, convert bitrate units, sort by time."""
    df = pd.read_csv(meta.path, dtype=str)
    ts = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    out = pd.DataFrame({"timestamp": ts})
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    out["dl_mbps"] = out.pop("DL_bitrate") / 1000.0
    out["ul_mbps"] = out.pop("UL_bitrate") / 1000.0
    out["state"] = df.get("State", pd.Series(dtype=str)).astype(str)
    nm = df.get("NetworkMode", pd.Series(dtype=str)).astype(str)
    out["is_5g_mode"] = (nm == "5G").astype(float)
    out["trace_id"] = meta.trace_id
    out["tech"] = meta.tech
    out["mobility"] = meta.mobility
    out["operator"] = meta.operator
    out["app"] = meta.app
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def load_all(raw_root: Path, min_rows: int = 60) -> pd.DataFrame:
    """Load every trace; drop traces shorter than min_rows samples."""
    frames = []
    dropped = []
    for meta in discover_traces(raw_root):
        df = load_trace(meta)
        if len(df) >= min_rows:
            frames.append(df)
        else:
            dropped.append((meta.trace_id, len(df)))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.attrs["dropped_traces"] = dropped
    return all_df


TRACE_ID = "trace_id"


def _with_trace_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with its trace identifier column named trace_id.

    The identifier is the single column whose name ends in _id; a file may store it
    under another name, which is replaced here so that all code uses trace_id.
    """
    ids = [c for c in df.columns if str(c).endswith("_id")]
    if len(ids) != 1:
        raise ValueError(f"expected one trace identifier column ending in _id, found {ids}")
    return df if ids[0] == TRACE_ID else df.rename(columns={ids[0]: TRACE_ID})


def _read_with_trace_id(path, usecols=None, **kwargs) -> pd.DataFrame:
    """pandas.read_csv followed by _with_trace_id; usecols may name trace_id."""
    if usecols is not None:
        wanted = set(usecols) - {TRACE_ID}
        kwargs["usecols"] = lambda c: c in wanted or c.endswith("_id")
    return _with_trace_id(pd.read_csv(path, **kwargs))


def read_feature_table(path, **kwargs) -> pd.DataFrame:
    """Read a feature table (data/processed/features_*.csv.gz) with its trace
    identifier column as trace_id. Keyword arguments go to pandas.read_csv; usecols
    may name trace_id."""
    return _read_with_trace_id(path, **kwargs)


def read_cell_rows(path, **kwargs) -> pd.DataFrame:
    """Read per-trace rows written by experiments.run_cell (results/locked/locked_rows.csv.gz,
    the per-cell files in results/locked/cells/ and the pilot files in results/pilot/)
    with the trace identifier column as trace_id. Keyword arguments as read_feature_table."""
    return _read_with_trace_id(path, **kwargs)


def append_csv_columns(src: Path, dst: Path, columns: dict) -> None:
    """Write a copy of the CSV src (gzip if the name ends in .gz) to dst with extra
    columns appended to every line; the existing text of each line is unchanged.

    columns maps a column name to an array with one value per data line; floats are
    written with repr() and NaN as an empty field. The line terminator of src is kept,
    and gzip output carries no timestamp, so equal inputs give identical files.
    """
    import gzip

    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("dst must differ from src")
    with (gzip.open(src, "rb") if src.suffix == ".gz" else open(src, "rb")) as f:
        lines = f.read().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    eol = b"\r\n" if lines and lines[0].endswith(b"\r") else b"\n"
    lines = [ln[:-1] if ln.endswith(b"\r") else ln for ln in lines]
    names = list(columns)
    values = [np.asarray(columns[c]) for c in names]
    n = len(lines) - 1
    if any(len(v) != n for v in values):
        raise ValueError(f"every column needs {n} values")
    if any(b'"' in ln for ln in lines):
        raise ValueError("quoted CSV fields are not supported")

    def fmt(v) -> bytes:
        v = float(v)
        return b"" if np.isnan(v) else repr(v).encode("ascii")

    out = [lines[0] + b"," + ",".join(names).encode("ascii")]
    cols = [[fmt(x) for x in v] for v in values]
    for i in range(n):
        out.append(lines[i + 1] + b"," + b",".join(c[i] for c in cols))
    payload = eol.join(out) + eol
    if dst.suffix == ".gz":
        with open(dst, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh,
                                                  mtime=0) as gz:
            gz.write(payload)
    else:
        dst.write_bytes(payload)
