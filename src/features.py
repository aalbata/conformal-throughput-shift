"""Causal (past-only) feature construction for short-horizon throughput prediction.

At prediction time t we may use anything observed up to and including t;
the target is dl_mbps at t + horizon. All rolling statistics are computed on
values up to t (inclusive) within the same trace, so no future leakage is possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RADIO_FEATURES = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed", "is_5g_mode"]
LAG_WINDOW = 10  # seconds of throughput history summarised


def build_features(df: pd.DataFrame, horizon: int = 1,
                   lag_window: int = LAG_WINDOW) -> pd.DataFrame:
    """Return a per-sample feature table with target column 'y'.

    Rows near trace start (insufficient history) and trace end (no target)
    are dropped. Instances are restricted to rows where the device is in
    downloading state ('D') at time t, per the pre-registered protocol.
    """
    parts = []
    for sid, g in df.groupby("trace_id", sort=False):
        g = g.reset_index(drop=True)
        y = g["dl_mbps"]
        feat = pd.DataFrame(index=g.index)
        feat["dl_last"] = y
        roll = y.rolling(lag_window, min_periods=lag_window)
        feat["dl_mean_w"] = roll.mean()
        feat["dl_std_w"] = roll.std()
        feat["dl_min_w"] = roll.min()
        feat["dl_max_w"] = roll.max()
        feat["dl_diff"] = y.diff()
        for c in RADIO_FEATURES:
            feat[c] = g[c]
        feat["y"] = y.shift(-horizon)
        feat["state"] = g["state"]
        feat["trace_id"] = sid
        for c in ("tech", "mobility", "operator", "app"):
            feat[c] = g[c].iloc[0]
        feat["t_index"] = g.index
        parts.append(feat)
    out = pd.concat(parts, ignore_index=True)
    out = out[out["state"] == "D"]
    out = out.dropna(subset=["y", "dl_mean_w"])
    # radio features may be sporadically missing; median-impute is fit later on train only
    return out.reset_index(drop=True)


FEATURE_COLS = ["dl_last", "dl_mean_w", "dl_std_w", "dl_min_w", "dl_max_w", "dl_diff",
                *RADIO_FEATURES]


def train_median_impute(train_X: pd.DataFrame) -> pd.Series:
    return train_X[FEATURE_COLS].median(numeric_only=True)


def apply_impute(X: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return X[FEATURE_COLS].fillna(medians).to_numpy(dtype=float)


def trace_seconds(df: pd.DataFrame) -> np.ndarray:
    """Seconds since the first row of each trace, for rows in load_all order."""
    t0 = df.groupby("trace_id", sort=False)["timestamp"].transform("first")
    return (df["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float)


def instance_timestamps(feat: pd.DataFrame, raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Times of row t (ts) and of target row t + horizon (ts_label) of every instance.

    raw is the row table of data_loading.load_all and feat a table that build_features
    produced from it with the same horizon. Times are seconds since the first row of
    the trace. Raises ValueError unless, for every instance, dl_last equals raw dl_mbps
    at row t and y equals raw dl_mbps at row t + horizon. The result is aligned with
    feat (same index and row order).
    """
    raw = raw.reset_index(drop=True)
    first = raw.drop_duplicates("trace_id")
    start = pd.Series(first.index.to_numpy(), index=first["trace_id"].to_numpy())
    size = raw.groupby("trace_id", sort=False).size()
    sid = feat["trace_id"]
    if not sid.isin(start.index).all():
        raise ValueError("feature table contains traces that are not in the raw table")
    t = feat["t_index"].to_numpy(dtype=np.int64)
    base = sid.map(start).to_numpy(dtype=np.int64)
    n_rows = sid.map(size).to_numpy(dtype=np.int64)
    if np.any(t < 0) or np.any(t + horizon >= n_rows):
        raise ValueError("t_index or t_index + horizon outside its trace")
    pos = base + t
    pos_label = pos + horizon
    dl = raw["dl_mbps"].to_numpy(dtype=float)
    if not np.array_equal(feat["dl_last"].to_numpy(dtype=float), dl[pos]):
        raise ValueError("dl_last differs from raw dl_mbps at row t")
    if not np.array_equal(feat["y"].to_numpy(dtype=float), dl[pos_label]):
        raise ValueError("y differs from raw dl_mbps at row t + horizon")
    sec = trace_seconds(raw)
    return pd.DataFrame({"ts": sec[pos], "ts_label": sec[pos_label]}, index=feat.index)


def _window_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, standard deviation (ddof=1), minimum and maximum, ignoring NaN."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.nan
    std = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    return float(np.mean(v)), std, float(np.min(v)), float(np.max(v))


def build_features_timestamp(df: pd.DataFrame, horizon: float = 1.0,
                             window_s: float = float(LAG_WINDOW)) -> pd.DataFrame:
    """Per-instance feature table with target and window defined on timestamps.

    tau(t) is the time of row t in seconds since the first row of its trace. The
    target y of the instance at row t is dl_mbps of the last row whose timestamp
    equals tau(t) + horizon; instances without such a row are dropped. dl_mean_w,
    dl_std_w, dl_min_w and dl_max_w summarise the rows with timestamps in
    (tau(t) - window_s, tau(t)], and an instance needs tau(t) >= window_s.
    dl_last, dl_diff (difference to the previous row) and the radio features are
    those of row t, and instances are rows in downloading state ('D'), as in
    build_features. Columns ts and ts_label hold tau(t) and tau(t) + horizon.
    """
    parts = []
    for sid, g in df.groupby("trace_id", sort=False):
        g = g.reset_index(drop=True)
        tau = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        y = g["dl_mbps"].to_numpy(dtype=float)
        last = np.searchsorted(tau, tau + horizon, side="right") - 1
        has_target = tau[last] == tau + horizon
        keep = has_target & (tau >= window_s) & (g["state"].to_numpy() == "D")
        rows = np.flatnonzero(keep)
        start = np.searchsorted(tau, tau - window_s, side="right")
        end = np.searchsorted(tau, tau, side="right")
        stats = np.array([_window_stats(y[start[i]:end[i]]) for i in rows],
                         dtype=float).reshape(len(rows), 4)
        feat = pd.DataFrame(index=pd.RangeIndex(len(rows)))
        feat["dl_last"] = y[rows]
        feat["dl_mean_w"] = stats[:, 0]
        feat["dl_std_w"] = stats[:, 1]
        feat["dl_min_w"] = stats[:, 2]
        feat["dl_max_w"] = stats[:, 3]
        feat["dl_diff"] = g["dl_mbps"].diff().to_numpy(dtype=float)[rows]
        for c in RADIO_FEATURES:
            feat[c] = g[c].to_numpy()[rows]
        feat["y"] = y[last[rows]]
        feat["state"] = g["state"].to_numpy()[rows]
        feat["trace_id"] = sid
        for c in ("tech", "mobility", "operator", "app"):
            feat[c] = g[c].iloc[0]
        feat["t_index"] = rows
        feat["ts"] = tau[rows]
        feat["ts_label"] = tau[rows] + horizon
        parts.append(feat)
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["y", "dl_mean_w"])
    return out.reset_index(drop=True)
