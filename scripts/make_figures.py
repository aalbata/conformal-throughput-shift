"""Manuscript figures and graphical abstract from the summaries in results/summary/.

Inputs (read only, written by scripts/analyze_amendment.py):
  results/summary/main_methods.csv       marginal coverage per regime and method
  results/summary/main_groups.csv        speed-group coverage per regime and method
  results/summary/locked_main.csv        seed-averaged trace coverage and reported median width
  results/summary/delay_grid.csv         ACI and Mondrian-ACI for reporting delays of 0 to 10 s
  results/summary/locked_downstream.csv  overshoot probability of the lower-bound rate selector
  results/summary/has_metrics.csv        streaming simulation metrics
  protocol/protocol.json                 regimes, shifted regimes, methods and alpha

Outputs in figures/ (or --out DIR):
  fig_marginal_coverage  mean marginal coverage, 7 regimes x 7 methods, 95% CIs over seeds
  fig_group_coverage     speed-group coverage under Tech 4G->5G, 95% CIs over seeds
  fig_tradeoff           mean seed-averaged trace coverage versus reported median width
  fig_delay              ACI and Mondrian-ACI versus reporting delay, three panels
  fig_downstream         overshoot probability of the lower-bound rate selector, 95% CIs
                         over traces
  each as vector PDF and 300-dpi PNG render;
  gagraphic.png (660 x 295 px, under 45,000 bytes), gagraphic.pdf, gagraphic_caption.txt

Every plotted value is read from the summaries. Axis limits are fixed per figure and checked
to contain every plotted value and confidence limit, so a changed summary that would be
clipped stops the script. PDFs carry no creation date, so reruns are byte-identical.

Usage: python scripts/make_figures.py [--summary DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
OUT = ROOT / "figures"

PROTO = json.loads((ROOT / "protocol" / "protocol.json").read_text(encoding="utf-8"))
ALPHA = PROTO["alpha_primary"]
NOMINAL = 1.0 - ALPHA
REGIMES = PROTO["regimes"]
SHIFTED = PROTO["shifted_regimes"]
METHODS = PROTO["methods"]
CONFORMAL = ["split", "cqr", "mondrian", "aci", "mondrian_aci"]
ONLINE = ["aci", "mondrian_aci"]
DOWNSTREAM_METHODS = ["split", "aci", "mondrian_aci"]
GROUPS = ["speed_0", "speed_low", "speed_med", "speed_high"]
DELAYS = [0.0, 1.0, 2.0, 5.0, 10.0]
N_SEEDS = 20
GROUP_REGIME = "cross_tech_4to5"
GA_DELAY_REGIMES = ["cross_tech_4to5", "cross_mobility_4g"]
HAS_REGIME = "cross_mobility_4g"
HAS_VARIANTS = ["mpc_hm", "mpc_robust", "mpc_split", "mpc_aci"]

ARROW = "\u2192"
REGIME_LABELS = {"in_domain_4g": "In-domain 4G", "in_domain_5g": "In-domain 5G",
                 "cross_mobility_4g": "Mobility 4G", "cross_mobility_5g": "Mobility 5G",
                 "cross_tech_4to5": f"Tech 4G{ARROW}5G", "cross_tech_5to4": f"Tech 5G{ARROW}4G",
                 "cross_app_5g": "App 5G"}
METHOD_LABELS = {"gaussian": "Gaussian", "quantile_raw": "Raw QR", "split": "Split CP",
                 "cqr": "CQR", "mondrian": "Mondrian", "aci": "ACI",
                 "mondrian_aci": "Mondrian-ACI"}
GROUP_LABELS = {"speed_0": "[0, 1)", "speed_low": "[1, 15)", "speed_med": "[15, 50)",
                "speed_high": "[50, \u221e)"}
HAS_LABELS = {"mpc_hm": "MPC-HM", "mpc_robust": "RobustMPC", "mpc_split": "Split CP",
              "mpc_aci": "ACI"}

# One fixed color per method in every figure (Okabe-Ito hues, dark gray for Gaussian).
METHOD_COLORS = {"gaussian": "#4D4D4D", "quantile_raw": "#CC79A7", "split": "#0072B2",
                 "cqr": "#56B4E9", "mondrian": "#009E73", "aci": "#E69F00",
                 "mondrian_aci": "#D55E00"}
POINT_COLOR = "#BBBBBB"
# One fixed color per regime where color encodes the regime (fig_delay, graphical abstract).
# No regime is black, so no data line shares the color of the dashed reference lines or of
# the black line-style keys in the fig_delay legend.
REGIME_COLORS = {"in_domain_4g": "#332288", "in_domain_5g": "#66AADD",
                 "cross_mobility_4g": "#117733", "cross_mobility_5g": "#999933",
                 "cross_tech_4to5": "#CC6677", "cross_tech_5to4": "#882255",
                 "cross_app_5g": "#8833EE"}
REGIME_MARKERS = {"cross_mobility_4g": "o", "cross_mobility_5g": "s", "cross_tech_4to5": "^",
                  "cross_tech_5to4": "v", "cross_app_5g": "D"}
ONLINE_STYLE = {"aci": {"ls": "-", "marker": "o", "mfc": None},
                "mondrian_aci": {"ls": (0, (3.0, 1.6)), "marker": "s", "mfc": "white"}}
HAS_COLORS = {"mpc_hm": "#4D4D4D", "mpc_robust": "#4D4D4D", "mpc_split": METHOD_COLORS["split"],
              "mpc_aci": METHOD_COLORS["aci"]}
HAS_MARKERS = {"mpc_hm": "o", "mpc_robust": "s", "mpc_split": "o", "mpc_aci": "D"}

COL_W = 3.45
FULL_W = 7.16
PDF_META = {"CreationDate": None}
PNG_DPI = 300
REF_LINE = {"color": "black", "ls": (0, (4.0, 2.0)), "lw": 0.7}
GRID = {"color": "#E3E3E3", "lw": 0.4}
ERR = {"fmt": "none", "ecolor": "black", "elinewidth": 0.5, "capsize": 1.0, "capthick": 0.5}

GA_SIZE_PX = (660, 295)
GA_DPI = 100
GA_MAX_BYTES = 45_000
GA_FONT_PT = 12.0  # 12 pt at 100 dpi is 16.7 px


def first_available(families: list[str], fallback: str) -> str:
    """First font family in families that matplotlib can resolve without falling back."""
    for name in families:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            return name
        except ValueError:
            continue
    return fallback


SERIF = first_available(["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "serif")
SANS = first_available(["Arial", "Liberation Sans", "DejaVu Sans"], "sans-serif")

STYLE = {
    "font.family": "serif", "font.serif": [SERIF, "DejaVu Serif"], "font.weight": "normal",
    "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.0,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "mathtext.fontset": "custom", "mathtext.rm": SERIF, "mathtext.it": f"{SERIF}:italic",
    "mathtext.bf": f"{SERIF}:bold",
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 2.0, "lines.linewidth": 0.8, "lines.markersize": 3.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
    "legend.labelspacing": 0.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 100, "savefig.dpi": PNG_DPI,
    "figure.constrained_layout.w_pad": 0.02, "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.wspace": 0.04, "figure.constrained_layout.hspace": 0.04,
}

GA_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": [SANS, "DejaVu Sans"],
    "font.weight": "normal", "font.size": GA_FONT_PT, "axes.labelsize": GA_FONT_PT,
    "xtick.labelsize": GA_FONT_PT, "ytick.labelsize": GA_FONT_PT,
    "legend.fontsize": GA_FONT_PT,
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 2.0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "pdf.fonttype": 42, "figure.dpi": GA_DPI,
}


# ---------------------------------------------------------------- data

def read_summaries(src: Path = SUMMARY) -> dict[str, pd.DataFrame]:
    """The summary tables used by the figures, keyed by short name."""
    files = {"main": "main_methods.csv", "groups": "main_groups.csv",
             "locked": "locked_main.csv", "delay": "delay_grid.csv",
             "downstream": "locked_downstream.csv", "has": "has_metrics.csv"}
    return {k: pd.read_csv(src / v) for k, v in files.items()}


def one(df: pd.DataFrame, **keys) -> pd.Series:
    """The single row of df whose columns equal keys; ValueError unless exactly one."""
    mask = np.ones(len(df), dtype=bool)
    for col, val in keys.items():
        mask &= (df[col] == val).to_numpy()
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {keys}, found {len(rows)}")
    return rows.iloc[0]


def require_seeds(row: pd.Series, what: str) -> None:
    """ValueError unless the row aggregates N_SEEDS seeds."""
    if int(row["n_seeds"]) != N_SEEDS:
        raise ValueError(f"{what}: n_seeds {row['n_seeds']} != {N_SEEDS}")


def check_within(values, lo: float, hi: float, what: str) -> None:
    """ValueError unless every value is finite and inside [lo, hi]."""
    v = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(v)) or v.min() < lo or v.max() > hi:
        raise ValueError(f"{what}: values {np.nanmin(v):.4f} to {np.nanmax(v):.4f} "
                         f"outside the axis range {lo} to {hi}")


def marginal_data(s: dict) -> pd.DataFrame:
    """Mean marginal coverage with its 95% CI over seeds, per regime and method."""
    recs = []
    for r in REGIMES:
        for m in METHODS:
            row = one(s["main"], regime=r, method=m)
            require_seeds(row, f"main_methods {r} {m}")
            recs.append({"regime": r, "method": m, "value": row["marginal_coverage"],
                         "lo": row["marginal_coverage_ci_lo"],
                         "hi": row["marginal_coverage_ci_hi"]})
    return pd.DataFrame(recs)


def group_data(s: dict) -> pd.DataFrame:
    """Speed-group coverage with its 95% CI over seeds under GROUP_REGIME."""
    recs = []
    for g in GROUPS:
        for m in CONFORMAL:
            row = one(s["groups"], regime=GROUP_REGIME, method=m, group=g)
            require_seeds(row, f"main_groups {g} {m}")
            recs.append({"group": g, "method": m, "value": row["coverage"],
                         "lo": row["coverage_ci_lo"], "hi": row["coverage_ci_hi"]})
    return pd.DataFrame(recs)


def tradeoff_data(s: dict) -> pd.DataFrame:
    """Mean seed-averaged trace coverage and reported median width (median across traces
    of the seed-averaged per-trace median width, Mbit/s) in the shifted regimes."""
    recs = []
    for r in SHIFTED:
        for m in CONFORMAL:
            row = one(s["locked"], regime=r, method=m)
            require_seeds(row, f"locked_main {r} {m}")
            recs.append({"regime": r, "method": m, "width": row["median_width"],
                         "coverage": row["trace_coverage"]})
    return pd.DataFrame(recs)


def delay_data(s: dict) -> pd.DataFrame:
    """Marginal coverage, instance-level infinite-interval percentage and finite-interval
    coverage of ACI and Mondrian-ACI per regime and reporting delay (h = 1)."""
    d = s["delay"][s["delay"]["tag"] == "main"]
    recs = []
    for m in ONLINE:
        for r in REGIMES:
            for D in DELAYS:
                row = one(d, method=m, regime=r, delay_s=D)
                require_seeds(row, f"delay_grid {m} {r} {D}")
                recs.append({"method": m, "regime": r, "delay_s": D,
                             "marginal_coverage": row["marginal_coverage"],
                             "inf_pct": 100.0 * row["frac_inf_instance"],
                             "finite_coverage": row["finite_coverage"]})
    return pd.DataFrame(recs)


def downstream_data(s: dict) -> pd.DataFrame:
    """Mean trace overshoot probability with its 95% CI over traces in the shifted regimes:
    the lower-bound selector of each method in DOWNSTREAM_METHODS and the point-prediction
    selector (identical for every method; checked)."""
    ds = s["downstream"]
    recs = []
    for r in SHIFTED:
        for m in DOWNSTREAM_METHODS:
            row = one(ds, regime=r, method=m, metric="overshoot_lb")
            recs.append({"regime": r, "series": m, "value": row["mean"],
                         "lo": row["mean_ci_lo"], "hi": row["mean_ci_hi"]})
        pts = ds[(ds.regime == r) & (ds.metric == "overshoot_point")]
        if len(pts) != len(METHODS):
            raise ValueError(f"overshoot_point {r}: {len(pts)} rows")
        vals = pts[["mean", "mean_ci_lo", "mean_ci_hi"]].to_numpy()
        if not np.all(vals == vals[0]):
            raise ValueError(f"overshoot_point {r} differs between methods")
        recs.append({"regime": r, "series": "point", "value": vals[0, 0],
                     "lo": vals[0, 1], "hi": vals[0, 2]})
    return pd.DataFrame(recs)


def has_data(s: dict) -> pd.DataFrame:
    """Mean average bitrate (Mbit/s) and mean stalls per trace of the MPC variants in
    HAS_VARIANTS under HAS_REGIME."""
    recs = []
    for v in HAS_VARIANTS:
        rec = {"variant": v}
        for metric in ("avg_bitrate_mbps", "stalls"):
            row = one(s["has"], regime=HAS_REGIME, variant=v, metric=metric)
            rec[metric] = row["mean"]
            rec["n_traces"] = int(row["n_traces"])
        recs.append(rec)
    return pd.DataFrame(recs)


def ga_data(s: dict) -> dict[str, pd.DataFrame]:
    """The three panels of the graphical abstract."""
    m = marginal_data(s)
    cov = m[m.regime.isin(SHIFTED) & m.method.isin(["split", "aci"])]
    d = delay_data(s)
    inf = d[(d.method == "aci") & d.regime.isin(GA_DELAY_REGIMES)]
    return {"coverage": cov.reset_index(drop=True), "delay": inf.reset_index(drop=True),
            "has": has_data(s)}


# ---------------------------------------------------------------- drawing helpers

def bar_series(ax, x, g: pd.DataFrame, width: float, color: str, label: str,
               base: float) -> None:
    """Bars from base to g.value with black 95% CI error bars from g.lo to g.hi."""
    ax.bar(x, g["value"] - base, width, bottom=base, color=color, edgecolor="white",
           linewidth=0.3, label=label, zorder=2)
    ax.errorbar(x, g["value"], yerr=[g["value"] - g["lo"], g["hi"] - g["value"]],
                zorder=4, **ERR)


def reference_line(ax, y: float) -> None:
    """Dashed horizontal reference line above the bars and below the error bars."""
    ax.axhline(y, zorder=3, **REF_LINE)


def y_grid(ax) -> None:
    """Light horizontal grid behind the data."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, **GRID)


def save(fig, out: Path, name: str) -> list[Path]:
    """Vector PDF without creation date and 300-dpi PNG render; closes the figure."""
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0], metadata=PDF_META)
    fig.savefig(paths[1], dpi=PNG_DPI)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- manuscript figures

def fig_marginal_coverage(s: dict):
    """Mean marginal coverage by regime and method with 95% CIs over seeds."""
    d = marginal_data(s)
    y0, y1 = 0.45, 1.0
    check_within(d[["value", "lo", "hi"]], y0, y1, "fig_marginal_coverage")
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4), layout="constrained")
    x = np.arange(len(REGIMES))
    w = 0.84 / len(METHODS)
    for i, m in enumerate(METHODS):
        g = d[d.method == m].set_index("regime").reindex(REGIMES)
        bar_series(ax, x + (i - (len(METHODS) - 1) / 2) * w, g, w, METHOD_COLORS[m],
                   METHOD_LABELS[m], y0)
    reference_line(ax, NOMINAL)
    y_grid(ax)
    ax.set_xticks(x, [REGIME_LABELS[r] for r in REGIMES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(REGIMES) - 0.5)
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Regime")
    ax.set_ylabel("Marginal coverage")
    fig.legend(loc="outside upper center", ncols=len(METHODS))
    return fig


def fig_group_coverage(s: dict):
    """Speed-group coverage under GROUP_REGIME with 95% CIs over seeds."""
    d = group_data(s)
    y0, y1 = 0.5, 0.95
    check_within(d[["value", "lo", "hi"]], y0, y1, "fig_group_coverage")
    fig, ax = plt.subplots(figsize=(COL_W, 2.2), layout="constrained")
    x = np.arange(len(GROUPS))
    w = 0.82 / len(CONFORMAL)
    for i, m in enumerate(CONFORMAL):
        g = d[d.method == m].set_index("group").reindex(GROUPS)
        bar_series(ax, x + (i - (len(CONFORMAL) - 1) / 2) * w, g, w, METHOD_COLORS[m],
                   METHOD_LABELS[m], y0)
    reference_line(ax, NOMINAL)
    y_grid(ax)
    ax.set_xticks(x, [GROUP_LABELS[g] for g in GROUPS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Speed bin (km/h)")
    ax.set_ylabel("Group coverage")
    fig.legend(loc="outside upper center", ncols=len(CONFORMAL))
    return fig


def fig_tradeoff(s: dict):
    """Mean seed-averaged trace coverage versus reported median width (log scale) for the
    conformal methods in the shifted regimes; color per method, marker per regime."""
    d = tradeoff_data(s)
    x0, x1, y0, y1 = 3.0, 45.0, 0.70, 1.0
    check_within(d["width"], x0, x1, "fig_tradeoff width")
    check_within(d["coverage"], y0, y1, "fig_tradeoff coverage")
    fig, ax = plt.subplots(figsize=(COL_W, 2.6), layout="constrained")
    reference_line(ax, NOMINAL)
    for m in CONFORMAL:
        for r in SHIFTED:
            row = one(d, regime=r, method=m)
            ax.plot(row["width"], row["coverage"], ls="none", marker=REGIME_MARKERS[r],
                    ms=4.2, mfc="none", mec=METHOD_COLORS[m], mew=0.8, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 40]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.7, 0.8, 0.9, 1.0]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    y_grid(ax)
    ax.set_xlabel("Median interval width (Mbit/s, log scale)")
    ax.set_ylabel("Mean trace coverage")
    method_handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none",
                             mec=METHOD_COLORS[m], mew=0.8, label=METHOD_LABELS[m])
                      for m in CONFORMAL]
    regime_handles = [Line2D([], [], ls="none", marker=REGIME_MARKERS[r], ms=4.2, mfc="none",
                             mec="black", mew=0.8, label=REGIME_LABELS[r]) for r in SHIFTED]
    fig.legend(handles=method_handles, loc="outside upper center", ncols=len(CONFORMAL))
    ax.legend(handles=regime_handles, loc="lower right", ncols=1)
    return fig


def fig_delay(s: dict):
    """(a) marginal coverage, (b) infinite-interval percentage and (c) finite-interval
    coverage of ACI and Mondrian-ACI versus reporting delay, per regime."""
    d = delay_data(s)
    panels = [("marginal_coverage", "Marginal coverage", (0.899, 0.905),
               [0.900, 0.902, 0.904], "%.3f", True),
              ("inf_pct", "Infinite intervals (%)", (0.0, 35.0), [0, 10, 20, 30], "%g", False),
              ("finite_coverage", "Finite-interval coverage", (0.84, 0.91),
               [0.84, 0.86, 0.88, 0.90], "%.2f", True)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), layout="constrained")
    for ax, (col, ylabel, ylim, ticks, fmt, ref), letter in zip(axes, panels, "abc"):
        check_within(d[col], ylim[0], ylim[1], f"fig_delay {col}")
        if ref:
            ax.axhline(NOMINAL, zorder=1, **REF_LINE)
        for m in ONLINE:
            st = ONLINE_STYLE[m]
            for r in REGIMES:
                g = d[(d.method == m) & (d.regime == r)].sort_values("delay_s")
                # Unclipped so that markers on the lower axis limit are drawn whole; every
                # value is inside the limits (check_within above).
                ax.plot(g["delay_s"], g[col], color=REGIME_COLORS[r], ls=st["ls"], lw=0.8,
                        marker=st["marker"], ms=2.6, mew=0.6,
                        mfc=st["mfc"] or REGIME_COLORS[r], mec=REGIME_COLORS[r], zorder=3,
                        clip_on=False)
        ax.set_xlim(-0.4, 10.4)
        ax.xaxis.set_major_locator(FixedLocator(DELAYS))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        y_grid(ax)
        ax.set_xlabel(f"Reporting delay $D$ (s)\n({letter})")
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=REGIME_COLORS[r], lw=1.2, label=REGIME_LABELS[r])
               for r in REGIMES]
    handles += [Line2D([], [], color="black", lw=0.8, ls=ONLINE_STYLE[m]["ls"],
                       marker=ONLINE_STYLE[m]["marker"], ms=2.6, mew=0.6,
                       mfc=ONLINE_STYLE[m]["mfc"] or "black", mec="black",
                       label=METHOD_LABELS[m]) for m in ONLINE]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles),
               handlelength=2.2, columnspacing=0.9)
    return fig


def fig_downstream(s: dict):
    """Overshoot probability of the lower-bound selectors and of the point-prediction
    selector in the shifted regimes, 95% CIs over traces, on a broken ordinate."""
    d = downstream_data(s)
    lower, upper = (0.0, 0.12), (0.20, 0.80)
    lb = d[d.series != "point"]
    pt = d[d.series == "point"]
    check_within(lb[["value", "lo", "hi"]], lower[0], lower[1], "fig_downstream bounds")
    check_within(pt[["value", "lo", "hi"]], upper[0], upper[1], "fig_downstream point")
    series = DOWNSTREAM_METHODS + ["point"]
    labels = {m: f"{METHOD_LABELS[m]} LB" for m in DOWNSTREAM_METHODS} | {
        "point": "Point prediction"}
    colors = {m: METHOD_COLORS[m] for m in DOWNSTREAM_METHODS} | {"point": POINT_COLOR}
    fig, (ax_u, ax_l) = plt.subplots(2, 1, sharex=True, figsize=(COL_W, 2.5),
                                     layout="constrained",
                                     gridspec_kw={"height_ratios": [1.0, 1.6]})
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.03)
    x = np.arange(len(SHIFTED))
    w = 0.82 / len(series)
    for i, sname in enumerate(series):
        g = d[d.series == sname].set_index("regime").reindex(SHIFTED)
        xi = x + (i - (len(series) - 1) / 2) * w
        for ax in (ax_u, ax_l):
            bar_series(ax, xi, g, w, colors[sname], labels[sname] if ax is ax_l else None,
                       0.0)
    reference_line(ax_l, ALPHA)
    ax_l.set_ylim(*lower)
    ax_u.set_ylim(*upper)
    ax_l.yaxis.set_major_locator(FixedLocator([0.0, 0.05, 0.10]))
    ax_l.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_u.yaxis.set_major_locator(FixedLocator([0.2, 0.4, 0.6, 0.8]))
    ax_u.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for ax in (ax_u, ax_l):
        y_grid(ax)
    ax_u.spines["bottom"].set_visible(False)
    ax_u.tick_params(axis="x", length=0, labelbottom=False)
    ax_l.tick_params(axis="x", length=0)
    mark = {"marker": [(-1.0, -0.6), (1.0, 0.6)], "markersize": 5, "linestyle": "none",
            "color": "black", "mec": "black", "mew": 0.6, "clip_on": False}
    ax_u.plot([0.0], [0.0], transform=ax_u.transAxes, **mark)
    ax_l.plot([0.0], [1.0], transform=ax_l.transAxes, **mark)
    ax_l.set_xticks(x, [REGIME_LABELS[r].replace(" ", "\n", 1) for r in SHIFTED])
    ax_l.set_xlim(-0.5, len(SHIFTED) - 0.5)
    ax_l.set_xlabel("Regime")
    fig.supylabel("Overshoot probability", fontsize=7.0)
    fig.legend(loc="outside upper center", ncols=len(series), columnspacing=0.8)
    return fig


# ---------------------------------------------------------------- graphical abstract

def px_axes(fig, left: float, bottom: float, width: float, height: float):
    """Axes placed in pixel coordinates of the graphical-abstract canvas."""
    w, h = GA_SIZE_PX
    return fig.add_axes((left / w, bottom / h, width / w, height / h))


def graphical_abstract(s: dict):
    """Three panels: split CP and ACI marginal coverage in the shifted regimes; ACI
    infinite-interval percentage versus reporting delay; MPC stalls versus bitrate.
    Expects the GA_STYLE rc context."""
    data = ga_data(s)
    w, h = GA_SIZE_PX
    fig = plt.figure(figsize=(w / GA_DPI, h / GA_DPI), dpi=GA_DPI)
    heading = {"ha": "center", "va": "top", "fontsize": GA_FONT_PT, "color": "#222222"}
    bottom, top = 46, 251  # axes edges in px from the bottom of the canvas
    height = top - bottom

    # Coverage of split CP and ACI in the shifted regimes.
    ax = px_axes(fig, 110, bottom, 120, height)
    cov = data["coverage"]
    x0, x1 = 0.55, 1.0
    check_within(cov["value"], x0, x1, "graphical abstract coverage")
    rows = np.arange(len(SHIFTED))[::-1]
    for yi, r in zip(rows, SHIFTED):
        sp = one(cov, regime=r, method="split")["value"]
        ac = one(cov, regime=r, method="aci")["value"]
        ax.plot([sp, ac], [yi, yi], color="#B0B0B0", lw=1.6, zorder=1)
        # Markers unclipped so that a value near the upper limit is drawn whole.
        ax.plot(ac, yi, ls="none", marker="D", ms=8.5, color=METHOD_COLORS["aci"],
                mec="white", mew=0.8, zorder=3, clip_on=False)
        ax.plot(sp, yi, ls="none", marker="o", ms=6.5, color=METHOD_COLORS["split"],
                mec="white", mew=0.8, zorder=4, clip_on=False)
    ax.axvline(NOMINAL, color="black", ls=(0, (3.0, 2.0)), lw=1.0, zorder=0)
    ax.set_yticks(rows, [REGIME_LABELS[r] for r in SHIFTED])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(SHIFTED) - 0.4)
    ax.set_xlim(x0, x1)
    ax.set_xticks([0.6, 0.8, 1.0], ["0.6", "0.8", "1.0"])
    ax.set_xlabel("Coverage (target 0.9)")
    fig.text(170 / w, 1 - 3 / h, "Coverage under shift", **heading)
    key = [Line2D([], [], ls="none", marker="o", ms=6.5, color=METHOD_COLORS["split"],
                  mec="white", mew=0.8, label=METHOD_LABELS["split"]),
           Line2D([], [], ls="none", marker="D", ms=8.5, color=METHOD_COLORS["aci"],
                  mec="white", mew=0.8, label=METHOD_LABELS["aci"])]
    fig.legend(handles=key, loc="upper center", bbox_to_anchor=(170 / w, 1 - 22 / h),
               ncols=2, frameon=False, handlelength=0.8, handletextpad=0.3,
               columnspacing=1.0, borderpad=0.0, borderaxespad=0.0)

    # ACI infinite-interval percentage versus reporting delay.
    ax = px_axes(fig, 292, bottom, 116, height)
    dd = data["delay"]
    y0, y1 = 0.0, 48.0
    check_within(dd["inf_pct"], y0, y1, "graphical abstract delay")
    for r in GA_DELAY_REGIMES:
        g = dd[dd.regime == r].sort_values("delay_s")
        ax.plot(g["delay_s"], g["inf_pct"], color=REGIME_COLORS[r], lw=2.0, marker="o",
                ms=5, mec="white", mew=0.6, label=REGIME_LABELS[r])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(y0, y1)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 20, 40])
    ax.set_xlabel("Reporting delay (s)")
    ax.set_ylabel("Infinite intervals (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0), frameon=False,
              handlelength=1.0, handletextpad=0.3, borderaxespad=0.0, borderpad=0.0,
              labelspacing=0.15)
    fig.text(350 / w, 1 - 3 / h, "ACI under delay", **heading)

    # Streaming simulation under HAS_REGIME.
    ax = px_axes(fig, 470, bottom, 176, height)
    hd = data["has"]
    x0, x1, y0, y1 = 4.0, 15.0, 0.0, 20.0
    check_within(hd["avg_bitrate_mbps"], x0, x1, "graphical abstract bitrate")
    check_within(hd["stalls"], y0, y1, "graphical abstract stalls")
    place = {"mpc_hm": (-0.5, 0.0, "right", "center"),
             "mpc_robust": (0.0, 1.3, "center", "bottom"),
             "mpc_split": (0.5, -0.7, "left", "center"),
             "mpc_aci": (-0.45, 0.0, "right", "center")}
    for _, row in hd.iterrows():
        v = row["variant"]
        ax.plot(row["avg_bitrate_mbps"], row["stalls"], ls="none", marker=HAS_MARKERS[v],
                ms=8, color=HAS_COLORS[v], mec="white", mew=0.8, zorder=3)
        dx, dy, ha, va = place[v]
        ax.text(row["avg_bitrate_mbps"] + dx, row["stalls"] + dy, HAS_LABELS[v], ha=ha,
                va=va, color="#222222")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([5, 10, 15])
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel("Bitrate (Mbit/s)")
    ax.set_ylabel("Stalls per trace")
    fig.text(558 / w, 1 - 3 / h, f"Streaming, {REGIME_LABELS[HAS_REGIME]}", **heading)
    return fig, data


def ga_caption(data: dict) -> str:
    """One-sentence caption naming exactly what the graphical abstract shows."""
    cov = data["coverage"]
    sp = cov[cov.method == "split"]["value"]
    ac = cov[cov.method == "aci"]["value"]
    dd = data["delay"]
    parts = []
    for r in GA_DELAY_REGIMES:
        g = dd[dd.regime == r].sort_values("delay_s")["inf_pct"].to_numpy()
        parts.append(f"{REGIME_LABELS[r]} ({g[0]:.1f}% at 0 s to {g[-1]:.1f}% at "
                     f"{DELAYS[-1]:g} s)")
    hd = data["has"].set_index("variant")
    has_txt = ", ".join(f"{HAS_LABELS[v]} {hd.loc[v, 'avg_bitrate_mbps']:.1f} Mbit/s and "
                        f"{hd.loc[v, 'stalls']:.1f} stalls" for v in HAS_VARIANTS)
    return (f"Left, mean marginal coverage of {NOMINAL:.0%} prediction intervals over "
            f"{N_SEEDS} seeds in the five shifted regimes for split conformal prediction "
            f"(Split CP, {sp.min():.3f} to {sp.max():.3f}) and adaptive conformal inference "
            f"(ACI, {ac.min():.3f} to {ac.max():.3f}), with the target {NOMINAL:.2f} dashed; "
            f"middle, percentage of ACI intervals that are infinite versus the delay with "
            f"which throughput labels are reported, in the {parts[0]} and {parts[1]} "
            f"regimes; right, mean stalls per trace versus mean average bitrate in the video "
            f"streaming simulation under {REGIME_LABELS[HAS_REGIME]} for the model predictive "
            f"control (MPC) player driven by the harmonic-mean throughput estimate (MPC-HM), "
            f"by RobustMPC, by the split CP lower bound and by the ACI lower bound "
            f"({has_txt}).\n")


def save_ga(fig, out: Path) -> list[Path]:
    """PNG of exactly GA_SIZE_PX quantized to a palette, under GA_MAX_BYTES, and PDF."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=GA_DPI)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if img.size != GA_SIZE_PX:
        raise ValueError(f"graphical abstract is {img.size} px, expected {GA_SIZE_PX}")
    png = out / "gagraphic.png"
    img.quantize(colors=96, method=Image.Quantize.MEDIANCUT,
                 dither=Image.Dither.NONE).save(png, optimize=True)
    if png.stat().st_size >= GA_MAX_BYTES:
        raise ValueError(f"{png.name} has {png.stat().st_size} bytes")
    pdf = out / "gagraphic.pdf"
    fig.savefig(pdf, metadata=PDF_META)
    plt.close(fig)
    return [png, pdf]


# ---------------------------------------------------------------- main

FIGURES = {"fig_marginal_coverage": fig_marginal_coverage,
           "fig_group_coverage": fig_group_coverage,
           "fig_tradeoff": fig_tradeoff,
           "fig_delay": fig_delay,
           "fig_downstream": fig_downstream}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = read_summaries(args.summary)
    written = []
    with plt.rc_context(STYLE):
        for name, build in FIGURES.items():
            written += save(build(s), args.out, name)
    with plt.rc_context(GA_STYLE):
        fig, data = graphical_abstract(s)
        written += save_ga(fig, args.out)
    cap = args.out / "gagraphic_caption.txt"
    cap.write_text(ga_caption(data), encoding="utf-8", newline="\n")
    written.append(cap)
    for p in written:
        print(f"{p.relative_to(args.out.parent) if p.is_relative_to(args.out.parent) else p}"
              f"  {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
