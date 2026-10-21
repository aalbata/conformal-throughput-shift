"""Write the LaTeX tables and the number macros of the article from the summaries.

Inputs: results/summary/summary.json (scripts/analyze_amendment.py) and
results/summary/pool_sizes.json (scripts/pool_sizes.py).
Outputs in results/summary/tex/: numbers.tex (one \\newcommand per number quoted in the
text) and tab_main.tex, tab_tests.tex, tab_inform.tex, tab_winkler.tex, tab_has.tex,
tab_hastests.tex, tab_ablations.tex. Every value is taken from the summaries at full
precision and rounded once, here. A missing or non-finite value stops the script.

Usage: python scripts/make_tables.py
"""
from __future__ import annotations

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"
OUT = SUM / "tex"

REGIMES = ["in_domain_4g", "in_domain_5g", "cross_mobility_4g", "cross_mobility_5g",
           "cross_tech_4to5", "cross_tech_5to4", "cross_app_5g"]
SHIFTED = REGIMES[2:]
HAS_SHIFTED = ["cross_mobility_4g", "cross_mobility_5g", "cross_tech_4to5", "cross_tech_5to4"]
METHODS = ["gaussian", "quantile_raw", "split", "cqr", "mondrian", "aci", "mondrian_aci"]
ONLINE = ["aci", "mondrian_aci"]
GROUPS = ["speed_0", "speed_low", "speed_med", "speed_high"]
GROUP_TEX = {"speed_0": "$[0,1)$", "speed_low": "$[1,15)$", "speed_med": "$[15,50)$",
             "speed_high": "$[50,\\infty)$"}
REGIME_TEX = {
    "in_domain_4g": "In-domain 4G", "in_domain_5g": "In-domain 5G",
    "cross_mobility_4g": "Mobility 4G", "cross_mobility_5g": "Mobility 5G",
    "cross_tech_4to5": "Tech 4G$\\rightarrow$5G", "cross_tech_5to4": "Tech 5G$\\rightarrow$4G",
    "cross_app_5g": "App 5G",
}
REGIME_TEXT = {
    "in_domain_4g": "in-domain 4G", "in_domain_5g": "in-domain 5G",
    "cross_mobility_4g": "4G mobility shift", "cross_mobility_5g": "5G mobility shift",
    "cross_tech_4to5": "$4\\text{G}\\!\\to\\!5\\text{G}$ shift",
    "cross_tech_5to4": "$5\\text{G}\\!\\to\\!4\\text{G}$ shift",
    "cross_app_5g": "application shift",
}
METHOD_TEX = {"gaussian": "Gaussian", "quantile_raw": "Raw QR", "split": "Split CP",
              "cqr": "CQR", "mondrian": "Mondrian", "aci": "ACI",
              "mondrian_aci": "Mondrian-ACI"}
VARIANTS = ["rate_hm", "rate_point", "rate_split", "rate_aci", "rate_mondrian_aci",
            "mpc_hm", "mpc_robust", "mpc_point", "mpc_split", "mpc_aci", "mpc_mondrian_aci"]
VARIANT_TEX = {
    "rate_hm": "Rate, harmonic mean", "rate_point": "Rate, point prediction",
    "rate_split": "Rate, split CP bound", "rate_aci": "Rate, ACI bound",
    "rate_mondrian_aci": "Rate, Mondrian-ACI bound", "mpc_hm": "MPC, harmonic mean",
    "mpc_robust": "RobustMPC", "mpc_point": "MPC, point prediction",
    "mpc_split": "MPC, split CP bound", "mpc_aci": "MPC, ACI bound",
    "mpc_mondrian_aci": "MPC, Mondrian-ACI bound",
}

S: dict = {}
P: dict = {}


def v(key: str) -> float:
    """Finite numeric value of a summary key."""
    if key not in S:
        raise KeyError(key)
    x = S[key]
    if x is None or isinstance(x, bool) or not math.isfinite(float(x)):
        raise ValueError(f"{key} is not a finite number: {x!r}")
    return float(x)


def fx(x: float, nd: int) -> str:
    """Round half up to nd decimals, with a math minus sign for negative values."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = format(q, "f")
    if s.startswith("-"):
        s = "$-$" + s[1:] if s.strip("-0.") else s[1:]
    return s


def fsci(x: float) -> str:
    """Two significant digits; a x 10^b below 0.001, plain decimals otherwise."""
    if x == 0:
        return "0"
    if abs(x) >= 0.001:
        nd = max(0, 1 - int(math.floor(math.log10(abs(x)))))
        return fx(x, nd)
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    mstr = fx(m, 1)
    if mstr.replace("$-$", "") == "10.0":
        mstr, e = ("$-$1.0" if x < 0 else "1.0"), e + 1
    return f"${mstr.replace('$-$', '-')}\\times10^{{{e}}}$"


def fdiff(x: float) -> str:
    """Three decimals, or a x 10^b when the value rounds to zero at three decimals."""
    return fx(x, 3) if abs(x) >= 0.0005 else fsci(x)

def fint(x: float) -> str:
    s = f"{int(round(x)):,}"
    return s.replace(",", "{,}")


def pct(x: float, nd: int = 1) -> str:
    return fx(100 * x, nd)


def pct_auto(x: float) -> str:
    return pct(x, 2 if 100 * x < 1 else 1)


def ceil_nd(x: float, nd: int) -> str:
    f = 10 ** nd
    return f"{math.ceil(x * f - 1e-12) / f:.{nd}f}"


def floor_nd(x: float, nd: int) -> str:
    f = 10 ** nd
    return f"{math.floor(x * f + 1e-12) / f:.{nd}f}"


def main_key(regime: str, method: str, metric: str) -> str:
    return f"main.{regime}.{method}.{metric}"


def locked_key(regime: str, method: str, metric: str) -> str:
    return f"locked.main.{regime}.{method}.{metric}"


def numbers() -> dict[str, str]:
    N: dict[str, str] = {}
    N["nTracesFourG"] = fint(v("audit.n_traces_4g"))
    N["nTracesFiveG"] = fint(v("audit.n_traces_5g"))
    N["nTracesFiveGDownload"] = fint(v("audit.n_traces_5g_download"))
    N["nInstances"] = fint(v("audit.n_instances"))
    N["nInstancesFourG"] = fint(v("audit.n_instances_4g"))
    N["nInstancesFiveG"] = fint(v("audit.n_instances_5g"))
    N["pctExactOneSec"] = pct(v("audit.h1_share_target_exactly_1s"))
    N["pctZeroFourG"] = pct(v("audit.zero_share_4g"))
    N["pctZeroFiveG"] = pct(v("audit.zero_share_5g"))
    N["pctZeroApp"] = pct(v("audit.zero_share_app_shift_test"))

    def cov(name, regime, method, ci=False):
        N[name] = fx(v(locked_key(regime, method, "marginal_coverage")), 3)
        if ci:
            N[name + "Lo"] = fx(v(locked_key(regime, method, "marginal_coverage_ci_lo")), 3)
            N[name + "Hi"] = fx(v(locked_key(regime, method, "marginal_coverage_ci_hi")), 3)

    cov("covSplitIDfour", "in_domain_4g", "split", True)
    cov("covSplitIDfive", "in_domain_5g", "split", True)
    cov("covCqrIDfour", "in_domain_4g", "cqr")
    cov("covCqrIDfive", "in_domain_5g", "cqr", True)
    cov("covSplitTechFtoF", "cross_tech_4to5", "split", True)
    cov("covCqrTechFtoF", "cross_tech_4to5", "cqr")
    cov("covGaussTechFtoF", "cross_tech_4to5", "gaussian")
    cov("covQrTechFtoF", "cross_tech_4to5", "quantile_raw")
    cov("covSplitTechFtoFour", "cross_tech_5to4", "split")
    cov("covSplitMobFour", "cross_mobility_4g", "split")
    cov("covSplitMobFive", "cross_mobility_5g", "split")
    cov("covSplitApp", "cross_app_5g", "split")
    N["gapSplitMobFourPts"] = fx(100 * (0.9 - v(locked_key("cross_mobility_4g", "split",
                                                            "marginal_coverage"))), 0)
    N["covPosSplitApp"] = fx(v(main_key("cross_app_5g", "split", "coverage_pos")), 3)
    N["covPosAciIDfive"] = fx(v(main_key("in_domain_5g", "aci", "coverage_pos")), 3)
    N["covPosAciMobFive"] = fx(v(main_key("cross_mobility_5g", "aci", "coverage_pos")), 3)

    def width(name, regime, method):
        N[name] = fx(v(locked_key(regime, method, "median_width")), 1)

    width("widthSplitApp", "cross_app_5g", "split")
    width("widthMaciApp", "cross_app_5g", "mondrian_aci")
    width("widthSplitMobFour", "cross_mobility_4g", "split")
    width("widthMaciMobFour", "cross_mobility_4g", "mondrian_aci")
    width("widthSplitTechFtoF", "cross_tech_4to5", "split")
    width("widthMaciTechFtoF", "cross_tech_4to5", "mondrian_aci")
    N["ratioWidthApp"] = fx(v("locked.derived.width_ratio_split_over_mondrian_aci_cross_app_5g"), 1)
    N["tracecovSplitTechFtoF"] = fx(v(locked_key("cross_tech_4to5", "split", "trace_coverage")), 3)

    for m, tag in (("aci", "Aci"), ("mondrian_aci", "Maci")):
        covs = [v(locked_key(r, m, "marginal_coverage")) for r in REGIMES]
        N[f"cov{tag}Min"], N[f"cov{tag}Max"] = fx(min(covs), 3), fx(max(covs), 3)
        infs = [v(main_key(r, m, "frac_inf_instance")) for r in REGIMES]
        N[f"inf{tag}Min"], N[f"inf{tag}Max"] = pct_auto(min(infs)), pct_auto(max(infs))
        fins = [v(main_key(r, m, "finite_coverage")) for r in REGIMES]
        N[f"finCov{tag}Min"] = fx(min(fins), 3)
        d10 = [v(f"delay.main.{m}.D10.{r}.finite_coverage") for r in REGIMES]
        N[f"finCov{tag}DtenMin"] = fx(min(d10), 3)
        N[f"inf{tag}MobFour"] = pct(v(main_key("cross_mobility_4g", m, "frac_inf_instance")))
        N[f"inf{tag}TechFtoFour"] = pct(v(main_key("cross_tech_5to4", m, "frac_inf_instance")))
    N["covDiffAciMaciMax"] = fx(max(abs(v(locked_key(r, "mondrian_aci", "marginal_coverage"))
                                        - v(locked_key(r, "aci", "marginal_coverage")))
                                    for r in REGIMES), 4)

    conf = [f"locked.test.confirmatory.{r}.mondrian_aci.split" for r in SHIFTED]
    N["pConfMax"] = fsci(max(v(c + ".p_bh") for c in conf)).strip("$")
    rbs = [round(v(c + ".rank_biserial"), 2) for c in conf]
    N["rbConfMin"], N["rbConfMax"] = fx(min(rbs), 2), fx(max(rbs), 2)
    meds = [v(f"delay_bound.main.{m}.D0.{r}.bound_applied_median") for m in ONLINE for r in REGIMES]
    N["boundMedMin"], N["boundMedMax"] = fx(min(meds), 3), fx(max(meds), 3)

    g = lambda r, m, grp: v(f"main_group.{r}.{m}.{grp}.coverage")  # noqa: E731
    N["covGrpSplitTechFtoFzero"] = fx(g("cross_tech_4to5", "split", "speed_0"), 3)
    N["covGrpSplitTechFtoFlow"] = fx(g("cross_tech_4to5", "split", "speed_low"), 3)
    N["covGrpSplitTechFtoFhigh"] = fx(g("cross_tech_4to5", "split", "speed_high"), 3)
    for m, tag in (("aci", "Aci"), ("mondrian_aci", "Maci")):
        gs = [g("cross_tech_4to5", m, grp) for grp in GROUPS]
        N[f"covGrp{tag}TechFtoFMin"], N[f"covGrp{tag}TechFtoFMax"] = fx(min(gs), 3), fx(max(gs), 3)
    N["covGrpMaciMobFivezero"] = fx(g("cross_mobility_5g", "mondrian_aci", "speed_0"), 3)
    N["covGrpSplitMobFivezero"] = fx(g("cross_mobility_5g", "split", "speed_0"), 3)
    N["worstGrpAciMin"] = floor_nd(min(v(main_key(r, "aci", "worst_group_coverage")) for r in REGIMES), 4)
    N["worstGrpMaciMin"] = fx(min(v(main_key(r, "mondrian_aci", "worst_group_coverage"))
                                  for r in REGIMES), 3)

    N["secAciRedMin"] = fsci(v("locked.derived.secondary_aci_significant_reduction_min"))
    N["secAciRedMax"] = fdiff(v("locked.derived.secondary_aci_significant_reduction_max"))
    N["pSecAciTechFtoF"] = fsci(v("locked.test.secondary.cross_tech_4to5.mondrian_aci.aci.p_bh"))

    pools = {k: P[k] for k in P}
    N["poolMobFourMedMin"] = fint(pools["cross_mobility_4g.speed_med"]["n_pool_min"])
    N["poolMobFourMedMax"] = fint(pools["cross_mobility_4g.speed_med"]["n_pool_max"])
    med = pools["cross_mobility_4g.speed_med"]["n_pool_median"]
    N["poolMobFourMedMed"] = fx(med, 1) if med != int(med) else fint(med)
    N["poolMobFourMedFallback"] = fint(pools["cross_mobility_4g.speed_med"]["n_seeds_fallback"])
    N["poolMobFourHighFallback"] = fint(pools["cross_mobility_4g.speed_high"]["n_seeds_fallback"])
    idt = [P[f"{r}.{grp}"] for r in ("in_domain_4g", "in_domain_5g", "cross_tech_4to5",
                                    "cross_tech_5to4") for grp in GROUPS]
    if any(p["n_seeds_fallback"] for p in idt):
        raise ValueError("a speed group uses the pooled fallback in-domain or under technology shift")
    N["poolMinIDTech"] = fint(min(p["n_pool_min"] for p in idt))

    dcov, ddev, ratio, nge1 = [], [], [], 0
    for m in ONLINE:
        for d in (0, 1, 2, 5, 10):
            for r in REGIMES:
                dcov.append(v(f"delay.main.{m}.D{d}.{r}.marginal_coverage"))
                ddev.append(v(f"delay.main.{m}.D{d}.{r}.abs_dev"))
                ratio.append(v(f"delay_bound.main.{m}.D{d}.{r}.max_ratio_applied_upper"))
                nge1 += int(v(f"delay_bound.main.{m}.D{d}.{r}.n_alpha_ge1"))
                if v(f"delay_bound.main.{m}.D{d}.{r}.n_alpha_ge1_covered") != 0:
                    raise ValueError("an interval issued with alpha >= 1 covered its target")
        for r in ("in_domain_5g", "cross_tech_4to5"):
            ratio.append(v(f"delay_bound.h5.{m}.D0.{r}.max_ratio_applied_upper"))
            nge1 += int(v(f"delay_bound.h5.{m}.D0.{r}.n_alpha_ge1"))
    N["covDelayMin"], N["covDelayMax"] = fx(min(dcov), 3), fx(max(dcov), 3)
    N["covDelayMinFour"], N["covDelayMaxFour"] = floor_nd(min(dcov), 4), ceil_nd(max(dcov), 4)
    N["absDevDelayMax"] = ceil_nd(max(ddev), 4)
    N["maxBoundRatio"] = ceil_nd(max(ratio), 3)
    N["nAlphaGeOne"] = fint(nge1)
    N["nDelayRows"] = fint(v("check.delay_bound_rows_checked"))
    if v("check.delay_bound_rows_violated_applied") != 0 or v("check.delay_bound_rows_violated_all_form") != 0:
        raise ValueError("Proposition 1 violated in some trace-seed row")
    for m, tag in (("aci", "Aci"), ("mondrian_aci", "Maci")):
        for r, rt in (("cross_tech_4to5", "TechFtoF"), ("cross_mobility_4g", "MobFour")):
            N[f"inf{tag}{rt}Dzero"] = pct(v(f"delay.main.{m}.D0.{r}.frac_inf_instance"))
            N[f"inf{tag}{rt}Dten"] = pct(v(f"delay.main.{m}.D10.{r}.frac_inf_instance"))
    h = "h5.cross_tech_4to5.aci"
    N["covAciHfiveDelTechFtoF"] = fx(v(f"{h}.delayed_D0.marginal_coverage"), 3)
    N["covAciHfiveImmTechFtoF"] = fx(v(f"{h}.immediate.marginal_coverage"), 3)
    N["infAciHfiveDelTechFtoF"] = pct(v(f"{h}.delayed_D0.frac_inf_instance"))
    N["infAciHfiveImmTechFtoF"] = pct(v(f"{h}.immediate.frac_inf_instance"))
    N["finCovAciHfiveDelTechFtoF"] = fx(v(f"{h}.delayed_D0.finite_coverage"), 3)
    N["finCovAciHfiveImmTechFtoF"] = fx(v(f"{h}.immediate.finite_coverage"), 3)

    N["overshootMaciMin"] = fx(v("locked.derived.mondrian_aci_overshoot_lb_min"), 3)
    N["overshootMaciMax"] = fx(v("locked.derived.mondrian_aci_overshoot_lb_max"), 3)
    N["overshootPointMinPct"] = pct(v("locked.derived.overshoot_point_min_shifted"), 0)
    N["overshootPointMaxPct"] = pct(v("locked.derived.overshoot_point_max_shifted"), 0)
    ds = "locked.downstream"
    N["utilSplitApp"] = fx(v(f"{ds}.cross_app_5g.split.util_lb.mean"), 4)
    N["utilMaciApp"] = fx(v(f"{ds}.cross_app_5g.mondrian_aci.util_lb.mean"), 4)
    N["ratioUtilApp"] = fx(v("locked.derived.util_lb_ratio_mondrian_aci_over_split_cross_app_5g"), 2)
    for r, rt in (("cross_tech_4to5", "TechFtoF"), ("cross_mobility_5g", "MobFive")):
        N[f"overshootSplit{rt}"] = fx(v(f"{ds}.{r}.split.overshoot_lb.mean"), 3)
        N[f"overshootMaci{rt}"] = fx(v(f"{ds}.{r}.mondrian_aci.overshoot_lb.mean"), 3)

    hk = lambda r, var, met: v(f"has.{r}.{var}.{met}.mean")  # noqa: E731
    for var, tag in (("mpc_aci", "MpcAci"), ("mpc_hm", "MpcHm"), ("mpc_point", "MpcPoint"),
                     ("mpc_robust", "MpcRobust")):
        N[f"stalls{tag}MobFour"] = fx(hk("cross_mobility_4g", var, "stalls"), 2)
        N[f"br{tag}MobFour"] = fx(hk("cross_mobility_4g", var, "avg_bitrate_mbps"), 2)
    N["qoeMpcSplitMobFour"] = fx(hk("cross_mobility_4g", "mpc_split", "qoe_per_segment"), 2)
    N["qoeMpcRobustMobFour"] = fx(hk("cross_mobility_4g", "mpc_robust", "qoe_per_segment"), 2)
    rs = [f"has_test.{r}.mpc_mondrian_aci.{b}" for r in HAS_SHIFTED for b in ("mpc_split", "mpc_robust")]
    N["pHasMaxRobSplit"] = fsci(max(v(k + ".p_bh") for k in rs)).strip("$")
    rbs = [round(v(k + ".rank_biserial"), 2) for k in rs]
    N["rbHasMin"], N["rbHasMaxRobSplit"] = fx(min(rbs), 2), fx(max(rbs), 2)
    ra = [f"has_test.{r}.mpc_mondrian_aci.mpc_aci" for r in HAS_SHIFTED]
    mds = [v(k + ".median_diff") for k in ra]
    N["mdHasAciMin"], N["mdHasAciMax"] = fx(min(mds), 2), fx(max(mds), 2)
    N["pHasAciMobFive"] = fsci(v("has_test.cross_mobility_5g.mpc_mondrian_aci.mpc_aci.p_bh"))

    N["ablMaciMaxDevPts"] = ceil_nd(100 * v("locked.derived.ablation_mondrian_aci_max_abs_marginal_minus_target"), 1)
    split45 = [v(k) for k in S if k.startswith("locked.ablation.") and ".cross_tech_4to5." in k
               and k.endswith(".split.marginal_coverage") and ".alpha0.1." in k]
    N["ablSplitFtoFMin"], N["ablSplitFtoFMax"] = fx(min(split45), 3), fx(max(split45), 3)
    N["ablAlphaInfMax"] = pct(v("locked.derived.ablation_alpha05_mondrian_aci_frac_inf_rows_max"))
    N["ablAlphaFinCovMobFour"] = fx(v("locked.ablation.abl_alpha05.cross_mobility_4g.gbm.alpha0.05."
                                      "gamma0.05.mondrian_aci.finite_coverage"), 3)
    diffs = {(r, m): abs(v(f"ts.{r}.{m}.ts_marginal_coverage") - v(f"ts.{r}.{m}.row_marginal_coverage"))
             for r in REGIMES for m in METHODS}
    N["tsOnlineMaxDiff"] = ceil_nd(max(d for (r, m), d in diffs.items() if m in ONLINE), 4)
    (rw, mw), dw = max(((k, d) for k, d in diffs.items() if k[1] not in ONLINE), key=lambda t: t[1])
    N["tsOfflineMaxDiff"] = fx(dw, 3)
    N["tsOfflineMaxWhere"] = f"{METHOD_TEX[mw]} under {REGIME_TEXT[rw]}"
    rt = "runtime.timing_log.full"
    N["runMainMin"] = fx(v(f"{rt}.main.wall_minutes"), 1)
    N["runHfiveTsMin"] = fx(v(f"{rt}.h5.wall_minutes") + v(f"{rt}.ts.wall_minutes"), 1)
    N["runHasMin"] = fx(v("runtime.timeline.batch3_ladder_and_HAS.wall_minutes"), 1)
    return N


def tab_main() -> str:
    rows = []
    for m in METHODS:
        cells = [f"{fx(v(locked_key(r, m, 'marginal_coverage')), 3)} "
                 f"({fx(v(locked_key(r, m, 'median_width')), 1)})" for r in REGIMES]
        rows.append(f"{METHOD_TEX[m]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(REGIME_TEX[r] for r in REGIMES)
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Mean marginal test coverage over 20 seeds at nominal level 0.90, with the "
        "median width in Mbit/s in parentheses (median across traces of the seed-averaged "
        "per-trace median width, infinite widths included). CIs are shown in "
        "Fig.~\\ref{fig:marginal}. QR, quantile regression; CP, conformal prediction; "
        "CQR, conformalized QR; ACI, adaptive conformal inference; CIs, confidence intervals; "
        "Tech and App, technology and application shift.}",
        "\\label{tab:main}", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccccccc}", "\\toprule",
        f"Method & {head} \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
        "\\end{table*}"])


def tab_tests() -> str:
    lines = []
    for fam, comps in (("confirmatory", ["split"]), ("secondary", ["aci", "mondrian", "cqr"])):
        lines.append("\\multicolumn{6}{l}{\\emph{%s family}} \\\\" % fam.capitalize())
        for r in SHIFTED:
            for c in comps:
                k = f"locked.test.{fam}.{r}.mondrian_aci.{c}"
                lines.append(f"{REGIME_TEX[r]} & {METHOD_TEX[c]} & {fint(v(k + '.n'))} & "
                             f"{fdiff(v(k + '.median_diff'))} & {fx(v(k + '.rank_biserial'), 2)} & "
                             f"{fsci(v(k + '.p_bh'))} \\\\")
        if fam == "confirmatory":
            lines.append("\\midrule")
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Paired two-sided Wilcoxon signed-rank tests on the absolute trace coverage "
        "deviation $\\Delta_s$ of \\eqref{eq:tracemetrics}, with traces paired, comparing "
        "Mondrian-ACI with each listed method. The five split CP comparisons form the "
        "pre-specified confirmatory family. $n$ is the number of paired traces, med.\\ diff.\\ the "
        "median of the Mondrian-ACI minus comparator differences of $\\Delta_s$ (negative favors "
        "Mondrian-ACI), $r_{rb}$ the matched-pairs rank-biserial correlation, and "
        "$p_{\\mathrm{BH}}$ the Benjamini--Hochberg adjusted $p$-value within each family. Abbreviations as in Table~\\ref{tab:main}.}",
        "\\label{tab:tests}", "\\centering", "\\footnotesize",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Regime & vs. & $n$ & med.\\ diff. & $r_{rb}$ & $p_{\\mathrm{BH}}$ \\\\", "\\midrule",
        *lines, "\\bottomrule", "\\end{tabular}", "\\end{table*}"])


def tab_inform() -> str:
    lines = []
    for i, r in enumerate(REGIMES):
        if i:
            lines.append("\\midrule")
        for j, m in enumerate(["split", "aci", "mondrian_aci"]):
            wg = v(main_key(r, m, "worst_group_coverage"))
            grp = S[main_key(r, m, "worst_group")]
            lines.append(
                f"{REGIME_TEX[r] if j == 0 else ''} & {METHOD_TEX[m]} & "
                f"{pct(v(main_key(r, m, 'frac_inf_instance')), 2)} & "
                f"{fx(v(main_key(r, m, 'finite_coverage')), 3)} & "
                f"{fx(v(main_key(r, m, 'width_finite_median')), 2)} & "
                f"{fx(v(main_key(r, m, 'winkler_median')), 1)} & "
                f"{fx(wg, 3)} {GROUP_TEX[grp]} \\\\")
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Informativeness of split CP, ACI and Mondrian-ACI at nominal level 0.90 "
        "(20 seeds, $h=1$, immediate feedback). Inf.\\ is the instance-level percentage of "
        "infinite intervals. Fin.\\ cov.\\ is the coverage of the finite intervals, pooled within "
        "each seed and averaged over seeds. Fin.\\ width is the median across traces of the "
        "seed-averaged per-trace median finite width, and Winkler the median across traces of "
        "the seed-averaged per-trace mean Winkler score \\eqref{eq:winkler} over finite "
        "intervals, both in Mbit/s; lower is better. Worst group is the lowest speed-group "
        "coverage (mean over seeds) and its speed bin in km/h. The Inf., Fin.\\ cov.\\ and "
        "Worst group columns are post hoc descriptive analyses. Abbreviations as in Table~\\ref{tab:main}.}",
        "\\label{tab:inform}", "\\centering", "\\footnotesize",
        "\\begin{tabular}{llrrrrl}", "\\toprule",
        "Regime & Method & Inf.\\ (\\%) & Fin.\\ cov. & Fin.\\ width & Winkler & Worst group \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}", "\\end{table*}"])


def tab_winkler() -> str:
    rows = []
    for m in METHODS:
        cells = [fx(v(main_key(r, m, "winkler_median")), 1) for r in REGIMES]
        rows.append(f"{METHOD_TEX[m]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(REGIME_TEX[r] for r in REGIMES)
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Winkler interval score \\eqref{eq:winkler} in Mbit/s at $\\alpha=0.10$, given as the "
        "median across traces of the seed-averaged per-trace mean score over finite intervals "
        "(20 seeds). Lower is better. Only ACI and Mondrian-ACI issue infinite intervals; their "
        "fractions are given in Table~\\ref{tab:inform}. Intervals whose upper endpoint lies "
        "below the zero-clipped lower endpoint (crossing quantiles or a negative point "
        "prediction) count as empty sets of zero width. Abbreviations as in Table~\\ref{tab:main}.}",
        "\\label{tab:winkler}", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccccccc}", "\\toprule",
        f"Method & {head} \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
        "\\end{table*}"])


def tab_has() -> str:
    lines = []
    for i, r in enumerate(HAS_SHIFTED):
        if i:
            lines.append("\\midrule")
        n = fint(v(f"has.{r}.mpc_robust.qoe_per_segment.n_traces"))
        for j, var in enumerate(VARIANTS):
            hk = lambda met: v(f"has.{r}.{var}.{met}.mean")  # noqa: E731
            label = f"{REGIME_TEX[r]} ({n})" if j == 0 else ""
            lines.append(
                f"{label} & {VARIANT_TEX[var]} & {fx(hk('avg_bitrate_mbps'), 2)} & "
                f"{pct(hk('rebuffer_ratio'), 2)} & {fx(hk('stalls'), 2)} & "
                f"{fx(hk('switches_per_min'), 2)} & {fx(hk('qoe_per_segment'), 2)} \\\\")
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Trace-driven adaptive-streaming simulation in the four shifted regimes with "
        "capacity traces (number of traces in parentheses), 20 seeds. Means across traces of "
        "per-trace values averaged over seeds: average bitrate in Mbit/s, rebuffering ratio in "
        "percent of playback plus rebuffering time, number of stalls after startup, quality "
        "switches per minute of playback, and the QoE \\eqref{eq:qoe} per segment (higher is "
        "better). Rate denotes the rate-based rule; bounds are interval lower bounds at "
        "$\\alpha=0.10$. Startup delay is identical across variants because the first segment "
        "is always fetched at the lowest representation. MPC, model predictive control; other "
        "abbreviations as in Table~\\ref{tab:main}.}",
        "\\label{tab:has}", "\\centering", "\\footnotesize",
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Regime (traces) & Player and estimate & Bitrate & Rebuf.\\ (\\%) & Stalls & "
        "Switches/min & QoE \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}",
        "\\end{table*}"])


def tab_hastests() -> str:
    lines = []
    for r in HAS_SHIFTED:
        for b in ("mpc_aci", "mpc_split", "mpc_robust"):
            k = f"has_test.{r}.mpc_mondrian_aci.{b}"
            lines.append(f"{REGIME_TEX[r]} & {VARIANT_TEX[b]} & {fint(v(k + '.n'))} & "
                         f"{fx(v(k + '.mean_a'), 2)} & {fx(v(k + '.mean_b'), 2)} & "
                         f"{fx(v(k + '.median_diff'), 2)} & {fx(v(k + '.rank_biserial'), 2)} & "
                         f"{fsci(v(k + '.p_bh'))} \\\\")
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Pre-specified paired two-sided Wilcoxon signed-rank tests on per-trace QoE "
        "\\eqref{eq:qoe}, comparing MPC with the Mondrian-ACI bound (a) with each listed "
        "comparator (b). Means are across traces; med.\\ diff.\\ is the median of a minus b "
        "(negative favors b), $r_{rb}$ the matched-pairs rank-biserial correlation, and "
        "$p_{\\mathrm{BH}}$ the Benjamini--Hochberg adjusted $p$-value over the 12 tests. MPC, model predictive control; "
        "other abbreviations as in Table~\\ref{tab:main}.}",
        "\\label{tab:hastests}", "\\centering", "\\footnotesize",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Regime & Comparator (b) & $n$ & QoE a & QoE b & med.\\ diff. & $r_{rb}$ & "
        "$p_{\\mathrm{BH}}$ \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}",
        "\\end{table*}"])


def tab_ablations() -> str:
    base_rows = [("cross_mobility_5g", "persistence"), ("cross_mobility_5g", "ridge"),
                 ("cross_tech_4to5", "persistence"), ("cross_tech_4to5", "ridge"),
                 ("in_domain_4g", "persistence"), ("in_domain_4g", "ridge")]
    alpha_regimes = ["cross_app_5g", "cross_mobility_4g", "cross_mobility_5g", "cross_tech_4to5",
                     "cross_tech_5to4", "in_domain_4g", "in_domain_5g"]
    gamma_regimes = ["cross_app_5g", "cross_mobility_4g", "cross_mobility_5g", "cross_tech_4to5",
                     "cross_tech_5to4"]

    def row(label, regime, setting, method, key):
        cov = v(key + ".marginal_coverage")
        fin = v(key + ".finite_coverage")
        inf = v(key + ".frac_inf_trace_rows")
        return (f"{label} & {REGIME_TEX[regime]} & {setting} & {METHOD_TEX[method]} & "
                f"{fx(cov, 3)} & {fx(fin, 3)} & {fx(inf, 3)} \\\\")

    lines = []
    for r, b in base_rows:
        for m in ("mondrian_aci", "split"):
            lines.append(row("Base model", r, b, m,
                             f"locked.ablation.abl_base.{r}.{b}.alpha0.1.gamma0.05.{m}"))
    for r in alpha_regimes:
        for m in ("mondrian_aci", "split"):
            lines.append(row("$\\alpha=0.05$", r, "$\\alpha=0.05$", m,
                             f"locked.ablation.abl_alpha05.{r}.gbm.alpha0.05.gamma0.05.{m}"))
    for r in gamma_regimes:
        for gm in ("0.01", "0.02"):
            for m in ("mondrian_aci", "split"):
                lines.append(row("$\\gamma$", r, f"$\\gamma={gm}$", m,
                                 f"locked.ablation.abl_gamma.{r}.gbm.alpha0.1.gamma{gm}.{m}"))
    for r in ("cross_tech_4to5", "in_domain_5g"):
        for m in ("mondrian_aci", "split"):
            lines.append(row("Horizon 5\\,s", r, "immediate", m,
                             f"locked.ablation.abl_h5.{r}.gbm.alpha0.1.gamma0.05.{m}"))
        k = f"h5.{r}.mondrian_aci.delayed_D0"
        lines.append(f"Horizon 5\\,s & {REGIME_TEX[r]} & label delay & Mondrian-ACI & "
                     f"{fx(v(k + '.marginal_coverage'), 3)} & {fx(v(k + '.finite_coverage'), 3)} & "
                     f"{fx(v(k + '.frac_inf_trace_rows'), 3)} \\\\")
    return "\n".join([
        "\\begin{table*}[!t]",
        "\\caption{Ablation and sensitivity results with seeds 0 to 4: marginal coverage (Cov.), "
        "coverage of the finite intervals (Fin.\\ cov.) and mean fraction of infinite intervals "
        "over trace and seed pairs (Inf.\\ frac.). The target is 0.90 except in the "
        "$\\alpha=0.05$ rows (0.95). Split CP does not depend on $\\gamma$; differences from "
        "Table~\\ref{tab:main} reflect the smaller seed set. The 5-s rows marked immediate apply "
        "each label before the next interval is issued; the label-delay rows apply each label "
        "when it is observed, five rows later. Abbreviations as in Table~\\ref{tab:main}.}",
        "\\label{tab:ablations}", "\\centering", "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llllrrr}", "\\toprule",
        "Ablation & Regime & Setting & Method & Cov. & Fin.\\ cov. & Inf.\\ frac. \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}", "\\end{table*}"])


def main() -> None:
    global S, P
    S = json.loads((SUM / "summary.json").read_text(encoding="utf-8"))
    P = json.loads((SUM / "pool_sizes.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    nums = numbers()
    text = "".join(f"\\newcommand{{\\{k}}}{{{val}}}\n" for k, val in nums.items())
    (OUT / "numbers.tex").write_text(text, encoding="utf-8", newline="\n")
    tables = {"tab_main": tab_main(), "tab_tests": tab_tests(), "tab_inform": tab_inform(),
              "tab_winkler": tab_winkler(), "tab_has": tab_has(), "tab_hastests": tab_hastests(),
              "tab_ablations": tab_ablations()}
    for name, body in tables.items():
        (OUT / f"{name}.tex").write_text(body + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(nums)} numbers and {len(tables)} tables to {OUT}")


if __name__ == "__main__":
    main()
