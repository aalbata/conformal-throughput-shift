# Pre-registered Protocol — Distribution-Free Prediction Intervals for Cellular Downlink Throughput under Distribution Shift

**Version:** 1.0 (frozen). **Date:** 2026-08-28.
**Frozen before:** any locked (confirmatory) experiment is executed or examined.
**Pilot status at freeze:** exploratory pilots on pilot seeds {100, 101, 102} were run to
verify feasibility, metric behaviour, and to select the ACI step size γ by the
pre-specified rule below. No locked seed (0–19) has been run.

## 1. Research question and hypotheses

**RQ:** Do distribution-free prediction intervals for short-horizon cellular downlink
throughput remain valid under mobility-, technology-, and application-induced
distribution shift on public traces, and can group-conditional (Mondrian) calibration
combined with online adaptive conformal updating restore validity at acceptable
efficiency cost?

Falsifiable hypotheses (α_stat = 0.05, BH-corrected within the confirmatory family):

- **H1 (in-domain validity).** In the in-domain regimes, split conformal and CQR achieve
  marginal test coverage within ±0.02 of the nominal 0.90. (Evaluated descriptively with
  95% bootstrap CIs over seeds; no significance test.)
- **H2 (degradation under shift).** In at least one shifted regime, the marginal coverage
  of split conformal falls below 0.85 (i.e., ≥5 points below nominal), with the 95%
  bootstrap CI over seeds excluding 0.90.
- **H3 (repair).** In the shifted regimes, Mondrian-ACI attains smaller per-session
  absolute coverage deviation |coverage − 0.90| than split conformal
  (paired one-comparison-per-regime Wilcoxon signed-rank on seed-averaged session values;
  confirmatory family defined in §7).
- **H4 (efficiency cost).** Restoring coverage increases interval width; the width ratio
  Mondrian-ACI / split is reported descriptively with 95% bootstrap CIs (no test).
- **H5 (conditional validity).** Within shifted regimes, per-speed-group coverage of
  Mondrian-ACI is closer to nominal than split conformal's for the worst-covered group
  (descriptive with CIs; group coverages reported for all groups with ≥500 test samples).
- **H6 (downstream risk control).** Selecting a transmission/streaming rate equal to the
  interval lower bound yields session-mean overshoot probability ≤ α + 0.02 for
  Mondrian-ACI in shifted regimes, versus ≈0.5 for point-prediction rate selection
  (descriptive with CIs).

Negative or null outcomes for any hypothesis are reported as such.

## 2. Data, unit of analysis, and replication unit

- **Datasets (public, downloaded 2026-08-28, SHA-256 in `data/raw/download_manifest.json`):**
  UCC MISL 4G LTE dataset (Raca et al., MMSys 2018; 135 sessions; mobility folders
  bus/car/pedestrian/static/train; operators A/B) and UCC MISL 5G dataset (Raca et al.,
  MMSys 2020; 83 sessions; Static/Driving × {Netflix, Amazon_Prime, Download}).
- **Prediction instance:** time t within a session with device state 'D' (downloading) at
  t and ≥10 s of within-session history; target y = downlink throughput (Mbit/s) at t+1 s.
- **Unit of analysis for intervals:** the prediction instance.
- **Independent replication unit for all confirmatory statistics: the session** (trace).
  Packets/samples within a session are autocorrelated and are never treated as
  independent.
- **Session inclusion:** sessions with ≥60 samples; sessions with ≥1 valid instance enter
  training pools; sessions with <30 test instances are excluded from session-level
  metric rows (rule fixed before locked runs).

## 3. Features and preprocessing (fixed)

Causal features only (no future information): last throughput, rolling mean/std/min/max
over trailing 10 s, first difference, RSRP, RSRQ, SNR, CQI, RSSI, speed, 5G-mode flag.
'-' parsed as missing; median imputation fitted on training split only. DL_bitrate
converted kbit/s → Mbit/s. Leakage is guarded by unit test `test_no_future_leakage`.

## 4. Regimes (source → target), fixed

| Regime | Train+Cal source | Test target |
|---|---|---|
| in_domain_4g | 4G random 60/20/20 session split | held-out 4G sessions |
| in_domain_5g | 5G random 60/20/20 session split | held-out 5G sessions |
| cross_mobility_4g | 4G static+pedestrian (80/20 train/cal) | 4G car+bus+train |
| cross_mobility_5g | 5G static (80/20) | 5G driving |
| cross_tech_4to5 | all 4G (80/20) | all 5G |
| cross_tech_5to4 | all 5G (80/20) | all 4G |
| cross_app_5g | 5G Download (80/20) | 5G Netflix+Amazon_Prime |

"Shifted regimes" = the last five rows.

## 5. Methods under comparison (fixed)

Base point predictor: `HistGradientBoostingRegressor` (300 iters, lr 0.1, per-seed
random_state). Quantile heads: HistGBM pinball at α/2 and 1−α/2.
Interval methods: (1) Gaussian (train-residual σ, non-conformal baseline);
(2) raw quantile regression (non-conformal); (3) split conformal; (4) CQR;
(5) Mondrian split conformal; (6) ACI (per test session, canonical Gibbs–Candès
unclipped α_t); (7) **Mondrian-ACI (proposed)**: group-specific score pools + per-session
online α_t adaptation. Groups: runtime-observable speed bins [0,1), [1,15), [15,50),
[50,∞) km/h; groups with <50 calibration scores fall back to the pooled pool.
Nominal miscoverage α = 0.10 (primary).

## 6. Seeds, splits, and pilot/locked separation

- **Locked seeds: 0–19** (20 replicates). Session-level splits and model seeds are both
  driven by the seed. Pilot seeds 100–102 are never reported as confirmatory.
- **γ (ACI step size): selected on pilot data only**, from {0.01, 0.02, 0.05}, minimising
  mean |session coverage − 0.90| for the `aci` method averaged over the four pilot
  regimes; ties broken toward the smaller γ. **Selected value: γ = 0.05** (pilot rule
  table: 0.01 → 0.01791, 0.02 → 0.01044, 0.05 → 0.00480 mean |session coverage − 0.90|;
  see `results/pilot/pilot_rows.csv.gz`; recorded 2026-08-28 before any locked run).
- Checkpoint/resume per (regime, base_model, seed) cell; failed cells are recorded in
  `audit/failed_runs.md` and rerun only for infrastructure failures (never because of
  their results).

## 7. Confirmatory statistical analysis (fixed)

- For each (regime, method, session): average each session metric over the seeds in
  which that session appeared in the test split. Sessions are then the paired unit.
- **Confirmatory family (BH-corrected together, 5 tests):** for each shifted regime,
  paired two-sided Wilcoxon signed-rank of per-session |coverage − 0.90|:
  mondrian_aci vs split.
- **Secondary comparisons (BH-corrected as a second family, 15 tests):** mondrian_aci vs
  {aci, mondrian, cqr} in each shifted regime.
- Effect size: matched-pairs rank-biserial correlation. 95% percentile bootstrap CIs
  (10,000 resamples over sessions, seed 0) for all reported means/medians.
- Coverage-vs-nominal claims (H1, H2) use marginal coverage with bootstrap-over-seeds
  CIs; per-session coverage distributions are reported alongside.
- All failed/missing runs, and any session excluded by §2, are reported.

## 8. Ablations and sensitivity (locked, secondary; 5 seeds 0–4 each)

a. Base model: ridge and persistence instead of GBM (regimes: in_domain_4g,
   cross_mobility_5g, cross_tech_4to5).
b. Nominal level α = 0.05 (all regimes).
c. γ ∈ {0.01, 0.05} for the ACI methods (shifted regimes) — robustness to the
   pilot-selected γ.
d. Horizon h = 5 s (regimes: in_domain_5g, cross_tech_4to5).

## 9. Compute budget and stopping rules

Estimated locked cost: ≈180 cells × ≈30–60 s ≈ 1.5–3 h CPU sequential, run with ≤6
parallel workers (OMP_NUM_THREADS=1 per worker), peak RAM ≪ 10 GB, disk ≪ 2 GB.
No run continues past 2 h wall-clock without owner approval. No additional seeds,
regimes, metrics, or methods after locked runs begin; any unavoidable change requires a
dated amendment in `protocol/amendments.md` stating whether results had been viewed.

## 10. Reporting commitments

All locked outputs are preserved unmodified in `results/locked/`. The manuscript reports
every regime and every method, including unfavourable, null, and negative results;
prohibited claims are listed in `audit/topic_selection.md`. Every manuscript number is
generated programmatically from locked result files.
