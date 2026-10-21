# Protocol amendments

The original protocol (`preregistered_protocol.md`, `protocol.json`, hashes in
`protocol_hashes.json`) is unchanged. Terminology: a "trace" in this amendment is one
drive-test recording, called a "session" in the original protocol.

## Amendment 1 (2026-10-21)

### 1.1 Outcomes examined before this amendment

The following quantities were computed from the existing outputs in `results/locked/`
and examined before this amendment was written. They are reported as post hoc
descriptive analyses:
finite-interval coverage derived from stored coverage and infinite-interval fractions;
finite-interval mean width; instance-level and per-trace infinite-interval fractions;
speed-group coverage in every regime; the spread of per-trace coverage; mean width with
infinite widths replaced by a cap; the stored per-trace median interval scores; and the
share of zero-throughput targets by regime.

No outcome of the analyses specified in 1.4 to 1.8 (feedback delay, corrected 5-s
horizon, timestamp sensitivity, positive-throughput sensitivity, adaptive-streaming
simulation, mean Winkler score, finite-interval median width) had been computed when
this amendment was written.

### 1.2 Deviations from the original protocol

1. The step-size sensitivity of Section 8c was run with gamma in {0.01, 0.02}, not the
   specified {0.01, 0.05}; gamma = 0.05 is the primary value. The grid as run is reported.
2. The width-ratio confidence intervals of H4 and the group-coverage confidence intervals
   of H5 were not produced in the original analysis. The 500-instance group filter of H5
   excludes no group (the smallest speed group has 788 test instances in every seed).
3. The Winkler interval score was not part of the original protocol. It is added in 1.5.
4. Targets and the trailing window are defined on the row index of each trace (nominal
   1 Hz). For h = 1, 85.8% of targets lie exactly 1 s after the prediction time. The
   confirmatory analysis is unchanged; 1.7 adds a timestamp-based sensitivity analysis.
5. Seeds determine the trace-level splits only. The gradient-boosting models are
   deterministic for a given training set because early stopping is disabled.

### 1.3 Instance-level re-execution of the main and 5-s cells

The 140 main cells (7 regimes, seeds 0-19, gradient boosting, alpha = 0.10,
gamma = 0.05, h = 1) and the 10 cells of the 5-s horizon ablation (in_domain_5g and
cross_tech_4to5, seeds 0-4) are re-executed with code that also records the quantities
specified below. Acceptance criterion: for every (cell, method, trace) the trace length
n equals the locked value and coverage and infinite-interval fraction agree with the
locked values to within 1e-12. If the criterion fails, execution stops and no new
outcome is examined until the cause is identified and recorded here.

### 1.4 Feedback delay

Label-arrival model. For the instance issued at row t of a trace, with timestamp tau(t)
and horizon h, the label is the throughput at row t + h. It becomes available at time
tau(t + h) + D, where D >= 0 is an additional reporting delay in seconds. Before issuing
the interval at time tau(j), ACI and Mondrian-ACI apply the update
alpha <- alpha + gamma (alpha_nominal - err_i), in order of arrival (ties in issue
order), for every earlier instance i of the same trace whose label has arrived
(tau(t_i + h) + D <= tau(j)) and has not yet been applied. Labels that arrive after the
last instance of a trace are not applied. With h = 1 and D = 0 this reproduces the
sequencing of the locked runs, which 1.3 verifies.

Grid. D in {0, 1, 2, 5, 10} s at h = 1, all 7 regimes, seeds 0-19, alpha = 0.10,
gamma = 0.05, for ACI and Mondrian-ACI (the other methods do not use feedback).
Corrected 5-s horizon: the 10 cells of 1.3 at h = 5 with D = 0 under this model, so
labels arrive 5 rows after issue; reported beside the immediate-feedback results.

Recorded per trace: coverage, infinite-interval fraction, the quantities of 1.5, the
absolute coverage deviation, the maximum number of issued predictions whose labels were
outstanding at any issue time, and the numbers of issue times with alpha_t <= 0 and with
alpha_t >= 1, and the number of intervals issued with alpha_t >= 1 that covered their
target. The deterministic long-run bound evaluated at the recorded maximum is reported
per trace. Analysis is descriptive: means with 95% percentile bootstrap confidence
intervals over traces (10,000 resamples, bootstrap seed 0).

### 1.5 Finite intervals and the Winkler score

For every method and regime of the main cells, and of the cells in 1.4:
- infinite-interval fraction, at instance level (pooled within a seed, then averaged over
  seeds) and as the mean of per-trace fractions;
- finite-interval coverage, marginal (pooled over finite intervals within a seed, then
  averaged over seeds) and as the mean of seed-averaged per-trace values;
- finite-interval width: the median across traces of the seed-averaged per-trace median
  finite width, and the same with the per-trace mean finite width;
- Winkler score, Eq. (7) of the manuscript, computed per instance for finite intervals;
  per-trace mean over finite intervals, averaged over seeds, reported as the median across
  traces (primary) and the mean across traces. No cap is applied to infinite intervals;
  their fraction is reported alongside;
- coverage in every speed group of every regime, with 95% percentile bootstrap
  confidence intervals over the 20 seed-level values, and the worst-group coverage.
For the ablation rows of the original Section 8, finite-interval coverage is computed
exactly from the stored per-trace coverage and infinite-interval fractions.

### 1.6 Zero-throughput targets

The share of targets equal to 0 Mbit/s is reported for every regime, and marginal
coverage restricted to instances with a positive target is reported for every method
and regime of the main cells.

### 1.7 Timestamp sensitivity

Features and targets are rebuilt on timestamps: the target of the instance at time
tau(t) is the throughput of the last row whose timestamp equals tau(t) + h (instances
without such a row are dropped); the trailing window contains the rows with timestamps
in (tau(t) - 10 s, tau(t)]; at least 10 s of history is required. All 7 methods,
7 regimes, seeds 0-4, h = 1, D = 0. Reported: marginal coverage and infinite-interval
fraction.

### 1.8 Adaptive-streaming simulation

Traces. Only traces recorded during bulk downloads represent available capacity: all 4G
traces and the 5G Download traces. The 5G Netflix and Amazon Prime traces reflect the
players' own on/off demand and are excluded; the application-shift regime therefore has
no streaming evaluation. Regimes: in_domain_4g, in_domain_5g, cross_mobility_4g,
cross_mobility_5g, cross_tech_4to5, cross_tech_5to4, with the models, calibration sets and
seeds 0-19 of the main cells; the simulated traces are the eligible test traces of each
cell.

Time base. One row of a trace is one second of capacity at that row's downlink
throughput, as for the predictor. Download time follows from integrating capacity over
seconds; each segment request adds 80 ms of latency.

Video. Big Buck Bunny, x264, 4-s segments, with the per-segment sizes of the UCC 4K DASH
dataset; six representations of 237, 752, 1815, 4282, 14817 and 39537 kbit/s (the
available representations nearest to a geometric sequence between the lowest and the
highest). Segment sizes repeat cyclically when a trace outlasts the video.

Player. Buffer capacity 30 s. The first segment is fetched at the lowest representation
and playback starts when it is downloaded. When the buffer is full the player waits until
one segment of space is free. Playback runs from the first instance of the trace to the
end of the trace; an incomplete final download is discarded.

Throughput estimates at a request issued during second s (the most recent instance at or
before s is used): point prediction; lower bound of split CP; lower bound of ACI; lower
bound of Mondrian-ACI (all from the main configuration, h = 1, D = 0); harmonic mean of
the last five segment throughputs; and the RobustMPC estimate, the harmonic mean divided
by one plus the maximum relative error of the last five harmonic-mean estimates. Before
any segment has completed, the lowest representation is used. Estimates below
0.01 Mbit/s, including the zero lower bound of an infinite interval, are set to
0.01 Mbit/s.

Rules. Rate-based: the highest representation whose bitrate does not exceed the
estimate (the lowest if none). MPC: exhaustive search over the next five segments that
maximises the QoE below with the estimate held constant, executing the first decision.
Variants: rate-based with harmonic mean, point prediction, split CP, ACI and
Mondrian-ACI bounds; MPC with harmonic mean, RobustMPC, point prediction, split CP, ACI
and Mondrian-ACI bounds.

QoE (linear form of Yin et al., SIGCOMM 2015): sum of segment bitrates in Mbit/s, minus
the sum of absolute bitrate changes between consecutive segments, minus mu times the
rebuffering time in seconds, minus mu_s times the startup delay, with mu = mu_s = 39.537
(the highest bitrate in Mbit/s), normalised per segment.

Metrics per trace: average bitrate (Mbit/s), rebuffering time (s), rebuffering ratio
(rebuffering time divided by playback plus rebuffering time), number of stalls after
startup, startup delay (s), quality switches per minute of playback, mean absolute
bitrate change per switch (Mbit/s), and QoE per segment.

Statistics. Per-trace metrics are averaged over the seeds in which the trace was tested.
All variants are reported with means and 95% percentile bootstrap confidence intervals
over traces. Pre-specified paired two-sided Wilcoxon signed-rank tests on per-trace QoE,
Benjamini-Hochberg corrected as one family of 12 tests, with matched-pairs rank-biserial
effect sizes: within the MPC family, Mondrian-ACI bound versus ACI bound, versus split CP
bound, and versus RobustMPC, in cross_mobility_4g, cross_mobility_5g, cross_tech_4to5 and
cross_tech_5to4. Results are reported whatever their direction.

Verification. The simulator is checked against hand-computed outcomes on synthetic
traces, and its rate-based rule is compared with an independent reference implementation
(Sabre) on identical synthetic inputs.

### 1.9 Unchanged

No other method, regime, seed, base model, tuning parameter or hypothesis is added or
changed. The original confirmatory family, its tests and its results are unchanged and
reported.
