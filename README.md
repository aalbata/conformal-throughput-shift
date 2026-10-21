# Conformal Prediction Intervals for Cellular Throughput Under Distribution Shift

Code, protocol and result files for the article

"Conformal Prediction Intervals for Cellular Throughput Under Distribution Shift"

by Ali Al Bataineh, Majed Bouchahma, Ashraf Abou Tabl, Munther Gdeisat and Yacouba Moumouni
(Higher Colleges of Technology, Abu Dhabi, United Arab Emirates).
Corresponding author: Ali Al Bataineh, aalbataineh@hct.ac.ae.

The package regenerates the tables, the figures and the number macros of the article from
the shipped per-trace result files (fast path), and the per-trace result files themselves
from the public 4G and 5G datasets (full path).

## What the package reproduces

- The locked confirmatory analysis of the protocol fixed before the locked runs
  (`protocol/preregistered_protocol.md`, `protocol/protocol.json`): seven interval methods in
  seven transfer regimes over 20 seeds, the pre-specified Wilcoxon tests, the ablations and the
  rate-selection proxy.
- The analyses of protocol amendment 1 (`protocol/amendments.md`): finite-interval coverage,
  infinite-interval fractions, widths and the Winkler score; speed-group coverage in every
  regime; zero-throughput targets; reporting delays of 0 to 10 s and the check of the
  deterministic coverage bound of Proposition 1; the 5-s horizon with delayed labels;
  timestamp-based features; and the trace-driven adaptive-streaming simulation with its
  pre-specified tests. The Winkler score cited as Eq. (7) in `protocol/amendments.md` is
  Eq. (12) of the article.

| Article | File |
| --- | --- |
| Table 1 | `results/summary/tex/tab_main.tex` |
| Table 2 | `results/summary/tex/tab_tests.tex` |
| Table 3 | `results/summary/tex/tab_inform.tex` |
| Table 4 | `results/summary/tex/tab_winkler.tex` |
| Table 5 | `results/summary/tex/tab_has.tex` |
| Table 6 | `results/summary/tex/tab_hastests.tex` |
| Table 7 | `results/summary/tex/tab_ablations.tex` |
| Fig. 1 | `figures/fig_marginal_coverage.pdf` |
| Fig. 2 | `figures/fig_group_coverage.pdf` |
| Fig. 3 | `figures/fig_tradeoff.pdf` |
| Fig. 4 | `figures/fig_delay.pdf` |
| Fig. 5 | `figures/fig_downstream.pdf` |
| Graphical abstract | `figures/gagraphic.png`, `figures/gagraphic.pdf`, `figures/gagraphic_caption.txt` |
| Numbers quoted in the text | `results/summary/tex/numbers.tex` (one LaTeX macro per number) |

## Environment

All results were computed on Windows 11 with an Intel Core Ultra 7 155U (12 cores, 14
threads), 15.5 GB of memory and no GPU, with Python 3.12.10 and the package versions in
`requirements.txt`. The figure files are byte-identical when the fonts used for them (Times
New Roman and Arial, as installed with Windows 11) are available; with substitute fonts the
figure files differ, while the numbers, tables and summaries do not depend on fonts. Windows limits paths
to 259 characters unless long paths are enabled. The longest path of an extracted trace file
has 103 characters relative to the folder that holds `data/`, and 112 relative to the package
folder in the full path (`full_run/data/...`), so the path of the package folder should stay
below about 145 characters.

## Installation

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On Linux or macOS, use `python3.12 -m venv .venv` and `.venv/bin/python`. The commands
below assume the package folder as the working directory and the environment's Python.

## Fast path: summaries, tables and figures from the shipped results

```
python reproduce.py
```

`reproduce.py` checks the shipped input files against `SHA256SUMS`, runs
`scripts/analyze_locked.py`, `scripts/analyze_amendment.py --stored-audit`,
`scripts/make_tables.py` and `scripts/make_figures.py`, and compares each of the
44 regenerated files of `results/summary/` and `figures/` with its SHA-256 in
`SHA256SUMS`, printing `PASS` or `FAIL` per file and a summary line (exit code 0 when every
file passes). It needs no raw or processed data: the data audit and the ladder bitrates
(`results/summary/audit.json`) and the calibration pool sizes (`results/summary/pool_sizes.csv`,
`pool_sizes.json`) are read from the shipped files. Measured time: 0.6 min.
`SHA256SUMS` lists 69 files (protocol, data manifest, results and figures).
`python scripts/make_figures.py` also writes 300-dpi PNG renders of the figures.

## Full path: from the public datasets

```
python reproduce.py --full
```

The full path works in `full_run/` (option `--workdir`), so the shipped files stay
untouched. It downloads the datasets and the video ladder, rebuilds the feature tables, reruns
every experiment, recomputes `audit.json` and the pool sizes, and compares the regenerated
files with the shipped ones: the per-trace row files row by row (the gzip header, which
carries a write time, and the name of the identifier column are not compared),
`audit.json`, `pool_sizes.csv`, `pool_sizes.json` and `pilot_meta.json` byte by byte. It then
runs the fast path. The run scripts skip finished cells, so an interrupted full path resumes
when the same command is run again. `--n-jobs N` sets the worker processes of
`run_amendment.py` and `run_has.py` (default 6, the maximum of the protocol);
`run_locked.py` uses up to 6. `--offline` skips the downloads when `LTE_Dataset.zip` and
`5G-production-dataset.zip` are in `full_run/data/raw/` and `4K_segment_distribution.tar` in
`full_run/data/external/`. Measured times of the steps (6 workers where parallel):

| Step | Command | Measured time |
| --- | --- | --- |
| Download and verify the datasets | `python scripts/download_data.py` | download of 4.4 MB; 1 s to verify and extract |
| Feature tables, h = 1 | `python scripts/prepare_data.py --horizon 1` | 16 s |
| Feature tables, h = 5 | `python scripts/prepare_data.py --horizon 5` | 10 s |
| Timestamps and timestamp-based features | `python scripts/prepare_timestamps.py` | 32 s |
| Calibration pool sizes | `python scripts/pool_sizes.py` | 6 s |
| Pilot (seeds 100 to 102, one process) | `python scripts/run_pilot.py` | about 23 min |
| Locked runs (265 cells) | `python scripts/run_locked.py` | 60.2 min |
| Amendment runs, main (140 cells) | `python scripts/run_amendment.py --tag main` | 20.3 min |
| Amendment runs, h5 (10 cells) | `python scripts/run_amendment.py --tag h5` | 1.6 min |
| Amendment runs, ts (35 cells) | `python scripts/run_amendment.py --tag ts` | 3.9 min |
| Video ladder and streaming simulation (120 cells) | `python scripts/fetch_ladder.py` and `python scripts/run_has.py` | 31.4 min |
| Data audit, comparisons and fast path | (reproduce.py) | about 2 min |

In total about 2.4 h. The steps can also be run one by one with the commands
above from the package folder, which writes the data to `data/` and the results in place.

## Tests

```
python -m pytest -p no:cacheprovider tests
```

Result on the reference machine: `181 passed, 3 skipped in 100.83s (0:01:40)` (1.7 min). The
three skipped tests compare the streaming simulator with the Sabre simulator (Spiteri,
Sitaraman and Sparacio, ACM TOMM 2019; github.com/UMass-LIDS/sabre, BSD-2-Clause, not
included); they run when the environment variable `SABRE_PATH` names Sabre's `src/sabre.py`.

## Data sources and licences

The datasets are not redistributed; the scripts download them from their publishers.

| Data | Source | Licence |
|---|---|---|
| 4G traces (135 traces): D. Raca, J. J. Quinlan, A. H. Zahran and C. J. Sreenan, ACM MMSys 2018, DOI 10.1145/3204949.3208123 | Zenodo, DOI 10.5281/zenodo.1219679 (`LTE_Dataset.zip`) | CC BY 4.0 |
| 5G traces (83 traces): D. Raca, D. Leahy, C. J. Sreenan and J. J. Quinlan, ACM MMSys 2020, DOI 10.1145/3339825.3394938 | github.com/uccmisl/5Gdataset (`5G-production-dataset.zip`) | GPL-3.0 |
| Video segment sizes: J. J. Quinlan and C. J. Sreenan, ACM MMSys 2018, DOI 10.1145/3204949.3208130 | Zenodo, DOI 10.5281/zenodo.1219788 (`4K_segment_distribution.tar`) | CC BY 4.0 |

The GPL-3.0 licence governs the 5G dataset; that dataset is not covered by the MIT licence of
this package's code. `scripts/download_data.py` accepts an archive only when its size and
SHA-256 equal those in `data/raw/download_manifest.json`, and `scripts/fetch_ladder.py` checks
the ladder archive and the extracted segment-size file against their SHA-256.

## Results

Values from `results/summary/summary.json`, rounded half up as in the article. Keys of
regimes, methods and variants are listed under Keys below.

### Coverage and width (Table 1 of the article)

Mean marginal test coverage over 20 seeds at nominal level 0.90, with the median width in
Mbit/s in parentheses (median across traces of the seed-averaged per-trace median width,
infinite widths included).

| Method | In-domain 4G | In-domain 5G | Mobility 4G | Mobility 5G | Tech 4G→5G | Tech 5G→4G | App 5G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 0.887 (12.4) | 0.923 (18.8) | 0.838 (10.5) | 0.935 (21.2) | 0.732 (7.7) | 0.921 (24.6) | 0.990 (33.6) |
| Raw QR | 0.874 (7.8) | 0.926 (2.5) | 0.777 (7.9) | 0.919 (7.0) | 0.504 (3.8) | 0.882 (10.6) | 0.573 (9.1) |
| Split CP | 0.896 (12.5) | 0.914 (19.0) | 0.830 (10.6) | 0.897 (20.7) | 0.738 (7.9) | 0.819 (17.2) | 0.986 (38.6) |
| CQR | 0.901 (7.9) | 0.926 (2.5) | 0.835 (8.2) | 0.926 (7.2) | 0.667 (3.8) | 0.882 (10.6) | 0.774 (9.7) |
| Mondrian | 0.890 (11.6) | 0.872 (13.6) | 0.776 (7.7) | 0.826 (16.0) | 0.744 (8.1) | 0.783 (14.3) | 0.965 (28.6) |
| ACI | 0.902 (8.4) | 0.903 (5.4) | 0.901 (8.9) | 0.903 (7.1) | 0.901 (7.9) | 0.902 (9.9) | 0.904 (10.0) |
| Mondrian-ACI | 0.902 (8.4) | 0.902 (5.2) | 0.901 (9.7) | 0.902 (7.1) | 0.901 (8.1) | 0.902 (10.1) | 0.904 (9.9) |

### Informativeness (Table 3)

Split CP, ACI and Mondrian-ACI at nominal level 0.90 (20 seeds, h = 1, immediate feedback).
Inf. is the instance-level percentage of infinite intervals; Fin. cov. the coverage of the
finite intervals, pooled within each seed and averaged over seeds; Fin. width the median
across traces of the seed-averaged per-trace median finite width; Winkler the median across
traces of the seed-averaged per-trace mean Winkler score (Eq. (12)) over finite
intervals (both in Mbit/s, lower is better); Worst group the lowest speed-group coverage (mean
over seeds) and its speed bin in km/h.

| Regime | Method | Inf. (%) | Fin. cov. | Fin. width | Winkler | Worst group |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| In-domain 4G | Split CP | 0.00 | 0.896 | 12.54 | 17.4 | 0.879 [0, 1) |
|  | ACI | 4.55 | 0.898 | 8.37 | 14.7 | 0.901 [50, ∞) |
|  | Mondrian-ACI | 5.10 | 0.897 | 8.18 | 14.6 | 0.899 [50, ∞) |
| In-domain 5G | Split CP | 0.00 | 0.914 | 18.99 | 24.1 | 0.862 [0, 1) |
|  | ACI | 3.80 | 0.899 | 5.45 | 18.2 | 0.901 [15, 50) |
|  | Mondrian-ACI | 6.24 | 0.895 | 5.18 | 18.4 | 0.884 [50, ∞) |
| Mobility 4G | Split CP | 0.00 | 0.830 | 10.57 | 21.8 | 0.814 [0, 1) |
|  | ACI | 9.93 | 0.890 | 8.49 | 14.7 | 0.900 [50, ∞) |
|  | Mondrian-ACI | 17.59 | 0.878 | 7.06 | 14.5 | 0.898 [15, 50) |
| Mobility 5G | Split CP | 0.00 | 0.897 | 20.68 | 24.2 | 0.882 [0, 1) |
|  | ACI | 4.81 | 0.898 | 7.03 | 16.8 | 0.902 [15, 50) |
|  | Mondrian-ACI | 7.60 | 0.893 | 7.11 | 17.1 | 0.878 [1, 15) |
| Tech 4G→5G | Split CP | 0.00 | 0.738 | 7.91 | 28.4 | 0.554 [0, 1) |
|  | ACI | 15.88 | 0.882 | 7.85 | 23.2 | 0.900 [0, 1) |
|  | Mondrian-ACI | 16.04 | 0.882 | 7.76 | 23.3 | 0.898 [50, ∞) |
| Tech 5G→4G | Split CP | 0.00 | 0.819 | 17.16 | 31.0 | 0.769 [0, 1) |
|  | ACI | 3.58 | 0.898 | 9.78 | 18.4 | 0.900 [50, ∞) |
|  | Mondrian-ACI | 6.89 | 0.894 | 9.79 | 19.0 | 0.896 [50, ∞) |
| App 5G | Split CP | 0.00 | 0.986 | 38.62 | 41.5 | 0.974 [0, 1) |
|  | ACI | 0.05 | 0.904 | 9.98 | 23.9 | 0.904 [0, 1) |
|  | Mondrian-ACI | 1.07 | 0.903 | 9.90 | 24.8 | 0.870 [50, ∞) |

### Feedback delay (Fig. 4)

ACI and Mondrian-ACI when each label is applied D seconds after the timestamp of its target
row (h = 1, 20 seeds): range of the mean marginal coverage over D = 0, 1, 2, 5 and 10 s,
instance-level percentage of infinite intervals at each D, and coverage of the finite
intervals at D = 0 and 10 s. Every one of the 95,390 trace and seed rows of the delay
grid and the 5-s horizon satisfies Eq. (9) with its recorded K and M; the ratio of
the observed deviation to the bound is at most 0.953 (`results/summary/delay_bound.csv`).

| Regime | Method | Coverage, D = 0 to 10 s | Inf. % D = 0 | D = 1 | D = 2 | D = 5 | D = 10 | Fin. cov. D = 0 | D = 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| In-domain 4G | ACI | 0.902 to 0.902 | 4.5 | 6.1 | 7.4 | 11.0 | 16.8 | 0.898 | 0.882 |
| In-domain 4G | Mondrian-ACI | 0.902 to 0.902 | 5.1 | 6.8 | 8.0 | 11.7 | 17.4 | 0.897 | 0.881 |
| In-domain 5G | ACI | 0.902 to 0.903 | 3.8 | 5.0 | 5.9 | 8.3 | 12.3 | 0.899 | 0.887 |
| In-domain 5G | Mondrian-ACI | 0.901 to 0.902 | 6.2 | 8.0 | 9.3 | 12.4 | 17.1 | 0.895 | 0.879 |
| Mobility 4G | ACI | 0.900 to 0.901 | 9.9 | 12.5 | 14.4 | 19.5 | 26.9 | 0.890 | 0.862 |
| Mobility 4G | Mondrian-ACI | 0.900 to 0.901 | 17.6 | 20.3 | 22.1 | 26.8 | 33.4 | 0.878 | 0.846 |
| Mobility 5G | ACI | 0.902 to 0.903 | 4.8 | 6.6 | 8.0 | 10.9 | 15.1 | 0.898 | 0.884 |
| Mobility 5G | Mondrian-ACI | 0.901 to 0.902 | 7.6 | 10.3 | 12.4 | 17.3 | 24.4 | 0.893 | 0.866 |
| Tech 4G→5G | ACI | 0.900 to 0.901 | 15.9 | 18.8 | 22.0 | 26.5 | 32.0 | 0.882 | 0.853 |
| Tech 4G→5G | Mondrian-ACI | 0.900 to 0.901 | 16.0 | 19.0 | 21.9 | 26.3 | 32.0 | 0.882 | 0.853 |
| Tech 5G→4G | ACI | 0.901 to 0.902 | 3.6 | 6.4 | 9.3 | 16.8 | 26.2 | 0.898 | 0.863 |
| Tech 5G→4G | Mondrian-ACI | 0.900 to 0.902 | 6.9 | 9.9 | 12.9 | 20.4 | 30.2 | 0.894 | 0.855 |
| App 5G | ACI | 0.904 to 0.904 | 0.0 | 0.1 | 0.2 | 0.6 | 1.8 | 0.904 | 0.902 |
| App 5G | Mondrian-ACI | 0.903 to 0.904 | 1.1 | 1.4 | 2.0 | 3.1 | 5.1 | 0.903 | 0.897 |

### Adaptive streaming (Tables 5 and 6)

MPC players in the four shifted regimes with capacity traces (number of traces in
parentheses), 20 seeds: means across traces of per-trace values averaged over seeds. Bitrate
in Mbit/s, rebuffering ratio in percent of playback plus rebuffering time, stalls after
startup, quality switches per minute, and QoE (Eq. (14)) per segment (higher is
better). Bounds are interval lower bounds at α = 0.10. The rate-based rows are in
`results/summary/tex/tab_has.tex`.

| Regime (traces) | Player and estimate | Bitrate | Rebuf. (%) | Stalls | Switches/min | QoE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mobility 4G (89) | MPC, harmonic mean | 12.79 | 11.26 | 16.98 | 4.37 | −27.73 |
|  | RobustMPC | 10.59 | 5.83 | 3.63 | 3.87 | −15.72 |
|  | MPC, point prediction | 12.67 | 9.56 | 12.99 | 5.18 | −26.42 |
|  | MPC, split CP bound | 9.80 | 5.10 | 2.23 | 3.81 | −13.27 |
|  | MPC, ACI bound | 6.67 | 4.49 | 1.69 | 5.49 | −17.15 |
|  | MPC, Mondrian-ACI bound | 5.96 | 4.43 | 1.69 | 5.45 | −17.81 |
| Mobility 5G (16) | MPC, harmonic mean | 17.14 | 6.52 | 19.44 | 4.03 | 3.50 |
|  | RobustMPC | 12.49 | 1.44 | 2.69 | 3.82 | 7.60 |
|  | MPC, point prediction | 14.54 | 4.99 | 11.93 | 5.72 | 1.15 |
|  | MPC, split CP bound | 6.95 | 1.05 | 1.39 | 2.91 | 1.38 |
|  | MPC, ACI bound | 2.47 | 1.00 | 1.04 | 3.44 | −2.19 |
|  | MPC, Mondrian-ACI bound | 2.26 | 0.96 | 1.02 | 3.55 | −2.27 |
| Tech 4G→5G (21) | MPC, harmonic mean | 21.13 | 6.38 | 20.24 | 3.45 | 7.78 |
|  | RobustMPC | 15.87 | 1.68 | 2.95 | 3.56 | 10.40 |
|  | MPC, point prediction | 20.17 | 5.45 | 18.13 | 5.05 | 6.23 |
|  | MPC, split CP bound | 15.55 | 1.69 | 2.83 | 4.94 | 7.32 |
|  | MPC, ACI bound | 4.07 | 1.09 | 1.38 | 5.87 | −2.40 |
|  | MPC, Mondrian-ACI bound | 4.04 | 1.07 | 1.37 | 5.85 | −2.44 |
| Tech 5G→4G (135) | MPC, harmonic mean | 12.42 | 8.97 | 15.22 | 4.31 | −17.69 |
|  | RobustMPC | 10.61 | 4.27 | 2.96 | 3.77 | −8.08 |
|  | MPC, point prediction | 10.77 | 5.66 | 5.62 | 4.97 | −15.00 |
|  | MPC, split CP bound | 2.79 | 2.82 | 0.88 | 2.59 | −11.13 |
|  | MPC, ACI bound | 1.11 | 2.87 | 1.02 | 3.20 | −12.67 |
|  | MPC, Mondrian-ACI bound | 1.09 | 2.88 | 1.02 | 3.21 | −12.78 |

Pre-specified paired two-sided Wilcoxon signed-rank tests on per-trace QoE, MPC with the
Mondrian-ACI bound (a) against each comparator (b); med. diff. is the median of a − b
(negative favours b), r_rb the matched-pairs rank-biserial correlation, p_BH the
Benjamini-Hochberg adjusted p-value over the 12 tests.

| Regime | Comparator (b) | n | QoE a | QoE b | Med. diff. | r_rb | p_BH |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mobility 4G | MPC, ACI bound | 89 | −17.81 | −17.15 | −0.45 | −0.83 | 9.6 × 10^−11 |
| Mobility 4G | MPC, split CP bound | 89 | −17.81 | −13.27 | −1.97 | −0.72 | 1.1 × 10^−8 |
| Mobility 4G | RobustMPC | 89 | −17.81 | −15.72 | −3.22 | −0.56 | 8.2 × 10^−6 |
| Mobility 5G | MPC, ACI bound | 16 | −2.27 | −2.19 | 0.02 | 0.01 | 0.98 |
| Mobility 5G | MPC, split CP bound | 16 | −2.27 | 1.38 | −1.75 | −0.99 | 8.1 × 10^−5 |
| Mobility 5G | RobustMPC | 16 | −2.27 | 7.60 | −8.59 | −1.00 | 4.6 × 10^−5 |
| Tech 4G→5G | MPC, ACI bound | 21 | −2.44 | −2.40 | −0.09 | −0.65 | 0.0077 |
| Tech 4G→5G | MPC, split CP bound | 21 | −2.44 | 7.32 | −8.55 | −0.97 | 8.2 × 10^−6 |
| Tech 4G→5G | RobustMPC | 21 | −2.44 | 10.40 | −10.27 | −1.00 | 2.3 × 10^−6 |
| Tech 5G→4G | MPC, ACI bound | 135 | −12.78 | −12.67 | −0.02 | −0.37 | 2.6 × 10^−4 |
| Tech 5G→4G | MPC, split CP bound | 135 | −12.78 | −11.13 | −0.55 | −0.74 | 8.8 × 10^−13 |
| Tech 5G→4G | RobustMPC | 135 | −12.78 | −8.08 | −3.98 | −0.73 | 8.8 × 10^−13 |

## Keys

| Key | Meaning |
|---|---|
| `in_domain_4g`, `in_domain_5g` | Random 60/20/20 trace split into training, calibration and test within one technology |
| `cross_mobility_4g` | 4G static and pedestrian traces to car, bus and train traces |
| `cross_mobility_5g` | 5G static traces to driving traces |
| `cross_tech_4to5`, `cross_tech_5to4` | All 4G traces to all 5G traces, and the reverse |
| `cross_app_5g` | 5G file-download traces to Netflix and Amazon Prime traces |
| `gaussian`, `quantile_raw` | Gaussian intervals around the point prediction ("Gaussian") and raw quantile regression ("Raw QR") |
| `split`, `cqr`, `mondrian` | Split conformal prediction ("Split CP"), conformalized quantile regression ("CQR"), Mondrian split CP over speed groups ("Mondrian") |
| `aci`, `mondrian_aci` | Adaptive conformal inference run per trace ("ACI") and its speed-group variant ("Mondrian-ACI") |
| `speed_0`, `speed_low`, `speed_med`, `speed_high` | Speed groups [0, 1), [1, 15), [15, 50) and [50, ∞) km/h |
| `rate_<estimator>`, `mpc_<estimator>` | Streaming variants: rate-based rule (highest representation not above the estimate) or model predictive control over five segments |
| `hm`, `robust`, `point` | Harmonic mean of the last five segment throughputs; RobustMPC (that mean divided by 1 plus the largest relative error of the last five estimates); point prediction |
| `split`, `aci`, `mondrian_aci` as estimators | Lower bound of that method's interval at α = 0.10 |

In cross-domain regimes the source traces are split 80/20 into training and calibration and
every target trace is tested. Seeds: 0 to 19 for the main configuration, 0 to 4 for the
ablations and the tags h5 and ts, 100 to 102 for the pilot.

## Directory layout

```
reproduce.py           fast path and full path
SHA256SUMS             SHA-256 of the protocol, data manifest, result and figure files
requirements.txt       package versions
src/                   data loading, features, models, interval methods, metrics,
                       statistics and the streaming simulator
scripts/               pipeline scripts (table below)
tests/                 unit tests
protocol/              protocol fixed before the locked runs, its hashes, amendment 1
data/                  dataset manifest and data notes (README.md)
results/locked/        locked rows (265 cells), analysis provenance, run record
results/pilot/         pilot rows and settings
results/amendment1/    rows of tags main, h5 and ts, streaming rows (has/), run
                       configurations, timing log and batch timeline
results/summary/       summaries (CSV, summary.json, digest.md), audit.json, pool sizes,
                       LaTeX tables and number macros (tex/), locked analysis (locked/)
figures/               figures of the article (PDF) and graphical abstract
```

| Script | Role |
| --- | --- |
| `download_data.py` | Downloads the 4G and 5G archives, verifies them against `data/raw/download_manifest.json` and extracts them into `data/raw/`. |
| `prepare_data.py` | Feature tables `data/processed/features_h1.csv.gz` and `features_h5.csv.gz` with their data audits. |
| `prepare_timestamps.py` | Instance timestamps (`features_h{1,5}_timed.csv.gz`) and the timestamp-based feature table (`features_timebase_h1.csv.gz`). |
| `pool_sizes.py` | Calibration pool sizes of the speed groups. |
| `run_pilot.py` | Pilot used to select γ (`results/pilot/`). |
| `run_locked.py` | Locked runs of `protocol/protocol.json` (`results/locked/`). |
| `run_amendment.py` | Runs of amendment 1, tags main, h5 and ts (`results/amendment1/`). |
| `fetch_ladder.py` | Downloads and verifies the UCC 4K DASH segment sizes (`data/external/`, `data/ladder/`). |
| `run_has.py` | Streaming simulation (`results/amendment1/has/`). |
| `analyze_locked.py` | Locked analysis (`results/summary/locked/`). |
| `analyze_amendment.py` | Summaries of amendment 1 and full-precision locked summaries (`results/summary/`). |
| `make_tables.py` | LaTeX tables and number macros (`results/summary/tex/`). |
| `make_figures.py` | Figures and graphical abstract (`figures/`). |

`results/summary/summary.json` holds every scalar of the summaries by name at full precision
and `results/summary/digest.md` the same summaries at 3 to 4 significant digits;
`results/summary/audit.json` holds the data audit (trace and instance counts, target timing,
zero-target shares) and the ladder bitrates. NaN is an empty CSV field and null in JSON.

## Column definitions

### `results/locked/locked_rows.csv.gz`

Locked confirmatory and ablation rows written by scripts/run_locked.py: one row per cell (tag, regime, base model, seed, α, γ), method and test trace.

| Column | Definition |
| --- | --- |
| `tag` | Run block: `main` (7 regimes, seeds 0 to 19), `abl_base` (ridge and persistence base models), `abl_alpha05` (α = 0.05), `abl_gamma` (γ = 0.01 and 0.02) or `abl_h5` (horizon of 5 rows). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `alpha` | Miscoverage level α; the intervals target coverage 1 − α. |
| `gamma` | Step size γ of ACI and Mondrian-ACI. |
| `base_model` | Point model: `gbm` (histogram gradient boosting), `ridge` or `persistence`. |
| `method` | Interval method (see Keys). |
| `trace_id` | Trace identifier: technology, folder path and file name of the trace without `.csv`. In this file the column has a different name ending in `_id`; `src.data_loading.read_cell_rows` returns it as `trace_id`. |
| `n` | Test instances of the trace. |
| `coverage` | Fraction of the trace's test targets inside their intervals. |
| `frac_inf` | Fraction of the trace's intervals that are infinite. |
| `mean_width` | Mean width of the trace's finite intervals, Mbit/s (inf when all are infinite). |
| `median_width` | Median width of all of the trace's intervals, infinite widths included, Mbit/s. |
| `winkler` | Median over the trace's instances of the interval (Winkler) score at level α, Mbit/s (infinite for an infinite interval). |
| `overshoot_lb` | Over the trace's instances with a positive target, fraction whose interval lower bound exceeds the target. |
| `util_lb` | Over the same instances, mean of min(lower bound, target) / target. |
| `overshoot_point` | As `overshoot_lb` with the point prediction. |
| `util_point` | As `util_lb` with the point prediction. |
| `marginal_coverage` | Coverage pooled over all test instances of the cell (regime, seed, setting, method), repeated on every row of the cell. |
| `rmse_point` | Root mean squared error of the point prediction over the test instances of the cell, Mbit/s. |
| `mae_point` | Mean absolute error of the point prediction over the test instances of the cell, Mbit/s. |
| `group_coverage_json` | JSON object giving, for each speed group, the coverage pooled over the test instances of the cell in that group. |

### `results/pilot/pilot_rows.csv.gz`

Pilot rows written by scripts/run_pilot.py (regimes in_domain_4g, in_domain_5g, cross_mobility_5g and cross_tech_4to5, seeds 100 to 102, γ = 0.01, 0.02 and 0.05, plus the persistence and ridge base models on in_domain_4g, seed 100, γ = 0.05); columns as in locked_rows.csv.gz without tag.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `alpha` | Miscoverage level α; the intervals target coverage 1 − α. |
| `gamma` | Step size γ of ACI and Mondrian-ACI. |
| `base_model` | Point model: `gbm` (histogram gradient boosting), `ridge` or `persistence`. |
| `method` | Interval method (see Keys). |
| `trace_id` | Trace identifier: technology, folder path and file name of the trace without `.csv`. In this file the column has a different name ending in `_id`; `src.data_loading.read_cell_rows` returns it as `trace_id`. |
| `n` | Test instances of the trace. |
| `coverage` | Fraction of the trace's test targets inside their intervals. |
| `frac_inf` | Fraction of the trace's intervals that are infinite. |
| `mean_width` | Mean width of the trace's finite intervals, Mbit/s (inf when all are infinite). |
| `median_width` | Median width of all of the trace's intervals, infinite widths included, Mbit/s. |
| `winkler` | Median over the trace's instances of the interval (Winkler) score at level α, Mbit/s (infinite for an infinite interval). |
| `overshoot_lb` | Over the trace's instances with a positive target, fraction whose interval lower bound exceeds the target. |
| `util_lb` | Over the same instances, mean of min(lower bound, target) / target. |
| `overshoot_point` | As `overshoot_lb` with the point prediction. |
| `util_point` | As `util_lb` with the point prediction. |
| `marginal_coverage` | Coverage pooled over all test instances of the cell (regime, seed, setting, method), repeated on every row of the cell. |
| `rmse_point` | Root mean squared error of the point prediction over the test instances of the cell, Mbit/s. |
| `mae_point` | Mean absolute error of the point prediction over the test instances of the cell, Mbit/s. |
| `group_coverage_json` | JSON object giving, for each speed group, the coverage pooled over the test instances of the cell in that group. |

### `results/amendment1/main_traces.csv.gz`

Per-trace rows of tag main written by scripts/run_amendment.py: one row per regime, seed, method, feedback, delay and test trace.

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1, 7 regimes, seeds 0 to 19), `h5` (h = 5, in_domain_5g and cross_tech_4to5, seeds 0 to 4) or `ts` (timestamp-based features, 7 regimes, seeds 0 to 4). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `base_model` | Point model (`gbm`). |
| `alpha` | Miscoverage level α (0.1). |
| `gamma` | Step size γ (0.05). |
| `horizon` | Forecast horizon h: rows ahead for tags main and h5, seconds ahead on the timestamp grid for tag ts. |
| `method` | Interval method (see Keys). |
| `feedback` | `none` for the five methods without feedback; `delayed` for ACI and Mondrian-ACI under the delay model of amendment 1.4. |
| `delay_s` | Reporting delay D in seconds: a label is applied D s after the timestamp of its target row (empty when feedback is none). |
| `trace_id` | Trace identifier: technology, folder path and file name of the trace without `.csv`. |
| `n` | Test instances of the trace. |
| `n_cov` | Covered test targets. |
| `coverage` | n_cov / n. |
| `abs_dev` | Absolute deviation of coverage from 1 − α. |
| `n_inf` | Infinite intervals. |
| `frac_inf` | n_inf / n. |
| `n_finite` | Finite intervals. |
| `n_cov_finite` | Covered targets with a finite interval. |
| `coverage_finite` | n_cov_finite / n_finite (empty when n_finite = 0). |
| `n_inverted` | Finite intervals with upper end below lower end (empty sets). |
| `width_finite_median` | Median of max(hi − lo, 0) over the finite intervals, Mbit/s. |
| `width_finite_mean` | Mean of max(hi − lo, 0) over the finite intervals, Mbit/s. |
| `winkler_finite_mean` | Mean interval (Winkler) score over the finite intervals, inverted intervals with zero width, Mbit/s. |
| `n_y0` | Targets equal to 0 Mbit/s. |
| `n_pos` | Targets above 0 Mbit/s (empty for D > 0). |
| `n_cov_pos` | Covered positive targets (empty for D > 0). |
| `coverage_pos` | n_cov_pos / n_pos (empty for D > 0). |
| `max_outstanding` | K: largest number of earlier instances of the trace whose labels had not been applied when an interval was issued (empty when feedback is none). |
| `n_applied` | M: labels applied before the end of the trace (empty when feedback is none). |
| `n_alpha_le0` | Intervals issued with α_t ≤ 0, which are infinite. |
| `n_alpha_ge1` | Intervals issued with α_t ≥ 1, which have zero radius. |
| `n_alpha_ge1_covered` | Those of them that covered their target. |
| `bound` | Right side of Proposition 1 (ii), (max(α, 1 − α) + (K + 1)γ) / (γM); inf when M = 0. |

### `results/amendment1/main_groups.csv.gz`

Speed-group rows of tag main: one row per regime, seed, method, feedback, delay and speed group.

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1, 7 regimes, seeds 0 to 19), `h5` (h = 5, in_domain_5g and cross_tech_4to5, seeds 0 to 4) or `ts` (timestamp-based features, 7 regimes, seeds 0 to 4). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `base_model` | Point model (`gbm`). |
| `alpha` | Miscoverage level α (0.1). |
| `gamma` | Step size γ (0.05). |
| `horizon` | Forecast horizon h: rows ahead for tags main and h5, seconds ahead on the timestamp grid for tag ts. |
| `method` | Interval method (see Keys). |
| `feedback` | `none` for the five methods without feedback; `delayed` for ACI and Mondrian-ACI under the delay model of amendment 1.4. |
| `delay_s` | Reporting delay D in seconds: a label is applied D s after the timestamp of its target row (empty when feedback is none). |
| `group` | Speed group (see Keys). |
| `n` | Test instances of the cell in the group. |
| `n_cov` | Covered test targets in the group. |
| `coverage` | n_cov / n. |

### `results/amendment1/h5_traces.csv.gz`

Per-trace rows of tag h5 (horizon of 5 rows; ACI and Mondrian-ACI apply each label when it is observed).

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1, 7 regimes, seeds 0 to 19), `h5` (h = 5, in_domain_5g and cross_tech_4to5, seeds 0 to 4) or `ts` (timestamp-based features, 7 regimes, seeds 0 to 4). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `base_model` | Point model (`gbm`). |
| `alpha` | Miscoverage level α (0.1). |
| `gamma` | Step size γ (0.05). |
| `horizon` | Forecast horizon h: rows ahead for tags main and h5, seconds ahead on the timestamp grid for tag ts. |
| `method` | Interval method (see Keys). |
| `feedback` | `none` for the five methods without feedback; `delayed` for ACI and Mondrian-ACI under the delay model of amendment 1.4. |
| `delay_s` | Reporting delay D in seconds: a label is applied D s after the timestamp of its target row (empty when feedback is none). |
| `trace_id` | Trace identifier: technology, folder path and file name of the trace without `.csv`. |
| `n` | Test instances of the trace. |
| `n_cov` | Covered test targets. |
| `coverage` | n_cov / n. |
| `abs_dev` | Absolute deviation of coverage from 1 − α. |
| `n_inf` | Infinite intervals. |
| `frac_inf` | n_inf / n. |
| `n_finite` | Finite intervals. |
| `n_cov_finite` | Covered targets with a finite interval. |
| `coverage_finite` | n_cov_finite / n_finite (empty when n_finite = 0). |
| `n_inverted` | Finite intervals with upper end below lower end (empty sets). |
| `width_finite_median` | Median of max(hi − lo, 0) over the finite intervals, Mbit/s. |
| `width_finite_mean` | Mean of max(hi − lo, 0) over the finite intervals, Mbit/s. |
| `winkler_finite_mean` | Mean interval (Winkler) score over the finite intervals, inverted intervals with zero width, Mbit/s. |
| `max_outstanding` | K: largest number of earlier instances of the trace whose labels had not been applied when an interval was issued (empty when feedback is none). |
| `n_applied` | M: labels applied before the end of the trace (empty when feedback is none). |
| `n_alpha_le0` | Intervals issued with α_t ≤ 0, which are infinite. |
| `n_alpha_ge1` | Intervals issued with α_t ≥ 1, which have zero radius. |
| `n_alpha_ge1_covered` | Those of them that covered their target. |
| `bound` | Right side of Proposition 1 (ii), (max(α, 1 − α) + (K + 1)γ) / (γM); inf when M = 0. |

### `results/amendment1/h5_groups.csv.gz`

Speed-group rows of tag h5.

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1, 7 regimes, seeds 0 to 19), `h5` (h = 5, in_domain_5g and cross_tech_4to5, seeds 0 to 4) or `ts` (timestamp-based features, 7 regimes, seeds 0 to 4). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `base_model` | Point model (`gbm`). |
| `alpha` | Miscoverage level α (0.1). |
| `gamma` | Step size γ (0.05). |
| `horizon` | Forecast horizon h: rows ahead for tags main and h5, seconds ahead on the timestamp grid for tag ts. |
| `method` | Interval method (see Keys). |
| `feedback` | `none` for the five methods without feedback; `delayed` for ACI and Mondrian-ACI under the delay model of amendment 1.4. |
| `delay_s` | Reporting delay D in seconds: a label is applied D s after the timestamp of its target row (empty when feedback is none). |
| `group` | Speed group (see Keys). |
| `n` | Test instances of the cell in the group. |
| `n_cov` | Covered test targets in the group. |
| `coverage` | n_cov / n. |

### `results/amendment1/ts_traces.csv.gz`

Per-trace rows of tag ts (timestamp-based features of amendment 1.7, D = 0).

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1, 7 regimes, seeds 0 to 19), `h5` (h = 5, in_domain_5g and cross_tech_4to5, seeds 0 to 4) or `ts` (timestamp-based features, 7 regimes, seeds 0 to 4). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `base_model` | Point model (`gbm`). |
| `alpha` | Miscoverage level α (0.1). |
| `gamma` | Step size γ (0.05). |
| `horizon` | Forecast horizon h: rows ahead for tags main and h5, seconds ahead on the timestamp grid for tag ts. |
| `method` | Interval method (see Keys). |
| `feedback` | `none` for the five methods without feedback; `delayed` for ACI and Mondrian-ACI under the delay model of amendment 1.4. |
| `delay_s` | Reporting delay D in seconds: a label is applied D s after the timestamp of its target row (empty when feedback is none). |
| `trace_id` | Trace identifier: technology, folder path and file name of the trace without `.csv`. |
| `n` | Test instances of the trace. |
| `n_cov` | Covered test targets. |
| `coverage` | n_cov / n. |
| `n_inf` | Infinite intervals. |
| `frac_inf` | n_inf / n. |

### `results/amendment1/timing_log.csv`

One row per cell execution of scripts/run_amendment.py.

| Column | Definition |
| --- | --- |
| `finished_utc` | Time the cell finished (UTC). |
| `mode` | `full` (metrics written) or `reproduce_only` (reproduction check of amendment 1.3 only). |
| `tag` | Run tag (main, h5 or ts). |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `status` | `ok`, `failed` (reproduction criterion not met) or `error`. |
| `seconds_total` | Wall-clock seconds of the cell in its worker process. |
| `seconds_fit` | Seconds for fitting the base models and calibrating. |
| `seconds_reproduction` | Seconds for the reproduction check against the locked rows. |
| `seconds_online` | Seconds for the online methods under the delay model (empty for reproduce_only). |
| `seconds_metrics` | Seconds for the per-trace and per-group metrics (empty for reproduce_only). |
| `n_test` | Test instances of the cell. |
| `n_traces` | Test traces of the cell. |
| `pid` | Worker process identifier. |
| `started_utc` | Time the cell started (UTC). |

### `results/amendment1/has/has_rows.csv.gz`

Streaming simulation rows written by scripts/run_has.py: one row per regime, seed, streaming-eligible test trace (4G traces and 5G Download traces) and variant.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). (the six regimes with capacity traces). |
| `seed` | Seed of the trace-level split. |
| `trace_id` | Trace identifier: technology, folder path and file name of the trace without `.csv`. |
| `tech` | `4G` or `5G`. |
| `mobility` | Mobility label of the trace folder. |
| `variant` | `<rule>_<estimator>` (see Keys). |
| `rule` | `rate` or `mpc`. |
| `estimator` | `hm`, `robust`, `point`, `split`, `aci` or `mondrian_aci`. |
| `trace_rows` | Rows of the raw trace. |
| `start_second` | Row of the first test instance of the trace; playback starts there. |
| `capacity_missing` | Missing throughput values from start_second on, replayed as zero capacity. |
| `n_segments` | Segments downloaded before the end of the trace. |
| `avg_bitrate_mbps` | Mean nominal bitrate of those segments, Mbit/s. |
| `rebuffer_s` | Stall time after startup up to the end of the trace, s. |
| `rebuffer_final_s` | Part of rebuffer_s spent on the last download, which the end of the trace cut off, s. |
| `rebuffer_ratio` | rebuffer_s / (play_s + rebuffer_s). |
| `stalls` | Stalls after startup. |
| `startup_s` | Time from start_second to the arrival of the first segment, s. |
| `play_s` | Playback time from startup to the end of the trace, s. |
| `switches` | Changes of representation between consecutive segments. |
| `switches_per_min` | Switches per minute of playback. |
| `mean_abs_switch_mbps` | Mean absolute bitrate change per switch, Mbit/s (empty without switches). |
| `qoe_total` | Sum of segment bitrates − sum of absolute bitrate changes − μ rebuffer_s − μ startup_s, with μ = 39.537 (highest bitrate in Mbit/s). |
| `qoe_per_segment` | qoe_total / n_segments (QoE per segment of the article). |
| `wait_s` | Time spent waiting for buffer space, s. |
| `discarded` | 1 if a download was cut off by the end of the trace. |
| `duration_s` | Seconds from start_second to the end of the capacity trace. |

### `results/summary/audit_has_eligible.csv`

Streaming-eligible test traces per regime and seed.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `n_eligible` | Streaming-eligible test traces (4G traces and 5G Download traces). |
| `n_eligible_4g` | Of which 4G. |
| `n_eligible_5g` | Of which 5G. |

### `results/summary/delay_bound.csv`

Check of Proposition 1 (ii) in every trace and seed row. The deviation of the miss fraction over the M applied labels is bracketed from N, M and the miss count over all N intervals, because the unapplied labels are not stored individually.

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1) or `h5` (h = 5, D = 0). |
| `method` | `aci` or `mondrian_aci`. |
| `delay_s` | Reporting delay D, s. |
| `regime` | Transfer regime (see Keys). |
| `n_trace_rows` | Trace and seed rows checked. |
| `K_min` | Minimum over rows of K, the largest number of earlier instances of the trace whose labels had not been applied when an interval was issued. |
| `K_median` | Median of K over rows. |
| `K_mean` | Mean of K over rows. |
| `K_max` | Maximum of K over rows. |
| `M_min` | Minimum over rows of M, the number of labels applied before the end of the trace. |
| `M_median` | Median of M over rows. |
| `M_max` | Maximum of M over rows. |
| `unapplied_max` | Largest N − M over rows (N intervals of the trace). |
| `n_rows_unapplied_above_K_plus_1` | Rows with N − M > K + 1. |
| `bound_applied_min` | Minimum over rows of the bound of Proposition 1 (ii), (max(α, 1 − α) + (K + 1)γ) / (γM). |
| `bound_applied_median` | Median of that bound over rows. |
| `bound_applied_max` | Maximum of that bound over rows. |
| `bound_all_min` | Minimum over rows of its form over all N intervals, (max(α, 1 − α) + (K + 1)γ) / (γN) + (K + 1) max(α, 1 − α) / N. |
| `bound_all_median` | Median of the all-interval form over rows. |
| `bound_all_max` | Maximum of the all-interval form over rows. |
| `bound_stored_max_abs_diff` | Largest absolute difference between the recomputed bound and the bound stored in the trace rows. |
| `n_alpha_le0` | Intervals issued with α_t ≤ 0 (infinite intervals), summed over rows. |
| `n_alpha_ge1` | Intervals issued with α_t ≥ 1 (zero radius), summed over rows. |
| `n_alpha_ge1_covered` | Those of them that covered their target. |
| `n_rows_alpha_ge1` | Rows with at least one interval issued with α_t ≥ 1. |
| `n_rows_bound_not_applicable` | Rows where the bound does not apply (M = 0, or an interval issued with α_t ≥ 1 covered). |
| `max_ratio_applied_upper` | Largest ratio of the upper end of the bracket of the applied-label deviation to the bound (a value ≤ 1 proves the bound in every row). |
| `max_ratio_applied_lower` | Largest ratio of the lower end of the bracket to the bound. |
| `max_ratio_all` | Largest ratio of the deviation of the miss fraction over all N intervals from α to the all-interval form. |
| `n_rows_applied_proven` | Rows whose upper ratio is at most 1. |
| `n_rows_applied_violated` | Rows whose lower ratio exceeds 1 (proven violations). |
| `n_rows_applied_unresolved` | Rows between the two. |
| `n_rows_all_violated` | Rows whose all-interval ratio exceeds 1. |

### `results/summary/delay_grid.csv`

ACI and Mondrian-ACI for D = 0, 1, 2, 5 and 10 s (tag main) and the 5-s horizon (tag h5): interval summary and the check of Proposition 1 (ii), one row per tag, method, delay and regime.

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1) or `h5` (h = 5, D = 0). |
| `method` | `aci` or `mondrian_aci`. |
| `delay_s` | Reporting delay D, s. |
| `regime` | Transfer regime (see Keys). |
| `n_seeds` | Seeds with rows. |
| `n_traces` | Distinct test traces. |
| `n_trace_rows` | Trace and seed rows. |
| `n_instances` | Test instances, summed over the rows. |
| `marginal_coverage` | Coverage pooled over the test instances of each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `marginal_coverage_ci_lo` and `marginal_coverage_ci_hi`. |
| `marginal_coverage_ci_lo` | Lower bound of the CI of `marginal_coverage`. |
| `marginal_coverage_ci_hi` | Upper bound of the CI of `marginal_coverage`. |
| `trace_coverage` | Mean over traces of the per-trace coverage averaged over the seeds in which the trace was tested; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `trace_coverage_ci_lo` and `trace_coverage_ci_hi`. |
| `trace_coverage_ci_lo` | Lower bound of the CI of `trace_coverage`. |
| `trace_coverage_ci_hi` | Upper bound of the CI of `trace_coverage`. |
| `abs_dev` | Mean over traces of the absolute deviation of the seed-averaged trace coverage from 1 − α (Δ_s of the article); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `abs_dev_ci_lo` and `abs_dev_ci_hi`. |
| `abs_dev_ci_lo` | Lower bound of the CI of `abs_dev`. |
| `abs_dev_ci_hi` | Upper bound of the CI of `abs_dev`. |
| `frac_inf_instance` | Fraction of infinite intervals pooled within each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `frac_inf_instance_ci_lo` and `frac_inf_instance_ci_hi`. |
| `frac_inf_instance_ci_lo` | Lower bound of the CI of `frac_inf_instance`. |
| `frac_inf_instance_ci_hi` | Upper bound of the CI of `frac_inf_instance`. |
| `frac_inf_trace_rows` | Mean of the per-trace fraction of infinite intervals over the trace and seed rows. |
| `frac_inf_trace` | Mean over traces of the seed-averaged per-trace fraction of infinite intervals; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `frac_inf_trace_ci_lo` and `frac_inf_trace_ci_hi`. |
| `frac_inf_trace_ci_lo` | Lower bound of the CI of `frac_inf_trace`. |
| `frac_inf_trace_ci_hi` | Upper bound of the CI of `frac_inf_trace`. |
| `finite_coverage` | Coverage of the finite intervals pooled within each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `finite_coverage_ci_lo` and `finite_coverage_ci_hi`. |
| `finite_coverage_ci_lo` | Lower bound of the CI of `finite_coverage`. |
| `finite_coverage_ci_hi` | Upper bound of the CI of `finite_coverage`. |
| `finite_coverage_trace` | Mean over traces of the seed-averaged per-trace coverage of the finite intervals; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `finite_coverage_trace_ci_lo` and `finite_coverage_trace_ci_hi`. |
| `finite_coverage_trace_ci_lo` | Lower bound of the CI of `finite_coverage_trace`. |
| `finite_coverage_trace_ci_hi` | Upper bound of the CI of `finite_coverage_trace`. |
| `n_trace_rows_without_finite` | Rows without any finite interval. |
| `width_finite_median` | Median over traces of the seed-averaged per-trace median width of the finite intervals, Mbit/s. |
| `width_finite_mean_median` | Median over traces of the seed-averaged per-trace mean width of the finite intervals, Mbit/s. |
| `winkler_median` | Median over traces of the seed-averaged per-trace mean interval (Winkler) score over the finite intervals, Mbit/s. |
| `winkler_mean` | Mean over traces of the same per-trace values, Mbit/s; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `winkler_mean_ci_lo` and `winkler_mean_ci_hi`. |
| `winkler_mean_ci_lo` | Lower bound of the CI of `winkler_mean`. |
| `winkler_mean_ci_hi` | Upper bound of the CI of `winkler_mean`. |
| `n_inverted` | Finite intervals with upper end below lower end, summed over rows (−1 when not recorded). |
| `coverage_pos` | Coverage of the positive targets pooled within each seed, mean over seeds (empty when not recorded); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `coverage_pos_ci_lo` and `coverage_pos_ci_hi`. |
| `coverage_pos_ci_lo` | Lower bound of the CI of `coverage_pos`. |
| `coverage_pos_ci_hi` | Upper bound of the CI of `coverage_pos`. |
| `zero_share` | Share of targets equal to 0 Mbit/s pooled within each seed, mean over seeds (empty when not recorded). |
| `K_min` | Minimum over rows of K, the largest number of earlier instances of the trace whose labels had not been applied when an interval was issued. |
| `K_median` | Median of K over rows. |
| `K_mean` | Mean of K over rows. |
| `K_max` | Maximum of K over rows. |
| `M_min` | Minimum over rows of M, the number of labels applied before the end of the trace. |
| `M_median` | Median of M over rows. |
| `M_max` | Maximum of M over rows. |
| `unapplied_max` | Largest N − M over rows (N intervals of the trace). |
| `n_rows_unapplied_above_K_plus_1` | Rows with N − M > K + 1. |
| `bound_applied_min` | Minimum over rows of the bound of Proposition 1 (ii), (max(α, 1 − α) + (K + 1)γ) / (γM). |
| `bound_applied_median` | Median of that bound over rows. |
| `bound_applied_max` | Maximum of that bound over rows. |
| `bound_all_min` | Minimum over rows of its form over all N intervals, (max(α, 1 − α) + (K + 1)γ) / (γN) + (K + 1) max(α, 1 − α) / N. |
| `bound_all_median` | Median of the all-interval form over rows. |
| `bound_all_max` | Maximum of the all-interval form over rows. |
| `bound_stored_max_abs_diff` | Largest absolute difference between the recomputed bound and the bound stored in the trace rows. |
| `n_alpha_le0` | Intervals issued with α_t ≤ 0 (infinite intervals), summed over rows. |
| `n_alpha_ge1` | Intervals issued with α_t ≥ 1 (zero radius), summed over rows. |
| `n_alpha_ge1_covered` | Those of them that covered their target. |
| `n_rows_alpha_ge1` | Rows with at least one interval issued with α_t ≥ 1. |
| `n_rows_bound_not_applicable` | Rows where the bound does not apply (M = 0, or an interval issued with α_t ≥ 1 covered). |
| `max_ratio_applied_upper` | Largest ratio of the upper end of the bracket of the applied-label deviation to the bound (a value ≤ 1 proves the bound in every row). |
| `max_ratio_applied_lower` | Largest ratio of the lower end of the bracket to the bound. |
| `max_ratio_all` | Largest ratio of the deviation of the miss fraction over all N intervals from α to the all-interval form. |
| `n_rows_applied_proven` | Rows whose upper ratio is at most 1. |
| `n_rows_applied_violated` | Rows whose lower ratio exceeds 1 (proven violations). |
| `n_rows_applied_unresolved` | Rows between the two. |
| `n_rows_all_violated` | Rows whose all-interval ratio exceeds 1. |

### `results/summary/delay_outstanding.csv`

Distribution of K over the trace and seed rows.

| Column | Definition |
| --- | --- |
| `tag` | `main` (h = 1) or `h5` (h = 5, D = 0). |
| `method` | `aci` or `mondrian_aci`. |
| `delay_s` | Reporting delay D, s. |
| `regime` | Transfer regime (see Keys). `all` pools the regimes. |
| `K` | Value of K. |
| `n_trace_rows` | Rows with that K. |
| `share` | Fraction of the rows of the tag, method, delay and regime. |

### `results/summary/has_metrics.csv`

Streaming metrics per regime, variant and metric, over traces of per-trace values averaged over the seeds in which the trace was tested.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `variant` | Streaming variant (see Keys). |
| `metric` | Column of has_rows.csv.gz. |
| `n_traces` | Traces with a defined value. |
| `n_traces_undefined` | Traces without a defined value. |
| `mean` | Mean over traces; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `mean_ci_lo` and `mean_ci_hi`. |
| `mean_ci_lo` | Lower bound of the CI of `mean`. |
| `mean_ci_hi` | Upper bound of the CI of `mean`. |
| `median` | Median over traces. |

### `results/summary/has_tests.csv`

The 12 pre-specified paired Wilcoxon signed-rank tests on per-trace QoE per segment (MPC with the Mondrian-ACI bound against MPC with the ACI bound, MPC with the split CP bound and RobustMPC in the four shifted regimes with capacity traces), one BH family. Inputs and differences are rounded to 12 decimals and pairs with zero difference are dropped from the test.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `variant_a` | `mpc_mondrian_aci`. |
| `variant_b` | Comparator variant. |
| `metric` | `qoe_per_segment`. |
| `mean_a` | Mean of variant_a over the paired traces. |
| `mean_b` | Mean of variant_b over the paired traces. |
| `n` | Paired traces. |
| `p` | Two-sided Wilcoxon signed-rank p-value. |
| `stat` | Test statistic returned by scipy.stats.wilcoxon. |
| `rank_biserial` | Matched-pairs rank-biserial correlation (W+ − W−) / (W+ + W−) of the differences a − b. |
| `median_diff` | Median of the differences a − b over the paired traces. |
| `p_bh` | Benjamini-Hochberg adjusted p-value within the family. |
| `significant_bh` | True when p_bh ≤ 0.05. |

### `results/summary/horizon5.csv`

5-s horizon (seeds 0 to 4): labels applied when observed beside immediate feedback.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | `split`, `aci` or `mondrian_aci`. |
| `feedback` | `delayed_D0` (tag h5, labels applied when observed), `immediate` (locked abl_h5 rows, each label applied before the next interval), `none` (split CP, tag h5) or `none_locked` (split CP, locked abl_h5 rows). |
| `n_seeds` | Seeds with rows. |
| `n_traces` | Distinct test traces. |
| `n_trace_rows` | Trace and seed rows. |
| `n_instances` | Test instances, summed over the rows. |
| `marginal_coverage` | Coverage pooled over the test instances of each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `marginal_coverage_ci_lo` and `marginal_coverage_ci_hi`. |
| `marginal_coverage_ci_lo` | Lower bound of the CI of `marginal_coverage`. |
| `marginal_coverage_ci_hi` | Upper bound of the CI of `marginal_coverage`. |
| `trace_coverage` | Mean over traces of the per-trace coverage averaged over the seeds in which the trace was tested; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `trace_coverage_ci_lo` and `trace_coverage_ci_hi`. |
| `trace_coverage_ci_lo` | Lower bound of the CI of `trace_coverage`. |
| `trace_coverage_ci_hi` | Upper bound of the CI of `trace_coverage`. |
| `abs_dev` | Mean over traces of the absolute deviation of the seed-averaged trace coverage from 1 − α (Δ_s of the article); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `abs_dev_ci_lo` and `abs_dev_ci_hi`. |
| `abs_dev_ci_lo` | Lower bound of the CI of `abs_dev`. |
| `abs_dev_ci_hi` | Upper bound of the CI of `abs_dev`. |
| `frac_inf_instance` | Fraction of infinite intervals pooled within each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `frac_inf_instance_ci_lo` and `frac_inf_instance_ci_hi`. |
| `frac_inf_instance_ci_lo` | Lower bound of the CI of `frac_inf_instance`. |
| `frac_inf_instance_ci_hi` | Upper bound of the CI of `frac_inf_instance`. |
| `frac_inf_trace_rows` | Mean of the per-trace fraction of infinite intervals over the trace and seed rows. |
| `frac_inf_trace` | Mean over traces of the seed-averaged per-trace fraction of infinite intervals; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `frac_inf_trace_ci_lo` and `frac_inf_trace_ci_hi`. |
| `frac_inf_trace_ci_lo` | Lower bound of the CI of `frac_inf_trace`. |
| `frac_inf_trace_ci_hi` | Upper bound of the CI of `frac_inf_trace`. |
| `finite_coverage` | Coverage of the finite intervals pooled within each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `finite_coverage_ci_lo` and `finite_coverage_ci_hi`. |
| `finite_coverage_ci_lo` | Lower bound of the CI of `finite_coverage`. |
| `finite_coverage_ci_hi` | Upper bound of the CI of `finite_coverage`. |
| `finite_coverage_trace` | Mean over traces of the seed-averaged per-trace coverage of the finite intervals; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `finite_coverage_trace_ci_lo` and `finite_coverage_trace_ci_hi`. |
| `finite_coverage_trace_ci_lo` | Lower bound of the CI of `finite_coverage_trace`. |
| `finite_coverage_trace_ci_hi` | Upper bound of the CI of `finite_coverage_trace`. |
| `n_trace_rows_without_finite` | Rows without any finite interval. |
| `width_finite_median` | Median over traces of the seed-averaged per-trace median width of the finite intervals, Mbit/s. |
| `width_finite_mean_median` | Median over traces of the seed-averaged per-trace mean width of the finite intervals, Mbit/s. |
| `winkler_median` | Median over traces of the seed-averaged per-trace mean interval (Winkler) score over the finite intervals, Mbit/s. |
| `winkler_mean` | Mean over traces of the same per-trace values, Mbit/s; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `winkler_mean_ci_lo` and `winkler_mean_ci_hi`. |
| `winkler_mean_ci_lo` | Lower bound of the CI of `winkler_mean`. |
| `winkler_mean_ci_hi` | Upper bound of the CI of `winkler_mean`. |
| `n_inverted` | Finite intervals with upper end below lower end, summed over rows (−1 when not recorded). |
| `coverage_pos` | Coverage of the positive targets pooled within each seed, mean over seeds (empty when not recorded); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `coverage_pos_ci_lo` and `coverage_pos_ci_hi`. |
| `coverage_pos_ci_lo` | Lower bound of the CI of `coverage_pos`. |
| `coverage_pos_ci_hi` | Upper bound of the CI of `coverage_pos`. |
| `zero_share` | Share of targets equal to 0 Mbit/s pooled within each seed, mean over seeds (empty when not recorded). |

### `results/summary/locked_ablations.csv`

Locked ablation rows (seeds 0 to 4), one row per tag, regime, base model, α, γ and method; interval-summary columns are computed from the per-trace counts n × coverage and n × frac_inf, against the target 1 − α of the row.

| Column | Definition |
| --- | --- |
| `tag` | Ablation block (see locked_rows.csv.gz). |
| `regime` | Transfer regime (see Keys). |
| `base_model` | Point model. |
| `alpha` | α. |
| `gamma` | γ. |
| `method` | Interval method (see Keys). |
| `coverage_rows_mean` | Mean per-trace coverage over rows. |
| `marginal_coverage_rows_mean` | Mean cell marginal coverage over rows. |
| `median_width_rows_mean` | Mean per-trace median width over rows, Mbit/s. |
| `frac_inf_rows_mean` | Mean per-trace fraction of infinite intervals over rows. |
| `n_seeds` | Seeds. |
| `n_traces` | Test traces. |
| `marginal_coverage` | Coverage pooled over the test instances of each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `marginal_coverage_ci_lo` and `marginal_coverage_ci_hi`. |
| `marginal_coverage_ci_lo` | Lower bound of the CI of `marginal_coverage`. |
| `marginal_coverage_ci_hi` | Upper bound of the CI of `marginal_coverage`. |
| `trace_coverage` | Mean over traces of the per-trace coverage averaged over the seeds in which the trace was tested. |
| `abs_dev` | Mean over traces of the absolute deviation of the seed-averaged trace coverage from 1 − α (Δ_s of the article). |
| `frac_inf_instance` | Fraction of infinite intervals pooled within each seed, mean over seeds. |
| `frac_inf_trace_rows` | Mean of the per-trace fraction of infinite intervals over the trace and seed rows. |
| `frac_inf_trace` | Mean over traces of the seed-averaged per-trace fraction of infinite intervals. |
| `finite_coverage` | Coverage of the finite intervals pooled within each seed, mean over seeds. |
| `finite_coverage_trace` | Mean over traces of the seed-averaged per-trace coverage of the finite intervals. |
| `target` | 1 − α. |

### `results/summary/locked_downstream.csv`

Rate-selection proxy of the locked main rows.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `metric` | `overshoot_lb`, `util_lb`, `overshoot_point` or `util_point` (see locked_rows.csv.gz). |
| `mean` | Mean over traces of seed-averaged per-trace values; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `mean_ci_lo` and `mean_ci_hi`. |
| `mean_ci_lo` | Lower bound of the CI of `mean`. |
| `mean_ci_hi` | Upper bound of the CI of `mean`. |

### `results/summary/locked_groups.csv`

Speed-group coverage of the locked main rows.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `group` | Speed group (see Keys). |
| `n_seeds` | Seeds. |
| `coverage` | Mean over seeds of the stored per-seed group coverage; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `coverage_ci_lo` and `coverage_ci_hi`. |
| `coverage_ci_lo` | Lower bound of the CI of `coverage`. |
| `coverage_ci_hi` | Upper bound of the CI of `coverage`. |
| `coverage_rows_mean` | Mean over trace and seed rows (definition of scripts/analyze_locked.py). |

### `results/summary/locked_main.csv`

Locked main rows (seeds 0 to 19) at full precision, one row per regime and method.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `n_traces` | Test traces. |
| `n_seeds` | Seeds. |
| `marginal_coverage` | Mean over seeds of the stored cell marginal coverage (Table 1); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `marginal_coverage_ci_lo` and `marginal_coverage_ci_hi`. |
| `marginal_coverage_ci_lo` | Lower bound of the CI of `marginal_coverage`. |
| `marginal_coverage_ci_hi` | Upper bound of the CI of `marginal_coverage`. |
| `trace_coverage` | Mean over traces of the seed-averaged trace coverage; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `trace_coverage_ci_lo` and `trace_coverage_ci_hi`. |
| `trace_coverage_ci_lo` | Lower bound of the CI of `trace_coverage`. |
| `trace_coverage_ci_hi` | Upper bound of the CI of `trace_coverage`. |
| `abs_dev` | Mean over traces of Δ_s; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `abs_dev_ci_lo` and `abs_dev_ci_hi`. |
| `abs_dev_ci_lo` | Lower bound of the CI of `abs_dev`. |
| `abs_dev_ci_hi` | Upper bound of the CI of `abs_dev`. |
| `median_width` | Median over traces of the seed-averaged per-trace median width, infinite widths included, Mbit/s (Table 1). |
| `frac_inf_trace_rows` | Mean per-trace fraction of infinite intervals over rows. |
| `winkler_median_rows` | Median over rows of the stored per-trace median Winkler score, Mbit/s. |
| `rmse_point` | Mean over rows of the cell RMSE of the point prediction, Mbit/s. |
| `frac_inf_instance` | Fraction of infinite intervals pooled within seed, mean over seeds. |
| `finite_coverage` | Coverage of the finite intervals pooled within seed, mean over seeds. |
| `marginal_coverage_from_counts` | Marginal coverage recomputed from the per-trace counts (consistency check). |

### `results/summary/locked_tests.csv`

Confirmatory and secondary tests of the locked analysis on the per-trace absolute coverage deviation Δ_s, recomputed at full precision (a negative rank-biserial correlation or median difference favours method_a).

| Column | Definition |
| --- | --- |
| `family` | `confirmatory` (Mondrian-ACI against split CP in the five shifted regimes) or `secondary` (against ACI, Mondrian and CQR). |
| `regime` | Transfer regime (see Keys). |
| `method_a` | `mondrian_aci`. |
| `method_b` | Comparator. |
| `n` | Paired traces. |
| `p` | Two-sided Wilcoxon signed-rank p-value. |
| `stat` | Test statistic returned by scipy.stats.wilcoxon. |
| `rank_biserial` | Matched-pairs rank-biserial correlation (W+ − W−) / (W+ + W−) of the differences a − b. |
| `median_diff` | Median of the differences a − b over the paired traces. |
| `p_bh` | Benjamini-Hochberg adjusted p-value within the family. |
| `significant_bh` | True when p_bh ≤ 0.05. |

### `results/summary/main_groups.csv`

Speed-group coverage of the main configuration, one row per regime, method and group.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `group` | Speed group (see Keys). |
| `n_seeds` | Seeds with rows. |
| `n_test_min` | Smallest number of test instances of the group in one seed. |
| `n_test_max` | Largest number in one seed. |
| `n_test_mean` | Mean number per seed. |
| `n_test_total` | Total over seeds. |
| `coverage` | Coverage pooled over the group's test instances of each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `coverage_ci_lo` and `coverage_ci_hi`. |
| `coverage_ci_lo` | Lower bound of the CI of `coverage`. |
| `coverage_ci_hi` | Upper bound of the CI of `coverage`. |

### `results/summary/main_methods.csv`

Main configuration (h = 1, offline methods and ACI-type methods with D = 0, seeds 0 to 19), one row per regime and method. Rows of traces with fewer than 30 test instances are excluded here and in every summary below.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `n_seeds` | Seeds with rows. |
| `n_traces` | Distinct test traces. |
| `n_trace_rows` | Trace and seed rows. |
| `n_instances` | Test instances, summed over the rows. |
| `marginal_coverage` | Coverage pooled over the test instances of each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `marginal_coverage_ci_lo` and `marginal_coverage_ci_hi`. |
| `marginal_coverage_ci_lo` | Lower bound of the CI of `marginal_coverage`. |
| `marginal_coverage_ci_hi` | Upper bound of the CI of `marginal_coverage`. |
| `trace_coverage` | Mean over traces of the per-trace coverage averaged over the seeds in which the trace was tested; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `trace_coverage_ci_lo` and `trace_coverage_ci_hi`. |
| `trace_coverage_ci_lo` | Lower bound of the CI of `trace_coverage`. |
| `trace_coverage_ci_hi` | Upper bound of the CI of `trace_coverage`. |
| `abs_dev` | Mean over traces of the absolute deviation of the seed-averaged trace coverage from 1 − α (Δ_s of the article); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `abs_dev_ci_lo` and `abs_dev_ci_hi`. |
| `abs_dev_ci_lo` | Lower bound of the CI of `abs_dev`. |
| `abs_dev_ci_hi` | Upper bound of the CI of `abs_dev`. |
| `frac_inf_instance` | Fraction of infinite intervals pooled within each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `frac_inf_instance_ci_lo` and `frac_inf_instance_ci_hi`. |
| `frac_inf_instance_ci_lo` | Lower bound of the CI of `frac_inf_instance`. |
| `frac_inf_instance_ci_hi` | Upper bound of the CI of `frac_inf_instance`. |
| `frac_inf_trace_rows` | Mean of the per-trace fraction of infinite intervals over the trace and seed rows. |
| `frac_inf_trace` | Mean over traces of the seed-averaged per-trace fraction of infinite intervals; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `frac_inf_trace_ci_lo` and `frac_inf_trace_ci_hi`. |
| `frac_inf_trace_ci_lo` | Lower bound of the CI of `frac_inf_trace`. |
| `frac_inf_trace_ci_hi` | Upper bound of the CI of `frac_inf_trace`. |
| `finite_coverage` | Coverage of the finite intervals pooled within each seed, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `finite_coverage_ci_lo` and `finite_coverage_ci_hi`. |
| `finite_coverage_ci_lo` | Lower bound of the CI of `finite_coverage`. |
| `finite_coverage_ci_hi` | Upper bound of the CI of `finite_coverage`. |
| `finite_coverage_trace` | Mean over traces of the seed-averaged per-trace coverage of the finite intervals; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `finite_coverage_trace_ci_lo` and `finite_coverage_trace_ci_hi`. |
| `finite_coverage_trace_ci_lo` | Lower bound of the CI of `finite_coverage_trace`. |
| `finite_coverage_trace_ci_hi` | Upper bound of the CI of `finite_coverage_trace`. |
| `n_trace_rows_without_finite` | Rows without any finite interval. |
| `width_finite_median` | Median over traces of the seed-averaged per-trace median width of the finite intervals, Mbit/s. |
| `width_finite_mean_median` | Median over traces of the seed-averaged per-trace mean width of the finite intervals, Mbit/s. |
| `winkler_median` | Median over traces of the seed-averaged per-trace mean interval (Winkler) score over the finite intervals, Mbit/s. |
| `winkler_mean` | Mean over traces of the same per-trace values, Mbit/s; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the traces in `winkler_mean_ci_lo` and `winkler_mean_ci_hi`. |
| `winkler_mean_ci_lo` | Lower bound of the CI of `winkler_mean`. |
| `winkler_mean_ci_hi` | Upper bound of the CI of `winkler_mean`. |
| `n_inverted` | Finite intervals with upper end below lower end, summed over rows (−1 when not recorded). |
| `coverage_pos` | Coverage of the positive targets pooled within each seed, mean over seeds (empty when not recorded); 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `coverage_pos_ci_lo` and `coverage_pos_ci_hi`. |
| `coverage_pos_ci_lo` | Lower bound of the CI of `coverage_pos`. |
| `coverage_pos_ci_hi` | Upper bound of the CI of `coverage_pos`. |
| `zero_share` | Share of targets equal to 0 Mbit/s pooled within each seed, mean over seeds (empty when not recorded). |
| `worst_group` | Speed group with the lowest seed-averaged coverage. |
| `worst_group_coverage` | Its coverage, mean over seeds; 95% percentile bootstrap CI (10,000 resamples, seed 0) over the seed values in `worst_group_coverage_ci_lo` and `worst_group_coverage_ci_hi`. |
| `worst_group_coverage_ci_lo` | Lower bound of the CI of `worst_group_coverage`. |
| `worst_group_coverage_ci_hi` | Upper bound of the CI of `worst_group_coverage`. |
| `max_group_abs_dev` | Largest absolute deviation of a group coverage from 0.90. |
| `min_group_n_test` | Smallest number of test instances of a group in one seed. |

### `results/summary/pool_sizes.csv`

Calibration score-pool sizes of the speed groups (scripts/pool_sizes.py), one row per regime, seed and group of the main configuration.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `seed` | Seed of the trace-level split. |
| `group` | Speed group (see Keys). |
| `n_pool` | Calibration instances in the group (size of its score pool). |
| `n_pooled` | Calibration instances of the cell. |
| `fallback` | True when n_pool < 50, so that Mondrian and Mondrian-ACI use the pooled scores for the group. |

### `results/summary/runtime.csv`

Compute time per tag (timing log) and per batch (batch timeline).

| Column | Definition |
| --- | --- |
| `source` | `timing_log` (results/amendment1/timing_log.csv) or `timeline` (results/amendment1/batch_timeline.txt). |
| `mode` | `full` or `reproduce_only` (empty for timeline rows). |
| `tag` | Run tag, or batch step of the timeline (`total` spans its first to last line). |
| `n_cells` | Cells (empty for timeline rows). |
| `n_ok` | Cells with status ok. |
| `n_workers` | Distinct worker processes. |
| `worker_seconds_total` | Sum of seconds_total over cells. |
| `worker_seconds_mean` | Mean of seconds_total. |
| `worker_seconds_max` | Maximum of seconds_total. |
| `first_start_utc` | First start (UTC). |
| `last_finish_utc` | Last finish (UTC). |
| `wall_minutes` | Minutes from first start to last finish. |
| `busy_minutes` | Length of the union of the cell intervals in minutes (differs from wall_minutes when cells ran in separate launches). |

### `results/summary/timestamp.csv`

Timestamp-based features (prefix `ts_`, tag ts, D = 0) beside the row-based locked main rows of the same seeds 0 to 4 (prefix `row_`), one row per regime and method.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `ts_n_seeds` | Timestamp-based: Seeds. |
| `ts_n_traces` | Timestamp-based: Test traces. |
| `ts_n_instances` | Timestamp-based: Test instances. |
| `ts_marginal_coverage` | Timestamp-based: Coverage pooled within seed, mean over seeds. |
| `ts_marginal_coverage_ci_lo` | Timestamp-based: Lower bound of its 95% percentile bootstrap CI over the seed values. |
| `ts_marginal_coverage_ci_hi` | Timestamp-based: Upper bound of that CI. |
| `ts_trace_coverage` | Timestamp-based: Mean over traces of the seed-averaged trace coverage. |
| `ts_frac_inf_instance` | Timestamp-based: Fraction of infinite intervals pooled within seed, mean over seeds. |
| `ts_frac_inf_trace_rows` | Timestamp-based: Mean per-trace fraction of infinite intervals over rows. |
| `ts_frac_inf_trace` | Timestamp-based: Mean over traces of the seed-averaged per-trace fraction. |
| `row_n_seeds` | Row-based: Seeds. |
| `row_n_traces` | Row-based: Test traces. |
| `row_n_instances` | Row-based: Test instances. |
| `row_marginal_coverage` | Row-based: Coverage pooled within seed, mean over seeds. |
| `row_marginal_coverage_ci_lo` | Row-based: Lower bound of its 95% percentile bootstrap CI over the seed values. |
| `row_marginal_coverage_ci_hi` | Row-based: Upper bound of that CI. |
| `row_trace_coverage` | Row-based: Mean over traces of the seed-averaged trace coverage. |
| `row_frac_inf_instance` | Row-based: Fraction of infinite intervals pooled within seed, mean over seeds. |
| `row_frac_inf_trace_rows` | Row-based: Mean per-trace fraction of infinite intervals over rows. |
| `row_frac_inf_trace` | Row-based: Mean over traces of the seed-averaged per-trace fraction. |

### `results/summary/locked/ablations_abl_alpha05.csv`

Ablation abl_alpha05 of scripts/analyze_locked.py: means over trace and seed rows, rounded to 4 decimals.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `base_model` | Point model. |
| `alpha` | α. |
| `gamma` | γ. |
| `method` | Interval method (see Keys). |
| `coverage` | Mean per-trace coverage. |
| `marginal_coverage` | Mean cell marginal coverage. |
| `median_width` | Mean per-trace median width, Mbit/s. |
| `frac_inf` | Mean per-trace fraction of infinite intervals. |

### `results/summary/locked/ablations_abl_base.csv`

Ablation abl_base of scripts/analyze_locked.py: means over trace and seed rows, rounded to 4 decimals.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `base_model` | Point model. |
| `alpha` | α. |
| `gamma` | γ. |
| `method` | Interval method (see Keys). |
| `coverage` | Mean per-trace coverage. |
| `marginal_coverage` | Mean cell marginal coverage. |
| `median_width` | Mean per-trace median width, Mbit/s. |
| `frac_inf` | Mean per-trace fraction of infinite intervals. |

### `results/summary/locked/ablations_abl_gamma.csv`

Ablation abl_gamma of scripts/analyze_locked.py: means over trace and seed rows, rounded to 4 decimals.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `base_model` | Point model. |
| `alpha` | α. |
| `gamma` | γ. |
| `method` | Interval method (see Keys). |
| `coverage` | Mean per-trace coverage. |
| `marginal_coverage` | Mean cell marginal coverage. |
| `median_width` | Mean per-trace median width, Mbit/s. |
| `frac_inf` | Mean per-trace fraction of infinite intervals. |

### `results/summary/locked/ablations_abl_h5.csv`

Ablation abl_h5 of scripts/analyze_locked.py: means over trace and seed rows, rounded to 4 decimals.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `base_model` | Point model. |
| `alpha` | α. |
| `gamma` | γ. |
| `method` | Interval method (see Keys). |
| `coverage` | Mean per-trace coverage. |
| `marginal_coverage` | Mean cell marginal coverage. |
| `median_width` | Mean per-trace median width, Mbit/s. |
| `frac_inf` | Mean per-trace fraction of infinite intervals. |

### `results/summary/locked/downstream.csv`

Rate-selection proxy of scripts/analyze_locked.py, 4 decimals.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `metric` | `overshoot_lb`, `util_lb`, `overshoot_point` or `util_point`. |
| `mean` | Mean over traces of seed-averaged per-trace values. |
| `lo` | Lower bound of its 95% CI over traces. |
| `hi` | Upper bound of its 95% CI over traces. |

### `results/summary/locked/group_coverage.csv`

Speed-group coverage of scripts/analyze_locked.py.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `group` | Speed group (see Keys). |
| `coverage_mean` | Mean over trace and seed rows of the cell's group coverage, 4 decimals. |
| `n_cells` | Number of trace and seed rows averaged. |

### `results/summary/locked/summary_main.csv`

Main summary of scripts/analyze_locked.py, rounded as written by that script.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method` | Interval method (see Keys). |
| `n_traces` | Test traces. |
| `n_seeds` | Seeds. |
| `marginal_coverage` | Mean over seeds of the cell marginal coverage. |
| `marg_lo` | Lower bound of its 95% CI over seeds. |
| `marg_hi` | Upper bound of its 95% CI over seeds. |
| `trace_coverage` | Mean over traces of the seed-averaged trace coverage. |
| `trace_lo` | Lower bound of its 95% CI over traces. |
| `trace_hi` | Upper bound of its 95% CI over traces. |
| `abs_dev` | Mean over traces of Δ_s. |
| `dev_lo` | Lower bound of its 95% CI over traces. |
| `dev_hi` | Upper bound of its 95% CI over traces. |
| `median_width` | Median over traces of the seed-averaged per-trace median width, Mbit/s. |
| `frac_inf` | Mean per-trace fraction of infinite intervals over rows. |
| `winkler_median` | Median over rows of the stored per-trace median Winkler score, Mbit/s. |
| `rmse_point` | Mean over rows of the cell RMSE, Mbit/s. |

### `results/summary/locked/tests_confirmatory.csv`

The five confirmatory tests (mondrian-aci against split cp) on Δ_s, as written by scripts/analyze_locked.py.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method_a` | `mondrian_aci`. |
| `method_b` | Comparator. |
| `n` | Paired traces. |
| `p` | Two-sided Wilcoxon signed-rank p-value. |
| `stat` | Test statistic returned by scipy.stats.wilcoxon. |
| `rank_biserial` | Matched-pairs rank-biserial correlation (W+ − W−) / (W+ + W−) of the differences a − b. |
| `median_diff` | Median of the differences a − b over the paired traces. |
| `p_bh` | Benjamini-Hochberg adjusted p-value within the family. |
| `significant_bh` | True when p_bh ≤ 0.05. |

### `results/summary/locked/tests_secondary.csv`

The 15 secondary tests (mondrian-aci against aci, mondrian and cqr) on Δ_s, as written by scripts/analyze_locked.py.

| Column | Definition |
| --- | --- |
| `regime` | Transfer regime (see Keys). |
| `method_a` | `mondrian_aci`. |
| `method_b` | Comparator. |
| `n` | Paired traces. |
| `p` | Two-sided Wilcoxon signed-rank p-value. |
| `stat` | Test statistic returned by scipy.stats.wilcoxon. |
| `rank_biserial` | Matched-pairs rank-biserial correlation (W+ − W−) / (W+ + W−) of the differences a − b. |
| `median_diff` | Median of the differences a − b over the paired traces. |
| `p_bh` | Benjamini-Hochberg adjusted p-value within the family. |
| `significant_bh` | True when p_bh ≤ 0.05. |

## Licence

The code is released under the MIT licence (`LICENSE`). The datasets keep their own licences
(see Data sources and licences).
