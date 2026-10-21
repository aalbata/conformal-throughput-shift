# Data

The datasets are not included in this package. `python scripts/download_data.py` downloads
the two trace archives from their publishers, checks each archive against
`raw/download_manifest.json` (size and SHA-256 of the archives used for the article; the
script reads this file and never writes it) and extracts them:

| Archive | Extracted to | Source | Licence |
|---|---|---|---|
| `LTE_Dataset.zip` | `raw/4G/Dataset/<mobility>/<trace>.csv` | 4G dataset of Raca et al., ACM MMSys 2018; Zenodo, DOI 10.5281/zenodo.1219679 | CC BY 4.0 |
| `5G-production-dataset.zip` | `raw/5G/5G-production-dataset/<application>/<mobility>/<trace>.csv` | 5G dataset of Raca et al., ACM MMSys 2020; github.com/uccmisl/5Gdataset | GPL-3.0 |

`python scripts/download_data.py --offline` uses archives placed in `raw/` by hand.
`python scripts/prepare_data.py --horizon 1`, `python scripts/prepare_data.py --horizon 5`
and `python scripts/prepare_timestamps.py` write the feature tables to `processed/`.
`python scripts/fetch_ladder.py` downloads the UCC 4K DASH segment-size archive (Quinlan and
Sreenan, ACM MMSys 2018; Zenodo, DOI 10.5281/zenodo.1219788; CC BY 4.0) to `external/` and
writes the per-segment sizes of Big Buck Bunny (x264, 4-s segments) to `ladder/`.
