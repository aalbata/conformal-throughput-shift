"""Differential test of the rate-based rule against the Sabre simulator.

Sabre (Spiteri, Sitaraman and Sparacio, ACM MMSys 2018 and ACM TOMM 2019;
github.com/UMass-LIDS/sabre, BSD-2-Clause) is run unmodified as a subprocess with a
custom bitrate rule that implements the rate-based rule of src.has: highest bitrate
not above the harmonic mean of the last five segment throughputs, lowest bitrate
for the first segment. Sabre options: 30-s buffer (-b 30), no abandonment (-noa),
a one-segment sliding window (-ma sliding -ws 1) so that its throughput estimate is
the throughput of the last segment, and verbose per-segment output (-v).

Both simulators receive the same per-second capacity trace (80 ms latency per
request) and the same segment sizes; the trace is long enough that Sabre does not
wrap around it. Compared per segment: representation and download time; in total:
rebuffering time and number of rebuffering events.

Skipped unless the environment variable SABRE_PATH names Sabre's src/sabre.py.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.has import LADDER_KBPS, simulate  # noqa: E402

SABRE = os.environ.get("SABRE_PATH", "")
pytestmark = pytest.mark.skipif(not SABRE or not Path(SABRE).is_file(),
                                reason="SABRE_PATH does not name sabre.py")

ABR_MODULE = '''
class HarmonicRate:
    window = 5

    def __init__(self, config):
        self.throughputs = []

    def get_first_quality(self):
        return 0

    def get_quality_delay(self, segment_index):
        info = getattr(self, "session")
        self.throughputs.append(info.get_throughput())
        last = self.throughputs[-self.window:]
        estimate = len(last) / sum(1.0 / x for x in last)
        bitrates = info.manifest.bitrates
        quality = 0
        for q in range(len(bitrates)):
            if bitrates[q] <= estimate:
                quality = q
        return (quality, 0)

    def report_delay(self, delay):
        pass

    def report_download(self, metrics, is_replacement):
        pass

    def report_seek(self, where):
        pass

    def check_abandon(self, progress, buffer_level):
        return None
'''

SEGMENT_LINE = re.compile(r"^\[\d+-\d+\]\s+(\d+): q=(\d+) s=\d+/\d+ t=(\d+)=\d+\+\d+")


def run_sabre(tmp_path: Path, capacity: np.ndarray, segment_bits: np.ndarray):
    """Run Sabre; return per-segment (quality, download ms), rebuffer s, rebuffer events."""
    network = [{"duration_ms": 1000, "bandwidth_kbps": float(c) * 1000.0, "latency_ms": 80}
               for c in capacity]
    movie = {"segment_duration_ms": 4000, "bitrates_kbps": list(LADDER_KBPS),
             "segment_sizes_bits": segment_bits.tolist()}
    (tmp_path / "network.json").write_text(json.dumps(network), encoding="utf-8")
    (tmp_path / "movie.json").write_text(json.dumps(movie), encoding="utf-8")
    (tmp_path / "HarmonicRate.py").write_text(ABR_MODULE, encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(Path(SABRE).resolve()), "-n", "network.json", "-m", "movie.json",
         "-a", "HarmonicRate.py", "-b", "30", "-noa", "-ma", "sliding", "-ws", "1", "-v"],
        cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    segments = {}
    for line in out.splitlines():
        hit = SEGMENT_LINE.match(line)
        if hit:
            segments[int(hit.group(1))] = (int(hit.group(2)), int(hit.group(3)))
    rebuffer = float(re.search(r"^total rebuffer: ([0-9.]+)$", out, re.M).group(1))
    events = int(float(re.search(r"^total rebuffer events: ([0-9.]+)$", out, re.M).group(1)))
    return [segments[i] for i in range(len(segment_bits))], rebuffer, events


def traces():
    rng = np.random.default_rng(11)
    smooth = rng.uniform(1.0, 40.0, 600)
    rough = rng.gamma(1.2, 8.0, 600)
    rough[60:75] = 0.0
    rough[150:190] = 0.3
    rough[rng.random(600) < 0.05] = 0.0
    # alternating high (20-60 Mbit/s) and low (0.2-3 Mbit/s) blocks: a high rung chosen
    # at the end of a high block meets a low block and the download outlasts the buffer
    blocks = []
    while sum(len(b) for b in blocks) < 900:
        blocks.append(rng.uniform(20.0, 60.0, int(rng.integers(30, 80))))
        blocks.append(rng.uniform(0.2, 3.0, int(rng.integers(20, 70))))
    drops = np.concatenate(blocks)[:900]
    drops[rng.random(900) < 0.05] = 0.0
    return {"smooth": smooth, "rough": rough, "drops": drops}


@pytest.mark.parametrize("name,n_seg", [("smooth", 40), ("rough", 40), ("drops", 120)])
def test_rate_rule_matches_sabre(tmp_path, name, n_seg):
    capacity = traces()[name]
    rng = np.random.default_rng(12)
    segment_bits = np.round(rng.uniform(0.5, 1.5, (n_seg, len(LADDER_KBPS)))
                            * np.array(LADDER_KBPS) * 4000.0)
    sabre_segments, sabre_rebuffer, sabre_events = run_sabre(tmp_path, capacity, segment_bits)

    _, log = simulate(capacity, 0, None, LADDER_KBPS, segment_bits, "rate", "hm")
    assert np.all(np.isfinite(log["finish_s"][:n_seg]))
    reps = log["rep"][:n_seg].tolist()
    ms = ((log["finish_s"] - log["request_s"])[:n_seg] * 1000.0).tolist()
    assert reps == [q for q, _ in sabre_segments]
    for mine, (_, theirs) in zip(ms, sabre_segments):
        assert abs(mine - theirs) < 1.0
    assert log["rebuffer_s"][:n_seg].sum() == pytest.approx(sabre_rebuffer, abs=1e-5)
    assert int((log["rebuffer_s"][:n_seg] > 0).sum()) == sabre_events
