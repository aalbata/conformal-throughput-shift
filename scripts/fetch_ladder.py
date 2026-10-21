"""Download the UCC 4K DASH segment-size archive and extract the Big Buck Bunny ladder.

Source: J. J. Quinlan and C. J. Sreenan, "Multi-profile Ultra High Definition (UHD)
AVC and HEVC 4K DASH Datasets," ACM MMSys 2018; data on Zenodo,
DOI 10.5281/zenodo.1219788, licence CC BY 4.0.

Usage: python scripts/fetch_ladder.py
Downloads 4K_segment_distribution.tar into data/external/ (skipped if present), verifies
its size and SHA-256, extracts the per-segment byte costs of Big Buck Bunny, x264,
4-s segments, main profile, into data/ladder/, verifies that file's SHA-256, checks
that the six representations of the streaming simulation are present, and writes
data/ladder/ladder_manifest.json.
"""
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.has import LADDER_KBPS, load_ladder  # noqa: E402

URL = "https://zenodo.org/api/records/1219788/files/4K_segment_distribution.tar/content"
DOI = "10.5281/zenodo.1219788"
TAR_NAME = "4K_segment_distribution.tar"
TAR_BYTES = 2513408
TAR_SHA256 = "672c23e90f762c63b0c2710714e18f45082c4dde551d14f4d75bf848dddae97d"
MEMBER = ("4K_segment_distribution/4K_segment_distribution_output_x264_4sec/bbb/"
          "DASH_Files/main/segmentByteCostinColumns-bbb-x264.csv")
CSV_NAME = "segmentByteCostinColumns-bbb-x264-4sec-main.csv"
CSV_SHA256 = "26230d258686cb33e1fa589e3ec51a79a3c458444dddbef4fb9705efb62ec7bb"

EXTERNAL = ROOT / "data" / "external"
LADDER_DIR = ROOT / "data" / "ladder"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path) -> None:
    """Stream URL into dest.part and rename it to dest once the bytes are complete."""
    import truststore
    truststore.inject_into_ssl()
    import requests

    part = dest.with_name(dest.name + ".part")
    print(f"downloading {URL}")
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    part.replace(dest)


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = EXTERNAL / TAR_NAME
    if tar_path.exists():
        print(f"{tar_path.name}: already present, skipping download")
    else:
        download(tar_path)
    size = tar_path.stat().st_size
    digest = sha256_file(tar_path)
    if size != TAR_BYTES or digest != TAR_SHA256:
        raise SystemExit(f"{tar_path}: size {size} / sha256 {digest} do not match the "
                         f"expected {TAR_BYTES} / {TAR_SHA256}")
    print(f"{tar_path.name}: {size} bytes, sha256 verified")

    with tarfile.open(tar_path) as tf:
        member = tf.getmember(MEMBER)
        if not member.isfile():
            raise SystemExit(f"{MEMBER} is not a regular file in the archive")
        data = tf.extractfile(member).read()
    csv_digest = hashlib.sha256(data).hexdigest()
    if csv_digest != CSV_SHA256:
        raise SystemExit(f"{MEMBER}: sha256 {csv_digest} does not match {CSV_SHA256}")
    csv_path = LADDER_DIR / CSV_NAME
    csv_path.write_bytes(data)
    print(f"wrote {csv_path} ({len(data)} bytes, sha256 verified)")

    ladder = load_ladder(csv_path, LADDER_KBPS)
    mean_kbps = (ladder.segment_bits.mean(axis=0) / ladder.segment_s / 1000.0).round(1)
    manifest = {
        "source_doi": DOI,
        "source_url": URL,
        "licence": "CC BY 4.0",
        "archive": {"file": f"data/external/{TAR_NAME}", "bytes": TAR_BYTES,
                    "sha256": TAR_SHA256},
        "member": MEMBER,
        "ladder_csv": {"file": f"data/ladder/{CSV_NAME}", "bytes": len(data),
                       "sha256": CSV_SHA256},
        "video": "Big Buck Bunny, x264, main profile",
        "segment_s": ladder.segment_s,
        "n_segments": int(ladder.segment_bits.shape[0]),
        "representations_kbps": [int(x) for x in ladder.bitrates_kbps],
        "mean_segment_bitrate_kbps": [float(x) for x in mean_kbps],
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (LADDER_DIR / "ladder_manifest.json").write_text(json.dumps(manifest, indent=1),
                                                     encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
