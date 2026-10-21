"""Download the public UCC MISL 4G and 5G trace datasets, verify them against the
checksum manifest and extract them into the folder layout read by src/data_loading.py.

Datasets:
  4G  LTE_Dataset.zip: D. Raca, J. J. Quinlan, A. H. Zahran and C. J. Sreenan, ACM MMSys
      2018, DOI 10.1145/3204949.3208123. Zenodo record DOI 10.5281/zenodo.1219679,
      licence CC BY 4.0.
  5G  5G-production-dataset.zip: D. Raca, D. Leahy, C. J. Sreenan and J. J. Quinlan, ACM
      MMSys 2020, DOI 10.1145/3339825.3394938. Repository github.com/uccmisl/5Gdataset,
      licence GPL-3.0.

The expected size and SHA-256 of each archive are read from data/raw/download_manifest.json;
this script never writes that file. An archive already in data/raw/ is not downloaded
again. A download is written to <name>.part and renamed only when its size and SHA-256
match the manifest; an archive in data/raw/ that does not match stops the script.

Extraction (every run, overwriting earlier copies of the same files):
  LTE_Dataset.zip            -> data/raw/4G/Dataset/<mobility>/<trace>.csv
  5G-production-dataset.zip  -> data/raw/5G/5G-production-dataset/<application>/<mobility>/<trace>.csv
macOS folder metadata (__MACOSX/ entries and .DS_Store files) is not extracted. An entry
outside the expected top-level folder, or with an absolute path or a '..' component,
stops the script before anything is written.

Usage: python scripts/download_data.py [--offline]
  --offline  use only the archives already in data/raw/ (no network access)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "download_manifest.json"

# archive name -> target folder under data/raw/, top-level folder inside the archive,
# download URLs tried in order
ARCHIVES = {
    "LTE_Dataset.zip": {
        "target": "4G", "top": "Dataset",
        "urls": ["https://zenodo.org/api/records/1219679/files/LTE_Dataset.zip/content",
                 "https://zenodo.org/records/1219679/files/LTE_Dataset.zip?download=1"]},
    "5G-production-dataset.zip": {
        "target": "5G", "top": "5G-production-dataset",
        "urls": ["https://raw.githubusercontent.com/uccmisl/5Gdataset/master/"
                 "5G-production-dataset.zip",
                 "https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip"]},
}
METADATA_NAMES = {"__MACOSX", ".DS_Store"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    """{archive name: {"bytes": int, "sha256": str}} from the checksum manifest."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {}
    for e in entries:
        name = PurePosixPath(e["file"]).name
        expected[name] = {"bytes": int(e["bytes"]), "sha256": str(e["sha256"]).lower()}
    missing = sorted(set(ARCHIVES) - set(expected))
    if missing:
        raise SystemExit(f"{path}: no entry for {', '.join(missing)}")
    return expected


def matches(path: Path, expected: dict) -> tuple[bool, int, str]:
    """(size and SHA-256 equal the manifest entry, size, SHA-256) of a file."""
    size = path.stat().st_size
    digest = sha256_file(path)
    return size == expected["bytes"] and digest == expected["sha256"], size, digest


def download(urls: list[str], dest: Path, expected: dict) -> None:
    """Fetch the first URL whose content matches the manifest entry into dest."""
    import truststore
    truststore.inject_into_ssl()
    import requests

    part = dest.with_name(dest.name + ".part")
    for url in urls:
        print(f"{dest.name}: downloading {url}")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        except requests.RequestException as e:
            print(f"  failed: {e}")
            part.unlink(missing_ok=True)
            continue
        ok, size, digest = matches(part, expected)
        if ok:
            part.replace(dest)
            return
        print(f"  {size} bytes, sha256 {digest}: does not match the manifest; discarded")
        part.unlink()
    raise SystemExit(f"{dest.name}: no source delivered the archive listed in the manifest")


def member_parts(name: str, top: str) -> tuple[str, ...] | None:
    """Path components of an archive entry below the target folder, or None for macOS
    metadata; stops the script for an entry outside `top` or with an unsafe path."""
    p = PurePosixPath(name.replace("\\", "/"))
    parts = tuple(x for x in p.parts if x not in ("", "."))
    if any(x in METADATA_NAMES for x in parts):
        return None
    if p.is_absolute() or ".." in parts or (parts and ":" in parts[0]):
        raise SystemExit(f"unsafe path in archive: {name!r}")
    if not parts or parts[0] != top:
        raise SystemExit(f"entry {name!r} is outside the expected folder {top}/")
    return parts


def extract(archive: Path, target: Path, top: str) -> int:
    """Extract the entries of archive below top/ into target; returns the number of
    files written."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        plan = []
        for info in zf.infolist():
            parts = member_parts(info.filename, top)
            if parts is None:
                continue
            dest = target.joinpath(*parts)
            if target != dest.resolve() and target not in dest.resolve().parents:
                raise SystemExit(f"unsafe path in archive: {info.filename!r}")
            plan.append((info, dest))
        n_files = 0
        for info, dest in plan:
            try:
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            except OSError as e:
                raise SystemExit(f"cannot write {dest} ({len(str(dest))} characters): {e}. "
                                 "Without long-path support, Windows limits file paths to 259 "
                                 "characters and folder paths to 247; use a shorter folder "
                                 "path.") from e
            n_files += 1
    return n_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true",
                    help="use only the archives already in data/raw/")
    args = ap.parse_args()

    expected = load_manifest(MANIFEST)
    RAW.mkdir(parents=True, exist_ok=True)
    for name, spec in ARCHIVES.items():
        dest = RAW / name
        if dest.exists():
            print(f"{name}: present, not downloaded")
        elif args.offline:
            raise SystemExit(f"{dest} is missing (--offline)")
        else:
            download(spec["urls"], dest, expected[name])
        ok, size, digest = matches(dest, expected[name])
        if not ok:
            raise SystemExit(f"{dest}: {size} bytes, sha256 {digest}; the manifest lists "
                             f"{expected[name]['bytes']} bytes, sha256 "
                             f"{expected[name]['sha256']}. Remove the file and run again.")
        print(f"{name}: {size} bytes, sha256 {digest} matches the manifest")
        n = extract(dest, RAW / spec["target"], spec["top"])
        print(f"{name}: {n} files extracted to {RAW / spec['target'] / spec['top']}")

    from src.data_loading import discover_traces
    traces = discover_traces(RAW)
    n4 = sum(t.tech == "4G" for t in traces)
    n5 = sum(t.tech == "5G" for t in traces)
    print(f"src.data_loading.discover_traces finds {n4} 4G and {n5} 5G trace files")
    if n4 == 0 or n5 == 0:
        raise SystemExit("no traces found in the extracted layout")


if __name__ == "__main__":
    main()
