"""Checksum verification and extraction of scripts/download_data.py on small archives."""
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def dd():
    spec = importlib.util.spec_from_file_location("download_data",
                                                  ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zip(path: Path, entries: dict) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _entry(path: Path) -> dict:
    data = path.read_bytes()
    return {"file": f"data/raw/{path.name}", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _archives(raw: Path) -> list[Path]:
    four = _zip(raw / "LTE_Dataset.zip", {
        "Dataset/": "", "Dataset/car/": "",
        "Dataset/car/A_2017.12.04_15.24.12.csv": "Timestamp,DL_bitrate\n",
        "Dataset/static/B_2018.01.27_10.58.49.csv": "Timestamp,DL_bitrate\n"})
    five = _zip(raw / "5G-production-dataset.zip", {
        "5G-production-dataset/Download/Driving/B_2019.12.16_13.40.04.csv": "x\n",
        "5G-production-dataset/.DS_Store": "meta",
        "__MACOSX/5G-production-dataset/._Download": "meta"})
    return [four, five]


def test_member_parts_skips_metadata_and_rejects_unsafe_paths(dd):
    assert dd.member_parts("Dataset/car/a.csv", "Dataset") == ("Dataset", "car", "a.csv")
    assert dd.member_parts("__MACOSX/Dataset/._a.csv", "Dataset") is None
    assert dd.member_parts("Dataset/.DS_Store", "Dataset") is None
    for bad in ("../Dataset/a.csv", "Dataset/../../a.csv", "/Dataset/a.csv",
                "C:/Dataset/a.csv", "Other/a.csv"):
        with pytest.raises(SystemExit):
            dd.member_parts(bad, "Dataset")


def test_extract_writes_the_layout_of_data_loading(dd, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    four, five = _archives(raw)
    assert dd.extract(four, raw / "4G", "Dataset") == 2
    assert dd.extract(five, raw / "5G", "5G-production-dataset") == 1
    assert (raw / "4G" / "Dataset" / "car" / "A_2017.12.04_15.24.12.csv").read_text() == \
        "Timestamp,DL_bitrate\n"
    assert not (raw / "5G" / "__MACOSX").exists()
    assert not (raw / "5G" / "5G-production-dataset" / ".DS_Store").exists()
    from src.data_loading import discover_traces
    ids = sorted(t.trace_id for t in discover_traces(raw))
    assert ids == ["4G/car/A_2017.12.04_15.24.12", "4G/static/B_2018.01.27_10.58.49",
                   "5G/Download/Driving/B_2019.12.16_13.40.04"]


def test_extract_refuses_an_archive_with_an_unsafe_entry(dd, tmp_path):
    bad = _zip(tmp_path / "bad.zip", {"Dataset/a.csv": "x", "Dataset/../../evil.csv": "x"})
    with pytest.raises(SystemExit):
        dd.extract(bad, tmp_path / "out", "Dataset")
    assert not (tmp_path / "out" / "Dataset" / "a.csv").exists()
    assert not (tmp_path / "evil.csv").exists()


def test_offline_run_verifies_against_the_manifest_without_rewriting_it(dd, tmp_path,
                                                                        monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    archives = _archives(raw)
    manifest = raw / "download_manifest.json"
    manifest.write_text(json.dumps([_entry(p) for p in archives], indent=1), encoding="utf-8")
    before = manifest.read_bytes()
    monkeypatch.setattr(dd, "RAW", raw)
    monkeypatch.setattr(dd, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--offline"])
    dd.main()
    assert manifest.read_bytes() == before
    assert (raw / "5G" / "5G-production-dataset" / "Download" / "Driving").is_dir()

    with zipfile.ZipFile(archives[0], "a") as zf:
        zf.writestr("Dataset/car/extra.csv", "x\n")
    with pytest.raises(SystemExit):
        dd.main()
    assert manifest.read_bytes() == before


def test_offline_run_stops_when_an_archive_is_missing(dd, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    archives = _archives(raw)
    manifest = raw / "download_manifest.json"
    manifest.write_text(json.dumps([_entry(p) for p in archives]), encoding="utf-8")
    archives[1].unlink()
    monkeypatch.setattr(dd, "RAW", raw)
    monkeypatch.setattr(dd, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["download_data.py", "--offline"])
    with pytest.raises(SystemExit):
        dd.main()
