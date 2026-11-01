"""External-only storage policy; all test files are fabricated bytes."""
import hashlib
from pathlib import Path
import sys

import pytest

from harness.common import write_json
from harness.storage import external_directory, private_package_file
from scripts import download_real_data as downloader


def entry(path, data=b"test input"):
    return dict(task="001", path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest(), drive_id="unused")


def write_entry(root, e, data=b"test input"):
    path = root / e["task"] / e["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_private_classification_includes_oracles_and_validation_logs():
    assert not private_package_file(entry("data/development/101.parquet"))
    assert not private_package_file(entry("code/realdata.py"))
    for path in ("data/heldout/201.parquet", "data/heldout/manifest_201.json",
                 "data/reference_outputs/101.npz", "data/validation.jsonl", "data/data_checks.jsonl"):
        assert private_package_file(entry(path))


def test_external_root_rejects_repo_paths_and_symlink_aliases(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    assert external_directory(outside, workspace=workspace) == outside
    with pytest.raises(ValueError, match="outside"):
        external_directory(workspace / "private", workspace=workspace)
    (workspace / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        external_directory(workspace / "alias", workspace=workspace)
    (tmp_path / "alias-in").symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        external_directory(tmp_path / "alias-in" / "data", workspace=workspace)
    (workspace / ".git").mkdir()
    with pytest.raises(ValueError, match="outside"):
        external_directory(workspace / "data", workspace=workspace / "runs/snapshot")


def test_relocation_verifies_then_moves_only_private_files_and_is_idempotent(tmp_path, monkeypatch):
    workspace, private = tmp_path / "repo", tmp_path / "private"
    monkeypatch.setattr(downloader, "ROOT", workspace)
    public_root = workspace / "datasets"
    entries = [entry("data/development/101.parquet"), entry("data/heldout/201.parquet"),
               entry("data/reference_outputs/101.npz"), entry("data/validation.jsonl")]
    for e in entries:
        write_entry(public_root, e)
    assert downloader.relocate_private(entries, public_root, private) == 3
    assert (public_root / "001/data/development/101.parquet").read_bytes() == b"test input"
    assert not (public_root / "001/data/heldout").exists()
    assert not (public_root / "001/data/reference_outputs").exists()
    assert not (public_root / "001/data/validation.jsonl").exists()
    for e in entries[1:]:
        assert downloader.valid(private / e["task"] / e["path"], e)
    assert downloader.relocate_private(entries, public_root, private) == 3


def test_relocation_never_overwrites_or_removes_on_preflight_failure(tmp_path, monkeypatch):
    workspace, private = tmp_path / "repo", tmp_path / "private"
    monkeypatch.setattr(downloader, "ROOT", workspace)
    public = workspace / "datasets"
    files = [entry("data/heldout/201.parquet"), entry("data/heldout/202.parquet")]
    sources = [write_entry(public, e) for e in files]
    conflict = write_entry(private, files[1], b"do not overwrite")
    with pytest.raises(ValueError, match="refusing overwrite"):
        downloader.relocate_private(files, public, private)
    assert all(p.read_bytes() == b"test input" for p in sources)
    assert conflict.read_bytes() == b"do not overwrite"
    assert not (private / "001/data/heldout/201.parquet").exists()


def test_relocation_retains_all_sources_on_copy_failure(tmp_path, monkeypatch):
    workspace, private = tmp_path / "repo", tmp_path / "private"
    monkeypatch.setattr(downloader, "ROOT", workspace)
    public = workspace / "datasets"
    files = [entry("data/heldout/201.parquet"), entry("data/heldout/202.parquet")]
    sources = [write_entry(public, e) for e in files]
    original = downloader.shutil.copyfileobj
    calls = 0
    def fail_second(src, dst, length):
        nonlocal calls
        calls += 1
        if calls == 2:
            dst.write(b"partial")
            raise OSError("test copy failure")
        return original(src, dst, length)
    monkeypatch.setattr(downloader.shutil, "copyfileobj", fail_second)
    with pytest.raises(OSError, match="test copy failure"):
        downloader.relocate_private(files, public, private)
    assert all(p.read_bytes() == b"test input" for p in sources)
    assert not (private / "001/data/heldout/202.parquet").exists()


def test_complete_download_routes_private_files_outside_repo(tmp_path, monkeypatch):
    workspace, private = tmp_path / "repo", tmp_path / "private"
    monkeypatch.setattr(downloader, "ROOT", workspace)
    (workspace / "data_sources").mkdir(parents=True)
    entries = [entry("data/development/101.parquet"), entry("data/heldout/201.parquet"),
               entry("data/reference_outputs/101.npz")]
    write_json(workspace / "data_sources/packages.json", dict(files=entries))
    calls = []
    def collect(e, root, verify):
        calls.append((e["path"], root, verify))
        return True
    monkeypatch.setattr(downloader, "download", collect)
    monkeypatch.setattr(sys, "argv", ["download_real_data.py", "--complete-package", "--verify-only",
                                     "--private-root", str(private)])
    assert downloader.main() == 0
    assert sorted(calls) == sorted([(entries[0]["path"], workspace / "datasets", True),
                                   (entries[1]["path"], private, True), (entries[2]["path"], private, True)])
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["download_real_data.py", "--verify-only"])
    assert downloader.main() == 0
    assert calls == [(entries[0]["path"], workspace / "datasets", True)]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["download_real_data.py", "--complete-package", "--private-root", str(workspace / "bad")])
    with pytest.raises(SystemExit) as error:
        downloader.main()
    assert error.value.code == 2 and not calls
