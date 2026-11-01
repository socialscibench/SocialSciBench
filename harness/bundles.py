"""Portable run export and replay; never replace historical measurements."""
import fnmatch
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import read_json, safe_relative, sha256, write_json


def verify_run(path):
    directory = Path(path).resolve()
    metadata = read_json(directory / "run.json")
    if metadata.get("protocol_version") not in (2, 3):
        raise ValueError("Export/replay requires a portable protocol version 2 or 3 run")
    for name in ("summary.json", "traces.jsonl", "snapshot/tasks.json"):
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete run: missing {name}")
    snap = directory / "snapshot"
    hashes = metadata.get("source_sha256", {})
    if not hashes:
        raise ValueError("Run has no source hashes")
    for name, digest in hashes.items():
        source = snap / safe_relative(name)
        if not source.resolve().is_relative_to(snap) or not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
    return directory, metadata


def export_run(path, out):
    directory, metadata = verify_run(path)
    out = Path(out).resolve()
    if out.is_relative_to(directory):
        raise ValueError("Export destination must be outside the source run")

    def ignore(source, names):
        omitted = {"__pycache__"} & set(names)
        if Path(source).is_relative_to(directory / "workers"):
            omitted.update(name for name in names if fnmatch.fnmatch(name, "output-*.npz")
                           or name in ("output-ready.json", "output-ack.json"))
        return omitted

    # Preserve declared artifacts even if their names resemble transient worker files.
    shutil.copytree(directory, out, ignore=ignore)
    write_json(out / "bundle.json", dict(schema_version=1, source_run_id=metadata["run_id"],
                                        protocol_version=metadata["protocol_version"],
                                        output_arrays_included=False))
    (out / "README.md").write_text(
        "# SocialSciBench run bundle\n\nThis bundle preserves the original measurements, "
        "source snapshot, research/build artifacts, worker configurations and logs. "
        "Transient output arrays are excluded. See `summary.json` and `traces.jsonl` for "
        "the evaluated scope and failures; export does not certify correctness.\n\n"
        "Install the recorded candidate dependencies and build requirements first. "
        "From `snapshot/`, run:\n\n```bash\n"
        "python -m harness replay --run .. --out /path/to/new-runs\n```\n\n"
        "Replay verifies snapshot hashes and uses the saved cases/settings in a new run. "
        "Hardware and installed dependencies are recorded anew, not recreated automatically.\n")
    if metadata.get("dataset"):
        with (out / "README.md").open("a") as stream:
            stream.write("\nExternal dataset files are not embedded. The expected paths, byte sizes and "
                         "SHA-256 hashes are recorded in run.json. Supply --data-root /path/to/package/data "
                         "when replaying relocated development runs.\n")
    if metadata.get("final_evaluation"):
        (out / "README.md").write_text(
            "# Frozen final-evaluation run bundle\n\nThis bundle preserves measurements, source hashes, "
            "snapshots, logs and the frozen-round identity from run.json. Dataset files and temporary "
            "output arrays are not embedded. Preserve and publish the associated freeze record and "
            "all numbered round records, including failures. Ordinary replay is disabled for held-out "
            "workloads; a repeat requires another recorded `harness final` round from that freeze.\n")
    return out


def replay_run(path, out, data_root=None):
    directory, metadata = verify_run(path)
    if metadata.get("final_evaluation") or any(c.get("suite") == "real-heldout" for c in metadata["cases"]):
        raise ValueError("Held-out replay must use another recorded 'harness final' round from its freeze")
    if data_root is not None and not metadata.get("dataset"):
        raise ValueError("This run has no external dataset")
    snap = directory / "snapshot"
    out = Path(out).resolve()
    if out.is_relative_to(directory):
        raise ValueError("Replay output must be outside the saved run")
    env = dict(os.environ, PYTHONPATH=str(snap), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "harness.replay", str(directory), str(out)]
    if data_root is not None:
        command.append(str(Path(data_root).resolve()))
    return subprocess.run(command,
                          cwd=snap, env=env, check=False).returncode
