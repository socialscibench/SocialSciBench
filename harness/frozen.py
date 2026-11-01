"""Dated final-evaluation snapshots and append-only attempt records (not a sandbox)."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

from .common import ROOT, read_json, safe_relative, sha256, task_hook, task_info, write_json
from .datasets import select_cases
from .storage import external_directory


def now():
    return datetime.now(timezone.utc).isoformat()


def create_freeze(entry, manifest, args):
    from .runner import load_solution, snapshot
    solution, source = load_solution(args.solution, entry["id"])
    task_dir = ROOT / entry["directory"]
    cases = task_hook(task_dir, manifest, "development_entry_point")(entry["id"], args.validation_seed)
    cases += select_cases(entry, manifest, "real-heldout", args.ids, all_workloads=not args.ids)
    out = external_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    _, _, _, hashes = snapshot(out, entry, manifest, solution, source)
    settings = {k: getattr(args, k) for k in ("threads", "warmups", "repeats", "timeout", "validation_seed", "keep_outputs")}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    except OSError:  # git is not installed: the commit is recorded as unknown, as for a copy without .git
        git = dirty = subprocess.CompletedProcess([], 1)
    frozen = dict(schema_version=1, freeze_id=uuid4().hex, created_at=now(), task=entry["id"],
                  git_commit=git.stdout.strip() if git.returncode == 0 else None,
                  git_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                  source_sha256=hashes, cases=cases, settings=settings, max_rounds=args.rounds)
    write_json(out / "freeze.json", frozen)
    (out / "rounds").mkdir()
    return out


def verify_freeze(path):
    path = Path(path).resolve()
    frozen = read_json(path / "freeze.json")
    if frozen.get("schema_version") != 1 or not frozen.get("source_sha256"):
        raise ValueError("Invalid freeze record")
    for name, digest in frozen["source_sha256"].items():
        p = path / "snapshot" / safe_relative(name)
        if not p.resolve().is_relative_to(path / "snapshot") or not p.is_file() or sha256(p) != digest:
            raise ValueError(f"Frozen file changed: {name}; create and disclose a new freeze")
    return path, frozen


def reserve_round(path, frozen, out, note):
    """Failures and interrupted attempts consume a round; never overwrite a reservation."""
    record_hash = sha256(path / "freeze.json")
    for previous in (path / "rounds").glob("*.json"):
        if read_json(previous)["freeze_sha256"] != record_hash:
            raise ValueError("Freeze settings changed after a previous round")
    for number in range(1, frozen["max_rounds"] + 1):
        attempt = path / "rounds" / f"{number:02d}.json"
        try:
            with attempt.open("x") as stream:
                json.dump(dict(round=number, status="STARTED", started_at=now(), note=note,
                               freeze_sha256=record_hash, output_parent=str(out)), stream, indent=2)
                stream.write("\n")
            return attempt, number
        except FileExistsError:
            continue
    raise ValueError("All frozen evaluation rounds have been used; report every attempt before a new study decision")


def run_final(path, data_root, out, note=""):
    path = external_directory(path)
    path, frozen = verify_freeze(path)
    data_root = external_directory(data_root, other_forbidden=(path,))
    out = external_directory(out or data_root.parent / "runs", other_forbidden=(path,))
    # Validate local bytes before consuming a round; no workloads are loaded or solved here.
    from .datasets import verify_files
    entry = read_json(path / "snapshot" / "tasks.json")["tasks"][0]
    manifest = read_json(path / "snapshot" / entry["directory"] / "task.json")
    raw = read_json(path / "snapshot" / entry["directory"] / manifest["real_workload_catalog"])
    selected = {c["params"]["id"] for c in frozen["cases"] if c["kind"] == "real"}
    verify_files(data_root, [f for w in raw["workloads"] if w["id"] in selected for f in w["files"]])
    # Prevent simultaneous rounds, including runs using the same freeze from another terminal.
    lock = path / "active-round.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as error:
        raise ValueError("A final round is active; if interrupted, inspect its log before removing active-round.lock") from error
    try:
        attempt, number = reserve_round(path, frozen, out, note)
        env = dict(os.environ, PYTHONPATH=str(path / "snapshot"), PYTHONDONTWRITEBYTECODE="1")
        code = subprocess.run([sys.executable, "-m", "harness.frozen", str(path), str(data_root.resolve()),
                               str(out), str(number)], cwd=path / "snapshot", env=env, check=False).returncode
        record = read_json(attempt)
        record.update(status="COMPLETED" if code == 0 else "FAILED", exit_code=code, finished_at=now())
        write_json(attempt, record)
        return code
    finally:
        lock.unlink(missing_ok=True)


def main():
    from .runner import evaluate
    path, frozen = verify_freeze(sys.argv[1])
    if ROOT != path / "snapshot":
        raise ValueError("Final evaluation must execute the frozen evaluator snapshot")
    entry, manifest = task_info(frozen["task"])
    provenance = dict(freeze_id=frozen["freeze_id"], created_at=frozen["created_at"],
                      freeze_sha256=sha256(path / "freeze.json"), round=int(sys.argv[4]))
    options = SimpleNamespace(**frozen["settings"], out=Path(sys.argv[3]), data_root=Path(sys.argv[2]),
                              suite="real-heldout", final_evaluation=provenance,
                              final_attempt=path / "rounds" / f"{int(sys.argv[4]):02d}.json")
    return evaluate(entry, manifest, ROOT / "solution/solution.json", frozen["cases"], options)


if __name__ == "__main__":
    raise SystemExit(main())
