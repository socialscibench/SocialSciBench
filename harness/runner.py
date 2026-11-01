"""Sequential isolated workers, independent validation, and immutable run records."""
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import PROTOCOL_VERSION
from .common import ROOT, read_json, safe_relative, sha256, task_hook, write_json


def load_solution(path, task):
    path = Path(path).resolve()
    solution = read_json(path)
    if not isinstance(solution, dict):
        raise ValueError("Solution manifest must be a JSON object")
    for field in ("name", "task", "entry_point", "sources", "target_hardware", "dependencies"):
        if field not in solution:
            raise ValueError(f"Missing solution field: {field}")
    if solution.get("schema_version") != 1 or solution["task"] != task:
        raise ValueError("Solution schema_version must be 1 and task must match")
    for field in ("name", "entry_point", "target_hardware"):
        if not isinstance(solution[field], str) or not solution[field].strip():
            raise ValueError(f"Solution {field} must be a nonempty string")
    if not isinstance(solution["dependencies"], list) or not all(isinstance(d, str) for d in solution["dependencies"]):
        raise ValueError("Solution dependencies must be a list of package declarations")
    if not isinstance(solution["sources"], list) or not solution["sources"]:
        raise ValueError("Solution sources must be a nonempty list")
    if not all(isinstance(s, str) for s in solution["sources"]):
        raise ValueError("Solution sources must contain relative path strings")
    if len(solution["sources"]) != len(set(solution["sources"])):
        raise ValueError("Duplicate solution source paths")
    artifacts = solution.get("artifacts", [])
    if not isinstance(artifacts, list) or not all(isinstance(s, str) for s in artifacts):
        raise ValueError("Solution artifacts must contain relative path strings")
    # Preserve conventional research/build records even for older manifests.
    artifacts = list(artifacts)
    for name in ("research.md", "requirements.lock", "build.md"):
        if (path.parent / name).is_file() and name not in artifacts + solution["sources"]:
            artifacts.append(name)
    if len(artifacts) != len(set(artifacts)) or set(artifacts) & set(solution["sources"]):
        raise ValueError("Duplicate solution artifact paths")
    solution["artifacts"] = artifacts
    for source in solution["sources"] + artifacts:
        p = path.parent / safe_relative(source)
        if not p.is_file() or not p.resolve().is_relative_to(path.parent):
            raise ValueError(f"Missing source or source outside solution directory: {source}")
    for key in ("entry_point", "synchronize"):
        if solution.get(key):
            if not isinstance(solution[key], str) or solution[key].count("::") != 1:
                raise ValueError(f"{key} must use filename::function syntax")
            file, name = solution[key].split("::")
            if file not in solution["sources"] or not name.isidentifier():
                raise ValueError(f"{key} must reference a declared source and callable name")
    if solution["target_hardware"] != "cpu" and not solution.get("synchronize"):
        raise ValueError("Accelerator candidates must declare a synchronize entry point")
    return solution, path.parent


def new_run_directory(base):
    if Path(base).resolve().is_relative_to(ROOT / "harness"):
        raise ValueError("Run output must be outside the evaluator source directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path(base).resolve() / (stamp + "-" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot(run_dir, task_entry, manifest, solution, source_dir):
    snap = run_dir / "snapshot"
    task_dir = snap / task_entry["directory"]
    task_dir.mkdir(parents=True)
    for name in manifest["prompt_files"]:
        dest = task_dir / safe_relative(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / task_entry["directory"] / name, dest)
    shutil.copytree(ROOT / "harness", snap / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    write_json(snap / "tasks.json", dict(schema_version=1, tasks=[task_entry]))
    for name in ("requirements.txt", "requirements.lock", "AGENTS.md"):
        shutil.copy2(ROOT / name, snap / name)
    agents = snap / "AGENTS.md"
    agents.write_text(agents.read_text().replace("001, 002, and 003", task_entry["id"]))
    (snap / "README.md").write_text(
        "# Run snapshot\n\nInstall the recorded candidate environment using "
        "`solution/build.md` and `solution/requirements.lock` when present. "
        "`requirements.lock` records the benchmark's tested dependencies. "
        "No packages or native builds are installed automatically.\n\n"
        "From this directory, run `python -m harness list` or "
        "`python -m harness replay --run .. --out /path/to/new-runs` to repeat the saved cases "
        "and measurement settings in a new run. Original measurements remain unchanged.\n")
    solution_dir = snap / "solution"
    solution_dir.mkdir()
    for name in solution["sources"] + solution["artifacts"]:
        dest = solution_dir / safe_relative(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / name, dest)
    write_json(solution_dir / "solution.json", solution)
    hashes = {str(p.relative_to(snap)): sha256(p) for p in sorted(snap.rglob("*")) if p.is_file()}
    return snap, task_dir, solution_dir, hashes


def run_worker(config, snapshot_dir, timeout, on_output=None):
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True)
    saved = dict(config, stream_outputs=on_output is not None)
    # Resolve these relative to config.json, so a relocated bundle remains usable.
    for key in ("task_dir", "source_dir", "output_dir"):
        saved[key] = os.path.relpath(config[key], directory)
    write_json(directory / "config.json", saved)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(snapshot_dir), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[key] = str(config["threads"])
    t0 = time.monotonic()
    validation_s = 0.0
    correctness = []
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen([sys.executable, "-m", "harness.worker", str(directory / "config.json")],
                                   cwd=directory, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                ready = directory / "output-ready.json"
                if on_output is not None and ready.exists():
                    item = read_json(ready)
                    started = time.monotonic()
                    report = on_output(item, directory)
                    correctness.append(dict(phase=item["phase"], **report))
                    validation_s += time.monotonic() - started
                    ready.unlink()
                    write_json(directory / "output-ack.json", dict(index=item["index"]))
                if time.monotonic() - t0 - validation_s > timeout:
                    return dict(status="TIMEOUT", wall_s=time.monotonic()-t0, validation_s=validation_s,
                                correctness=correctness, error=f"Worker exceeded {timeout}s excluding validation")
                time.sleep(0.01)
            code = process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if code != 0 or not (directory / "result.json").exists():
        return dict(status="RUNTIME_ERROR", exit_code=code, wall_s=time.monotonic()-t0)
    try:
        result = read_json(directory / "result.json")
    except (ValueError, OSError) as error:
        return dict(status="RUNTIME_ERROR", error=str(error))
    result["wall_s"] = time.monotonic() - t0
    result["validation_s"] = validation_s
    if on_output is not None:
        result["correctness"] = correctness
    return result


def compare_workers(reference, candidate):
    if reference["status"] != "COMPLETED":
        return "INVALID_REFERENCE", {"reference_status": reference["status"]}
    if candidate["status"] != "COMPLETED":
        return candidate["status"], {}
    if reference["input_sha256"] != candidate["input_sha256"]:
        return "INPUT_MISMATCH", {}
    details = {"reference": [], "candidate": []}
    for role, worker in (("reference", reference), ("candidate", candidate)):
        details[role] = worker["correctness"]
        if (not details[role] or len(details[role]) != len(worker["outputs"])
                or not all(r["passed"] for r in details[role])):
            return ("INVALID_REFERENCE" if role == "reference" else "INCORRECT"), details
    return "PASSED", details


def output_validator(task, manifest, task_dir, keep_outputs):
    """Hold only the first reference file between calls; validate and release each output."""
    import numpy as np

    validate = task_hook(task_dir, manifest, "validator")
    anchor = {}

    def consume(role, item, directory):
        path = directory / safe_relative(item["file"])
        if role == "reference" and not anchor:
            anchor.update(path=path, dimensions=item["dimensions"], input_sha256=item["input_sha256"])
        try:
            if not anchor:
                return dict(passed=False, checks={}, error="No reference output")
            if item["input_sha256"] != anchor["input_sha256"]:
                return dict(passed=False, checks={}, error="Input hashes differ")
            with np.load(anchor["path"], allow_pickle=False) as file:
                expected = dict(file)
            if path == anchor["path"]:
                actual = expected
            else:
                with np.load(path, allow_pickle=False) as file:
                    actual = dict(file)
            return validate(task, actual, expected, manifest["tolerances"], anchor["dimensions"])
        except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
            return dict(passed=False, checks={}, error=str(error))
        finally:
            if not keep_outputs and path != anchor.get("path"):
                path.unlink(missing_ok=True)

    return consume


def timing(samples):
    return dict(samples_s=samples, median_s=statistics.median(samples),
                min_s=min(samples), max_s=max(samples))


def host_info():
    info = dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
                processor=platform.processor())
    if sys.platform == "darwin":
        for field, key in (("cpu_model", "machdep.cpu.brand_string"), ("physical_memory_bytes", "hw.memsize")):
            try:
                value = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
                if value.returncode == 0:
                    info[field] = value.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    return info


def evaluate(entry, manifest, solution_path, cases, options):
    from .datasets import prepare_data, verify_files
    if any(c.get("suite") == "real-heldout" for c in cases) and not getattr(options, "final_evaluation", None):
        raise ValueError("Held-out workloads require 'harness freeze' followed by 'harness final'")
    if any(c.get("suite") == "real-heldout" for c in cases):
        from .storage import external_directory
        external_directory(options.out)
    solution, source_dir = load_solution(solution_path, entry["id"])
    dataset = prepare_data(entry, manifest, cases, getattr(options, "data_root", None))
    run_dir = new_run_directory(options.out)
    if getattr(options, "final_attempt", None):
        attempt = read_json(options.final_attempt)
        attempt.update(run_directory=str(run_dir), run_id=run_dir.name)
        write_json(options.final_attempt, attempt)
    snap, task_dir, solution_dir, hashes = snapshot(run_dir, entry, manifest, solution, source_dir)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    except OSError:  # git is not installed: the commit is recorded as unknown, as for a copy without .git
        git = dirty = subprocess.CompletedProcess([], 1)
    metadata = dict(protocol_version=PROTOCOL_VERSION, run_id=run_dir.name,
                    task=entry["id"], contract_version=manifest["contract_version"],
                    solution=solution, created_at=datetime.now(timezone.utc).isoformat(),
                    git_commit=git.stdout.strip() if git.returncode == 0 else None,
                    git_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
                    source_sha256=hashes, command=sys.argv, host=host_info(),
                    config=dict(threads=options.threads, warmups=options.warmups, repeats=options.repeats,
                                timeout_s=options.timeout, validation_seed=options.validation_seed,
                                keep_outputs=options.keep_outputs),
                    cases=cases, environment_policy="same Python environment for sequential reference/candidate workers")
    metadata["dataset"] = dataset
    metadata["suite"] = getattr(options, "suite", "synthetic")
    if getattr(options, "final_evaluation", None):
        metadata["final_evaluation"] = options.final_evaluation
    if getattr(options, "replay_of", None):
        metadata["replay_of"] = options.replay_of
    write_json(run_dir / "run.json", metadata)
    traces = []
    gate = True
    for case in cases:
        scored = case.get("scored", False)
        row = dict(protocol_version=PROTOCOL_VERSION, task=entry["id"], solution=solution["name"],
                   case=case, status="SKIPPED", performance=None, correctness=None)
        if scored and not gate:
            row["reason"] = "Development correctness gate failed"
        else:
            workers = {}
            # Establish the comparison anchor before consuming candidate outputs.
            order = ("reference", "candidate")
            consume = output_validator(entry["id"], manifest, task_dir, options.keep_outputs)
            for role in order:
                directory = run_dir / "workers" / case["id"] / role
                config = dict(task=entry["id"], task_dir=str(task_dir), case=case,
                              output_dir=str(directory), threads=options.threads,
                              warmups=options.warmups if scored else 0, repeats=options.repeats if scored else 0,
                              source_dir=str(task_dir if role == "reference" else solution_dir),
                              entry_point=manifest["reference_entry_point"] if role == "reference" else solution["entry_point"],
                              synchronize=solution.get("synchronize") if role == "candidate" else None)
                if case["kind"] == "real":
                    config["data_root"] = dataset["root"]
                workers[role] = run_worker(config, snap, options.timeout,
                                          lambda item, directory, role=role: consume(role, item, directory))
            ref_dir = run_dir / "workers" / case["id"] / "reference"
            cand_dir = ref_dir.parent / "candidate"
            try:
                row["status"], row["correctness"] = compare_workers(workers["reference"], workers["candidate"])
            except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
                row.update(status="INVALID_OUTPUT", correctness={"error": str(error)})
            row["workers"] = workers
            row["worker_order"] = list(order)
            if row["status"] == "PASSED" and scored:
                reference_time = timing(workers["reference"]["samples_s"])
                candidate_time = timing(workers["candidate"]["samples_s"])
                row["performance"] = dict(reference=reference_time, candidate=candidate_time,
                                          speedup=reference_time["median_s"] / candidate_time["median_s"])
            if not scored and row["status"] != "PASSED":
                gate = False
            if not options.keep_outputs:
                for directory in (ref_dir, cand_dir):
                    for output in directory.glob("output-*.npz"):
                        output.unlink()
        traces.append(row)
        with (run_dir / "traces.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"{case['id']}: {row['status']}", flush=True)
    changed = [name for name, digest in hashes.items() if not (snap / name).is_file() or sha256(snap / name) != digest]
    data_error = None
    if dataset:
        try:
            verify_files(dataset["root"], dataset["files"])
        except (ValueError, OSError) as error:
            data_error = str(error)
    if changed or data_error:
        # Results from a modified evaluator/reference/submission snapshot are never scored.
        for row in traces:
            row.update(status="INVALIDATED", performance=None, reason="Snapshot or dataset changed during evaluation")
        with (run_dir / "traces.jsonl").open("w") as stream:
            for row in traces:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    passed = not changed and not data_error and all(t["status"] == "PASSED" for t in traces)
    summary = dict(status="PASSED" if passed else "FAILED", task=entry["id"], solution=solution["name"],
                   development_gate_passed=gate, evaluated_cases=sum(t["status"] != "SKIPPED" for t in traces),
                   requested_workloads=[t["case"]["params"]["id"] for t in traces if t["case"].get("scored")],
                   scored_workloads=[t["case"]["params"]["id"] for t in traces if t["performance"] is not None],
                   scope="Selected cases only; smoke runs do not certify standard workloads",
                   modified_snapshot_files=changed, dataset_error=data_error,
                   suite=metadata["suite"], run_directory=".")
    write_json(run_dir / "summary.json", summary)
    print(f"{summary['status']}: {run_dir}", flush=True)
    return 0 if passed else 1
