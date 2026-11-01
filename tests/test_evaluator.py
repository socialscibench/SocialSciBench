import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.__main__ import main
from harness.cases import development_cases
from harness.common import ROOT, read_json, task_info
from harness.runner import evaluate, load_solution, run_worker
from harness.bundles import export_run, verify_run


@pytest.mark.parametrize("task", ["001", "002", "003"])
def test_reference_candidate_end_to_end(task, tmp_path):
    solution = tmp_path / "solution"
    assert main(["init-solution", "--task", task, "--out", str(solution)]) == 0
    assert main(["evaluate", "--task", task, "--solution", str(solution / "solution.json"),
                 "--ids", "1", "--warmups", "0", "--repeats", "2", "--out", str(tmp_path / "runs")]) == 0
    run = next((tmp_path / "runs").iterdir())
    summary = read_json(run / "summary.json")
    assert summary["scored_workloads"] == [1]
    metadata = read_json(run / "run.json")
    assert metadata["source_sha256"]["solution/solution.py"]
    rows = [json.loads(line) for line in (run / "traces.jsonl").read_text().splitlines()]
    assert all(row["status"] == "PASSED" for row in rows)
    assert all(row["performance"] is None for row in rows[:-1])
    row = rows[-1]
    assert len(row["performance"]["candidate"]["samples_s"]) == 2
    assert len(row["correctness"]["candidate"]) == 3
    assert row["workers"]["candidate"]["environment"]["packages"]["numpy"]
    assert row["workers"]["candidate"]["peak_rss_bytes"] > 0
    assert not list(run.rglob("*.npz"))


def candidate(tmp_path, body):
    path = tmp_path / "candidate"
    path.mkdir()
    source = (ROOT / "003_reclink" / "reference.py").read_text()
    (path / "solution.py").write_text(source + "\n_original_run = run\n" + body)
    manifest = dict(schema_version=1, name="test", task="003", entry_point="solution.py::run",
                    sources=["solution.py"], target_hardware="cpu", dependencies=["numpy", "pandas", "scipy"])
    (path / "solution.json").write_text(json.dumps(manifest))
    return path


def options(tmp_path):
    return SimpleNamespace(out=tmp_path / "runs", threads=1, warmups=1, repeats=2, timeout=30,
                           validation_seed=1729, keep_outputs=False)


def test_later_repetition_failure_is_not_scored(tmp_path):
    source = candidate(tmp_path, """
calls = 0
def run(**kwargs):
    global calls
    calls += 1
    i, j, score, n_perfect = _original_run(**kwargs)
    if calls >= 3:
        score = score + 0.01
    return i, j, score, n_perfect
""")
    entry, manifest = task_info("003")
    case = dict(development_cases("003")[0], scored=True)
    assert evaluate(entry, manifest, source / "solution.json", [case], options(tmp_path)) == 1
    row = json.loads(next((tmp_path / "runs").glob("*/traces.jsonl")).read_text())
    assert row["status"] == "INCORRECT"
    assert row["performance"] is None
    assert row["correctness"]["candidate"][0]["passed"]
    assert not row["correctness"]["candidate"][-1]["passed"]


def test_development_failure_prevents_workload_scoring(tmp_path):
    source = candidate(tmp_path, "def run(**kwargs):\n    return [], [], [], 0\n")
    entry, manifest = task_info("003")
    first = development_cases("003")[0]
    cases = [first, dict(first, id="workload-1", scored=True)]
    assert evaluate(entry, manifest, source / "solution.json", cases, options(tmp_path)) == 1
    rows = [json.loads(line) for line in next((tmp_path / "runs").glob("*/traces.jsonl")).read_text().splitlines()]
    assert rows[0]["status"] == "INCORRECT"
    assert rows[1]["status"] == "SKIPPED"
    assert rows[1]["performance"] is None
    summary = read_json(next((tmp_path / "runs").glob("*/summary.json")))
    assert summary["requested_workloads"] == [1]
    assert summary["scored_workloads"] == []
    assert summary["evaluated_cases"] == 1


def worker_config(tmp_path, source):
    return dict(task="003", task_dir=str(ROOT / "003_reclink"),
                case=development_cases("003")[2], output_dir=str(tmp_path / "worker"),
                threads=1, warmups=0, repeats=1, source_dir=str(source), entry_point="solution.py::run")


@pytest.mark.parametrize("body,status", [
    ("def run(**kwargs):\n    raise RuntimeError('test failure')\n", "RUNTIME_ERROR"),
    ("def run(**kwargs):\n    return None\n", "INVALID_OUTPUT"),
    ("def run(**kwargs):\n    kwargs['master']['name'].flags.writeable = True\n"
     "    kwargs['master']['name'][0] = 'CHANGED'\n    return _original_run(**kwargs)\n", "INPUT_MUTATION"),
])
def test_worker_failure_modes(tmp_path, body, status):
    source = candidate(tmp_path, body)
    result = run_worker(worker_config(tmp_path, source), ROOT, 30)
    assert result["status"] == status
    assert (tmp_path / "worker" / "stderr.log").exists()


def test_worker_timeout(tmp_path):
    source = candidate(tmp_path, "def run(**kwargs):\n    import time\n    time.sleep(60)\n")
    result = run_worker(worker_config(tmp_path, source), ROOT, 2)
    assert result["status"] == "TIMEOUT"


def test_snapshot_mutation_invalidates_results(tmp_path):
    source = candidate(tmp_path, """
def run(**kwargs):
    from pathlib import Path
    source = Path(__file__)
    source.write_text(source.read_text() + '\\n# changed during evaluation\\n')
    return _original_run(**kwargs)
""")
    entry, manifest = task_info("003")
    assert evaluate(entry, manifest, source / "solution.json", development_cases("003")[:1], options(tmp_path)) == 1
    summary = read_json(next((tmp_path / "runs").glob("*/summary.json")))
    assert summary["modified_snapshot_files"] == ["solution/solution.py"]
    row = json.loads(next((tmp_path / "runs").glob("*/traces.jsonl")).read_text())
    assert row["status"] == "INVALIDATED"
    assert row["performance"] is None


def test_accelerator_requires_sync_and_paths_cannot_escape(tmp_path):
    source = candidate(tmp_path, "")
    path = source / "solution.json"
    manifest = read_json(path)
    manifest["target_hardware"] = "cuda"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="synchronize"):
        load_solution(path, "003")
    manifest["target_hardware"] = "cpu"
    manifest["sources"] = ["../outside.py"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative path"):
        load_solution(path, "003")


def test_sync_hook_shares_module_state(tmp_path):
    source = candidate(tmp_path, """
synced = False
def synchronize():
    global synced
    synced = True
def run(**kwargs):
    assert synced
    return _original_run(**kwargs)
""")
    config = dict(worker_config(tmp_path, source), synchronize="solution.py::synchronize")
    assert run_worker(config, ROOT, 30)["status"] == "COMPLETED"


def test_staged_workspace_is_self_contained_and_excludes_results(tmp_path):
    staged = tmp_path / "staged"
    assert main(["stage", "--task", "003", "--out", str(staged)]) == 0
    assert not any(p.name in {"reports", "baselines", "deferred", "solutions", "results"} for p in staged.rglob("*"))
    assert read_json(staged / "tasks.json")["tasks"][0]["id"] == "003"
    guide = (staged / "harness" / "README.md").read_text()
    assert "requirements-dev.txt" not in guide
    assert "pytest" not in guide
    assert "--task 001" not in guide
    assert "--task 003" in guide
    solution = staged / "solutions" / "first"
    subprocess.run([sys.executable, "-m", "harness", "init-solution", "--task", "003", "--out", str(solution)],
                   cwd=staged, check=True, capture_output=True)
    manifest = read_json(solution / "solution.json")
    assert set(manifest["artifacts"]) == {"research.md", "build.md", "requirements.lock"}
    assert all((solution / name).is_file() for name in manifest["artifacts"])
    result = subprocess.run([sys.executable, "-m", "harness", "evaluate", "--task", "003",
                             "--solution", str(solution / "solution.json"), "--smoke"],
                            cwd=staged, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert next((staged / "runs").glob("*/summary.json"))


@pytest.mark.parametrize("keep_outputs", [False, True])
def test_output_storage_is_bounded_and_every_repetition_is_validated(tmp_path, monkeypatch, keep_outputs):
    import harness.runner as runner

    source = candidate(tmp_path, "")
    entry, manifest = task_info("003")
    settings = options(tmp_path)
    settings.repeats = 5
    settings.keep_outputs = keep_outputs
    counts = []
    original = runner.output_validator

    def tracked(*args):
        consume = original(*args)

        def observe(role, item, directory):
            counts.append(len(list((tmp_path / "runs").rglob("output-*.npz"))))
            return consume(role, item, directory)
        return observe

    monkeypatch.setattr(runner, "output_validator", tracked)
    case = dict(development_cases("003")[0], scored=True)
    assert evaluate(entry, manifest, source / "solution.json", [case], settings) == 0
    expected_calls = 2 * (1 + settings.warmups + settings.repeats)
    assert len(counts) == expected_calls
    assert max(counts) == (expected_calls if keep_outputs else 2)
    assert len(list((tmp_path / "runs").rglob("output-*.npz"))) == (expected_calls if keep_outputs else 0)
    row = json.loads(next((tmp_path / "runs").glob("*/traces.jsonl")).read_text())
    assert sum(len(checks) for checks in row["correctness"].values()) == expected_calls
    assert row["worker_order"] == ["reference", "candidate"]


@pytest.mark.parametrize("key", ["workload_entry_point", "workload_catalog", "adapter", "validator",
                                 "development_entry_point"])
def test_invalid_declared_task_hooks_fail_early(tmp_path, key):
    staged = tmp_path / "staged"
    main(["stage", "--task", "003", "--out", str(staged)])
    path = staged / "003_reclink" / "task.json"
    manifest = read_json(path)
    manifest[key] = "missing_module::missing_function"
    path.write_text(json.dumps(manifest))
    result = subprocess.run([sys.executable, "-m", "harness", "list"], cwd=staged,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert key in result.stderr


def test_declared_task_hooks_control_catalog_generation_adaptation_and_validation(tmp_path):
    staged = tmp_path / "staged"
    main(["stage", "--task", "003", "--out", str(staged)])
    source = tmp_path / "solution"
    main(["init-solution", "--task", "003", "--out", str(source)])
    task_dir = staged / "003_reclink"
    (task_dir / "hooks.py").write_text('''
from harness.cases import generate_inputs
from harness.validation import validate as original_validate
from workloads import generate as original_generate

CATALOG = [dict(id=91, seed=100, n_master=30, n_using=40, overlap=0.6)]

def development(task, seed):
    return []

def generate(params):
    result = original_generate(params)
    result['hook_marker'] = 0.77
    return result

def adapt(task_dir, task, case, generator):
    result = generate_inputs(task_dir, task, case, generator)
    assert result.pop('hook_marker') == 0.77
    result['minscore'] = 0.88
    return result

def validate(task, actual, expected, tolerances, dimensions):
    result = original_validate(task, actual, expected, tolerances, dimensions)
    assert (actual['score'] >= 0.88).all()
    result['custom_validator'] = True
    return result
''')
    path = task_dir / "task.json"
    manifest = read_json(path)
    manifest["prompt_files"].append("hooks.py")
    manifest.update(workload_catalog="hooks.py::CATALOG", workload_entry_point="hooks.py::generate",
                    development_entry_point="hooks.py::development", adapter="hooks.py::adapt",
                    validator="hooks.py::validate")
    path.write_text(json.dumps(manifest))
    result = subprocess.run([sys.executable, "-m", "harness", "evaluate", "--task", "003",
                             "--solution", str(source / "solution.json"), "--all-workloads",
                             "--warmups", "0", "--repeats", "1"], cwd=staged,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in next((staged / "runs").glob("*/traces.jsonl")).read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case"]["params"]["id"] == 91
    assert all(report["custom_validator"] for reports in rows[0]["correctness"].values() for report in reports)


def test_exported_run_replays_after_relocation_and_preserves_artifacts(tmp_path):
    source = candidate(tmp_path, "")
    for name in ("research.md", "build.md", "requirements.lock"):
        (source / name).write_text("# Actual candidate research/build record\n")
    (source / "output-build-data.npz").write_bytes(b"declared artifact, not worker output")
    solution_path = source / "solution.json"
    solution = read_json(solution_path)
    solution["artifacts"] = ["output-build-data.npz"]
    solution_path.write_text(json.dumps(solution))
    entry, manifest = task_info("003")
    case = dict(development_cases("003")[0], scored=True)
    settings = options(tmp_path)
    settings.keep_outputs = True
    assert evaluate(entry, manifest, solution_path, [case], settings) == 0
    run = next((tmp_path / "runs").iterdir())
    original = read_json(run / "run.json")
    original_traces = (run / "traces.jsonl").read_bytes()
    bundle = export_run(run, tmp_path / "bundle")
    with pytest.raises(FileExistsError):
        export_run(run, bundle)
    assert not list((bundle / "workers").rglob("*.npz"))
    assert (bundle / "traces.jsonl").read_bytes() == original_traces
    for name in ("research.md", "build.md", "requirements.lock", "output-build-data.npz"):
        assert (bundle / "snapshot" / "solution" / name).read_bytes() == (source / name).read_bytes()
        assert f"solution/{name}" in original["source_sha256"]
    for name in ("requirements.txt", "requirements.lock", "AGENTS.md", "tasks.json"):
        assert (bundle / "snapshot" / name).is_file()
    moved = tmp_path / "relocated-bundle"
    bundle.rename(moved)
    # Remove both original locations used by the evaluation; replay must use the bundle.
    shutil.rmtree(run)
    shutil.rmtree(source)
    for path in (moved / "workers").rglob("config.json"):
        config = read_json(path)
        for key in ("task_dir", "source_dir", "output_dir"):
            assert not Path(config[key]).is_absolute()
            assert (path.parent / config[key]).exists()
    verify_run(moved)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-m", "harness", "list"], cwd=moved / "snapshot",
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    replay_out = tmp_path / "replays"
    result = subprocess.run([sys.executable, "-m", "harness", "replay", "--run", "..", "--out", str(replay_out)],
                            cwd=moved / "snapshot", env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    new = read_json(next(replay_out.glob("*/run.json")))
    assert new["cases"] == original["cases"]
    assert new["config"] == original["config"]
    assert new["replay_of"]["run_id"] == original["run_id"]
    assert new["run_id"] != original["run_id"]
    assert (moved / "traces.jsonl").read_bytes() == original_traces
    (moved / "snapshot" / "solution" / "research.md").write_text("modified notes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_run(moved)


def test_artifact_paths_cannot_escape_candidate(tmp_path):
    source = candidate(tmp_path, "")
    path = source / "solution.json"
    solution = read_json(path)
    solution["artifacts"] = ["../private-notes.md"]
    path.write_text(json.dumps(solution))
    with pytest.raises(ValueError, match="relative path"):
        load_solution(path, "003")
