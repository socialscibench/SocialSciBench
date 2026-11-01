"""Small fabricated files exercise I/O/protocol; no official held-out inputs are opened."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from harness.__main__ import main
from harness.common import ROOT, read_json, sha256, task_hook, task_info, write_json
from harness.datasets import case_spec, select_cases, verify_files
from harness.frozen import reserve_round, verify_freeze
from harness.runner import load_solution


def loader(task):
    entry, manifest = task_info(task)
    return task_hook(ROOT / entry["directory"], manifest, "realdata_entry_point")


def parquet(root, name, data):
    path = root / "development" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path)
    return path


@pytest.mark.parametrize("task", ["001", "002", "003"])
def test_loader_rejects_unknown_ids_before_any_file_access(task, tmp_path, monkeypatch):
    namespace = loader(task).__globals__

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Invalid workload IDs must be rejected before reading files")

    monkeypatch.setattr(pq, "read_table", unexpected_read)
    monkeypatch.setattr(Path, "read_text", unexpected_read)
    invalid = [0, 110, 999, 101.0, "101", None]
    if task not in ("001", "002"):
        invalid.append(109)
    if task == "002":
        invalid += [201, 110, 300, "301"]           # the combined task: 101-109, 301, 302, 401-404
    for helper in ("load", "manifest", "ids"):
        if helper in namespace:
            for wid in invalid:
                with pytest.raises(ValueError, match=f"Unknown real workload ID for task {task}"):
                    namespace[helper](wid, tmp_path)


@pytest.mark.parametrize("task", ["001", "002", "003"])
def test_loader_splits_match_the_pinned_workload_catalog(task):
    entry, manifest = task_info(task)
    rows = read_json(ROOT / entry["directory"] / manifest["real_workload_catalog"])["workloads"]
    expected = {w["id"]: w["suite"].removeprefix("real-") for w in rows}
    namespace = loader(task).__globals__
    if task == "001":
        actual = {wid: spec["role"] for wid, spec in namespace["CATALOG"].items()}
    else:
        development, heldout = namespace["DEVELOPMENT"], namespace["HELDOUT"]
        assert len(set(development + heldout)) == len(development) + len(heldout)
        actual = {wid: role for role, ids in (("development", development), ("heldout", heldout)) for wid in ids}
    assert actual == expected


def test_reghdfe_compatibility_workload_needs_no_files_or_manifest(tmp_path, monkeypatch):
    load = loader("001")
    namespace = load.__globals__
    generated = object()

    def generate(kind):
        assert kind == "network"
        return generated

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Compatibility workload 109 must not read data files")

    monkeypatch.setattr(namespace["synthetic_extra"], "workload_109", generate)
    monkeypatch.setattr(pq, "read_table", unexpected_read)
    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert load(np.int64(109), tmp_path) is generated
    with pytest.raises(ValueError, match="no file manifest"):
        namespace["manifest"](109, tmp_path)


def test_csdid_column_order_and_zero_covariate_alias(tmp_path):
    parquet(tmp_path, "101.parquet", dict(unit=[3, 8, 12], gvar=[0, 2, 3],
            x_b=[1., 2., 3.], y003=[7., 8., 9.], y001=[1., 2., 3.],
            x_a=[4., 5., 6.], y002=[4., 5., 6.]))
    load = loader("002")
    full, empty = load(np.int64(101), tmp_path), load(106, tmp_path)
    np.testing.assert_array_equal(full["y"], [[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    np.testing.assert_array_equal(full["X"], [[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(full["y"], empty["y"])
    assert empty["X"].shape == (3, 0)
    assert full["gvar"].dtype == np.int64 and full["y"].dtype == np.float64


def test_reghdfe_event_dummies_determinism_and_variants(tmp_path):
    data = dict(fe_inventor=[0, 0, 1, 1], fe_firm=[0, 1, 0, 1], fe_year=[0, 1, 0, 1],
                cluster=[0, 1, 0, 1])
    parquet(tmp_path, "108.parquet", dict(data, d_m2=[1, 0, 0, 1], d_0=[0, 1, 1, 0]))
    parquet(tmp_path, "104.parquet", data)
    load = loader("001")
    first, repeat = load(108, tmp_path), load(108, tmp_path)
    assert first["X"].shape == (4, 22)
    np.testing.assert_array_equal(first["X"][:, :2], [[1, 0], [0, 1], [0, 1], [1, 0]])
    for name in first:
        np.testing.assert_array_equal(first[name], repeat[name])
    assert load(104, tmp_path, n_regressors=5)["X"].shape == (4, 5)
    assert load(104, tmp_path, n_regressors=50)["X"].shape == (4, 50)


def test_reclink_preserves_row_identity_missing_fields_and_duplicates(tmp_path):
    fields = dict(id=[81, 12], name=["ACME", "ACME"], addr=["", ""],
                  city=["PORTLAND", "PORTLAND"], zip=["", ""], state=["OR", "OR"])
    parquet(tmp_path, "102_master.parquet", {k: v[:1] for k, v in fields.items()})
    parquet(tmp_path, "102_using.parquet", fields)
    data = loader("003")(102, tmp_path)
    assert set(data["using"]) == {"name", "addr", "city", "zip", "state"}
    assert data["using"]["name"].tolist() == ["ACME", "ACME"]
    entry, manifest = task_info("003")
    reference = task_hook(ROOT / entry["directory"], manifest, "reference_entry_point")
    i, j, score, perfect = reference(**data)
    assert sorted(zip(i.tolist(), j.tolist())) == [(0, 0), (0, 1)]
    assert perfect == 2 and np.all(score == 1)


def test_real_selection_keeps_splits_and_variants_distinct():
    entry, manifest = task_info("001")
    cases = select_cases(entry, manifest, "real-development", all_workloads=True)
    assert [c["params"]["id"] for c in cases] == list(range(101, 109))
    for ids in ([109], [201], [101, 101]):
        with pytest.raises(ValueError):
            select_cases(entry, manifest, "real-development", ids)
    variant = select_cases(entry, manifest, "real-development", [104], n_regressors=50)[0]
    assert variant["id"].endswith("104-k50")
    with pytest.raises(ValueError):
        select_cases(entry, manifest, "real-development", [101], n_regressors=50)
    smoke = select_cases(entry, manifest, "real-smoke")[0]
    assert not smoke["scored"]
    with pytest.raises(ValueError, match="cannot be scored"):
        case_spec(ROOT / entry["directory"], manifest, dict(smoke, scored=True))


def test_command_line_workload_ids_map_to_the_catalog():
    entry, manifest = task_info("002")
    assert [c["params"]["id"] for c in select_cases(entry, manifest, "real-development", ["104", 101])] == [101, 104]
    for ids in ([301], [110], ["101", 101]):
        with pytest.raises(ValueError):
            select_cases(entry, manifest, "real-development", ids)
    heldout = select_cases(entry, manifest, "real-heldout", all_workloads=True)
    assert [c["params"]["id"] for c in heldout] == [301, 302, 401, 402, 403, 404]


def test_corrupt_or_missing_data_fails_before_execution(tmp_path):
    path = tmp_path / "a.parquet"
    path.write_bytes(b"abc")
    files = [dict(path=path.name, size=3, sha256=sha256(path))]
    verify_files(tmp_path, files)
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="checksum"):
        verify_files(tmp_path, files)
    path.unlink()
    with pytest.raises(ValueError, match="Missing dataset"):
        verify_files(tmp_path, files)


def test_heldout_is_rejected_by_normal_cli_before_candidate_or_data_access(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--task", "002", "--suite", "real-heldout", "--ids", "301",
              "--solution", str(tmp_path / "nonexistent.json")])
    assert error.value.code == 2


def isolated_command(root, *args):
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "harness", *map(str, args)], cwd=root,
                          env=env, capture_output=True, text=True, timeout=90)


def test_fabricated_real_run_stage_replay_and_frozen_rounds(tmp_path):
    # Build a separate tiny benchmark workspace: official held-out bytes are never accessed.
    work = tmp_path / "workspace"
    assert main(["stage", "--task", "002", "--out", str(work)]) == 0
    task_dir = work / "002_csdid"
    catalog_path = task_dir / "real_workloads.json"
    assert not any(w["suite"] == "real-heldout" for w in read_json(catalog_path)["workloads"])
    assert not any(p.name in {"reference_outputs", "validation.jsonl", "data_sources"} for p in work.rglob("*"))
    entry, manifest = task_info("002")
    generate = task_hook(ROOT / entry["directory"], manifest, "workload_entry_point")
    w = dict(task_hook(ROOT / entry["directory"], manifest, "workload_catalog")[0])
    w.update(n_units=60, n_periods=4, n_cohorts=2, n_covariates=1)
    inputs = generate(w)
    data = dict(unit=np.arange(60), gvar=inputs["gvar"], x1=inputs["X"][:, 0])
    data.update({f"y{i+1:03}": inputs["y"][:, i] for i in range(4)})
    root = tmp_path / "fake-data"
    path = parquet(root, "101.parquet", data)
    file = dict(path="development/101.parquet", size=path.stat().st_size, sha256=sha256(path))
    fake_final = root / "heldout/301.parquet"
    fake_final.parent.mkdir()
    fake_final.write_bytes(path.read_bytes())
    final_file = dict(file, path="heldout/301.parquet")
    # Final protocol is exercised on a fabricated fixture, not an official final workload.
    write_json(catalog_path, dict(schema_version=1, task="002", workloads=[
        dict(id=101, suite="real-development", files=[file]),
        dict(id=106, suite="real-development", files=[file]),
        dict(id=301, suite="real-heldout", files=[final_file])]))
    result = isolated_command(work, "stage", "--task", "002", "--suite", "real-development",
                              "--ids", "101", "106", "--data-root", root, "--out", tmp_path / "agent")
    assert result.returncode == 0, result.stderr
    staged = tmp_path / "agent"
    assert len(list((staged / "datasets").rglob("*.parquet"))) == 1
    assert {w["id"] for w in read_json(staged / "002_csdid/real_workloads.json")["workloads"]} == {101, 106}
    source = work / "candidate"
    assert isolated_command(work, "init-solution", "--task", "002", "--out", source).returncode == 0
    runs = tmp_path / "runs"
    result = isolated_command(work, "evaluate", "--task", "002", "--suite", "real-development",
                              "--ids", "101", "106", "--data-root", root, "--solution", source / "solution.json",
                              "--warmups", "0", "--repeats", "1", "--out", runs)
    assert result.returncode == 0, result.stdout + result.stderr
    run = next(runs.iterdir())
    metadata = read_json(run / "run.json")
    assert metadata["dataset"]["files"] == [file]
    assert read_json(run / "summary.json")["scored_workloads"] == [101, 106]
    moved = tmp_path / "private/data"
    moved.parent.mkdir()
    root.rename(moved)
    result = isolated_command(work, "replay", "--run", run, "--out", tmp_path / "replay", "--data-root", moved)
    assert result.returncode == 0, result.stdout + result.stderr
    result = isolated_command(work, "freeze", "--task", "002", "--solution", source / "solution.json",
                              "--out", work / "bad-freeze")
    assert result.returncode == 2 and "outside" in result.stderr
    assert not (work / "bad-freeze").exists()
    frozen = tmp_path / "frozen"
    result = isolated_command(work, "freeze", "--task", "002", "--solution", source / "solution.json",
                              "--out", frozen, "--rounds", "1", "--warmups", "0", "--repeats", "1")
    assert result.returncode == 0, result.stderr
    assert not list((frozen / "rounds").iterdir())
    for data_root, output in ((work / "bad-data", tmp_path / "final"), (moved, work / "bad-results")):
        result = isolated_command(work, "final", "--freeze", frozen, "--data-root", data_root, "--out", output)
        assert result.returncode == 2 and "outside" in result.stderr
        assert not list((frozen / "rounds").iterdir())
    (work / "frozen-alias").symlink_to(frozen, target_is_directory=True)
    result = isolated_command(work, "final", "--freeze", work / "frozen-alias", "--data-root", moved)
    assert result.returncode == 2 and "outside" in result.stderr
    assert not list((frozen / "rounds").iterdir())
    # Change the working candidate; final must still execute the frozen copy.
    (source / "solution.py").write_text('raise RuntimeError("working candidate must not execute")\n')
    final = moved.parent / "runs"
    result = isolated_command(work, "final", "--freeze", frozen, "--data-root", moved)
    assert result.returncode == 0, result.stdout + result.stderr
    attempt = read_json(frozen / "rounds/01.json")
    assert attempt["status"] == "COMPLETED"
    final_run = next(final.iterdir())
    assert read_json(final_run / "run.json")["final_evaluation"]["round"] == 1
    result = isolated_command(work, "final", "--freeze", frozen, "--data-root", moved, "--out", final)
    assert result.returncode == 2 and "rounds have been used" in result.stderr
    result = isolated_command(work, "replay", "--run", final_run, "--out", tmp_path / "bad-replay")
    assert result.returncode == 2 and "Held-out replay" in result.stderr
    (frozen / "snapshot/solution/solution.py").write_text("changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        verify_freeze(frozen)


def test_failed_or_interrupted_rounds_are_not_overwritten(tmp_path):
    write_json(tmp_path / "freeze.json", dict(settings={}))
    (tmp_path / "rounds").mkdir()
    frozen = dict(max_rounds=2)
    first, n1 = reserve_round(tmp_path, frozen, tmp_path / "runs", "first")
    second, n2 = reserve_round(tmp_path, frozen, tmp_path / "runs", "second")
    assert (n1, n2) == (1, 2) and read_json(first)["status"] == "STARTED"
    assert first != second
    with pytest.raises(ValueError, match="rounds have been used"):
        reserve_round(tmp_path, frozen, tmp_path / "runs", "third")
    write_json(tmp_path / "freeze.json", dict(settings={"changed": True}))
    with pytest.raises(ValueError, match="settings changed"):
        reserve_round(tmp_path, frozen, tmp_path / "runs", "fourth")
