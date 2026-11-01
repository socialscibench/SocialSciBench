"""Pinned real workloads, explicit splits and external data provenance."""
from pathlib import Path
import shutil

from .common import ROOT, read_json, safe_relative, sha256, task_hook, write_json
from .storage import external_directory

SUITES = ("synthetic", "real-development", "real-smoke", "real-heldout")


def catalog(task_dir, manifest, suite):
    if suite == "synthetic":
        return task_hook(task_dir, manifest, "workload_catalog")
    name = manifest.get("real_workload_catalog")
    if not name or name not in manifest["prompt_files"]:
        raise ValueError("This task has no real-data catalog")
    data = read_json(task_dir / safe_relative(name))
    if data.get("task") != manifest["id"] or data.get("schema_version") != 1:
        raise ValueError("Invalid real-data catalog")
    rows = [w for w in data["workloads"] if w["suite"] == suite]
    if not rows:
        raise ValueError(f"No workloads available in {suite} for task {manifest['id']}")
    return rows


def select_cases(entry, manifest, suite, ids=None, all_workloads=False, n_regressors=None):
    task_dir = ROOT / entry["directory"]
    rows = catalog(task_dir, manifest, suite)
    ids = [w["id"] for w in rows] if all_workloads else (ids or [rows[0]["id"]])
    # Command-line IDs arrive as integers or strings; map them to the catalog's own IDs
    known = {str(w["id"]): w["id"] for w in rows}
    if len({str(i) for i in ids}) != len(ids) or {str(i) for i in ids} - set(known):
        raise ValueError(f"Invalid or duplicate workload IDs for {suite}: {ids}")
    ids = [known[str(i)] for i in ids]
    if n_regressors is not None and not (entry["id"] == "001" and suite == "real-development"
                                         and ids == [104] and n_regressors in (5, 20, 50)):
        raise ValueError("--n-regressors is only allowed for task 001, real-development ID 104: 5, 20 or 50")
    cases = []
    for w in rows:
        if w["id"] not in ids:
            continue
        if suite == "synthetic":
            cases.append(dict(id=f"workload-{w['id']}", kind="generated", params=w, scored=True))
        else:
            options = {"n_regressors": n_regressors or 20} if entry["id"] == "001" and w["id"] == 104 else {}
            variant = f"-k{options['n_regressors']}" if options else ""
            cases.append(dict(id=f"{suite}-{w['id']}{variant}", kind="real", suite=suite,
                              params=dict(id=w["id"], loader_options=options),
                              scored=suite != "real-smoke"))
    return cases


def case_spec(task_dir, manifest, case):
    if case.get("suite") not in SUITES[1:]:
        raise ValueError("Real case must declare a real-data suite")
    matches = [w for w in catalog(task_dir, manifest, case["suite"]) if w["id"] == case["params"]["id"]]
    if len(matches) != 1:
        raise ValueError("Real case is not in the declared suite")
    options = case["params"].get("loader_options", {})
    if options and not (manifest["id"] == "001" and case["suite"] == "real-development"
                        and case["params"]["id"] == 104 and set(options) == {"n_regressors"}
                        and options["n_regressors"] in (5, 20, 50)):
        raise ValueError("Unsupported real workload variant")
    if case["suite"] == "real-smoke" and case.get("scored"):
        raise ValueError("Compatibility workload 109 cannot be scored")
    return matches[0]


def verify_files(root, files):
    root = Path(root).resolve()
    for f in files:
        path = root / safe_relative(f["path"])
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing dataset {path}; download it or set --data-root to the package's data directory")
        if path.stat().st_size != f["size"] or sha256(path) != f["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {f['path']}")


def prepare_data(entry, manifest, cases, data_root=None):
    real = [c for c in cases if c["kind"] == "real"]
    if not real:
        return None
    if any(c.get("suite") == "real-heldout" for c in real):
        if data_root is None:
            raise ValueError("Held-out data require an explicit external --data-root")
        data_root = external_directory(data_root)
    files = {}
    for case in real:
        spec = case_spec(ROOT / entry["directory"], manifest, case)
        for f in spec["files"]:
            if f["path"] in files and files[f["path"]] != f:
                raise ValueError("Conflicting dataset declarations")
            files[f["path"]] = f
    root = Path(data_root or ROOT / "datasets" / entry["id"] / "data").resolve()
    selected = sorted(files.values(), key=lambda f: f["path"])
    verify_files(root, selected)
    return dict(root=str(root), files=selected, included_in_snapshot=False)


def load_inputs(task_dir, manifest, case, data_root):
    case_spec(task_dir, manifest, case)
    load = task_hook(task_dir, manifest, "realdata_entry_point")
    return load(case["params"]["id"], data_root=data_root, **case["params"].get("loader_options", {}))


def stage_data(out, entry, manifest, suite, ids, data_root):
    """Only public development inputs; never copy package README, answers or logs."""
    if suite not in ("synthetic", "real-development", "real-smoke"):
        raise ValueError("Staging does not expose held-out data")
    task_dir = out / entry["directory"]
    name = manifest.get("real_workload_catalog")
    if name:
        data = read_json(task_dir / name)
        data["workloads"] = [w for w in data["workloads"] if w["suite"] != "real-heldout"]
        write_json(task_dir / name, data)
    if suite == "synthetic":
        if ids:
            raise ValueError("Stage --ids applies only to real datasets")
        return None
    cases = select_cases(entry, manifest, suite, ids=ids, all_workloads=not ids)
    selected = prepare_data(entry, manifest, cases, data_root)
    dest_root = out / "datasets" / entry["id"] / "data"
    for f in selected["files"]:
        dest = dest_root / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(selected["root"]) / f["path"], dest)
    verify_files(dest_root, selected["files"])
    if name:
        data["workloads"] = [w for w in data["workloads"] if w["id"] in {c["params"]["id"] for c in cases}]
        write_json(task_dir / name, data)
    return dict(suite=suite, files=selected["files"], workload_ids=[c["params"]["id"] for c in cases])
