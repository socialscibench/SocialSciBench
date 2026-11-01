"""File loading, declared task hooks and provenance helpers."""
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tasks():
    return read_json(ROOT / "tasks.json")["tasks"]


def task_info(task):
    matches = [t for t in tasks() if task in (t["id"], t["directory"], *t.get("legacy_aliases", []))]
    if len(matches) != 1:
        raise ValueError(f"Unsupported task {task!r}; use 'python -m harness list'")
    entry = matches[0]
    task_dir = ROOT / safe_relative(entry["directory"])
    manifest = read_json(task_dir / "task.json")
    if manifest.get("schema_version") != 1 or any(manifest.get(k) != entry[k] for k in ("id", "directory")):
        raise ValueError("Task manifest schema, ID and directory must match the registry")
    for key in ("reference_entry_point", "workload_entry_point", "workload_catalog",
                "development_entry_point", "adapter", "validator"):
        task_hook(task_dir, manifest, key)
    if "realdata_entry_point" in manifest or "real_workload_catalog" in manifest:
        name = manifest.get("real_workload_catalog")
        if name not in manifest["prompt_files"]:
            raise ValueError("Real workload catalog must be declared in prompt_files")
        data = read_json(task_dir / safe_relative(name))
        if data.get("schema_version") != 1 or data.get("task") != entry["id"]:
            raise ValueError("Real workload catalog schema and task ID must match")
        task_hook(task_dir, manifest, "realdata_entry_point")
    return entry, manifest


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative path without '..': {name!r}")
    return path


def symbol(root, value):
    if not isinstance(value, str) or value.count("::") != 1:
        raise ValueError(f"Expected filename::symbol: {value!r}")
    filename, function = value.split("::")
    path = root / safe_relative(filename)
    if not path.is_file() or not function.isidentifier():
        raise ValueError(f"Invalid entry point: {value}")
    sys.path.insert(0, str(root))
    module_name = "submitted_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    module = sys.modules.get(module_name)
    if module is None:
        module = load_module(path, module_name)
    try:
        return getattr(module, function)
    except AttributeError as error:
        raise ValueError(f"Missing symbol: {value}") from error


def entrypoint(root, value):
    result = symbol(root, value)
    if not callable(result):
        raise ValueError(f"Entry point is not callable: {value}")
    return result


def task_hook(task_dir, manifest, key):
    """Resolve task-local files or installed module symbols declared by a task."""
    value = manifest.get(key)
    if not isinstance(value, str) or value.count("::") != 1:
        raise ValueError(f"Invalid task hook {key}: {value!r}")
    location, name = value.split("::")
    if location.endswith(".py"):
        if location not in manifest["prompt_files"]:
            raise ValueError(f"Task hook file must be in prompt_files: {location}")
        result = symbol(task_dir, value)
    else:
        if key in ("reference_entry_point", "workload_entry_point", "workload_catalog", "realdata_entry_point"):
            raise ValueError(f"Task hook {key} must use a task-local filename.py::symbol")
        if not all(part.isidentifier() for part in location.split(".")) or not name.isidentifier():
            raise ValueError(f"Invalid task hook {key}: {value}")
        try:
            result = getattr(importlib.import_module(location), name)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot load task hook {key}: {value}") from error
    if key == "workload_catalog":
        if not isinstance(result, (list, tuple)) or not all(isinstance(w, dict) and "id" in w for w in result):
            raise ValueError("workload_catalog must be a sequence of workload dictionaries")
    elif not callable(result):
        raise ValueError(f"Task hook {key} is not callable: {value}")
    return result
