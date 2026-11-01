"""Run a task-002 baseline adapter on the evaluator's development fixtures against the reference, with
the evaluator's validator (study tooling). Bootstrap outputs are reported, not compared.

usage (repository root, nothing else running):
  OMP_NUM_THREADS=1 solutions/baselines/.venv/py/bin/python solutions/baselines/check_adapter.py ADAPTER.py [FIXTURE ...]

Without fixture IDs, the fixtures that follow 1.81's rules are run.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from harness.cases import development_cases, generate_inputs  # noqa: E402
from harness.common import read_json  # noqa: E402
from harness.validation import normalize, validate  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


adapter = load(sys.argv[1], "adapter_under_test")
reference = load(ROOT / "002_csdid" / "reference.py", "task002_reference")
tolerances = read_json(ROOT / "002_csdid" / "task.json")["tolerances"]
wanted = sys.argv[2:]
NOT_COMPARABLE = ("se_boot_", "crit_")
for case in development_cases("002"):
    params = case.get("params", {})
    rules = params.get("rules", "csdid181")
    if wanted and case["id"] not in wanted:
        continue
    if not wanted and rules != "csdid181":
        continue
    d = generate_inputs(ROOT / "002_csdid", "002", case)
    if "boot" in params and "boot_weights" not in d:
        print(case["id"], "has no boot_weights in its inputs; keys:", sorted(d))
    kwargs = {k: v for k, v in d.items()}
    ref = normalize("002", reference.run(**kwargs))
    t0 = time.perf_counter()
    try:
        out = normalize("002", adapter.run(**kwargs))
    except Exception as exc:  # report and go on
        print(f"{case['id']}: adapter error {type(exc).__name__}: {exc}")
        continue
    dt = time.perf_counter() - t0
    drop = lambda z: {k: v for k, v in z.items() if not k.startswith(NOT_COMPARABLE)}
    rep = validate("002", drop(out), drop(ref), tolerances, {})
    failed = {k: v for k, v in rep["checks"].items() if not v.get("passed", True)}
    print(f"{case['id']}: {'PASS' if rep['passed'] else 'FAIL'} ({dt:.1f} s; base {kwargs.get('base_period', 'varying')}, "
          f"control {kwargs.get('control', 'never')}, method {kwargs.get('method', 'dripw')})")
    for k, v in failed.items():
        print("   ", k, {a: b for a, b in v.items() if a != "passed"})
    print("    event_times", out["event_times"].tolist(), "reference", ref["event_times"].tolist())
    boot = sorted(k for k in ref if k.startswith(NOT_COMPARABLE))
    for k in boot:
        a, b = np.asarray(out.get(k, np.nan), dtype=float), np.asarray(ref[k], dtype=float)
        if a.shape != b.shape:
            print(f"    {k}: shape {a.shape} against {b.shape}")
            continue
        ok = np.isfinite(a).all()
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        print(f"    {k}: finite {ok}, relative difference to the reference median {np.median(rel):.3f}, "
              f"max {rel.max():.3f}")
