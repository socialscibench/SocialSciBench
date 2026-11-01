"""The plugin's cores against the frozen candidates on the benchmark's development inputs (fixtures,
synthetic and real-development workloads; never held-out ones). Every output of the frozen candidate
must be reproduced: integer outputs exactly, floating-point outputs within 1e-12 relative (plus 1e-13
of the array's largest magnitude), the frozen code's own run-to-run variation (Apple Accelerate's BLAS
is not bit-reproducible across memory alignments; cell-lean differs from itself by up to ~2e-15).
"bitwise" reports whether every output was also bit-identical. Outputs the plugin adds are not
compared here.

usage: NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
       solutions/.venv/bin/python plugin/tools/parity.py TASK SUITE [IDS...] [--variant=speed|memory|both]
       SUITE: fixtures, synthetic or real-development; IDS: workload ids (default: all)
"""
RTOL = 1e-12
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plugin/turbo-socialsci/engine/src"))

from harness.cases import development_cases, generate_inputs          # noqa: E402
from harness.common import read_json                                   # noqa: E402
from harness.datasets import load_inputs, prepare_data, select_cases   # noqa: E402

FROZEN = {("001", "speed"): "solutions/001/schur-cg/solution.py", ("001", "memory"): "solutions/001/schur-lean/solution.py",
          ("002", "speed"): "solutions/002/cell-fused/solution.py", ("002", "memory"): "solutions/002/cell-lean/solution.py",
          ("003", "speed"): "solutions/003/block-index/solution.py", ("003", "memory"): "solutions/003/block-lean/solution.py"}
CORE = {"001": "hdfe", "002": "did", "003": "reclink"}


def load_frozen(task, variant):
    path = ROOT / FROZEN[(task, variant)]
    spec = importlib.util.spec_from_file_location(f"frozen_{task}_{variant}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    sys.path.remove(str(path.parent))
    return mod.run


def cases_and_inputs(task, suite, ids):
    entry = next(e for e in read_json(ROOT / "tasks.json")["tasks"] if e["id"] == task)
    task_dir = ROOT / entry["directory"]
    manifest = read_json(task_dir / "task.json")
    if suite == "fixtures":
        for case in development_cases(task):
            if not ids or case["id"] in ids:
                yield case["id"], generate_inputs(task_dir, task, case)
        return
    if suite == "real-development" and task == "001" and (not ids or "104" in ids):
        extra = [select_cases(entry, manifest, suite, [104], n_regressors=k)[0] for k in (5, 50)]
    else:
        extra = []
    cases = select_cases(entry, manifest, suite, [int(i) for i in ids] if ids else None, all_workloads=not ids)
    cases += extra
    data = prepare_data(entry, manifest, cases)
    for case in cases:
        if case["kind"] == "real":
            yield case["id"], load_inputs(task_dir, manifest, case, data["root"])
        else:
            yield case["id"], generate_inputs(task_dir, task, case)


def same(a, b):
    """Compare nested outputs; returns (within tolerance, bit-identical, max relative difference, detail)."""
    if isinstance(a, dict):
        parts = {k: same(a[k], b.get(k)) for k in a}
        bad = [k for k, r in parts.items() if not r[0]]
        return (not bad, all(r[1] for r in parts.values()), max([r[2] for r in parts.values()] + [0.0]),
                f"keys differ: {bad}" if bad else "")
    if isinstance(a, (tuple, list)) and not (len(a) and isinstance(a[0], (int, float))):
        parts = [same(x, y) for x, y in zip(a, b)]
        bad = [i for i, r in enumerate(parts) if not r[0]]
        return (not bad and len(a) == len(b), all(r[1] for r in parts), max([r[2] for r in parts] + [0.0]),
                f"items differ: {bad}" if bad else "")
    if b is None:
        return False, False, float("inf"), "missing"
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        return False, False, float("inf"), f"shape {x.shape} vs {y.shape}"
    if x.dtype.kind not in "fc":
        eq = bool(np.array_equal(x, y))
        return eq, eq, 0.0 if eq else float("inf"), ""
    bit = bool(np.array_equal(x, y, equal_nan=True))
    if bit or x.size == 0:
        return True, True, 0.0, ""
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False, False, float("inf"), "NaN positions differ"
    f = np.isfinite(x)
    scale = float(np.max(np.abs(x[f]))) if f.any() else 0.0
    diff = np.abs(x[f] - y[f])
    ok = bool(np.all(diff <= RTOL * np.abs(x[f]) + 1e-13 * scale))
    rel = float(np.max(diff / np.maximum(np.abs(x[f]), 1e-300))) if diff.size else 0.0
    return ok, False, rel, ""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    variant = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--variant=")), "both")
    task, suite, ids = args[0], args[1], args[2:]
    variants = ["speed", "memory"] if variant == "both" else [variant]
    from importlib import import_module
    runs = {v: (load_frozen(task, v), import_module(f"turbo_socialsci.core.{CORE[task]}_{v}").run) for v in variants}
    failures = 0
    for cid, kw in cases_and_inputs(task, suite, ids):
        for v, (frozen, plugin) in runs.items():
            t0 = time.perf_counter()
            a = frozen(**kw)
            t1 = time.perf_counter()
            b = plugin(**kw)
            t2 = time.perf_counter()
            ok, bit, rel, why = same(a, b)
            failures += not ok
            print(json.dumps(dict(task=task, case=cid, variant=v, equal=ok, bitwise=bit, max_rel_diff=rel,
                                  detail=why, frozen_s=round(t1 - t0, 4), plugin_s=round(t2 - t1, 4))), flush=True)
    print(f"{failures} mismatches")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
