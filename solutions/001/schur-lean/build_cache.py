"""Compile schur_lean.py's Numba kernels into Numba's on-disk cache (__pycache__ next to it), so
that the evaluated process loads compiled code instead of compiling (data-independent; see
build.md). Run from the repository root with the candidate environment:

    solutions/.venv/bin/python solutions/001/schur-lean/build_cache.py

It calls the candidate on small synthetic inputs from the task's generator and the evaluator's
development fixtures, read-only as the evaluator passes them (Numba compiles read-only arrays
separately), steering every branch (one to four fixed effects; dense, approximate
Cholesky and diagonal preconditioner blocks; dense, explicit and implicit reduced operators) so
that every kernel is compiled for the argument types the evaluation uses, then checks that a
fresh process finds everything in the cache.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["SOCIALSCIBENCH_BUILD_CACHE"] = "1"      # solution.py then uses the in-tree cache (__pycache__)
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def build():
    from harness.cases import development_cases, generate_inputs
    from harness.common import load_module
    from harness.worker import freeze             # the evaluator passes read-only inputs
    load_module(HERE / "solution.py", "schur_lean_build")
    mod = sys.modules["socialscibench_001_schur_lean"]
    task = ROOT / "001_reghdfe"
    wl = load_module(task / "workloads.py", "wl001_build")
    inputs = [generate_inputs(task, "001", case) for case in development_cases("001")]
    for w in wl.WORKLOADS:
        if w["id"] in (1, 4, 6, 7):
            small = dict(w, **{k: max(2, v // 20) for k, v in w.items() if k in ("n_units", "n_firms")})
            inputs.append(wl.generate(small))
    settings = [dict(), dict(DENSE_MAX=0), dict(DENSE_MAX=0, SPARSE_PAIRS=0), dict(DENSE_MAX=0, SPARSE_PAIRS=10**9)]
    for force in (None, "dense", "explicit", "implicit"):
        for setting in settings:
            saved = {k: getattr(mod, k) for k in setting}
            for k, v in setting.items():
                setattr(mod, k, v)
            mod.FORCE = force
            for d in inputs:
                d = {k: v for k, v in d.items() if k != "truth"}
                freeze(d)
                mod.run(**d)
            for k, v in saved.items():
                setattr(mod, k, v)
    mod.FORCE = None


def check():
    """A fresh process must not compile: Numba reports no cache misses."""
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); sys.argv = ['x', 'build']; "
            f"import runpy; runpy.run_path({str(HERE / 'build_cache.py')!r}, run_name='__check__')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"NUMBA_DEBUG_CACHE": "1", **__import__("os").environ})
    misses = [l for l in out.stdout.splitlines() if "index saved" in l or "data saved" in l]
    print("cache entries written by a fresh process:", len(misses))
    return not misses


def record():
    """List the cache files as artifacts in solution.json (their names carry source line numbers)."""
    import json
    path = HERE / "solution.json"
    sol = json.loads(path.read_text())
    cache = sorted(str(p.relative_to(HERE)) for p in (HERE / "__pycache__").glob("schur_lean.*.nb[ic]"))
    sol["artifacts"] = [a for a in sol["artifacts"] if not a.startswith("__pycache__/")] + cache
    path.write_text(json.dumps(sol, indent=2) + "\n")
    print("cache files recorded in solution.json:", len(cache))


if __name__ == "__main__":
    build()
    ok = check()
    print("cache complete" if ok else "cache incomplete: kernels compiled again in a fresh process")
    if ok:
        record()
    sys.exit(0 if ok else 1)
elif __name__ == "__check__":
    build()
