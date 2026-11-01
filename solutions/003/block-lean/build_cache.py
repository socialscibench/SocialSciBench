"""Compile block_lean.py's Numba kernels into Numba's on-disk cache (__pycache__ next to it), so
that the evaluated process loads compiled code instead of compiling (data-independent; see
build.md). Run from the repository root with the candidate environment:

    solutions/.venv/bin/python solutions/003/block-lean/build_cache.py

It calls the candidate on the evaluator's development fixtures and small synthetic inputs from
the task's generator, read-only as the evaluator passes them (Numba compiles read-only arrays
separately), with one and several threads, then checks that a fresh process finds every kernel
in the cache and lists the cache files as artifacts in solution.json.
"""
import json
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
    from harness.worker import freeze
    load_module(HERE / "solution.py", "block_lean_build")
    mod = sys.modules["socialscibench_003_block_lean"]
    task = ROOT / "003_reclink"
    wl = load_module(task / "workloads.py", "wl003_build")
    inputs = [generate_inputs(task, "003", case) for case in development_cases("003")]
    for w in wl.WORKLOADS[:2]:
        d = wl.generate(dict(w, n_master=min(w["n_master"], 300), n_using=min(w["n_using"], 1500)))
        d.pop("truth", None)
        d.update(vars=wl.VARS, wmatch=wl.WMATCH, wnomatch=wl.WNOMATCH, required=wl.REQUIRED,
                 minscore=wl.MINSCORE, minbigram=wl.MINBIGRAM)
        inputs.append(d)
    for d in inputs:
        freeze(d)
        mod.run(**d)


def check():
    """A fresh process must not compile: Numba saves no new cache entries."""
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import runpy; "
            f"runpy.run_path({str(HERE / 'build_cache.py')!r}, run_name='__check__')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "NUMBA_DEBUG_CACHE": "1"})
    saved = [l for l in out.stdout.splitlines() if "index saved" in l or "data saved" in l]
    print("cache entries written by a fresh process:", len(saved))
    return not saved and out.returncode == 0


def record():
    """List the cache files as artifacts in solution.json (their names carry source line numbers)."""
    path = HERE / "solution.json"
    sol = json.loads(path.read_text())
    cache = sorted(str(p.relative_to(HERE)) for p in (HERE / "__pycache__").glob("block_lean.*.nb[ic]"))
    sol["artifacts"] = [a for a in sol["artifacts"] if not a.startswith("__pycache__/")] + cache
    path.write_text(json.dumps(sol, indent=2) + "\n")
    print("cache files recorded in solution.json:", len(cache))


if __name__ == "__main__":
    for threads in ("1", "4"):
        subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); import runpy; "
                        f"runpy.run_path({str(HERE / 'build_cache.py')!r}, run_name='__check__')"],
                       env={**os.environ, "NUMBA_NUM_THREADS": threads}, check=True)
    ok = check()
    print("cache complete" if ok else "cache incomplete: kernels compiled again in a fresh process")
    if ok:
        record()
    sys.exit(0 if ok else 1)
elif __name__ == "__check__":
    build()
