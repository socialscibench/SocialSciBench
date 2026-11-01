# Environment and build

No native build. The compute kernels are Numba functions compiled on the first call in each
process (the on-disk cache is disabled so the evaluator's snapshot is never written to); the
evaluator records first-call latency separately and excludes it from timed samples. One kernel
helper emits `llvm.prefetch` through `numba.extending.intrinsic` (tested with llvmlite 0.45.1,
LLVM 20).

All task 008 candidates in the same-session comparison, and the reference, run in one shared
environment, `solutions/008/.venv` (git-ignored): CPython 3.12.13 with the 14 October 2026
study's exact Python lock (`repro-socialscibench/research/2026-10-14/requirements.lock.txt`:
NumPy 2.3.5, SciPy 1.18.1, pandas 3.0.6, Numba 0.62.1, llvmlite 0.45.1, ...) plus rpy2 3.5.17
and threadpoolctl; it is identical to the task 002 environment. `requirements.lock` is its
freeze. From `solutions/008`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
    -r ../../../repro-socialscibench/research/2026-10-14/requirements.lock.txt \
    rpy2==3.5.17 threadpoolctl
```

Run the evaluator from the repository root with `solutions/008/.venv/bin/python -m harness
evaluate --task 008 --solution solutions/008/<candidate>/solution.json ...`.

`fuzz.py` compares the candidate with the reference on random small problems, bitwise
(`solutions/008/.venv/bin/python solutions/008/bigram-table-scan/fuzz.py 2000`).
