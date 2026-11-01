# Environment and build

No native build. The compute kernels are Numba functions compiled on the first call in each
process (the on-disk cache is disabled so the evaluator's snapshot is never written to); the
evaluator records first-call latency separately and excludes it from timed samples.

All task 002 candidates in the same-session comparison, and the reference, run in one shared
environment, `solutions/002/.venv` (git-ignored): CPython 3.12.13 with the 14 October 2026
study's exact Python lock (`repro-socialscibench/research/2026-10-14/requirements.lock.txt`:
NumPy 2.3.5, SciPy 1.18.1, pandas 3.0.6, Numba 0.62.1, ModernDiD 0.2.0, csdid 0.4.2, ...)
plus rpy2 3.5.17 and threadpoolctl. `requirements.lock` is that environment's freeze. From
`solutions/002`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
    -r ../../../repro-socialscibench/research/2026-10-14/requirements.lock.txt \
    rpy2==3.5.17 threadpoolctl
```

Run the evaluator from the repository root with `solutions/002/.venv/bin/python -m harness
evaluate --task 002 --solution solutions/002/<candidate>/solution.json ...`.

`fuzz.py` compares the candidate with the reference on random small problems
(`solutions/002/.venv/bin/python solutions/002/shared-nuisance/fuzz.py 400`).
