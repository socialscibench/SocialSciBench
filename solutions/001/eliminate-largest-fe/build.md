# Candidate environment and build

No native build. The compute kernels are Numba functions, compiled on the first call in each
process (the on-disk Numba cache is disabled so the evaluator's snapshot is never written to).
The evaluator records first-call latency separately and excludes it from timed samples.

## Environment

The evaluator runs the reference and candidate workers with the interpreter that launches it,
so the candidate's environment must also satisfy the benchmark's `requirements.txt`.

Numba needs llvmlite, which publishes no wheel for Python 3.14 on macOS x86_64; building
llvmlite 0.50.0 from source failed. The candidate therefore uses CPython 3.12 with the
versions in `requirements.lock` (NumPy 2.3.5 instead of the benchmark lock's 2.5.4):

```bash
cd solutions/001/eliminate-largest-fe
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --no-build -r requirements.lock
```

`.venv/` is ignored by Git.

## Evaluation

From the repository root:

```bash
solutions/001/eliminate-largest-fe/.venv/bin/python -m harness evaluate --task 001 \
    --solution solutions/001/eliminate-largest-fe/solution.json --smoke
solutions/001/eliminate-largest-fe/.venv/bin/python -m harness evaluate --task 001 \
    --solution solutions/001/eliminate-largest-fe/solution.json --ids 1
```

Larger workloads need a longer `--timeout`, because the reference runs five calls per workload
(first call, warmup, three measured); workload 9's reference takes about two minutes per call.
