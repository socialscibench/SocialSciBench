# Candidate environment and build

No native build: the kernels are compiled by Numba in each new process at the first call (first
calls of 32.7-38.7 s in the one-thread evaluator runs, which include the input's own work of at
most 3.8 s; Numba's on-disk cache is not used). The evaluator reports it as first-call latency,
outside the steady-state timing; the worker timeout includes it.

Environment used for the evaluator runs (`solutions/.venv`, shared by the task 001, 002 and 003
candidates of this study and by the reference they are compared with):

```bash
uv venv solutions/.venv --python /usr/local/bin/python3.12      # CPython 3.12.13 (Homebrew)
uv pip install --python solutions/.venv/bin/python -r solutions/002/cohort-fused/requirements.lock
```

The lock pins NumPy 2.3.5 because Numba 0.62.1, the last release with macOS x86_64 wheels
(llvmlite 0.45.1), requires NumPy < 2.4. NumPy uses Apple Accelerate for BLAS/LAPACK on this
host. Run the evaluator with this interpreter, e.g.

```bash
solutions/.venv/bin/python -m harness evaluate --task 002 \
  --solution solutions/002/cohort-fused/solution.json --suite real-development --ids 101 --threads 1
```

Threads: `--threads N` sets NUMBA_NUM_THREADS and the BLAS limits; the candidate's parallel
loops use the Numba thread pool (workqueue layer on this host) and stay within N threads.
`fuzz.py` (an artifact, not imported at runtime) compares the candidate with the reference on
random small panels: `python solutions/002/cohort-fused/fuzz.py solutions/002/cohort-fused/solution.py 400 1000`.
