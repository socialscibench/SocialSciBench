# Candidate environment and build

No native build: the kernels are compiled by Numba in each new process at the first call (about
20 s on the development host; Numba's on-disk cache is not used). The evaluator reports it as
first-call latency, outside the steady-state timing; the worker timeout includes it.

Environment (`solutions/.venv`, shared with the study's task 001, 002 and 003 candidates and with
the reference they are compared with):

```bash
uv venv solutions/.venv --python /usr/local/bin/python3.12      # CPython 3.12.13 (Homebrew)
uv pip install --python solutions/.venv/bin/python -r solutions/002b/cell-fused/requirements.lock
```

The lock pins NumPy 2.3.5 because Numba 0.62.1, the last release with macOS x86_64 wheels
(llvmlite 0.45.1), requires NumPy < 2.4. Run the evaluator with this interpreter, e.g.

```bash
solutions/.venv/bin/python -m harness evaluate --task 002b \
  --solution solutions/002b/cell-fused/solution.json --suite real-development --ids 501 503a --threads 1
```

Threads: `--threads N` sets NUMBA_NUM_THREADS and the BLAS limits; the candidate's parallel loops
(over cells, cohorts and blocks of units) use the Numba thread pool and stay within N threads; it
makes no BLAS calls in its parallel regions. `fuzz.py` (an artifact, not imported at runtime)
compares the candidate with the reference on random small panels:
`python solutions/002b/cell-fused/fuzz.py solutions/002b/cell-fused/solution.py 1500 2000`.
