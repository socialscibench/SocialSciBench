# Candidate environment and build

Environment: `solutions/.venv` (shared by the study's candidates and the reference):

```bash
uv venv solutions/.venv --python /usr/local/bin/python3.12      # CPython 3.12.13 (Homebrew)
uv pip install --python solutions/.venv/bin/python -r solutions/001/schur-lean/requirements.lock
```

**Build step (required for the memory profile, not for correctness):** compile the Numba kernels
into Numba's on-disk cache next to `schur_lean.py`, once per environment, host and source version:

```bash
solutions/.venv/bin/python solutions/001/schur-lean/build_cache.py
```

The script:
- runs the candidate on small synthetic inputs and the evaluator's fixtures (no development or
  held-out data), steering every solver branch so that every kernel is compiled for the
  argument types the evaluation uses;
- checks that a fresh process compiles nothing more;
- lists the cache files (`__pycache__/schur_lean.*.nbi`, `*.nbc`) as artifacts in
  `solution.json`, so that the evaluator's snapshot (`shutil.copy2`, which keeps file times)
  carries them.

The cache is code, not data; AGENTS.md allows data-independent compiled-code caches.

At run time `solution.py` copies the shipped cache to a private temporary directory and points
Numba there (`numba.config.CACHE_DIR`), so a compilation for an argument type the cache lacks is
written to that copy, never into the candidate's directory: the evaluator invalidates a run whose
snapshot changes. `build_cache.py` sets `SOCIALSCIBENCH_BUILD_CACHE=1` to write the cache in place.

Numba accepts the cache only for the same Python, Numba and CPU, and for a source file with the
same modification time and size. After a checkout or copy that changes `schur_lean.py`'s time,
rerun the script before evaluating or freezing. Without a valid cache the kernels compile at the
first call (about 10-45 s and 0.2-0.3 GB more peak memory), and the results are unchanged.

```bash
solutions/.venv/bin/python -m harness evaluate --task 001 \
  --solution solutions/001/schur-lean/solution.json --suite real-development --ids 101 --threads 1
```

Threads: `--threads N` sets NUMBA_NUM_THREADS and the BLAS limits; as in schur-cg, with more than
one thread one Python thread runs the approximate Cholesky factorizations (the kernel releases
the GIL) while N - 1 Numba threads assemble the operator. Memory: per-level arrays of at most
128 MB per batch of columns, 32 MB chunks of demeaned rows, and free pages returned to the OS
through ctypes (`malloc_zone_pressure_relief` on macOS, `malloc_trim` with glibc).
