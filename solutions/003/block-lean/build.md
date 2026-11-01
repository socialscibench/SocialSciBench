# Candidate environment and build

Environment: `solutions/.venv` (shared by the study's candidates and the reference):

```bash
uv venv solutions/.venv --python /usr/local/bin/python3.12      # CPython 3.12.13 (Homebrew)
uv pip install --python solutions/.venv/bin/python -r solutions/003/block-lean/requirements.lock
```

**Build step (required for the memory profile, not for correctness):** compile the Numba kernels
into Numba's on-disk cache next to `block_lean.py`, once per environment, host and source version:

```bash
solutions/.venv/bin/python solutions/003/block-lean/build_cache.py
```

The script:
- runs the candidate on the evaluator's fixtures and small synthetic inputs (no development or
  held-out data), read-only as the evaluator passes them, with one and four threads;
- checks that a fresh process compiles nothing more;
- lists the cache files (`__pycache__/block_lean.*.nbi`, `*.nbc`) as artifacts in
  `solution.json`, so that the evaluator's snapshot (`shutil.copy2`, which keeps file times)
  carries them.

The cache is compiled code, not data (allowed by AGENTS.md). Numba accepts it only for the same
Python, Numba and CPU and for a source file with the same modification time and size. Rerun the
script after a checkout or copy that changes `block_lean.py`'s time. Without a valid cache the
kernels compile at the first call (about 0.3 GB more peak memory and tens of seconds), and the
results are unchanged.

At run time `solution.py` copies the shipped cache to a private temporary directory and points
Numba there (`numba.config.CACHE_DIR`), so a compilation for an argument type the cache lacks is
written to that copy, never into the candidate's directory: the evaluator invalidates a run whose
snapshot changes. `build_cache.py` sets `SOCIALSCIBENCH_BUILD_CACHE=1` to write the cache in place.

```bash
solutions/.venv/bin/python -m harness evaluate --task 003 \
  --solution solutions/003/block-lean/solution.json --suite real-development --ids 101 --threads 1
```

Threads: `--threads N` sets NUMBA_NUM_THREADS; the search runs in parallel tasks of about equal
work, as in block-index.
