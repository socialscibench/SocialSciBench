# Candidate environment and build

No build step and no compiler: the candidate is NumPy only (the JIT compiler of the `cell-fused`
candidate, and its memory, are not in the process). Environment: `solutions/.venv`, shared with the
study's other candidates and the reference:

```bash
uv venv solutions/.venv --python /usr/local/bin/python3.12      # CPython 3.12.13 (Homebrew)
uv pip install --python solutions/.venv/bin/python -r solutions/002/cell-lean/requirements.lock
solutions/.venv/bin/python -m harness evaluate --task 002 \
  --solution solutions/002/cell-lean/solution.json --suite real-development --ids 101 107 --threads 1
```

Threads: `--threads N` sets the BLAS limits (VECLIB_MAXIMUM_THREADS for Accelerate); the candidate
itself runs in one thread and uses BLAS for its matrix products. Memory: working arrays are bounded
(512 KB per matrix of the pass over units, 4 MB of bootstrap draws at a time); the candidate asks
the C allocator to return free pages to the OS (`malloc_zone_pressure_relief` on macOS,
`malloc_trim` with glibc, through ctypes) after every 64 MB or so of large freed arrays.

`fuzz.py` (an artifact, not imported at runtime) compares the candidate with the reference on
random small panels: `python solutions/002/cell-lean/fuzz.py solutions/002/cell-lean/solution.py 1000 7000`.
