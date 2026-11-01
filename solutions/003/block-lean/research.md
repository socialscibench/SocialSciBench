# Candidate research: block-lean (task 003)

## Objective

The minimal-peak-memory variant of `solutions/003/block-index`, for the same contract:
`003_reclink/reference.py` (Stata reclink2) within the task's tolerances. "Memory" is the
evaluator's measure: the worker's peak resident set (`ru_maxrss`, including the interpreter,
imports, input loading, every call and the output serialization).

## What sets the peak

The loader builds the inputs as NumPy string arrays from Python lists, which peaks at 0.11-0.58
GiB on 101-107 and 8.89 GiB on 108. block-index adds about 0.5 GB above that on every workload,
mostly Numba's compilation at the first call (about 70 s and 470 MB on 101-106). Its working
arrays are small: UCS4 codes of the values, int32 ids, bigram lists per block.

## Method

block-index's computation unchanged (`block_lean.py` is its source with these changes), plus:
- **Prebuilt Numba cache:** the kernels load from Numba's on-disk cache, built by
  `build_cache.py` from the evaluator's fixtures and small synthetic inputs (data-independent
  compiled code, as AGENTS.md allows).
  - `solution.py` imports `block_lean.py` under a fixed module name, because the cached code
    refers to its module by name. The evaluator imports candidates as
    `submitted_<hash of the snapshot path>`.
  - The build passes read-only inputs, as the evaluator does; Numba compiles read-only arrays
    separately.
  - At run time the cache is read from a private temporary copy (`numba.config.CACHE_DIR`), so a
    compilation for an argument type it lacks never writes into the evaluator's snapshot.
- **Returning free pages:** the C allocator's free pages are returned to the OS at the start of
  the call (pages the loader freed) and after the dictionaries are built
  (`malloc_zone_pressure_relief` on macOS, `malloc_trim` with glibc).

## Measurements (development runs, fresh process per workload, one thread; 2026-10-25)

| Workload | After loading | block-index | block-lean | Time, block-index / block-lean (second call) |
|---|---:|---:|---:|---:|
| 101 | 172 MB | 745 MB | 277 MB | 0.59 / 0.60 s |
| 102 | 109 MB | 671 MB | 191 MB | 0.17 / 0.17 s |
| 104 | 577 MB | 1,209 MB | 754 MB | 1.86 / 1.92 s |
| 106 | 107 MB | 664 MB | 185 MB | 0.08 / 0.10 s |

First calls: 0.4-2.1 s for block-lean, against 69-74 s for block-index (compilation).

**Correctness:**
- the evaluator's four 003 fixtures pass (`harness evaluate --smoke`, no snapshot file changed);
- real development workloads 101-107 pass against saved reference outputs.

**Not attempted:** a lower-memory representation of the UCS4 value codes (one byte per
character instead of four). It would reduce the working arrays, which are not what sets the peak
on these workloads.

## Evaluator runs

One thread, 2026-10-25 (`harness evaluate`; every fixture, first call, warmup and measured output
passed):
- real-development 101-107: `runs/20261025T171340.478397Z-bcb2f9c7`;
- 108 (one measured call): `runs/20261025T174042.214822Z-00c1645d`;
- synthetic 1-5: `runs/20261025T175621.771892Z-c03f05a8`;
- smoke: `runs/20261025T171329.897429Z-8ab87674`.

| Workload | Reference | block-lean | Speedup | Peak RSS, reference / block-lean | First call |
|---|---:|---:|---:|---:|---:|
| 101 | 25.80 s | 0.609 s | 42.4x | 4.47 / 0.30 GiB | 0.96 s |
| 102 | 3.212 s | 0.174 s | 18.5x | 1.80 / 0.21 GiB | 0.53 s |
| 103 | 13.20 s | 0.530 s | 24.9x | 3.49 / 0.31 GiB | 0.89 s |
| 104 | 98.85 s | 1.884 s | 52.5x | 13.18 / 0.76 GiB | 2.45 s |
| 105 | 30.90 s | 0.711 s | 43.5x | 5.46 / 0.38 GiB | 1.07 s |
| 106 | 10.68 s | 0.096 s | 111.4x | 1.83 / 0.20 GiB | 0.43 s |
| 107 | 12.46 s | 0.430 s | 29.0x | 3.15 / 0.36 GiB | 0.97 s |
| 108 | 341.4 s | 8.681 s | 39.3x | 22.46 / 9.05 GiB | 9.79 s |
| s1 | 0.430 s | 0.025 s | 16.9x | 0.37 / 0.12 GiB | 0.59 s |
| s2 | 1.872 s | 0.068 s | 27.6x | 1.12 / 0.14 GiB | 0.42 s |
| s3 | 12.36 s | 0.383 s | 32.3x | 2.65 / 0.24 GiB | 0.75 s |
| s4 | 64.43 s | 1.927 s | 33.4x | 6.21 / 0.99 GiB | 2.25 s |
| s5 | 137.3 s | 4.653 s | 29.5x | 10.64 / 3.77 GiB | 4.85 s |

**Against block-index's one-thread evaluator runs of 2026-10-22:**
- peaks: 0.63-1.15 GiB on 101-107, 9.53 GiB on 108, 0.57 GiB on s1;
- steady-state times: 0.622, 0.175, 0.522, 1.974, 0.711, 0.079, 0.344 s on 101-107, and 6.36 s on
  108;
- first calls: about 70 s.

108's peak (9.05 GiB) is the loader's 8.89 GiB plus 0.16 GiB.

**Throttling:** these runs were throttled in most monitor samples
(`runs/study-20261025/thermal-cases.jsonl`). The reference's 108 took 341 s here against 276 s
on 2026-10-22, and block-lean's 8.68 s against block-index's 6.36 s then. The time ratio within
a run is the comparison; block-lean's code path is block-index's.
