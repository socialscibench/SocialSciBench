# Candidate research: schur-lean (task 001)

## Objective

The minimal-peak-memory variant of `solutions/001/schur-cg`, for the same contract:
`001_reghdfe/reference.py` within the task's tolerances (beta 1e-6, SE 1e-4, exact nobs, df_a and
n_clusters). "Memory" is the evaluator's measure: the worker's peak resident set (`ru_maxrss`),
which includes the interpreter, imports, input loading, every call and the output serialization.

## What sets the peak

The evaluator's loader builds the regressors with a list of n-vectors and `np.column_stack`, so
loading alone peaks at about twice the input (104 with 50 regressors: 5.79 GiB for 2.77 GiB of
inputs).

schur-cg needs more than the loader (11.25 GiB on 104k50):
- the demeaned n x (k + 1) matrix (2.57 GiB on 104k50);
- several (levels of the eliminated fixed effect) x (k + 1) arrays (0.66 GiB each);
- conjugate-gradient vectors for all columns at once;
- int64 index arrays;
- about 0.3 GB for Numba's compilation at the first call.

## Method

schur-cg's computation, with these changes:
- **Column batches:** the columns of [y, X] are solved in balanced batches whose per-level
  arrays take at most 128 MB: the eliminated and other fixed effects' sums, the conjugate-gradient
  vectors, the operator's buffers. Only the fixed-effect coefficients of every column are kept
  (0.66 + 0.10 GiB on 104k50).
- **No demeaned matrix:** a demeaned row is [y, X] minus its fixed-effect coefficients. The rows
  are formed in chunks of 32 MB, twice: once for W'W (BLAS), and once, cluster by cluster, for
  the residuals and the cluster scores (segment sums). The products are those of schur-cg; only
  the order of summation differs.
- **Releasing memory:** the first codes and the singleton mask are released after recoding, and
  the stacked-level index after building the incidence structures.
- **Returning free pages:** the C allocator's free pages are returned to the OS at the start of
  the call and after the solve (`malloc_zone_pressure_relief` on macOS, `malloc_trim` with
  glibc). Pages the loader freed are otherwise still resident when the call allocates.
- **Prebuilt Numba cache:** the kernels load from Numba's on-disk cache (`__pycache__`, built by
  `build_cache.py`; data-independent compiled code, which AGENTS.md allows), so the measured
  process does not run the LLVM compiler.
  - The cache only survives the evaluator's snapshot if the cached code's module name is fixed.
    The evaluator imports a candidate as `submitted_<hash of its path>`, so `solution.py` imports
    `schur_lean.py` under a fixed name. Three kernels that called `numba.get_num_threads()`
    inside compiled code (a dynamic global, which Numba does not cache) take the thread count as
    an argument.
  - Numba validates the cache by the source file's modification time and size. `shutil.copy2`
    (the evaluator's snapshot) preserves them; a checkout that changes the time invalidates the
    cache, and then the kernels are compiled at the first call, which is correct but slower and
    larger.
  - At run time the cache is read from a private temporary copy (`numba.config.CACHE_DIR`), so a
    compilation for an argument type it lacks never writes into the evaluator's snapshot, which
    must not change. The first smoke run was invalidated for that reason: two kernels had been
    cached for writable arrays only. The build now passes read-only inputs, as the evaluator does.

## Measurements (development runs, fresh process per workload, one thread; 2026-10-25)

Peak RSS in GiB; schur-cg and the reference from the one-thread evaluator runs of 2026-10-22/23:

| Workload | Loader alone | Reference | schur-cg | schur-lean | schur-lean time | schur-cg time |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 0.25 | 0.62 | 0.76 | 0.47 | 0.61 s | 0.42 s |
| 102 | 0.73 | 1.68 | 1.77 | 1.05 | 2.71 s | 2.21 s |
| 103 | 0.95 | 2.30 | 2.16 | 1.31 | 3.44 s | 2.69 s |
| 104 (5 regressors) | 1.26 | 3.15 | 3.26 | 2.50 | 9.1 s | 8.1 s |
| 104 (20) | 2.77 | 6.56 | 5.78 | 3.48 | 19.1 s | 12.9 s |
| 104 (50) | 5.79 | 13.48 | 11.25 | 5.82 | 28 s | 20.8 s |
| 105 | 2.82 | 6.85 | 5.95 | 3.58 | 43.5 s | 25.1 s |
| 106 | 0.28 | 0.71 | 0.72 | 0.54 | 0.38 s | 0.23 s |
| 107 | 0.85 | 2.11 | 2.16 | 1.46 | 4.06 s | 3.56 s |
| 108 | 5.39 | 10.10 | 7.69 | 5.39 | 22.8 s | 14.4 s |

These schur-lean measurements predate the balanced batches (2 threads afterwards: 105 24.2 s,
104k50 20.0 s); the evaluator runs replace them.

**How much each change saved:**
- **Prebuilt cache:** first calls take 0.3-0.8 s instead of 11-46 s, and the peak drops by about
  0.2 GB on the small workloads (106: 0.66 to 0.49 GiB; 101: 0.71 to 0.50 GiB).
- **Returning the loader's free pages at the start:** 104k50 went from 7.39 to 5.82 GiB, which is
  the loader's own peak; the call no longer raises it.
- **Remaining excess over the loader on the mid-size workloads** (about 0.25-1.2 GiB): the int64
  index arrays (codes, row orders, the incidence structures: about 1 GB on 104) and the
  per-level coefficient arrays. Converting the index arrays to int32 is the next step; not done.

**Speed:** the cost of the smaller footprint is time, 10-75% at one thread. The conjugate
gradients run once per batch, and each iteration's approximate Cholesky solves traverse the whole
factor whatever the number of columns. Workload 105 (four fixed effects; 21 columns in three
batches) is the slowest.

**Correctness:**
- the evaluator's five 001 fixtures pass (`harness evaluate --smoke`);
- all ten real development workloads pass against saved reference outputs (2 threads).

## Evaluator runs

One thread, 2026-10-25 (`harness evaluate`; every fixture, first call, warmup and measured output
passed). Median of five measured calls for 101-107, and one for the 104 variants, 105 and 108,
reference and candidate alike. schur-cg and its reference are from the evaluator runs of
2026-10-22/23:

| Workload | Reference | schur-lean | Speedup | Peak RSS, reference / schur-lean | schur-cg time | schur-cg peak |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 29.55 s | 0.532 s | 55.6x | 0.59 / 0.50 GiB | 0.422 s | 0.76 GiB |
| 102 | 102.5 s | 3.265 s | 31.4x | 1.70 / 1.14 GiB | 2.210 s | 1.77 GiB |
| 103 | 118.8 s | 3.842 s | 30.9x | 2.28 / 1.40 GiB | 2.690 s | 2.16 GiB |
| 104 (20 regressors) | 541.8 s | 18.25 s | 29.7x | 6.55 / 3.24 GiB | 12.88 s | 5.78 GiB |
| 104 (5) | 173.4 s | 9.834 s | 17.6x | 3.13 / 2.46 GiB | 8.144 s | 3.26 GiB |
| 104 (50) | 1,356 s | 30.85 s | 44.0x | 11.66 / 5.78 GiB | 20.78 s | 11.25 GiB |
| 106 | 3.60 s | 0.418 s | 8.6x | 0.71 / 0.54 GiB | 0.233 s | 0.72 GiB |
| 107 | 21.04 s | 4.171 s | 5.0x | 2.12 / 1.49 GiB | 3.563 s | 2.16 GiB |

**Runs:**
- 101-107: `runs/20261025T181953.890783Z-7c2f8ea2`;
- 104 (20, 5 and 50 regressors): `runs/20261025T185642.936804Z-9e839d4a`,
  `runs/20261025T191800.237299Z-b97583c5` and `runs/20261025T192624.376201Z-94bbf764`;
- 109 (compatibility only): `runs/20261025T185612.408476Z-9f0d2212`;
- smoke: `runs/20261025T171320.187251Z-6ba1e490`.

**Peak memory:** 15-51% below the reference's and 25-49% below schur-cg's. 104 with 50 regressors
peaks at the loader's own 5.79 GiB (5.78 GiB measured).

**Speed:** 1.2-1.8x schur-cg's time (the batches; see above).

**First calls:** 0.8-32 s here. They include one call's work; there is no compilation, against
11.5-46.1 s for schur-cg.

**105 and 108** (one measured call each, reference and candidate alike):

| Workload | Reference | schur-lean | Speedup | Peak RSS, reference / schur-lean | schur-cg time | schur-cg peak |
|---|---:|---:|---:|---:|---:|---:|
| 105 | 766.8 s | 41.40 s | 18.5x | 6.85 / 3.46 GiB | 25.12 s | 5.95 GiB |
| 108 | 654.5 s | 19.00 s | 34.4x | 10.10 / 5.36 GiB | 14.37 s | 7.69 GiB |

Runs: `runs/20261025T201428.660356Z-55c1b674` (105) and `runs/20261025T204408.349289Z-ca408a1f`
(108). 108 peaks at the loader's own 5.39 GiB. 105 is the slowest relative to schur-cg (1.65x):
its four fixed effects and 21 columns run as three batches of conjugate gradients.

**Synthetic** (`runs/20261025T210938.331957Z-5ca4a476`; median of three measured calls). schur-cg's
one-thread evaluator times and peaks of 2026-10-23 are given for comparison:

| Workload | Reference | schur-lean | Speedup | Peak RSS, reference / schur-lean | schur-cg time | schur-cg peak |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 0.25 s | 0.036 s | 6.9x | 0.12 / 0.16 GiB | 0.021 s | 0.26 GiB |
| s3 | 22.85 s | 3.594 s | 6.4x | 4.75 / 1.93 GiB | 1.846 s | 3.42 GiB |
| s4 | 17.72 s | 1.580 s | 11.2x | 1.22 / 0.87 GiB | 1.126 s | 1.25 GiB |
| s6 | 1.82 s | 0.238 s | 7.6x | 0.34 / 0.43 GiB | 0.181 s | 0.58 GiB |
| s7 | 24.75 s | 2.896 s | 8.5x | 2.38 / 1.21 GiB | 1.794 s | 2.19 GiB |
| s8 | 49.58 s | 1.195 s | 41.5x | 1.21 / 0.81 GiB | 0.721 s | 1.20 GiB |
| s9 | 120.0 s | 1.049 s | 114.4x | 1.25 / 0.75 GiB | 0.677 s | 1.17 GiB |
| s10 | 6.97 s | 1.351 s | 5.2x | 2.21 / 1.07 GiB | 0.593 s | 1.67 GiB |
| s12 | 69.50 s | 12.16 s | 5.7x | 14.37 / 6.78 GiB | 6.010 s | 11.79 GiB |

**Peak memory:** below schur-cg's on every synthetic workload. On s1 and s6 it is above the
reference's: Numba's import and cached code (about 0.1 GB) against the reference's SciPy, on
inputs of a few MB.

**Speed:** 1.3-2.3x schur-cg's time; s10 and s12, with 40 and 50 regressors, run as several
batches.

**s11** (`runs/20261025T214343.671021Z-325b5c8a`, one measured call; worker/firm graph with 1
million units and low mobility):
- reference: 1,996 s, 9.92 GiB peak;
- schur-lean: 13.23 s (150.9x), 4.17 GiB peak;
- schur-cg (2026-10-23): 8.486 s, 8.22 GiB peak.
