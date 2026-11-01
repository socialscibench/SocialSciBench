# Candidate research: cell-lean (task 002, combined task of 2026-10-25)

## Objective

The minimal-peak-memory variant of `solutions/002/cell-fused`, for the same contract:
`002_csdid/reference.py` under csdid 1.81's or 2.0.1's rules with every option, within the task's
tolerances. "Memory" is the evaluator's measure, the worker's peak resident set (`ru_maxrss`).
That covers the interpreter, imports, input loading (including the loader's generation of the
bootstrap multipliers), every call and the output serialization. Speed is secondary but should
stay well ahead of the reference.

## What sets the peak

Measured in a fresh process: load the workload as the evaluator does, import, call twice (one
BLAS thread; `ru_maxrss`, MB; 2026-10-25).

| Workload | After loading | Reference | cell-fused (speed) | cell-lean |
|---|---:|---:|---:|---:|
| 106 | 90 | 276 | 452 | 112 |
| 102 | 88 | - | - | 122 |
| 101 | 108 | - | - | 137 |
| 105 | 90 | 752 | 477 | 159 |
| 104 | 100 | 2,676 | 528 | 195 |
| 108 | 122 | 865 | 1,882 | 296 |
| 107 | 577-593 | 5,793 | 1,271 | 637 |

The table shows what sets each peak:
- **cell-fused:** Numba's compilation at the first call takes about 330 MB, and Numba's import
  about 40 MB.
- **The reference:** it keeps two n_units x n_cells arrays; with the bootstrap, it also keeps a
  dense n_units x n_columns array and the n_draws x n_columns draws.
- **Bootstrap workloads (101, 107, 108):** the evaluator's loader generates the multipliers
  before the call, using about three n_units x n_draws int8 temporaries (107: 200 MB each). That
  sets a floor no candidate can go below.

## Method

The same computation as `cell-fused` (see its `research.md`), organized for small working memory,
NumPy only:
- **Estimation:** one cohort at a time; consecutive cells with the same units and covariates share
  the fits and the sums that do not depend on the outcome change. The Newton Hessians and the
  regression matrices are accumulated over chunks of 2^16 elements, so no n_cell x k temporary
  arrays are formed.
- **Pass over units:** blocks of one cohort's units, with working matrices of at most 2^16
  elements (512 KB). A block's influence functions for all its cells form one units x cells
  matrix. The aggregate influence functions are only accumulated as sums and a cross-product
  (Gram) matrix (n_aggregates^2), which give every aggregate's standard error, the cohort
  average's (with 2.0.1's unit-level cohort weights) and the event-study covariance. No per-unit
  aggregate array is kept (cell-fused keeps an n_units x n_aggregates array: 111 MB on 107).
- **Bootstrap:** the same blocks times their multipliers (int8 converted to float64 in pieces),
  for tiles of cells and draws of at most 2^19 elements (4 MB). Workload 108, with 25,000 draws x
  812 cells (162 MB of draws), runs in about 40 tiles, each a pass over its 775 units.
  - Each tile's bootstrap SEs come from `np.partition`; its contribution to the critical value's
    maximum and to the aggregate draws is accumulated, then the tile is freed.
- **Returning memory:** on macOS the allocator keeps freed blocks of a few MB resident, and the
  resident peak then grows with every cohort even though the live arrays are small:
  - on 104, NumPy's traced peak was 39 MB while the resident set grew 100 to 286 MB;
  - with `MallocMediumZone=0` in the environment, 108 went from 279 to 222 MB.

  A candidate cannot set allocator variables in the evaluator's process, so it calls
  `malloc_zone_pressure_relief` (glibc: `malloc_trim`) after every 64 MB of large (4 MB or more)
  freed arrays.

## Measurements and tuning

Time per call, one BLAS thread, development runs (not evaluator runs): 106 0.05 s, 102 1.2-1.4 s,
101 0.4-0.5 s, 104 2.5-2.8 s, 105 3.2-4.0 s, 108 3.1-3.3 s, 107 7.1-7.6 s (cell-fused: 0.015,
0.23, 0.15, 0.44, 1.0, 1.27, 3.2 s).

**Unit-pass working-matrix size** (104, one thread):

| Size (elements) | 2^14 | 2^16 | 2^18 | 2^20 | 2^22 |
|---|---:|---:|---:|---:|---:|
| Peak | 148 MB | 164 MB | 294 MB | 762 MB | 652 MB |
| Time | 6.8 s | 3.5 s | 3.1 s | 3.5 s | 3.4 s |

2^16 is used.

**Other changes:**
- **Caching the units' outcome matrix per shared fit:** slower on 105, where no two cells share
  their units, and 16 MB per cohort on 107. Replaced by two column gathers per cell.
- **Returning free pages after every cohort and tile:** 107 went from 738 to 609 MB, but 105 took
  2x as long (3.3-4.1 to 6.5-7.4 s) and 104 1.8x, because the next allocations fault their pages
  again. Counting only freed arrays of 4 MB or more, and returning pages every 64 MB, keeps 104
  and 105 at their previous times. Then 107 reaches 637 MB at 7.1-7.6 s (6.1 s without
  returning pages) and 108 296 MB.

**Correctness:**
- the evaluator's 14 fixtures and real workloads 101-109 pass;
- `fuzz.py` (1,000 random panels, seed 7000) passes 783 of the 784 panels where the reference
  succeeds;
- the remaining panel (7276) is the degenerate one described in cell-fused's notes: a cell whose
  influence function is constant, so its bootstrap SE is rounding residue. The candidate raises
  the reference's own IndexError for an empty bootstrap maximum.

## Evaluator runs

One thread, 2026-10-25 (`harness evaluate`; every fixture, first call, warmup and measured output
passed). Median of five (real) or three (synthetic) measured calls, against the reference of the
same run:

| Workload | Reference | cell-lean | Speedup | Peak RSS, reference / cell-lean |
|---|---:|---:|---:|---:|
| 101 | 2.949 s | 0.424 s | 7.0x | 0.45 / 0.19 GiB |
| 102 | 4.020 s | 1.514 s | 2.7x | 0.46 / 0.16 GiB |
| 103 | 1.222 s | 0.365 s | 3.3x | 0.49 / 0.13 GiB |
| 104 | 82.44 s | 3.037 s | 27.1x | 2.45 / 0.23 GiB |
| 105 | 23.16 s | 4.158 s | 5.6x | 0.80 / 0.19 GiB |
| 106 | 0.305 s | 0.050 s | 6.1x | 0.37 / 0.11 GiB |
| 107 | 59.32 s | 8.102 s | 7.3x | 5.82 / 0.60 GiB |
| 108 | 2.031 s | 3.751 s | 0.5x (slower) | 0.91 / 0.35 GiB |
| 109 | 1.294 s | 0.549 s | 2.4x | 0.39 / 0.13 GiB |
| s1 | 0.031 s | 0.010 s | 3.0x | 0.04 / 0.04 GiB |
| s2 | 0.392 s | 0.051 s | 7.7x | 0.14 / 0.06 GiB |
| s3 | 2.794 s | 0.255 s | 11.0x | 0.31 / 0.09 GiB |
| s4 | 4.758 s | 0.546 s | 8.7x | 0.92 / 0.12 GiB |
| s5 | 92.06 s | 10.39 s | 8.9x | 12.37 / 0.22 GiB |
| s6 | 93.04 s | 14.77 s | 6.3x | 5.94 / 0.49 GiB |

**Runs:**
- real-development: `runs/20261025T143021.988145Z-13a88526`;
- synthetic: `runs/20261025T145326.956088Z-012394ea`;
- smoke: `runs/20261025T132212.867772Z-17799882`.

**Peak memory:** 0.04-0.60 GiB: 2.4-10.8x below the reference's on the real workloads and 1.0-57x
on the synthetic ones, and 2.1-9.8x below cell-fused's (0.37-1.87 GiB in its runs of the same
day). 107's 0.60 GiB is the loader's multiplier generation (about 0.58 GiB before the call).

**108 is slower than the reference** (0.5x): its 25,000 draws for 812 cells run as about 40 tiles
of 4 MB, each a pass over the units in NumPy. The reference holds the whole table at once
(0.91 GiB peak). This is the memory-for-time trade the variant makes.

**Throttling:** the runs were throttled in most samples (lowest speed limit 73%).

Four threads, real-development (`runs/20261025T162327.654759Z-b5659a9b`; every output passed):
cell-lean runs in one thread apart from BLAS, so its times are those of one thread.

| Workload | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cell-lean | 0.434 s | 1.529 s | 0.454 s | 3.781 s | 4.997 s | 0.054 s | 8.456 s | 4.404 s | 0.569 s |
| Speedup | 8.4x | 2.6x | 4.1x | 31.8x | 8.0x | 5.5x | 6.9x | 0.4x | 3.8x |
| Peak RSS | 0.18 GiB | 0.15 GiB | 0.14 GiB | 0.25 GiB | 0.18 GiB | 0.13 GiB | 0.62 GiB | 0.32 GiB | 0.13 GiB |

The speedups are against the four-thread reference of the same run (3.65, 3.91, 1.85, 120.2,
40.2, 0.30, 58.5, 1.67 and 2.14 s).

Four threads, synthetic (`runs/20261025T165253.800275Z-80a438ae`; every output passed):

| Workload | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 0.030 s | 0.470 s | 3.049 s | 6.926 s | 89.63 s | 94.90 s |
| cell-lean | 0.011 s | 0.053 s | 0.271 s | 0.769 s | 11.94 s | 17.31 s |
| Speedup | 2.7x | 8.9x | 11.2x | 9.0x | 7.5x | 5.5x |
| Peak RSS, reference / cell-lean | 0.04 / 0.04 GiB | 0.13 / 0.08 GiB | 0.26 / 0.09 GiB | 0.84 / 0.12 GiB | 11.65 / 0.24 GiB | 5.90 / 0.45 GiB |
