# Candidate research: cell-fused (task 002, combined task of 2026-10-25)

## Objective and constraints

Reproduce `002_csdid/reference.py` (the combined task: the former tasks 002, 002b and 002c) within
the task's tolerances, faster, on one and on several cores:

- **Estimator:** Stata `csdid` under 1.81's or 2.0.1's rules with the options of `reference.run`:
  - never or not-yet-treated controls, sampling weights, unbalanced panels with pair-balanced cells;
  - dripw, reg, ipw, stdipw and drimp;
  - varying or universal base periods, propensity-score trimming, and covariates that are fixed or vary over time;
  - csdid 2.0.1's overlap check and cohort-weight rules;
  - the simple, group, calendar and windowed event-study aggregates, event-time subsets and the event-study covariance;
  - analytic influence-function standard errors, and the multiplier bootstrap with the given +1/-1 multipliers (bootstrap standard errors and simultaneous critical values).
- **Tolerances:**
  - estimates: error / max(|reference|, its SE) ≤ 1e-6;
  - standard errors, bootstrap SEs and critical values: relative 1e-4;
  - covariance: error / (SE_i SE_j) ≤ 1e-4.
- **Workloads:**
  - real development: 101-109;
  - synthetic: 1-6 (the former task's, with the default options);
  - the evaluator's fixtures: 14 option combinations.
- **History:** this candidate continues the task-002b candidate of the same name. Its 002b
  version is described in "Previous versions" below; the former task 002's candidate,
  `solutions/002/cohort-fused`, belongs to the old task definition and is kept for the record.

## Method

The reference loops over the (cohort, period) cells, solves each as a separate small problem with
NumPy, keeps every cell's full-sample influence function and cohort-weight column in two
n_units x n_cells arrays, aggregates them with csdid's `aggte`, and multiplies the stacked
influence functions by the multipliers (a dense n_units x n_columns x n_draws product). The
candidate (`solution.py` docstring) computes the same quantities in four passes:

1. **Cells and their sizes:** the cells in csdid's order for either base-period rule.
   - Comparison thresholds follow the rule set: not-yet-treated units first treated after
     max(g, t), or after t under 2.0.1's rules with varying base periods.
   - Each cell's covariate period is its earlier period.
   - Cells whose two periods are observed for every unit (balanced panels) are counted from
     cohort tallies; the others unit by unit, in parallel over cells.
2. **Estimation** (parallel over cohorts, each cohort's cells in order):
   - **Overlap check:** csdid 2.0.1's check is an unweighted logit. Without weights it is the
     propensity fit itself; with weights, or for drimp's tilting, it is a separate warm-started
     fit.
   - **Fits:** the propensity fit (weighted logit or inverse probability tilting) uses Newton's
     method with the reference's stopping rule; the weighted least-squares regression follows.
   - **Shared fits:** consecutive cells of a cohort often use the same units and covariates
     (with never-treated controls and fixed covariates in a balanced panel, every cell of the
     cohort does). They then share the fits and the sums that do not depend on the outcome
     change, so each further cell costs one pass for 2 + 2k sums.
   - Other fits start from the previous cell's solution.
   - Cells the overlap check rejects are removed before the aggregation weights are formed.
3. **One pass over units** (parallel over blocks of 64 units of one cohort): for each unit, every
   cell it belongs to contributes its influence-function value to the cell's mean and sum of
   squares and, with the aggregation weights, to the unit's influence functions of the aggregates.
   - These include the terms for the estimated cohort weights; under 2.0.1's rules every unit of
     the cohort carries the weight, observed in the cell or not.
   - Consecutive cells with the same propensity fit reuse the lanes' scores.
   - The block's aggregate influence functions are accumulated in a block-local column-major
     buffer.
4. **Bootstrap:** the pass over units runs in chunks of one cohort's units. Each chunk's cell
   influence functions are multiplied by its multipliers with BLAS (Accelerate), together with
   the units' weights and ones.
   - Only the cells the cohort belongs to are included: with never-treated controls, a cohort's
     units appear in its own cells only.
   - The aggregate tables' draws are then linear combinations of the cell draws plus the cohort
     weights' terms, so no aggregate column is multiplied by the multipliers.
   - Standard errors and critical values come from order statistics by quickselect, not by sorting.

Exactness safeguards: a cell is recomputed with the reference's own NumPy operations when any of
the following holds:
- its fit is singular, or does not converge from the warm start and from the reference's start;
- a comparison unit's score lies within 1e-9 of the trimming cut;
- the overlap score lies within 1e-9 of 0.999;
- a warm-started overlap fit has an extreme index (|x'b| > 25) and does not converge from the
  reference's start.

Also, a cell whose standard error is at rounding level relative to its influence-function values
gets the reference's own ATT and standard error.

## Sources (accessed 2026-10-25)

- The combined task package (the task's `task.md`, `reference.py` and README, by the package's
  author): definitions, csdid 1.81 and 2.0.1 rules, the Stata comparison.
- Callaway & Sant'Anna (2021), J. Econometrics 225(2); Sant'Anna & Zhao (2020), J. Econometrics
  219(1); Abadie (2005), Rev. Econ. Stud. 72(1); for the multiplier bootstrap and uniform bands,
  Callaway & Sant'Anna (2021), section 4, and R `did` 2.5.1 (`mboot`, `compute.aggte`),
  https://github.com/bcallaway11/did.
- csdid / drdid (Stata) https://github.com/friosavila/stpackages (1.81/1.82 on SSC; 2.0.1, commit
  f98bcea, as cited by the task).
- Method of four Russians for products with +1/-1 matrices (Arlazarov et al. 1970; e.g. Albrecht
  et al., "M4RI", ACM TOMS 2010) — tried, see below.

## Measurements and unsuccessful attempts

Development measurements (single process per workload, second call after a first, compiling, call;
`solutions/.venv`; not evaluator runs; one thread: NUMBA_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1):

| Version | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3.0 (first passing combined-task version) | 0.255 s | 0.312 s | - | 1.314 s | 1.343 s | - | 5.456 s | 1.931 s | - |
| v3.1 | 0.149 s | 0.234 s | 0.066 s | 0.441 s | 1.015 s | 0.015 s | 3.176 s | 1.270 s | 0.141 s |
| v3.1, 4 threads | - | - | - | 0.218 s | 0.338 s | - | 1.640 s | 0.881 s | - |

The reference's development times (one call, multithreaded BLAS, 2026-10-25) are about 50 s on
107, 43-57 s on 104, 19-54 s on 105, 2-3 s on 101, 102 and 108, and under 3.6 s on the rest. They
vary with the host's throttling; the evaluator runs give the measured comparison.

**v3.0 to v3.1 (one thread):**
- **Balanced-panel cell counts from cohort tallies:** 107: 0.33 to 0.01 s; 104: 0.10 to 0.002 s.
- **Hoisted sums for cells sharing a fit:** one pass for 2 + 2k sums per cell instead of the
  per-cell constants' seven passes. 107 estimation: 1.11 to 0.70 s; 104: 0.38 to 0.18 s.
- **Unit pass:** a block-local column-major accumulator replaces strided updates of the
  n_units x n_aggregates array, and the scores of consecutive cells with the same fit are
  reused. 107: 1.28 to 0.45 s; 104: 0.81 to 0.22 s.
- **Bootstrap:** the aggregate tables' draws are derived from the cell draws, which removes the
  n_units x 70 aggregate columns (28%) from the product with the multipliers. A
  per-chunk fancy-index accumulation, which cost 2.2 s on 108 (25,000 draws), is replaced by row
  adds. Quickselect replaces `np.partition` for the order statistics. 107 bootstrap: 2.4 to
  1.8 s; 108: 1.6 to 1.0 s.
- **Logit covariance:** the inverse Hessian of the last Newton iteration is used. It is
  evaluated less than 1e-10 from the solution; the reference evaluates it at the solution. This
  saves one pass per fit; 105: 1.24 to 1.02 s.

**Unsuccessful attempts:**
- **Method of four Russians for the +1/-1 products** (tables of the 256 signed sums of 8 units'
  values per tile of 16 columns; each draw adds one table row). On a 2048-unit x 361-cell x
  999-draw chunk (one thread), Accelerate's `dgemm` took 32.5 ms (45 GFLOP/s, including the
  int8 to float64 conversion of the multipliers):
  - the table kernel took 57.7 ms (per-draw accumulator rows);
  - it took 44.9 ms with 16 scalar accumulators kept across blocks of 8 tables.

  The lookups did not vectorize. In the solution itself it was slower than BLAS: 107's product
  went from 2.24 s to 3.55 s plus 0.86 s for the bit patterns. Removed.
- **Lower-precision products** (float32 BLAS, or hi/lo splits): rejected before measuring. The
  float32 accumulation over 10^5 units gives draw errors of about 1e-5 relative, too close to
  the 1e-4 tolerance on bootstrap SEs and critical values.
- **Structure of never-treated controls:** factoring the never-treated units' cell influence
  functions into per-cohort features (outcome levels by period, covariates) needs T + 2k + 1
  features per cohort against T - 1 cells. Rejected (more work).

**Remaining costs** (v3.1, one thread):
- 107: the multiplier product, 1.6-1.8 s for 7.2e10 flops at Accelerate's 45 GFLOP/s (the
  reference's dense product is 1.65e11 flops).
- 105: the 1,681 separate warm-started logits (no two cells share their units under 2.0.1's
  not-yet-treated rule), 0.84 s. An incremental first Newton step from the previous cell's
  Hessian is the next idea; not attempted.
- 108: 25,000 draws for 812 cells, 0.55 s product and 0.42 s order statistics.

**Random panels** (`fuzz.py`: balanced or with gaps and exits; 0-3 covariates, fixed or varying
over time, sometimes missing; tiny cohorts; binary outcomes; weights; every option, including
20-300 bootstrap draws).
- v3.0: 464 of 464 panels where the reference succeeds pass (600 drawn).
- v3.1: 783 of 784 pass (1,000 drawn, seed 7000).
- The failing panel (7276) has two cells whose influence function is constant in exact
  arithmetic. Their analytic and bootstrap standard errors are rounding residue (3e-16 and
  6e-16 in the reference), and the gt table's critical value divides draws by those values, so
  the reference's critical value is set by rounding noise and cannot be reproduced.
- The reference fails outright (IndexError) when such a cell's draws are exactly zero; the
  candidate raises the same error then.
- None of the task's workloads has such a cell.
- Reference failures among the drawn panels:
  - no pre-treatment event in the window (non-finite pre_avg): 75;
  - an empty event set or no cells: 43;
  - an empty bootstrap maximum: 7;
  - non-convergence: 10.

## Previous versions (task 002b, 2026-10-23)

The task-002b version was the first to use the per-cohort shared fits, the cell-count-based
aggregation weights and the blocked, vectorized pass over units, with these gains:
- one-dimensional row views in the Newton and constant kernels: 20x on the Hessian kernel; 502
  went from 18.3 to 7.3 s;
- the blocked pass over units: 502 from 3.2 to 1.27 s.

Its 002b workloads are no longer part of the benchmark; its record is kept with its code in
`solutions/002/legacy-pre-merge/002b/cell-fused/research.md`.

## Evaluator runs

One thread, 2026-10-25 (`harness evaluate`, real-development 101-109 and synthetic 1-6; every
fixture, first call, warmup and measured output passed). Median of five measured calls (real) or
three (synthetic), against the reference of the same run:

| Workload | Reference | cell-fused | Speedup | Peak RSS, reference / cell-fused |
|---|---:|---:|---:|---:|
| 101 | 2.754 s | 0.152 s | 18.1x | 0.40 / 0.55 GiB |
| 102 | 3.644 s | 0.236 s | 15.4x | 0.44 / 0.45 GiB |
| 103 | 1.306 s | 0.075 s | 17.4x | 0.42 / 0.47 GiB |
| 104 | 102.3 s | 0.445 s | 229.9x | 2.66 / 0.57 GiB |
| 105 | 24.99 s | 0.994 s | 25.1x | 0.82 / 0.49 GiB |
| 106 | 0.298 s | 0.016 s | 19.0x | 0.37 / 0.44 GiB |
| 107 | 61.20 s | 3.306 s | 18.5x | 5.80 / 1.27 GiB |
| 108 | 2.015 s | 1.337 s | 1.5x | 0.90 / 1.87 GiB |
| 109 | 1.419 s | 0.135 s | 10.5x | 0.40 / 0.51 GiB |
| s1 | 0.028 s | 0.003 s | 10.3x | 0.04 / 0.37 GiB |
| s2 | 0.317 s | 0.010 s | 32.8x | 0.14 / 0.39 GiB |
| s3 | 2.901 s | 0.051 s | 56.4x | 0.30 / 0.39 GiB |
| s4 | 4.918 s | 0.198 s | 24.8x | 0.90 / 0.59 GiB |
| s5 | 92.56 s | 1.854 s | 49.9x | 12.31 / 0.85 GiB |
| s6 | 92.25 s | 3.271 s | 28.2x | 5.96 / 1.23 GiB |

**Runs:**
- real-development: `runs/20261025T132222.994847Z-325195eb`;
- synthetic: `runs/20261025T140015.833287Z-603526de`;
- smoke: `runs/20261025T131159.843213Z-7754c15c`.

**Largest relative error:** 1.1e-10, on 108.

**Throttling:** both runs were throttled in most monitor samples (lowest speed limit 71%;
`runs/study-20261025/thermal-cases.jsonl`). The reference's 104 took 101-107 s here, against
30.4 s in the saved-reference run earlier in the day and 67-85 s in the cell-lean run; the
speedups compare calls of the same run.

**Peak memory:** on the small workloads it is Numba's first-call compilation (about 0.33 GB),
not the computation. 108's 1.87 GiB is its 812 x 25,000 table of draws with temporaries.

**108 is the weak case:** 25,000 draws, so the multiplier products and the order statistics
dominate, as they do for the reference.

Four threads, 2026-10-25 (`runs/20261025T151318.614984Z-5e42de86`, real-development; every
output passed; no sleep, 409 of 461 monitor samples throttled, lowest speed limit 60%):

| Workload | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 3.524 s | 3.688 s | 1.776 s | 114.6 s | 38.67 s | 0.325 s | 58.64 s | 1.679 s | 1.851 s |
| cell-fused | 0.083 s | 0.079 s | 0.029 s | 0.199 s | 0.353 s | 0.013 s | 1.802 s | 0.929 s | 0.073 s |
| Speedup | 42.6x | 46.9x | 60.7x | 575x | 110x | 25.1x | 32.5x | 1.8x | 25.2x |

Scaling from one to four threads: 107 3.31 to 1.80 s (1.8x), 104 0.445 to 0.199 s (2.2x), 105
0.99 to 0.35 s (2.8x), 108 1.34 to 0.93 s (1.4x; the per-cohort products of 108 are small).

Four threads, synthetic (`runs/20261025T155317.049574Z-e43365d8`; every output passed; no sleep,
283 of 343 samples throttled, lowest limit 69%):

| Workload | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 0.028 s | 0.440 s | 3.162 s | 6.833 s | 89.00 s | 92.11 s |
| cell-fused | 0.003 s | 0.007 s | 0.021 s | 0.118 s | 1.393 s | 1.925 s |
| Speedup | 9.4x | 66.1x | 152.8x | 58.1x | 63.9x | 47.8x |
