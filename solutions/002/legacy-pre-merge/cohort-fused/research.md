# Candidate research: cohort-fused

## Objective and constraints

- Task 002, contract version 2: Callaway & Sant'Anna (2021) ATT(g,t) with the doubly robust IPW
  estimator of Sant'Anna & Zhao (2020) (`csdid ..., method(dripw)`: never-treated controls,
  varying base period, no propensity-score trimming since the task's correction of 2026-10-23
  (version 1 of this candidate followed the earlier reference, which trimmed at 0.995), analytic
  influence-function standard errors), the simple ATT, the event study and the pre/post averages. Must match `reference.run`:
  point estimates to 1e-6 and standard errors to 1e-4 relative error, labels exactly.
- Target hardware: CPU. Configurations: one thread (`--threads 1`) and six threads
  (`--threads 6`, the host's physical cores). Host: Intel Core i7-8700B (6 cores, 12 logical
  CPUs), 32 GiB, macOS 15.7.3. Environment: `solutions/.venv` (`build.md`): CPython 3.12.13,
  NumPy 2.3.5 (Accelerate BLAS), SciPy 1.18.1, Numba 0.62.1 (llvmlite 0.45.1).
- Research setting: developed by the same agent that, earlier in the same conversation, built
  task 002 and task 003 candidates. Those files and the repository's
  reports, baselines, published records and run directories were not opened in this session; the
  only prior knowledge carried over is what that conversation summary contained (the earlier
  002 candidate's name, `shared-nuisance` (now in `prestudy/solutions/002/`), and that it shared the per-cell nuisance fits; none
  of its code). The task files, the harness and the development data were used.

## Method

The reference estimates each (cohort g, period t) cell separately: a logit of cohort membership
on X among the cohort and the never-treated units (the propensity score), an OLS of the outcome
change on X among the never-treated units (the outcome model), and the doubly robust ATT with its
influence function; it stores every cell's influence function in an n_units x n_cells matrix and
aggregates it. Observations:

1. The propensity score depends on the cohort only (same units, same X, same D for every t), so
   one logit per cohort replaces (T - 1) identical logits.
2. The outcome model is an OLS on the never-treated units for every cell; it is linear in the
   outcome, so B = (X0'X0)^-1 X0'Y0 for all periods at once gives every cell's coefficients
   (B_t - B_b), and the residual of a cell is E_t - E_b with E = Y - X B (computed once, n x T).
3. Each cell's influence function is, per unit, a short expression in E_t - E_b, x_i, the
   cohort's propensity scores and a few cell constants that are sums over units (cohort sums of E,
   propensity-weighted control sums of E and of x E). Those sums are computed for all cohorts in
   one pass over the control units.
4. The aggregates' influence functions are linear combinations of the cells' plus the terms for
   the estimated cohort-size weights (which only involve a unit's own cohort). One fused pass over
   the units computes, per unit, all its cells' influence values, accumulates the per-cell sums
   the cell standard errors need, and forms its simple, event-time and pre/post aggregated values,
   accumulating their sums and sums of squares; the n x n_cells matrix is never formed.

Exactness safeguards (the reference defines the answer, including its numerical behaviour):

- A cohort whose fast Newton iteration does not converge (60 iterations, max |step| < 1e-12) or
  whose final Hessian has condition number above 1e8 (e.g. perfect separation with a handful of
  treated units) is recomputed cell by cell with the reference's own NumPy code (its logit and
  `drdid_panel`), and its cells enter the aggregates through their exact influence columns. On
  the development data this happens for one cohort of workload 108 (5 treated units, Newton runs
  the full 100 iterations, Hessian condition number about 1e21).
- A cell whose ATT is zero up to rounding relative to the outcome's level (|ATT| <= 1e-9 x the
  mean absolute outcome in its two periods) is also recomputed with the reference's code, and its
  mean is taken over the n_units x 1 column the way NumPy reduces the reference's matrix (axis 0).

## Sources (accessed 2026-10-22)

- Callaway, B. and Sant'Anna, P. H. C. (2021), "Difference-in-Differences with multiple time
  periods", Journal of Econometrics 225(2): 200-230.
- Sant'Anna, P. H. C. and Zhao, J. (2020), "Doubly robust difference-in-differences estimators",
  Journal of Econometrics 219(1): 101-122.
- R packages `did` 2.5.1 and `DRDID` 1.3.0 (conda-forge builds), https://bcallaway11.github.io/did/,
  https://psantanna.com/DRDID/ (definitions of `att_gt`, `aggte`, `drdid_panel`).
- R package `fastdid` (Tsai), https://tsailintung.github.io/fastdid/reference/fastdid.html.
- Numba documentation (performance tips, `@intrinsic`),
  https://numba.readthedocs.io/en/stable/user/performance-tips.html.
- Cody-Waite argument reduction for exp; FreeBSD msun `s_cbrtf.c` (constant for the float32
  cube-root estimate is used in task 003, not here).

## Hypotheses, measurements and unsuccessful attempts

Timings below are single calls after a first call, from development scripts (not evaluator runs),
one Numba thread and one BLAS thread unless stated.

| Change | 105 | 103 | 107 | s6 |
|---|---:|---:|---:|---:|
| Reference (one call) | 113.9 s | 26.8 s | 45.6 s | 133.5 s |
| 1. Per-cohort logit and OLS, fused pass (first version) | 3.07 s | 1.37 s | 1.54 s | - |
| 2. `error_model="numpy"` (divisions no longer block vectorization) | 1.06 s | 0.45 s | 0.68 s | 1.46 s |
| 3. Logit over column-blocked covariates, vectorizable exp, reassociated reductions | 0.90 s | 0.47 s | 0.67 s | 1.38 s |
| 4. Unsigned indices, no slices in hot loops | 0.93 s | - | - | 1.16 s |
| 5. Passes over units vectorized across blocks of 64 units | 0.68 s | 0.33 s | 0.49 s | 1.00 s |

- The logit dominated the first version (2.1 of 3.1 s on 105): Numba's default error model
  checks every division for zero, which kept the exp/division loop scalar (25 -> 12 ns per unit
  and iteration after `error_model="numpy"`); copying covariate blocks cost 2.8 ns per unit, the
  Hessian dot products 3.1 ns (vectorized only with reassociation allowed).
- A branch-free exp (Cody-Waite reduction, degree-13 Taylor polynomial, exponent from integer
  bits) is accurate to 1 ulp against libm and vectorizes.
- Warm-starting Newton at the intercept-only MLE diverged for rare-treatment cohorts (tiny initial
  curvature); a damped Newton from that start needed fewer Hessian evaluations but as many
  likelihood evaluations, so no gain; not used.
- Compressing duplicate covariate rows was rejected: 205k of 230k control rows are distinct in 104.
- The fused pass first ran at about 4 ns per unit-cell: slices created per (unit, cohort) and short
  loops of T - 1 cells; after restructuring to vectorize across 64 units per block it runs at about
  1 ns per unit-cell.
- `numpy.linalg.lstsq` (SVD) for the 256k x 6 control design took 0.28 s on s6; a QR solve gives
  the same least-squares solution (the reference itself needs full column rank: it inverts X0'X0).

## Numerical findings

- The reference is not bitwise reproducible: four calls in one process on a 400-unit random
  panel (fuzz case 5413, binary outcome, no covariates) gave four different bit patterns for
  `att_gt`, with one BLAS thread (alignment-dependent summation in NumPy/Accelerate). For ATTs
  that are zero in exact arithmetic the reference's value is rounding noise of 1e-19 to 1e-16,
  below the validator's relative-error floor of 1e-12, so no implementation can match it
  reliably. With the exact-cell safeguard, 5 of about 1,250 valid random panels (`fuzz.py`,
  seeds 1000-1399 and 5000-5999) still differ, all by about 1e-16 in such exactly-zero cells or
  event-time aggregates; every other output matches. No development workload has such cells.
- Random panels on which the reference itself returns non-finite values (141 of 1,400) are
  excluded from those counts.

## Version 2 (2026-10-23): the task's trimming correction

The task's reference was corrected on 2026-10-23 (at the maintainer's request): Stata's csdid
passes `pscoretrim(1)` to drdid, so no comparison unit is trimmed; outputs change only where a
comparison unit scores above 0.995 (development workloads 104 and 107, held-out 201). Version 2
of this candidate changes one constant, `TRIM = 1.0` (a raw score of at most 1.0 is never above
it; scores stay capped at 1 - 1e-6), in the fused score kernel and in the reference fallback path:
SHA-256 eae404914d6d... (version 1: 3ce68a25755d...). Random panels against the corrected
reference (`fuzz.py`, 600 panels, seed 5000): 535 of 536 checkable panels pass; the failure is a
cell whose ATT is zero in exact arithmetic (binary outcome, no covariates), where the reference
itself is not bitwise reproducible from run to run (its value is rounding residue that changes
between repetitions in one process, with one BLAS and one Numba thread). The evaluator runs below
the next heading were made with version 1 against the earlier reference; version 2's runs, made
after the correction under the study's thermal monitor, are listed under "Evaluator runs,
version 2". A spot check on 2026-10-24 (six threads, 102) measured the reference at 9.31 s
against 14.16 s in the version 1 run of 2026-10-22 (the candidate: 0.018 s both times), so the
six-thread reference times of the version 1 runs are likely inflated by load or throttling on the
host; the one-thread comparison is the conservative one.

## Evaluator runs (version 1, task before the correction)

`python -m harness evaluate` (protocol 3) on the development host, 2026-10-22/23, interpreter
`solutions/.venv`; the candidate's `solution.py` (SHA-256 3ce68a25755d...c8d6) is the same in
every run's snapshot. Every invocation first passed the development fixtures at its thread count
(and the smoke check `--smoke` passed: `runs/20261022T184744.223771Z-1fa8786f`), then recomputed
the reference in its own process with the same thread budget. Workload 104's single reference call
takes more than 150 s, so it ran with `--warmups 0 --repeats 1` (calls "1+0+1": a first call and
one measured call, for the reference and the candidate alike); the others used the evaluator's
defaults (real: first call, one warmup, five measured calls; synthetic: first call, one warmup,
three measured calls). Speedup = reference median / candidate median at the same thread count.
First call = the candidate's first call in its worker, including Numba compilation (not a steady-
state sample). Peak RSS = maximum resident set of each worker process, inputs included. Max. rel.
error = the validator's largest relative error of any output over all candidate calls. Every
development workload in scope (real 101-108, synthetic 1-6) passed at both thread counts.

### One thread (`--threads 1`)

Runs: `runs/20261022T202313.720306Z-685dfdff`, `runs/20261022T205204.620952Z-29a40f9d`, `runs/20261022T213104.757415Z-2aab95c1`

Real workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 101 | 1+1+5 | 1.171 s | 0.013 s (0.012-0.013) | 92.3x | 33.22 s | 0.36 / 0.44 GiB | 4.3e-11 | PASSED |
| 102 | 1+1+5 | 7.635 s | 0.038 s (0.037-0.040) | 198.4x | 32.71 s | 0.53 / 0.46 GiB | 2.4e-12 | PASSED |
| 103 | 1+1+5 | 32.17 s | 0.262 s (0.257-0.269) | 123.0x | 33.64 s | 1.62 / 0.68 GiB | 7.4e-11 | PASSED |
| 104 | 1+0+1 | 1076.1 s | 3.751 s | 286.9x | 38.68 s | 23.34 / 2.22 GiB | 3.0e-08 | PASSED |
| 105 | 1+1+5 | 112.4 s | 0.587 s (0.555-0.639) | 191.5x | 33.43 s | 6.41 / 0.67 GiB | 8.8e-11 | PASSED |
| 106 | 1+1+5 | 0.383 s | 0.009 s (0.008-0.009) | 44.3x | 33.52 s | 0.37 / 0.43 GiB | 1.0e-11 | PASSED |
| 107 | 1+1+5 | 39.44 s | 0.396 s (0.384-0.416) | 99.5x | 33.72 s | 5.66 / 0.94 GiB | 8.7e-11 | PASSED |
| 108 | 1+1+5 | 0.754 s | 0.023 s (0.022-0.023) | 33.4x | 32.76 s | 0.21 / 0.42 GiB | 1.6e-11 | PASSED |

Geometric-mean speedup (real): 107.8x over 8 cases.

Synthetic workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 1+1+3 | 0.030 s | 0.002 s (0.002-0.002) | 13.9x | 33.03 s | 0.04 / 0.37 GiB | 1.7e-14 | PASSED |
| s2 | 1+1+3 | 0.327 s | 0.005 s (0.005-0.005) | 66.0x | 33.21 s | 0.13 / 0.37 GiB | 1.1e-12 | PASSED |
| s3 | 1+1+3 | 3.411 s | 0.018 s (0.018-0.020) | 190.4x | 34.33 s | 0.30 / 0.38 GiB | 1.5e-12 | PASSED |
| s4 | 1+1+3 | 6.657 s | 0.103 s (0.100-0.108) | 64.7x | 33.81 s | 0.80 / 0.55 GiB | 3.7e-11 | PASSED |
| s5 | 1+1+3 | 90.07 s | 0.426 s (0.424-0.434) | 211.5x | 34.03 s | 12.52 / 0.76 GiB | 1.0e-09 | PASSED |
| s6 | 1+1+3 | 109.9 s | 0.813 s (0.795-0.825) | 135.3x | 35.16 s | 6.02 / 1.25 GiB | 1.8e-10 | PASSED |

Geometric-mean speedup (synthetic): 82.8x over 6 cases.

### Six threads (`--threads 6`)

Runs: `runs/20261023T033915.858286Z-d6c2921e`, `runs/20261023T041307.570718Z-4f62d207`, `runs/20261023T045834.269668Z-b6064dd7`

Real workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 101 | 1+1+5 | 1.802 s | 0.016 s (0.015-0.023) | 111.4x | 39.56 s | 0.38 / 0.45 GiB | 4.3e-11 | PASSED |
| 102 | 1+1+5 | 14.16 s | 0.018 s (0.018-0.019) | 766.0x | 51.42 s | 0.51 / 0.47 GiB | 2.4e-12 | PASSED |
| 103 | 1+1+5 | 32.62 s | 0.113 s (0.105-0.121) | 289.9x | 35.64 s | 1.73 / 0.73 GiB | 7.4e-11 | PASSED |
| 104 | 1+0+1 | 1227.3 s | 1.430 s | 858.1x | 37.23 s | 22.85 / 2.16 GiB | 3.0e-08 | PASSED |
| 105 | 1+1+5 | 120.1 s | 0.194 s (0.193-0.228) | 617.3x | 35.41 s | 6.44 / 0.68 GiB | 8.8e-11 | PASSED |
| 106 | 1+1+5 | 0.427 s | 0.009 s (0.007-0.009) | 48.7x | 34.19 s | 0.43 / 0.44 GiB | 1.0e-11 | PASSED |
| 107 | 1+1+5 | 55.52 s | 0.268 s (0.263-0.302) | 207.2x | 39.89 s | 4.99 / 0.91 GiB | 8.7e-11 | PASSED |
| 108 | 1+1+5 | 0.899 s | 0.025 s (0.024-0.027) | 35.9x | 39.38 s | 0.19 / 0.42 GiB | 1.6e-11 | PASSED |

Geometric-mean speedup (real): 216.0x over 8 cases.

Synthetic workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 1+1+3 | 0.028 s | 0.003 s (0.003-0.003) | 9.8x | 34.98 s | 0.04 / 0.37 GiB | 1.7e-14 | PASSED |
| s2 | 1+1+3 | 0.390 s | 0.005 s (0.004-0.005) | 84.3x | 33.79 s | 0.11 / 0.37 GiB | 1.1e-12 | PASSED |
| s3 | 1+1+3 | 3.001 s | 0.009 s (0.009-0.010) | 318.7x | 33.09 s | 0.27 / 0.38 GiB | 1.5e-12 | PASSED |
| s4 | 1+1+3 | 7.817 s | 0.065 s (0.061-0.080) | 119.9x | 33.66 s | 0.82 / 0.57 GiB | 3.7e-11 | PASSED |
| s5 | 1+1+3 | 89.62 s | 0.232 s (0.221-0.253) | 386.4x | 34.94 s | 12.00 / 0.81 GiB | 1.0e-09 | PASSED |
| s6 | 1+1+3 | 109.8 s | 0.474 s (0.451-0.484) | 231.6x | 35.10 s | 6.02 / 1.24 GiB | 1.8e-10 | PASSED |

Geometric-mean speedup (synthetic): 119.0x over 6 cases.

### Thread scaling (candidate median at one and six threads)

| Case | 1 thread | 6 threads | Ratio | Reference 1 / 6 threads |
|---|---:|---:|---:|---:|
| 101 | 0.013 s | 0.016 s | 0.78x | 1.171 s / 1.802 s |
| 102 | 0.038 s | 0.018 s | 2.08x | 7.635 s / 14.16 s |
| 103 | 0.262 s | 0.113 s | 2.33x | 32.17 s / 32.62 s |
| 104 | 3.751 s | 1.430 s | 2.62x | 1076.1 s / 1227.3 s |
| 105 | 0.587 s | 0.194 s | 3.02x | 112.4 s / 120.1 s |
| 106 | 0.009 s | 0.009 s | 0.99x | 0.383 s / 0.427 s |
| 107 | 0.396 s | 0.268 s | 1.48x | 39.44 s / 55.52 s |
| 108 | 0.023 s | 0.025 s | 0.90x | 0.754 s / 0.899 s |
| s1 | 0.002 s | 0.003 s | 0.75x | 0.030 s / 0.028 s |
| s2 | 0.005 s | 0.005 s | 1.07x | 0.327 s / 0.390 s |
| s3 | 0.018 s | 0.009 s | 1.90x | 3.411 s / 3.001 s |
| s4 | 0.103 s | 0.065 s | 1.58x | 6.657 s / 7.817 s |
| s5 | 0.426 s | 0.232 s | 1.84x | 90.07 s / 89.62 s |
| s6 | 0.813 s | 0.474 s | 1.71x | 109.9 s / 109.8 s |

The reference is slower with a six-thread budget than with one (cause not investigated) on several
workloads (102: 7.6 to 14.2 s; 107: 39.4 to 55.5 s; 104: 1,076 to 1,227 s), so the six-thread
speedups above are relative to a slower reference. Against the one-thread reference medians, the
six-thread candidate is 30-752x faster on the real workloads (108: 30x, 106: 44x, 101: 72x, 107:
147x, 103: 286x, 102: 413x, 105: 578x, 104: 752x) and 10-388x on the synthetic ones. The candidate
itself gains 1.5-3.0x from six threads on the larger workloads (102, 103, 104, 105, 107 and
synthetic 3-6) and loses a few milliseconds on the smallest (101: 0.013 to 0.016 s; 108: 0.023 to
0.025 s; synthetic 1: 0.002 to 0.003 s), where starting the parallel loops costs more than they
save.

### Commands

One thread:

```bash
solutions/.venv/bin/python -m harness evaluate --task 002 --solution solutions/002/cohort-fused/solution.json --threads 1 --suite real-development --ids 101 102 103 105 106 107 108 --timeout 3600   # runs/20261022T202313.720306Z-685dfdff
solutions/.venv/bin/python -m harness evaluate --task 002 --solution solutions/002/cohort-fused/solution.json --threads 1 --suite real-development --ids 104 --warmups 0 --repeats 1 --timeout 7200   # runs/20261022T205204.620952Z-29a40f9d
solutions/.venv/bin/python -m harness evaluate --task 002 --solution solutions/002/cohort-fused/solution.json --threads 1 --ids 1 2 3 4 5 6 --timeout 3600   # runs/20261022T213104.757415Z-2aab95c1
```

Six threads:

```bash
solutions/.venv/bin/python -m harness evaluate --task 002 --solution solutions/002/cohort-fused/solution.json --threads 6 --suite real-development --ids 101 102 103 105 106 107 108 --timeout 3600   # runs/20261023T033915.858286Z-d6c2921e
solutions/.venv/bin/python -m harness evaluate --task 002 --solution solutions/002/cohort-fused/solution.json --threads 6 --suite real-development --ids 104 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T041307.570718Z-4f62d207
solutions/.venv/bin/python -m harness evaluate --task 002 --solution solutions/002/cohort-fused/solution.json --threads 6 --ids 1 2 3 4 5 6 --timeout 3600   # runs/20261023T045834.269668Z-b6064dd7
```

## Evaluator runs, version 2

(added after the overnight measurement queue of 2026-10-24, `runs/baselines-20261023/plan-overnight.json`)
