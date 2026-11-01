# Candidate research: cell-fused (task 002b)

## Objective and constraints

Reproduce `002b_csdid_options/reference.py` (Stata `csdid` 1.81 with `drdid` 1.91: never or
not-yet-treated controls, sampling weights normalized within each cell, unbalanced panels with
pair-balanced cells, the dripw, reg, ipw, stdipw and drimp estimators, no propensity-score
trimming, the simple, group, calendar and windowed event-study aggregates with influence-function
standard errors) within the task's tolerances (1e-6 for estimates, 1e-4 for standard errors,
relative with a 1e-12 floor), faster, on one and on several cores. Task 002b was wired into the
shared evaluator in this study (2026-10-23, at the maintainer's request); its development
workloads are 501-506 (`DATA.md`).

## Method

The reference loops over the (cohort, period) cells, solves each as a separate small problem with
NumPy, keeps every cell's full-sample influence function and cohort-weight column in two
n_units x n_cells arrays, and aggregates them with csdid's `aggte`. The candidate (`solution.py`
docstring) computes the same quantities in three passes:

1. **Cells and their sizes** (Numba, parallel over cells): the cells in csdid's order and, for each,
   the units it uses (the cohort plus the comparison units observed in both of its periods), the
   cell size and the weight sums. The aggregation weights of every aggregate (cohort-size weights,
   csdid's normalization by the sum of the weight columns' means) depend only on these.
2. **Estimation** (parallel over cohorts, each cohort's cells in order): the propensity score
   (weighted logit, or inverse probability tilting for drimp) by Newton-Raphson with the
   reference's stopping rule (largest step below 1e-10; step halving for tilting), and the
   weighted least-squares outcome regression. In a balanced panel, consecutive cells of a cohort
   often use exactly the same units (with never-treated controls, every cell of the cohort does),
   and then share the propensity fit and the regression matrix; other fits start from the
   previous cell's solution (and fall back to the reference's starting point if that does not
   converge). Each cell is reduced to two coefficient vectors, two influence-function vectors and
   five scalars: any unit's influence-function value in the cell is then a short expression with
   four dot products with the unit's covariates (term by term the reference's `cell_rif`).
3. **One pass over units** (parallel over blocks of 64 units of one cohort, sorted by cohort):
   for every cell the block's cohort can belong to, the lanes' influence-function values are
   computed together (vectorized), summed per cell (mean and sum of squares around the
   closed-form ATT) and added, with the aggregation weights, to each unit's influence functions of
   the simple, group, calendar and event-study aggregates, including the terms for the estimated
   cohort weights. The n_units x n_cells arrays are never formed. Lanes outside a cell contribute
   exactly zero (their outcome change may be NaN in an unbalanced panel).

Exactness safeguards (cells recomputed with the reference's own NumPy operations, `_r_cell_rif`,
and the reference's axis-0 reductions for the cell's ATT and standard error): a cell whose fit is
singular or ill-conditioned (Cholesky pivot below 1e-12 of the diagonal) or does not converge;
a cell whose ATT is below 1e-7 of the magnitude of its terms (including the mean absolute outcome
change); a cell whose standard error is at rounding level relative to its influence-function
values (an influence function that is constant in exact arithmetic).

## Sources (accessed 2026-10-23)

- The task package (`csdid_002b_real_data_package`, README and its note to the task author, by the package's
  author): definitions, the Stata comparison and the csdid/csdid2 differences.
- Callaway & Sant'Anna (2021), J. Econometrics 225(2); Sant'Anna & Zhao (2020), J. Econometrics
  219(1) (dripw, drimp with inverse probability tilting); Abadie (2005), Rev. Econ. Stud. 72(1) (ipw).
- drdid 1.91 Mata functions (`drdid_panel`, `reg_panel`, `ipw_abadie_panel`, `std_ipw_panel`,
  `drdid_imp_panel`) and csdid 1.82 in https://github.com/friosavila/stpackages, as cited by the
  reference; DRDID 1.3.0 (R) https://github.com/cran/DRDID/tree/1.3.0/R for the same estimators.
- This study's task 002 candidate (`solutions/002/cohort-fused`): the branch-free vectorizable
  `exp` and the two-column trick for reproducing the reference's axis-0 reductions.

## Measurements and unsuccessful attempts

Development measurements (single process per workload, first call then three calls, median;
`solutions/.venv`; not evaluator runs; reference times are one call of `reference.run` per
workload with one BLAS thread, from `make_refs.py`):

| Version | 501 | 505 | 506 | 503a | 503b | 503d | 504 | 502 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference, 1 thread | 1.22 s | 2.15 s | 1.77 s | 6.64 s | 15.3 s | 14.2 s | 20.7 s | 42.2 s |
| v1 (first passing version), 1 thread | 0.38 s | 1.04 s | 1.20 s | 1.08 s | 4.00 s | 3.97 s | 4.47 s | 21.7 s |
| v2, 1 thread | 0.106 s | 0.268 s | 0.405 s | 1.05 s | 1.12 s | 1.16 s | 1.27 s | 5.58 s |
| v2, 6 threads | 0.050 s | 0.075 s | 0.114 s | 0.435 s | - | 0.490 s | 0.598 s | 2.81 s |

- **v1 -> v2, row views**: profiling v1 (502: 14.5 s of 18.3 s in estimation; 503b: 2.8 of
  4.2 s) showed one Newton fit from zero taking 114 ms on 503b (43,045 units, 11 unknowns, 9
  iterations): the Hessian accumulation indexed the gathered covariates as `Xg[a, lo + j]` and did
  not vectorize (3 ns per multiply-add). One-dimensional row views in the inner loops made the
  block kernel 20 times faster (60 to 3 us; one Hessian pass 11.4 to 1.5 ms; a fit 114 to 15 ms);
  the same change in the other kernels took 502 from 18.3 to 7.3 s and 503b from 3.9 to 1.7 s.
- **v2, blocked pass over units**: the first pass over units handled one unit at a time against
  every cell (3.2 s of 502). Units sorted by cohort into 64-lane blocks, with cells the block's
  cohort cannot belong to skipped and the lanes computed together: 3.2 to 1.27 s on 502, 1.2 to
  0.49 s on 503b.
- Rejected before measuring: storing every unit-cell's influence-function value (up to about
  55 million values, 440 MB, for the largest held-out shape) instead of recomputing it in the
  pass over units.
- Not attempted (time): sharing the outcome regression's residuals across cohorts where the
  comparison units are the same for every cohort (never-treated controls in a balanced panel:
  503a, 504), as the task 002 candidate does, which would make the regression part of every cell
  O(1) per unit; per-period sums for cells that share their units (estimated to save about 2x of
  the per-cell constants at these panel lengths, since a cohort's cells use almost every period).

Profile of v2 (one thread): 502: estimation 3.6 s (198,649 units, 361 cells: about 200 distinct
samples with not-yet-treated controls, a fit about 5-12 ms each, per-cell constants about 2 ms),
pass over units 1.3 s, cell counts 0.4 s; 503b: estimation 0.4 s, pass over units 0.5 s.

**Random panels** (`fuzz.py`, 1,500 panels: balanced or with gaps and exits, 0-3 covariates,
tiny cohorts, binary outcomes, weights, both comparison groups, all five methods, random event
windows): 1,368 of 1,374 panels where the reference itself succeeds pass (126 have a reference
failure: non-convergence or non-finite outputs). The six failures are standard errors of
aggregates that are exactly zero in exact arithmetic (groups or periods whose influence function
is constant, from single-unit cohorts with ipw): both sides report rounding residue of about
1e-16, which the validator's 1e-12 floor turns into relative errors of 2e-4 to 5e-4. The task's
workloads have no such aggregates (their smallest cohort in a cell has 4 units). The reference is
itself not bitwise reproducible from run to run on cells whose ATT is zero in exact arithmetic
(Accelerate's results depend on array alignment), so such cells can differ between any two runs.

## Evaluator runs

(added after the evaluator runs)
