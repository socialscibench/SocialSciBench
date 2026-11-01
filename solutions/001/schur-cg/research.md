# Candidate research: schur-cg

## Objective and constraints

- Task 001, contract version 2: OLS with high-dimensional fixed effects and one-way clustered
  standard errors with reghdfe's conventions (`reghdfe y X, absorb(...) vcov(cluster c)`):
  recursive singleton removal, partialling out, OLS, the CRV1 sandwich with the small-sample
  factor (n - 1)/(n - K) G/(G - 1), K = k + 1 + df_a and reghdfe's df_a rules. Must match
  `reference.run`: beta to 1e-6, standard errors to 1e-4, nobs/df_a/n_clusters exactly.
- Target hardware: CPU; configurations `--threads 1` and `--threads 6` (physical cores of the
  Intel Core i7-8700B host, 32 GiB, macOS 15.7.3). Environment `solutions/.venv` (`build.md`).
- Research setting: the same agent built task-001 candidates earlier in this conversation
  (2026-10-16/17; PR #2), one of them named `eliminate-largest-fe`, so the idea of eliminating
  the largest fixed effect exactly was known from that work and is not independent. None of
  the earlier candidates' files, nor the repository's reports, baselines, published records or
  runs, were opened in this session (only their directory names were seen in a listing). Task
  files, the harness and the development data were used.

## Method

1. **Singletons and levels.** Rows alone in a level of some fixed effect are peeled until none
   remain (the fixed point the reference's passes reach; sweeps also drop rows that become
   singletons during the sweep). Levels are recoded with a counting array (dense ids) or
   `np.unique`.
2. **Partialling out by elimination and conjugate gradients.** A fixed effect whose levels are
   unions of another's levels (year within technology-subclass x year in 105, year within
   industry x year in synthetic 7) is skipped: its indicators are sums of the other's, so the
   projection is unchanged (it still counts in df_a). The fixed effect with the most levels (the
   inventor, worker or pair) is eliminated exactly: given the other fixed effects'
   coefficients, its coefficients are group means. The remaining normal equations are the Schur
   complement S = A_oo - N_ob C_b^-1 N_bo over the other fixed effects' levels (A_oo: their
   cross-tabulation; N_bo: row counts of each eliminated level at each other level). y and all
   regressors are solved together (each sparse entry updates a row of all columns).
   - **Operator.** S is assembled (row-by-row accumulation) when a bound on its entries is at most
     twice the cost of applying it through A_oo and N_bo (inventor x firm x year panels: S has
     1.3-8.5 million entries). Otherwise (subclass x year in 105; pair-eliminated dyads in 107,
     where eliminating a 20-year pair couples 40 levels) it is applied as A_oo P minus a fused
     pass over the eliminated levels: each level's row of C_b^-1 N_bo P is formed and scattered
     straight back through N_ob, so that intermediate is never stored, and the levels are
     processed in order of their smallest other level so that consecutive levels reuse the same
     rows of P. With several threads each thread scatters into its own buffer. Small reduced
     systems (at most 4,000 levels, e.g. gravity designs) are dense. Rows of fixed effects with at
     most 512 levels (years) are kept dense and multiplied with BLAS.
   - **Preconditioner**, block by block (one block per remaining fixed effect): exact
     pseudo-inverse for blocks of at most 2,000 levels; for large blocks whose graph is sparse,
     a randomized approximate Cholesky factorization of the block's Laplacian (Kyng & Sachdeva
     2016; the one-sample "AC" variant of Gao, Kyng & Spielman 2023), eliminating vertices in
     order of increasing degree, stored in elimination order and solved 8 columns at a time
     (one cache line per row, accumulations in registers; column chunks in parallel);
     otherwise the diagonal. CG stops when every column's residual is below 1e-9 of its
     right-hand side.
3. The demeaned data, OLS (`W'W` by BLAS), cluster scores, the sandwich and df_a (nesting in the
   cluster variable; union-find components of the first two non-nested fixed effects; nesting
   among the others) follow the reference's definitions.

## Sources (accessed 2026-10-22)

- Correia, S. (2017), "Linear Models with High-Dimensional Fixed Effects: An Efficient and
  Feasible Estimator"; reghdfe 6.14.1, https://github.com/sergiocorreia/reghdfe (df_a rules,
  singleton removal).
- Bergé, L., Butts, K. and McDermott, G. (2026), "Fast and user-friendly econometrics
  estimations: The R package fixest", arXiv:2601.21749 (fixed-point demeaning with Irons-Tuck
  acceleration); fixest 0.14.2 `ssc()` and `feols()` documentation.
- Kyng, R. and Sachdeva, S. (2016), "Approximate Gaussian Elimination for Laplacians: Fast,
  Sparse, and Simple", arXiv:1605.02353; Gao, Y., Kyng, R. and Spielman, D. A. (2023), "Robust
  and Practical Solution of Laplacian Equations by Approximate Elimination", arXiv:2303.00709.
- py-econometrics `within` (LSMR with additive Schwarz preconditioning and approximate-Cholesky
  local solvers), https://github.com/py-econometrics/within; pyfixest 0.60.0.
- Gaure, S. (2013), "OLS with multiple high dimensional category variables", Computational
  Statistics & Data Analysis 66: 8-18 (alternating projections).

## Hypotheses, measurements and unsuccessful attempts

Development measurements (single calls after a first call, one thread unless stated; not
evaluator runs). Reference: one call each.

| Version | 101 | 102 | 103 | 104 | 105 | 107 | 108 | s9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 27.1 s | 95.7 s | 115.0 s | 714.4 s | 811 s | 21.0 s | 612.5 s | 120.2 s |
| 1. Eliminate largest FE, Jacobi-PCG on implicit S | 4.06 s | 14.6 s | 13.1 s | - | - | 5.24 s | - | 10.7 s |
| 2. + approximate Cholesky block preconditioner, explicit S | 0.44 s | 2.47 s | 2.97 s | 16.2 s | - | 4.22 s | 26.2 s | 1.23 s |
| 3. + cost model, CG tolerance 1e-9, dense small blocks, no data copies | 0.40 s | 2.24 s | 2.66 s | 13.2 s | 44.3 s | 3.34 s | 14.7 s | 0.59 s |
| 4. + redundant FE skipped, fused implicit operator, 8-column AC solves, compact peel counts | 0.42 s | 2.20 s | 2.65 s | 12.6 s | 25.3 s | 3.59 s | 14.1 s | 0.68 s |
| 3 at six threads | 0.31 s | 1.45 s | 1.65 s | 7.9 s | - | 1.66 s | 10.0 s | 0.46 s |
| 4 at six threads | 0.31 s | 1.29 s | 1.54 s | 6.5 s | 13.0 s | 1.69 s | 8.9 s | 0.54 s |

Version 4 (all checks pass at one and six threads: fixtures, 101-108, 104 with 5/50
regressors, 109 and synthetic 1, 3, 4, 6-12; one-thread errors against the reference at most
4.1e-7 in beta (s11) and 1.8e-6 relative in standard errors (101), well within 1e-6 and 1e-4).
Measured effect of each change:

- **Redundant fixed effects** (105: year is nested in subclass x year): 44.3 to 33.6 s at one
  thread with identical coefficients (2.4e-14 relative); synthetic 7 (year within ind x year):
  2.41 to 1.78 s. The nesting checks cost a few milliseconds on the others.
- **Fused implicit operator** (105, one product with 21 columns): 487 ms (three SpMMs through a
  stored C^-1 N_bo P of 892,132 x 21) to 386 ms fused, 276 ms with the locality order; at six
  threads 317 to 114 ms. 107 (small P that fits in cache): no change at one thread (132 ms),
  44 to 38 ms at six.
- **AC solves in 8-column chunks** (104's firm factor, 136,563 levels, 1.70 million entries, 21
  columns): 100 to 78 ms per solve at one thread, 67 to 30 ms at six; 51 columns: 206 to 189 ms
  and 131 to 74 ms. Synthetic 11 (very sparse factor) is 2.7% slower at one thread (7.86 to
  8.07 s), which was accepted.
- **Compact int32 level counts** in the singleton peel (one array for all fixed effects):
  104's peel 0.645 to 0.456 s (7 sweeps; the sweeps after the first remove 18,603, 2,174, 185,
  15, 1 and 0 rows).

Unsuccessful or rejected:

- Iterations (tolerance 1e-9): diagonal (Jacobi) preconditioning needs 85-232 on the worker/firm
  graphs; exact block solves (an LU of the firm block) need 8-20, approximate Cholesky 18-29.
  The year coupling is harmless: the difficulty is the firm Laplacian of the mover network.
- Incomplete Cholesky IC(0) of the reduced matrix broke down (S is not an M-matrix: the
  firm-year coupling has positive entries); not used. An exact sparse factorization of the firm
  block is cheap on 101 (fill 3.6x) but not on 102 (29x fill, 7 s with SuperLU) or larger.
- Workload 105 still needs 53 iterations after dropping year: the coupling between the firm and
  subclass x year blocks is not captured by block-Jacobi; approximate Cholesky on the
  subclass x year block (6.6 million edges) did not reduce iterations (48 both ways in a
  prototype) and costs 5.9 s to factorize.
- Relabelling the AC factor in elimination order alone (row-major solve over all columns): no
  gain (2.95 s of AC solves on 104 before and after); the gain came with the 8-column chunks.
- W'W (6.76 million x 21): Accelerate's dgemm takes 255 ms and is single-threaded for this
  shape even with six vecLib threads; Numba kernels took 564 ms at one thread and 146 ms at six;
  six Python threads each calling BLAS on a row block ran serially (CPU time = wall time). Kept
  BLAS.
- A round-based parallel singleton peel was slower (7.3 s on 104) than the sequential sweep
  (0.7 s): singleton chains need many rounds. A worklist peel needs per-level row lists, whose
  construction costs about as much as the extra sweeps; not pursued.
- Tolerances 1e-8, 1e-9 and 1e-11 give the same errors against the reference (dominated by the
  reference's own LSMR error, up to 4e-7 on s11's beta); 1e-9 is used.

## Evaluator runs

`python -m harness evaluate` (protocol 3) on the development host, 2026-10-22/23, interpreter
`solutions/.venv`; the candidate's `solution.py` (SHA-256 e86318a72225...20b3) is the same in
every run's snapshot. Every invocation first passed the development fixtures at its thread count
(and the smoke check `--smoke` passed: `runs/20261022T184642.648497Z-12889745`), then recomputed
the reference in its own process with the same thread budget. Cases whose single reference call
takes more than 150 s (104 with 5, 20 and 50 regressors, shown as 104k5/104k20/104k50; 105; 108;
synthetic 11) ran with `--warmups 0 --repeats 1` (calls "1+0+1": a first call and one measured
call, for the reference and the candidate alike); the others used the evaluator's defaults (real:
first call, one warmup, five measured calls; synthetic: first call, one warmup, three measured
calls). Speedup = reference median / candidate median at the same thread count. First call = the
candidate's first call in its worker, including Numba compilation (not a steady-state sample).
Peak RSS = maximum resident set of each worker process, inputs included. Max. rel. error = the
validator's largest relative error of any output over all candidate calls (tolerances: 1e-6 for
beta, 1e-4 for the standard errors; nobs, df_a and n_clusters exact). Every development workload
in scope (real 101-108 including 104's three widths; synthetic 1, 3, 4 and 6-12, the task's scored
synthetic workloads) passed at both thread counts. The compatibility workload 109 (real-smoke;
correctness only, no timing or score) passed at both thread counts
(`runs/20261022T223125.550138Z-03968a2c`, `runs/20261023T055833.910601Z-54235ec6`).

### One thread (`--threads 1`)

Runs: `runs/20261022T215434.426602Z-5729349d`, `runs/20261022T223311.828997Z-d8b691c5`, `runs/20261022T225246.061204Z-df2f14c7`, `runs/20261022T230015.850521Z-fa445d0e`, `runs/20261022T234408.288172Z-32ec7e50`, `runs/20261023T000948.005315Z-eba532c2`, `runs/20261023T003150.988899Z-dee49978`, `runs/20261023T010358.482359Z-a17b6a64`

Real workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 101 | 1+1+5 | 30.83 s | 0.422 s (0.410-0.430) | 73.1x | 20.30 s | 0.62 / 0.76 GiB | 1.8e-06 | PASSED |
| 102 | 1+1+5 | 105.5 s | 2.210 s (2.202-2.284) | 47.7x | 22.64 s | 1.68 / 1.77 GiB | 1.6e-07 | PASSED |
| 103 | 1+1+5 | 115.7 s | 2.690 s (2.671-2.773) | 43.0x | 22.40 s | 2.30 / 2.16 GiB | 2.2e-07 | PASSED |
| 104k20 | 1+0+1 | 523.7 s | 12.88 s | 40.7x | 33.16 s | 6.56 / 5.78 GiB | 2.4e-07 | PASSED |
| 104k5 | 1+0+1 | 168.7 s | 8.144 s | 20.7x | 28.72 s | 3.15 / 3.26 GiB | 2.3e-08 | PASSED |
| 104k50 | 1+0+1 | 1224.6 s | 20.78 s | 58.9x | 42.00 s | 13.48 / 11.25 GiB | 9.1e-07 | PASSED |
| 105 | 1+0+1 | 689.1 s | 25.12 s | 27.4x | 46.09 s | 6.85 / 5.95 GiB | 1.5e-07 | PASSED |
| 106 | 1+1+5 | 3.521 s | 0.233 s (0.227-0.245) | 15.1x | 11.49 s | 0.71 / 0.72 GiB | 1.6e-09 | PASSED |
| 107 | 1+1+5 | 20.45 s | 3.563 s (3.431-3.594) | 5.7x | 17.23 s | 2.11 / 2.16 GiB | 4.2e-10 | PASSED |
| 108 | 1+0+1 | 582.7 s | 14.37 s | 40.5x | 34.60 s | 10.10 / 7.69 GiB | 1.5e-08 | PASSED |

Geometric-mean speedup (real): 30.6x over 10 cases.

Synthetic workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 1+1+3 | 0.240 s | 0.021 s (0.019-0.021) | 11.6x | 11.51 s | 0.14 / 0.26 GiB | 1.1e-14 | PASSED |
| s3 | 1+1+3 | 21.85 s | 1.846 s (1.841-1.890) | 11.8x | 14.16 s | 4.77 / 3.42 GiB | 1.4e-14 | PASSED |
| s4 | 1+1+3 | 17.00 s | 1.126 s (1.123-1.145) | 15.1x | 20.96 s | 1.27 / 1.25 GiB | 3.0e-08 | PASSED |
| s6 | 1+1+3 | 1.711 s | 0.181 s (0.178-0.195) | 9.4x | 14.72 s | 0.37 / 0.58 GiB | 8.0e-15 | PASSED |
| s7 | 1+1+3 | 22.83 s | 1.794 s (1.791-1.795) | 12.7x | 13.98 s | 2.35 / 2.19 GiB | 1.4e-09 | PASSED |
| s8 | 1+1+3 | 47.73 s | 0.721 s (0.717-0.722) | 66.2x | 20.50 s | 1.32 / 1.20 GiB | 1.6e-07 | PASSED |
| s9 | 1+1+3 | 109.5 s | 0.677 s (0.676-0.678) | 161.8x | 20.58 s | 1.28 / 1.17 GiB | 5.5e-08 | PASSED |
| s10 | 1+1+3 | 6.430 s | 0.593 s (0.574-0.595) | 10.8x | 11.68 s | 2.29 / 1.67 GiB | 1.4e-14 | PASSED |
| s11 | 1+0+1 | 2079.4 s | 8.486 s | 245.0x | 30.05 s | 9.91 / 8.22 GiB | 4.1e-07 | PASSED |
| s12 | 1+1+3 | 65.30 s | 6.010 s (5.905-6.011) | 10.9x | 17.80 s | 15.16 / 11.79 GiB | 3.0e-14 | PASSED |

Geometric-mean speedup (synthetic): 24.5x over 10 cases.

### Six threads (`--threads 6`)

Runs: `runs/20261023T052206.604804Z-6ba4680c`, `runs/20261023T060022.670819Z-0e69e8e3`, `runs/20261023T061949.908034Z-298fe313`, `runs/20261023T062715.401175Z-cd3725b2`, `runs/20261023T071051.912648Z-43752d65`, `runs/20261023T073600.903087Z-286e661a`, `runs/20261023T075746.376884Z-4268a286`, `runs/20261023T082949.896955Z-c8bee552`

Real workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 101 | 1+1+5 | 32.37 s | 0.294 s (0.286-0.297) | 110.3x | 21.74 s | 0.61 / 0.71 GiB | 1.8e-06 | PASSED |
| 102 | 1+1+5 | 103.6 s | 1.328 s (1.320-1.358) | 78.0x | 22.68 s | 1.68 / 1.77 GiB | 1.6e-07 | PASSED |
| 103 | 1+1+5 | 116.6 s | 1.543 s (1.522-1.560) | 75.6x | 23.46 s | 2.29 / 2.19 GiB | 2.2e-07 | PASSED |
| 104k20 | 1+0+1 | 521.0 s | 6.371 s | 81.8x | 28.75 s | 6.59 / 5.85 GiB | 2.4e-07 | PASSED |
| 104k5 | 1+0+1 | 168.5 s | 4.702 s | 35.8x | 26.79 s | 3.18 / 3.28 GiB | 2.3e-08 | PASSED |
| 104k50 | 1+0+1 | 1223.8 s | 10.65 s | 114.9x | 33.62 s | 13.49 / 11.26 GiB | 9.1e-07 | PASSED |
| 105 | 1+0+1 | 683.3 s | 12.82 s | 53.3x | 35.49 s | 6.86 / 6.14 GiB | 1.7e-07 | PASSED |
| 106 | 1+1+5 | 3.470 s | 0.159 s (0.156-0.181) | 21.8x | 13.22 s | 0.73 / 0.67 GiB | 1.6e-09 | PASSED |
| 107 | 1+1+5 | 20.34 s | 1.563 s (1.512-1.605) | 13.0x | 17.01 s | 2.11 / 2.22 GiB | 4.2e-10 | PASSED |
| 108 | 1+0+1 | 578.4 s | 8.692 s | 66.5x | 31.03 s | 10.13 / 7.79 GiB | 1.5e-08 | PASSED |

Geometric-mean speedup (real): 54.2x over 10 cases.

Synthetic workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 1+1+3 | 0.217 s | 0.017 s (0.017-0.018) | 12.6x | 10.96 s | 0.14 / 0.26 GiB | 1.1e-14 | PASSED |
| s3 | 1+1+3 | 21.56 s | 1.245 s (1.235-1.258) | 17.3x | 14.92 s | 4.75 / 3.41 GiB | 6.2e-14 | PASSED |
| s4 | 1+1+3 | 16.93 s | 0.678 s (0.674-0.689) | 25.0x | 21.33 s | 1.26 / 1.26 GiB | 3.0e-08 | PASSED |
| s6 | 1+1+3 | 1.688 s | 0.106 s (0.105-0.107) | 15.9x | 14.95 s | 0.35 / 0.52 GiB | 8.2e-15 | PASSED |
| s7 | 1+1+3 | 23.02 s | 1.405 s (1.388-1.424) | 16.4x | 14.86 s | 2.41 / 2.19 GiB | 1.4e-09 | PASSED |
| s8 | 1+1+3 | 47.12 s | 0.545 s (0.536-0.548) | 86.4x | 21.60 s | 1.34 / 1.15 GiB | 1.6e-07 | PASSED |
| s9 | 1+1+3 | 111.8 s | 0.508 s (0.508-0.511) | 219.8x | 21.67 s | 1.31 / 1.13 GiB | 5.5e-08 | PASSED |
| s10 | 1+1+3 | 6.436 s | 0.453 s (0.450-0.457) | 14.2x | 13.43 s | 2.26 / 1.67 GiB | 2.9e-14 | PASSED |
| s11 | 1+0+1 | 1874.9 s | 5.710 s | 328.3x | 27.48 s | 9.88 / 8.26 GiB | 4.1e-07 | PASSED |
| s12 | 1+1+3 | 63.74 s | 3.823 s (3.817-4.378) | 16.7x | 17.14 s | 15.89 / 11.79 GiB | 1.1e-13 | PASSED |

Geometric-mean speedup (synthetic): 34.0x over 10 cases.

### Four threads (`--threads 4`, 2026-10-25)

The multicore configuration selected for the freeze (study report, "Host"). The candidate is
unchanged since 2026-10-22; every fixture and output passed. Real workloads only (the synthetic
ones were not run at four threads).

Runs:
- 101-103, 106, 107: `runs/20261026T031542.571084Z-d2779b4e`;
- 104 with 20 regressors: `runs/20261026T035408.896006Z-4531cec1`;
- 104 with 5: `runs/20261026T041515.937996Z-897e2553`;
- 104 with 50: `runs/20261026T042411.702249Z-26959f2a`;
- 105: `runs/20261026T051019.926857Z-a369e3de`;
- 108: `runs/20261026T053721.762823Z-ebd9b900`.

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate |
|---|---|---:|---:|---:|---:|---:|
| 101 | 1+1+5 | 31.14 s | 0.332 s (0.319-0.344) | 93.9x | 22.66 s | 0.58 / 0.75 GiB |
| 102 | 1+1+5 | 103.6 s | 1.418 s (1.361-1.551) | 73.0x | 23.23 s | 1.70 / 1.77 GiB |
| 103 | 1+1+5 | 114.9 s | 1.622 s (1.612-1.650) | 70.8x | 23.60 s | 2.29 / 2.19 GiB |
| 104 (20 regressors) | 1+0+1 | 541.6 s | 6.782 s | 79.9x | 29.51 s | 6.54 / 5.83 GiB |
| 104 (5) | 1+0+1 | 168.7 s | 5.159 s | 32.7x | 27.25 s | 3.16 / 3.26 GiB |
| 104 (50) | 1+0+1 | 1,246 s | 11.15 s | 111.8x | 34.85 s | 11.54 / 11.25 GiB |
| 105 | 1+0+1 | 697.7 s | 13.94 s | 50.0x | 35.62 s | 6.87 / 6.08 GiB |
| 106 | 1+1+5 | 3.57 s | 0.165 s (0.159-0.172) | 21.7x | 13.25 s | 0.75 / 0.69 GiB |
| 107 | 1+1+5 | 20.63 s | 1.747 s (1.703-1.756) | 11.8x | 17.14 s | 2.13 / 2.20 GiB |
| 108 | 1+0+1 | 589.9 s | 9.151 s | 64.5x | 31.55 s | 10.13 / 7.74 GiB |

**Geometric-mean speedup:** 50.9x (real, 10 cases).
- Four threads give 89-96% of six threads' speed (time ratio six / four threads: 0.89 on 101 to
  0.96 on 106).
- On 104 with 20 regressors, fixest at four threads took 351 s, 52x this candidate's 6.78 s
  (`runs/baselines-20261023/001-fixest-r-t4.jsonl`).

### Thread scaling (candidate median at one and six threads)

| Case | 1 thread | 6 threads | Ratio | Reference 1 / 6 threads |
|---|---:|---:|---:|---:|
| 101 | 0.422 s | 0.294 s | 1.44x | 30.83 s / 32.37 s |
| 102 | 2.210 s | 1.328 s | 1.66x | 105.5 s / 103.6 s |
| 103 | 2.690 s | 1.543 s | 1.74x | 115.7 s / 116.6 s |
| 104k20 | 12.88 s | 6.371 s | 2.02x | 523.7 s / 521.0 s |
| 104k5 | 8.144 s | 4.702 s | 1.73x | 168.7 s / 168.5 s |
| 104k50 | 20.78 s | 10.65 s | 1.95x | 1224.6 s / 1223.8 s |
| 105 | 25.12 s | 12.82 s | 1.96x | 689.1 s / 683.3 s |
| 106 | 0.233 s | 0.159 s | 1.47x | 3.521 s / 3.470 s |
| 107 | 3.563 s | 1.563 s | 2.28x | 20.45 s / 20.34 s |
| 108 | 14.37 s | 8.692 s | 1.65x | 582.7 s / 578.4 s |
| s1 | 0.021 s | 0.017 s | 1.20x | 0.240 s / 0.217 s |
| s3 | 1.846 s | 1.245 s | 1.48x | 21.85 s / 21.56 s |
| s4 | 1.126 s | 0.678 s | 1.66x | 17.00 s / 16.93 s |
| s6 | 0.181 s | 0.106 s | 1.71x | 1.711 s / 1.688 s |
| s7 | 1.794 s | 1.405 s | 1.28x | 22.83 s / 23.02 s |
| s8 | 0.721 s | 0.545 s | 1.32x | 47.73 s / 47.12 s |
| s9 | 0.677 s | 0.508 s | 1.33x | 109.5 s / 111.8 s |
| s10 | 0.593 s | 0.453 s | 1.31x | 6.430 s / 6.436 s |
| s11 | 8.486 s | 5.710 s | 1.49x | 2079.4 s / 1874.9 s |
| s12 | 6.010 s | 3.823 s | 1.57x | 65.30 s / 63.74 s |

The reference's medians are within 10% at one and six threads (synthetic 11: 2,079 and 1,875 s).
Six threads give the candidate 1.4-2.3x on the real workloads and 1.2-1.7x on the synthetic ones.
The conjugate-gradient products, the approximate-Cholesky solves (column chunks) and the fused
implicit operator run in parallel; the singleton peel is sequential (a round-based parallel peel
was slower, see above), and W'W with Accelerate's dgemm runs on one thread for these shapes. The
largest errors (101's standard errors, 1.8e-6 relative; synthetic 11's beta, 4.1e-7) are the same
at both thread counts and below the tolerances.

### Commands

One thread:

```bash
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-development --ids 101 102 103 106 107 --timeout 3600   # runs/20261022T215434.426602Z-5729349d
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-smoke --timeout 3600   # runs/20261022T223125.550138Z-03968a2c
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-development --ids 104 --warmups 0 --repeats 1 --timeout 7200   # runs/20261022T223311.828997Z-d8b691c5
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-development --ids 104 --n-regressors 5 --warmups 0 --repeats 1 --timeout 7200   # runs/20261022T225246.061204Z-df2f14c7
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-development --ids 104 --n-regressors 50 --warmups 0 --repeats 1 --timeout 7200   # runs/20261022T230015.850521Z-fa445d0e
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-development --ids 105 --warmups 0 --repeats 1 --timeout 7200   # runs/20261022T234408.288172Z-32ec7e50
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --suite real-development --ids 108 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T000948.005315Z-eba532c2
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --ids 1 3 4 6 7 8 9 10 12 --timeout 3600   # runs/20261023T003150.988899Z-dee49978
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 1 --ids 11 --warmups 0 --repeats 1 --timeout 10800   # runs/20261023T010358.482359Z-a17b6a64
```

Six threads:

```bash
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-development --ids 101 102 103 106 107 --timeout 3600   # runs/20261023T052206.604804Z-6ba4680c
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-smoke --timeout 3600   # runs/20261023T055833.910601Z-54235ec6
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-development --ids 104 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T060022.670819Z-0e69e8e3
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-development --ids 104 --n-regressors 5 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T061949.908034Z-298fe313
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-development --ids 104 --n-regressors 50 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T062715.401175Z-cd3725b2
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-development --ids 105 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T071051.912648Z-43752d65
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --suite real-development --ids 108 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T073600.903087Z-286e661a
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --ids 1 3 4 6 7 8 9 10 12 --timeout 3600   # runs/20261023T075746.376884Z-4268a286
solutions/.venv/bin/python -m harness evaluate --task 001 --solution solutions/001/schur-cg/solution.json --threads 6 --ids 11 --warmups 0 --repeats 1 --timeout 10800   # runs/20261023T082949.896955Z-c8bee552
```
