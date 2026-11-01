# Candidate research: eliminate-largest-fe

## Objective and constraints

- Task 001, contract version 2: OLS with multiple absorbed fixed effects and one-way
  cluster-robust standard errors, matching `reference.run` (coefficients within 1e-6, standard
  errors within 1e-4, exact `nobs`, `df_a` and `n_clusters`).
- Target hardware: CPU, one thread (the evaluator default). Host: Intel Core i7-8700B
  (6 cores), 32 GiB RAM, macOS 15.7.3. Environment: CPython 3.12.13, NumPy 2.3.5,
  SciPy 1.18.1, Numba 0.62.1 (see `build.md` for why not Python 3.14).
- Informed research session: the user authorized consulting the repository's prior
  results (`prestudy/reports/001_reghdfe.md`, historical baseline records). No prior candidates
  existed. The development fixtures and the reference were used for validation only.
- Budget: default single-thread protocol. The worker timeout was raised for larger
  workloads only because the reference's five calls exceed 300 s there.

## Method

1. **Preprocessing.** Recode labels (direct-address tables when codes are dense and
   nonnegative, `np.unique` otherwise), drop singletons recursively, and compute `df_a`
   with the reference's rule using O(n) nesting checks instead of sorting.
2. **Eliminate the largest fixed effect exactly.** Its block of the normal equations is
   diagonal (level counts), so it is eliminated in closed form. What remains is
   `S = D_R'D_R - D_R'D_e diag(1/n_e) D_e'D_R` on the other fixed effects' levels, a
   weighted graph Laplacian: for worker-firm data, firms linked by the workers who move
   between them. Its size is 10 (firm-year), 3,510 (four-way), 6,000 (gravity) and 10,010
   or 100,010 (worker-firm), instead of n = 0.45M-10M observations per iteration.
3. **Solve for y and all regressors at once,** with the solver chosen by structure:
   - Levels whose diagonal vanishes (no remaining variation, e.g. firms with only stayers)
     are free and set to zero.
   - Up to 4,000 levels: dense S, assembled after grouping identical cross-tabulation
     rows; Jacobi-preconditioned CG (at most 30 iterations), dense Cholesky fallback.
   - Larger and well connected (a component holds more than half the big-FE levels):
     Jacobi-preconditioned CG, with S applied implicitly when forming it would be dense.
   - Larger and forest-like (low mobility): exact. One level per connected component of
     the big-FE block is grounded, that block is factorized with SuperLU (no fill on
     forests), and small fixed effects (100 levels or fewer, e.g. year) are folded in
     through a dense Schur complement.
4. **Two passes over the data.** Pass 1: per-level sums of [y, X] and the Gram matrix;
   beta follows from `Z'MZ = Z'Z - (D'Z)'A`. Pass 2: residuals rebuilt per observation from
   the fixed-effect coefficients, accumulated into cluster scores. The sandwich uses the
   reference's small-sample factor.

`tol` and `maxiter` are accepted but unused: the reduced system is solved to near machine
precision, so coefficient differences from the reference reflect the reference's own LSMR
error (about 5.5e-8 on workload 9 and 4.1e-7 on workload 11, which matches the recorded
reference-versus-Stata gap of 4.13e-7 there).

## Sources (accessed 2026-10-17)

- Correia (2016), "reghdfe: Estimating linear models with multi-way fixed effects", Stata
  Conference: the two-way problem as a linear system on a graph.
  https://ideas.repec.org/p/boc/scon16/24.html
- Kelner, Orecchia, Sidford and Zhu (2013), "A Simple, Combinatorial Algorithm for Solving
  SDD Systems in Nearly-Linear Time", arXiv:1301.6628 (cited by Correia).
  https://arxiv.org/abs/1301.6628
- Abowd, Creecy and Kramarz (2002), "Computing Person and Firm Effects Using Linked
  Longitudinal Employer-Employee Data", LEHD Technical Paper 2002-06 (effects identified
  within connected groups). https://ideas.repec.org/p/cen/tpaper/2002-06.html
- SciPy 1.18.1 `scipy.sparse.linalg.splu` (SuperLU) and
  `scipy.sparse.csgraph.connected_components`; Numba 0.62.1.

General nearly-linear-time Laplacian solvers were not implemented; this candidate relies on
the reduced systems being small, well connected, or forest-like.

## Hypotheses and bottlenecks

- **Low mobility should be easy for a direct solve.** Confirmed: on workloads 9 and 11 the
  firm block splits into 1,390 and 13,975 components; the factorization's fill ratio is
  0.97-0.98 and the residual 8e-15 to 3e-14.
- **The remaining time is data passes and Python glue.** On workload 12 (prototype, quiet
  host): pass 1 2.27 s, pass 2 1.53 s, assembly 0.84 s, setup 0.48 s, solve 0.26 s. On
  workload 6 the data passes take 0.05 s of 0.45 s; setup and SciPy assembly take the rest.
- **Next steps (not implemented):** fuse pass 1 (one read of X instead of three), move
  setup and assembly into compiled code, multithread both passes by row blocks.

## Experiments

Prototype measurements before packaging were single `run` calls after a JIT warm-up,
compared with a reference cached in the same session; they are not evaluator results.

1. Explicit S from SciPy sparse products plus a full SuperLU factorization as the
   preconditioner: workload 9 took 0.94 s, but workload 4's factorization did not finish
   within several minutes (fill on a well-connected graph). Abandoned.
2. CG to a 1e-13 relative residual hung on the singular reduced systems: rounding leaves an
   inconsistent part of the right-hand side. Changed to a 1e-10 target or stagnation below
   1e-8, returning the best iterate.
3. Explicit S on gravity has 18M nonzeros; that version took 3.4 s, slower than the
   reference (1.65 s). Applying S implicitly through its factors: 0.45 s.
4. SuperLU incomplete LU with the year columns included, on workload 11: a 30 s solve, and
   CG stagnated at a 6e-7 residual. Replaced by the exact grounded factorization with a
   10x10 Schur complement for years: 1.4-2.3 s, residual 3e-14.
5. Dense Schur accumulation on workload 7 took 0.83 s (180M updates). Grouping identical
   cross-tabulation rows reduces 200,000 firm rows to 15,000 patterns (workload 12: 1M to 1).
6. Random small problems against the reference (singletons, arbitrary integer codes, nested
   fixed effects, and each solver path forced on): 538 cases, no failures except two
   non-identified problems where the fixed-effect dummies have rank n.

Prototype results against the same-session reference (single calls, one thread):

| Workload | Reference (s) | Prototype (s) | Max beta error | Max SE error |
|---|---:|---:|---:|---:|
| 1 | 0.23 | 0.021 | 1.6e-14 | 2.0e-15 |
| 3 | 29.6 | 1.90 | 4.4e-12 | 2.1e-14 |
| 4 | 16.7 | 0.90 | 3.6e-12 | 3.0e-8 |
| 6 | 1.65 | 0.45 | 2.0e-13 | 5.1e-15 |
| 7 | 24.0 | 2.37 | 4.5e-12 | 1.4e-9 |
| 8 | 58.8 | 1.12 | 7.2e-11 | 1.6e-7 |
| 9 | 113.6 | 0.62 | 5.5e-8 | 2.9e-8 |
| 10 | 7.83 | 0.75 | 1.3e-11 | 3.8e-14 |
| 11 | 2,510 (swapping) | 5.64 | 4.1e-7 | 2.1e-7 |
| 12 | 91.2 | 5.58 | 1.1e-11 | 3.8e-14 |

The mid-size prototype timings were taken while the reference for workload 11 ran on the
same host, which the workflow advises against; the evaluator runs below supersede them.

### Evaluator runs

- `runs/20261017T220642.321493Z-3b91df07`: `--smoke`, all 5 development fixtures passed.
- `runs/20261017T220722.585503Z-43273a66`: workload 1 passed. Reference median 0.2461 s,
  candidate median 0.0128 s (19.2x); first call 3.94 s (Numba compilation).
- `runs/20261017T220821.256103Z-0b07bbf3`: workloads 3, 4, 6, 7, 8, 9 and 10 passed
  (`--timeout 900`, otherwise default protocol: one thread, one warmup, three measured
  calls; all five outputs per workload validated). Medians:

  | Workload | Reference (s) | Candidate (s) | Speedup | First call (s) | Max beta error | Max SE error | Peak RSS ref / cand (GB) |
  |---|---:|---:|---:|---:|---:|---:|---|
  | 3 | 23.876 | 1.2961 | 18.4x | 5.35 | 4.4e-12 | 2.1e-14 | 5.08 / 2.76 |
  | 4 | 18.541 | 0.6640 | 27.9x | 4.80 | 3.6e-12 | 3.0e-8 | 1.32 / 0.83 |
  | 6 | 1.910 | 0.3247 | 5.9x | 2.31 | 2.0e-13 | 5.1e-15 | 0.35 / 0.60 |
  | 7 | 25.236 | 1.5836 | 15.9x | 5.63 | 4.5e-12 | 1.4e-9 | 2.52 / 1.89 |
  | 8 | 49.758 | 0.7978 | 62.4x | 4.77 | 7.2e-11 | 1.6e-7 | 1.31 / 0.85 |
  | 9 | 113.608 | 0.4427 | 256.6x | 4.39 | 5.5e-8 | 2.9e-8 | 1.40 / 0.92 |
  | 10 | 7.142 | 0.5119 | 14.0x | 4.46 | 1.3e-11 | 3.8e-14 | 2.34 / 1.28 |

  First calls include Numba compilation. Peak RSS is the worker's lifetime high-water mark,
  including input generation.
- `runs/20261017T223149.929641Z-924972d8`: workload 12 passed (`--timeout 1800`). Reference
  median 68.401 s, candidate median 4.0326 s (17.0x); max beta error 1.1e-11, SE 3.8e-14;
  first call 8.10 s.
- `runs/20261017T224035.082271Z-b76c2624`: workload 11 with `--repeats 1` **incomplete**. The
  session hosting the run ended during the reference's calls; no summary was written and no
  timing is claimed. W11 is rerun in the same-session comparison below.

### Same-session comparison

Five candidates (`eliminate-largest-fe` and four earlier ones now in `prestudy/solutions/001/`: `xhdfe`,
`within`, `julia-fem` and `fixest`) ran one after
another from `runs/compare-001.sh` in the shared environment `solutions/001/.venv`, with nothing
else benchmarking on the host. Each run measures its own reference.

- `runs/20261018T023431.390119Z-65be5196`: workloads 1, 3, 4, 6, 7, 8, 9, 10 and 12, default
  protocol (one thread, one warmup, three measured calls; `--timeout 3600`).
- `runs/20261018T055637.090305Z-f59421e6`: workload 11 with `--repeats 1` (`--timeout 21600`).

| Workload | Status | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Max beta error | Max SE error | Peak RSS (GB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | passed | 0.3287 | 0.0215 | 0.0196-0.0234 | 15.3x | 5.33 | 1.6e-14 | 2.0e-15 | 0.20 |
| 3 | passed | 30.251 | 1.906 | 1.887-1.917 | 15.9x | 7.56 | 4.4e-12 | 2.1e-14 | 2.76 |
| 4 | passed | 27.190 | 1.013 | 1.000-1.030 | 26.8x | 6.58 | 3.6e-12 | 3.0e-08 | 0.82 |
| 6 | passed | 1.840 | 0.3204 | 0.3137-0.3682 | 5.7x | 2.40 | 2.0e-13 | 5.1e-15 | 0.62 |
| 7 | passed | 24.706 | 1.558 | 1.526-1.650 | 15.9x | 5.64 | 4.5e-12 | 1.4e-09 | 1.88 |
| 8 | passed | 80.006 | 1.360 | 1.337-1.362 | 58.8x | 7.17 | 7.2e-11 | 1.6e-07 | 0.79 |
| 9 | passed | 119.8 | 0.4770 | 0.4690-0.4790 | 251.1x | 4.81 | 5.5e-08 | 2.9e-08 | 0.87 |
| 10 | passed | 7.120 | 0.5023 | 0.4954-0.5511 | 14.2x | 4.38 | 1.3e-11 | 3.8e-14 | 1.31 |
| 11 | passed | 1856.1 | 4.589 | 1 call | 404.5x | 9.08 | 4.1e-07 | 2.1e-07 | 6.33 |
| 12 | passed | 69.376 | 3.787 | 3.730-3.790 | 18.3x | 8.57 | 1.1e-11 | 3.8e-14 | 8.86 |

Medians are per `run` call; the first call includes compilation or package warm-up and is not
scored. Peak RSS is the worker's lifetime high-water mark, including input generation.

Workload 11's reference took 1,856 s here and 2,510 s in the prototype measurement above; that
earlier figure was taken while the host was swapping. Reference medians for workloads 1, 3, 4 and 8
in this run are 20-65% higher than in the other four candidates' runs of the same batch (for
example 80.0 s versus 47.8-49.6 s on workload 8), so speedups over the reference there are
inflated; ratios against other candidates use candidate medians only. See
`prestudy/reports/001_reghdfe.md` for the cross-candidate table.

The random-problem test script (`fuzz.py`) lived in a session scratch directory that was
cleared before it could be added to this candidate, so the 538-case result above is not
reproducible from this folder; the evaluator's development fixtures and `--validation-seed`
are the reproducible checks.

## Reproduction

See `build.md` and `requirements.lock`. The prototype and its test scripts were developed
in a scratch directory; this candidate's `solution.py` is the packaged version.
