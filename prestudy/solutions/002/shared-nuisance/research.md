# Candidate research: shared-nuisance

## Objective and constraints

- Task 002, contract version 2: Callaway–Sant'Anna staggered DiD with the traditional doubly
  robust estimator (logit propensity score, never-treated OLS outcome model, 0.995 trimming of
  comparison units, varying base period), analytic influence-function standard errors, and the
  simple, event-time and pre/post aggregates, matching `reference.run` (ATT relative error
  1e-6, standard errors 1e-4, exact labels).
- Target hardware: CPU, one thread (the evaluator default). Host: Intel Core i7-8700B
  (6 cores), 32 GiB RAM, macOS 15.7.3. Environment: the shared task 002 environment
  (`build.md`): CPython 3.12.13, NumPy 2.3.5, Numba 0.62.1.
- Informed research session: the user authorized consulting the repository's prior results
  (`reports/002_csdid.md`) and the 14 October 2026 study, whose recommendations (cache the
  cohort propensity fit, reuse the never-treated OLS factorization, avoid the dense influence
  function storage) this candidate implements. The development fixtures and the reference were
  used for validation only.

## Method

The reference runs one 2x2 doubly robust DiD per (cohort g, period t) cell: a logit of cohort
membership and an OLS of the outcome change `y_t - y_base` among never-treated units, then a
per-unit influence function rescaled to all units, then aggregation.

1. **One logit per cohort.** A cell's sample (never-treated plus cohort g), its treatment
   indicator and the time-invariant covariates do not depend on t, so all T-1 cells of a cohort
   share the same propensity fit: one fit per cohort instead of per cell (10-40 instead of
   190-2,360). Newton-Raphson as in the reference (same 1e-12 step criterion), started from the
   previous cohort's estimate (about 5 iterations instead of 7-10), with the Hessian accumulated
   over 2,048-row blocks with BLAS. The Hessian from the last iteration is used for the
   covariance; the reference recomputes it after the final step, which is below 1e-12.
2. **One outcome regression for all cells.** The OLS design is the covariates of all
   never-treated units in every cell, and `dy` is linear in y. So `R = y - X B` with
   `B = (X0'X0)^-1 X0'Y0` for all periods at once, and a cell's residual is
   `R[:, t] - R[:, base]`. `(X0'X0)^-1` also serves the outcome model's influence-function term.
3. **Closed-form cell influence functions.** With r the residual difference, each unit's
   influence function is
   - treated: `att + (r - eta_t)/m1 - (1 - ps) X.v / m0`
   - comparison: `att - r X.u/m1 - (w0 (r - eta_c) - ps X.v - r X.u0) / m0`

   where m1, m0, eta_t, eta_c, u, u0 are per-cohort sums and v (the propensity-score estimation
   effect) is `V (Q_t - Q_base - eta_c sum(w0 X))`, with `Q = sum_controls w0 X R` computed
   once per cohort.
4. **Streaming aggregation.** One compiled pass over never-treated units, visiting each unit's
   residual row once for all cohorts and periods, plus one pass over treated units, evaluates
   these expressions and accumulates the per-cell sums of the influence function and its square
   (giving ATT(g,t) and its SE), and the per-unit simple and event-time aggregate influence
   functions. The cohort-weight estimation terms depend only on a unit's cohort and are added
   afterwards. The reference's n_units x n_cells influence-function and weight matrices
   (1.9 GB each on workload 5) are never formed; the only large array is n_units x n_events.

## Sources (accessed 2026-10-18)

- Callaway and Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods",
  Journal of Econometrics 225(2). https://arxiv.org/abs/1803.09015
- Sant'Anna and Zhao (2020), "Doubly Robust Difference-in-Differences Estimators", Journal of
  Econometrics 219(1). https://arxiv.org/abs/1812.01723
- DRDID `drdid_panel` reference (the traditional DR panel estimator and its influence
  function). https://psantanna.com/DRDID/reference/drdid_panel.html
- `reports/002_csdid.md`, "Implementation study (14 October 2026)", recommendations section.

## Hypotheses and bottlenecks

- **Nuisance refits dominate the reference.** Confirmed: removing them alone (first version,
  per-cohort NumPy loop) gave 25-27x on workloads 5 and 6.
- **After that, memory traffic dominates.** The reference's dense influence-function storage
  and per-cell temporaries of full length; streaming them removes most of the remaining time.
- **Remaining time (workload 6, prototype timing):** logit 0.43 s, never-treated pass 0.51 s,
  per-cohort sums 0.21 s, outcome regression 0.09 s, aggregation 0.03 s (1.28 s in all).
  The pass does about 105 million (unit, cell) updates. The number of (unit, cell) pairs is
  inherent to the per-cell variances and event-time influence functions; reducing it would need
  a moment-based formulation of all cross-cell covariances, which costs more.

## Experiments

Prototype measurements are single `run` calls after a warm-up call, against reference
timings from the same session (cached), not evaluator results.

| Version | W1 | W2 | W3 | W4 | W5 | W6 |
|---|---:|---:|---:|---:|---:|---:|
| Reference (s) | 0.027 | 0.28 | 2.20 | 5.55 | 83.2 | 119.1 |
| 1. per-cohort fits, NumPy cell loop (s) | 0.0032 | 0.021 | 0.094 | 0.51 | 3.39 | 4.40 |
| 2. compiled unit-major pass, NumPy logit (s) | 0.0020 | 0.012 | 0.055 | 0.21 | 1.64 | 2.44 |
| 3. pass vectorized over periods, blocked BLAS logit (s) | 0.0021 | 0.0082 | 0.037 | 0.17 | 1.05 | 1.78 |
| 4. last-Hessian reuse, fused event finalization (s) | 0.0019 | 0.0072 | 0.036 | 0.12 | 0.79 | 1.69 |

Version 4's W6 single call varied between 1.28 s and 1.69 s across sessions. Maximum relative
errors against the reference for version 4: ATT 1.0e-9, SE 2.2e-14 (all workloads).

- Version 2 first made the pass slower (0.86 s to 1.05 s on W6): the covariate dot product
  per (unit, cell) did not vectorize. Version 3 splits each cohort's periods into the pre- and
  post-treatment ranges, where base period and event-time column are contiguous in t.
- Logit Hessian kernels, per pass over W6's 256,397 never-treated units: NumPy 12.8 ms,
  compiled triangular loop 11.3 ms, compiled full-square loop 8.6 ms, blocked BLAS 6.6 ms.
- Random small problems against the reference (`fuzz.py`, 400 cases: 60-600 units, 3-11
  periods, 1-4 covariates, 1-4 cohorts including g = 2 and late cohorts, shuffled unit order,
  strong selection so that 15 cases have trimmed comparison units): no failures; worst error
  2.3% of the tolerance.

## Evaluator runs

- `runs/20261018T194239.626963Z-45b54cc7`: `--smoke` (in the task 001 environment), all 4
  development fixtures passed; `--validation-seed` 11, 12, 13:
  `runs/20261018T194255.885159Z-d5565432`, `runs/20261018T194311.765806Z-b03e7f22`,
  `runs/20261018T194328.352734Z-dc45ce91`, all passed.
- `runs/20261018T194652.060762Z-6c05aa79`: `--smoke` in the shared task 002 environment,
  passed.
- `runs/20261018T195143.154197Z-1fb46e17`: same-session comparison run, workloads 1-6, default
  protocol (one warmup, three measured calls, one thread), `--timeout 3600`. All 4 development
  fixtures and all 6 workloads passed.

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.0242 | 0.00187 | 0.00172-0.00196 | 13.0 | 3.07 | 0.16 |
  | W2 | 0.345 | 0.00708 | 0.00678-0.00819 | 48.8 | 3.15 | 0.17 |
  | W3 | 3.087 | 0.0394 | 0.0354-0.0399 | 78.5 | 3.22 | 0.19 |
  | W4 | 6.936 | 0.104 | 0.101-0.141 | 66.9 | 3.35 | 0.27 |
  | W5 | 96.72 | 0.807 | 0.787-0.815 | 119.9 | 4.37 | 0.49 |
  | W6 | 116.1 | 1.351 | 1.349-1.483 | 86.0 | 4.75 | 0.76 |

  The first call includes Numba compilation (about 3 s). The reference's peak RSS is 12.5 GiB
  on W5 and 6.0 GiB on W6. Maximum relative errors: ATT 1.0e-9 (W5), SE 2.2e-14 (W6).
