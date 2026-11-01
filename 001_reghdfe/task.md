# Task 001: Linear regression with high-dimensional fixed effects

## Workload priority

Prioritize performance on `real-development` workloads while maintaining correctness and
measuring performance on synthetic workloads. Report the suites separately. Held-out inputs
are reserved for final evaluation of frozen implementations. See [DATA.md](DATA.md) for the
workload catalog, development coverage, data locations and split rules. The computation,
accuracy requirements and timing boundary below apply to both real and synthetic inputs.

Workload 109 (`real-smoke`) is for compatibility/crash checks only, never speed or accuracy tuning.

## Computation

Linear regression with multiple high-dimensional fixed effects and one-way cluster-robust standard errors. This is the computation performed by

```stata
reghdfe y x1 ... xk, absorb(fe1 fe2 ... feG) vce(cluster cl)
```

The computation has four steps:

1. **Singleton removal.** Repeatedly drop any observation that is the only member of its level of some fixed effect, until none remain.
2. **Partialling out.** Remove all fixed effects from `y` and from every column of `X`. This is the Frisch–Waugh–Lovell step. It is equivalent to regressing each variable on the full set of FE dummies and keeping the residual.
3. **OLS.** Regress the demeaned `y` on the demeaned `X`.
4. **Cluster-robust variance.** Compute the sandwich estimator `(X'X)⁻¹ (Σ_g s_g s_g') (X'X)⁻¹` with the small-sample factor `q = (N−1)/(N−K) · G/(G−1)`. Here `K = k + 1 + df_a`, and fixed effects nested within the cluster variable add nothing to `df_a`.

`df_a` follows reghdfe's conventions, as implemented in `absorbed_df` in `reference.py`:

- Fixed effects nested within the cluster variable contribute 0.
- The first non-nested fixed effect contributes all its levels.
- The second contributes its levels minus the number of connected components of the bipartite graph it forms with the first.
- Any further fixed effect contributes its levels minus 1, or minus the levels of an earlier fixed effect it is nested in (e.g. `region#year` within `year`).
- One level is then assigned to the intercept, so `df_a` matches reghdfe's `e(df_a)`.

**Where the time goes.** Step 2 accounts for almost all of the runtime. reghdfe solves it iteratively, by default with alternating projections (demean within each fixed effect in turn, repeat until convergence) accelerated by conjugate gradient. How fast it converges depends on how well the fixed effects are connected. In a worker–firm panel with low worker mobility, the bipartite worker–firm graph is sparse, so the number of iterations grows sharply. Workloads 4, 8 and 9 have the same size and differ only in mobility (10%, 2% and 0.5%). Balanced firm–year panels converge in a few sweeps, but large ones with many regressors still need one pass over all rows per variable and sweep: workload 12 (10M rows, 40 regressors) converges in 2 iterations.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| y | [n_obs] | float64 |
| X | [n_obs, n_regressors] | float64 |
| fe_ids | [n_obs, n_fe] | int64 |
| cluster_ids | [n_obs] | int64 |
| tol | scalar | float64 |
| maxiter | scalar | int64 |

- `fe_ids[:, g]` holds integer codes for the g-th absorbed factor. Codes can be any integers. Interactions such as `exp#year` are pre-encoded as a single code.
- `tol` defaults to 1e-8, matching reghdfe.

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| beta | [n_regressors] | float64 | `e(b)` without `_cons` |
| vcov | [n_regressors, n_regressors] | float64 | `e(V)` without `_cons` |
| nobs | scalar | int64 | `e(N)`, counted after singletons are dropped |
| df_a | scalar | int64 | `e(df_a)` |
| n_clusters | scalar | int64 | `e(N_clust)` |

## Interface

Provide a function with the same signature and return value as `reference.run`:

```python
run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000) -> dict(beta, vcov, nobs, df_a, n_clusters)
```

## Synthetic workloads

`workloads.py` (in this folder) generates the data deterministically (seed `1000 + id`): `generate(w)` returns the keyword arguments of `run` for each entry `w` in `WORKLOADS`. Workload ids are stable; ids 2 and 5 are retired.

| # | design | absorb() | cluster | n_obs | n_regressors | fe_levels | n_clusters | mobility |
|---|---|---|---|---|---|---|---|---|
| 1 | firm_year | `firm year` | firm | 100,000 | 3 | 10,000 × 10 | 10,000 | – |
| 3 | firm_year | `firm year` | firm | 5,000,000 | 20 | 500,000 × 10 | 500,000 | – |
| 4 | worker_firm | `worker firm year` | firm | 1,000,000 | 20 | 100,000 × 10,000 × 10 | 10,000 | 10% |
| 6 | gravity | `exp#year imp#year exp#imp` | exp#imp | 447,000 | 3 | 3,000 × 3,000 × 22,350 | 22,350 | – |
| 7 | four_way | `firm year ind#year reg#year` | firm | 2,000,000 | 20 | 200,000 × 10 × 3,000 × 500 | 200,000 | – |
| 8 | worker_firm | `worker firm year` | firm | 1,000,000 | 20 | 100,000 × 10,000 × 10 | 10,000 | 2% |
| 9 | worker_firm | `worker firm year` | firm | 1,000,000 | 20 | 100,000 × 10,000 × 10 | 10,000 | 0.5% |
| 10 | firm_year | `firm year` | firm | 1,000,000 | 50 | 100,000 × 10 | 100,000 | – |
| 11 | worker_firm | `worker firm year` | firm | 10,000,000 | 20 | 1,000,000 × 100,000 × 10 | 100,000 | 0.5% |
| 12 | firm_year | `firm year` | firm | 10,000,000 | 40 | 1,000,000 × 10 | 1,000,000 | – |

- `absorb()` lists the fixed effects in reghdfe syntax; `a#b` is one fixed effect per combination of `a` and `b`. `fe_levels` gives the number of levels of each, in the same order.
- `mobility` is the yearly probability that a worker switches firm.

Designs:

- **firm_year**: a balanced firm–year panel with firm and year fixed effects.
- **worker_firm**: an employer–employee panel in which workers move between firms, with worker, firm and year fixed effects (an AKM-style design). Low mobility makes these the slowest workloads.
- **gravity**: bilateral trade between countries, with exporter-by-year, importer-by-year and country-pair fixed effects.
- **four_way**: a firm panel with firm, year, industry-by-year and region-by-year fixed effects, including fixed effects nested in others.

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| beta | max relative error ≤ 1e-6 |
| sqrt(diag(vcov)) | max relative error ≤ 1e-4 |
| nobs, df_a, n_clusters | exact |

The partialling-out step must converge on every workload, including the poorly connected low-mobility panels (9, 11). A solver tolerance that is adequate on balanced panels can miss the `beta` tolerance there.

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable Python implementation (NumPy, SciPy). It defines what is computed and is not a performance target: it partials out the fixed effects with SciPy's general-purpose LSMR solver, one column at a time, with no problem-specific tuning.

## Shared evaluator (contract version 2)

`task.json` identifies the prompt files, generator, validator and tolerances. Inputs are read-only. Outputs must have the documented shapes and numeric types and contain only finite values. Relative errors use `abs(actual - reference) / max(abs(reference), 1e-12)` element by element, including near zero.

Run `python -m harness evaluate --task 001 --solution solutions/001/candidate/solution.json --smoke` from the repository root for public development fixtures. With the default `synthetic` suite, omit `--smoke` to validate those fixtures and time synthetic workload 1. Use `--ids` or `--all-workloads` for wider coverage within the selected suite; select `--suite real-development` explicitly for real inputs. Every first-call, warmup and measured output must pass. The score uses the ratio of reference and candidate median call times on the same host. Generation, imports, first-call setup, validation and serialization are excluded; input-dependent preprocessing and conversion to host-ready outputs are included. See `harness/README.md` for the full protocol. Historical baseline results predate this evaluator and are not automatically certified by it.

After the smoke check, start real-development evaluation from the repository or staged-workspace root:

```bash
python -m harness evaluate --task 001 --solution solutions/001/candidate/solution.json --suite real-development --ids 101 --threads 1
```

A staged workspace includes only the selected inputs; consult its `stage.json` and [DATA.md](DATA.md).
