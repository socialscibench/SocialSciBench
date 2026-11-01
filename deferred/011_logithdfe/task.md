# Task 011: Fixed-effects logit and average marginal effects

## Computation

Logit with **any number of high-dimensional fixed effects**, cluster-robust standard errors, and **average marginal effects (AMEs)** of a binary treatment and of continuous covariates on the probability, with delta-method standard errors. This is the computation behind

```stata
logit y i.d x1 ... xk i.fe1 i.fe2 ..., vce(cluster c)
margins, dydx(d x1 ... xk)
```

and, in R,

```r
m <- fixest::feglm(y ~ d + x1 + ... + xk | fe1 + fe2 + ..., family = binomial, cluster = ~c)
marginaleffects::avg_slopes(m)
```

A typical use: whether a firm adopts AI in a given year (0/1), explained by a treatment (e.g. a policy or an investment, 0/1) and firm characteristics, with firm fixed effects absorbing stable differences between firms and industry-by-year or state-by-year fixed effects absorbing shocks to whole industries or regions. The model is

> P(y_i = 1) = Λ(δ d_i + x_i′β + α¹[fe1_i] + α²[fe2_i] + …),  Λ(z) = 1 / (1 + e^(−z))

Each fixed effect is a set of intercepts, one per level: one per firm, one per industry-year, one per worker, and so on. The computation has four steps:

1. **Drop uninformative observations.** A level of any fixed effect whose outcome is 0 in every observation, or 1 in every observation, has an estimate of −∞ or +∞ and carries no information about the coefficients. Its observations are dropped (Stata reports "predicts failure perfectly"), and this repeats across all fixed effects until nothing more is dropped. With rare events, a large share of units is dropped.
2. **Maximum likelihood.** Estimate δ, β and every fixed effect. This is the ordinary *unconditional* fixed-effects logit: the fixed effects are estimated, not conditioned out, and no bias correction is applied. The reference uses iteratively reweighted least squares and partials the fixed effects out of each weighted regression, as `reghdfe` does (task 001). **This step dominates Stata's runtime**: Stata creates one dummy variable per fixed-effect level and works with dense matrices of size (number of parameters)².
3. **Cluster-robust variance.** The sandwich `q · H⁻¹ (Σ_c s_c s_c′) H⁻¹` over all parameters, where `s_c` is the score of cluster `c` and `q = G/(G−1)` (Stata's convention for maximum-likelihood models). Only the block for δ and β is reported.
4. **Average marginal effects**, averaged over the estimation sample and using each observation's estimated fixed effects:
   - **binary treatment `d`**: the average **discrete change** `(1/N) Σ_i [Λ(η_i with d=1) − Λ(η_i with d=0)]`, the change in the probability from switching the treatment on;
   - **continuous `x_k`**: the average **derivative** `(1/N) Σ_i β_k Λ′(η_i)`, the change in the probability from a one-unit increase.

   Standard errors use the delta method on the **full** variance matrix, so they include the estimation uncertainty of every fixed effect (Stata's `margins` convention).

**A statistical caveat.** With few observations per fixed-effect level (here 8–15 per unit), the unconditional fixed-effects logit is biased (the incidental-parameters problem; Fernández-Val & Weidner 2016). The benchmark computes the uncorrected estimator that Stata's `logit` with dummies and R's `feglm` both produce. Bias-corrected estimators (e.g. R's `alpaca` or `bife`) and the conditional logit (`clogit`) are different computations and are not covered.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| y | [n_obs] | float64 (0 or 1) |
| d | [n_obs] | float64 (0 or 1) |
| X | [n_obs, n_continuous] | float64 |
| fe_ids | [n_obs, n_fe] | int64 |
| cluster_ids | [n_obs] | int64 |
| tol | scalar | float64 |

- `fe_ids[:, g]` holds integer codes for the g-th fixed effect. Interactions such as `industry#year` are pre-encoded as one code, as in task 001.
- There is no separate constant; it is absorbed by the fixed effects.
- `tol` is the convergence tolerance on the coefficient update (reference: 1e-10).

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| beta | [1 + n_continuous] | float64 | `_b[1.d]`, `_b[x1]`, … after `logit` |
| se | [1 + n_continuous] | float64 | `_se[1.d]`, `_se[x1]`, … after `logit` |
| ame | [1 + n_continuous] | float64 | `_b[1.d]`, `_b[x1]`, … after `margins, dydx(d x1 …) post` |
| ame_se | [1 + n_continuous] | float64 | `_se[1.d]`, `_se[x1]`, … after `margins, … post` |
| nobs | scalar | int64 | `e(N)`, after dropping |
| n_clusters | scalar | int64 | `e(N_clust)` |
| fe_levels | [n_fe] | int64 | fixed-effect levels estimated, after dropping |

The treatment `d` comes first in every output.

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(y, d, X, fe_ids, cluster_ids, tol=1e-10, maxiter=100)
    -> dict(beta, se, ame, ame_se, nobs, n_clusters, fe_levels, ...)
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Every workload has one binary treatment `d` and continuous covariates; both are correlated with the unit fixed effect, so the fixed effects matter. Rare events mean many units never experience the event and are dropped ("obs used", "fe_levels (used)").

| # | design | absorb() | cluster | n_obs | n_covariates | n_fe | fe_levels (used) | n_clusters (used) | mobility | event rate | obs used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | `firm year` | firm | 3,000 | 4 | 2 | 250 × 10 | 250 | – | 29% | 2,500 |
| 2 | small multi-way check | `firm industry#year state#year` | firm | 6,000 | 4 | 3 | 516 × 100 × 100 | 516 | – | 31% | 5,160 |
| 3 | large public firms (S&P 1500) | `firm year` | firm | 22,500 | 6 | 2 | 1,192 × 15 | 1,192 | – | 20% | 17,880 |
| 4 | public firms + industry shocks (Compustat) | `firm industry#year` | firm | 75,000 | 11 | 2 | 3,404 × 720 | 3,404 | – | 15% | 51,060 |
| 5 | firms + industry and local shocks | `firm industry#year state#year` | state | 200,000 | 11 | 3 | 9,284 × 480 × 500 | 50 | – | 10% | 92,840 |
| 6 | workers + employers (résumé data) | `worker employer year` | employer | 500,000 | 6 | 3 | 24,212 × 1,000 × 10 | 1,000 | 5% | 10% | 242,120 |
| 7 | large private firms / establishments | `firm industry#year` | firm | 1,000,000 | 11 | 2 | 46,783 × 480 | 46,783 | – | 10% | 467,830 |

- `absorb()` lists the fixed effects in reghdfe syntax; `a#b` is one fixed effect per combination of `a` and `b`.
- `fe_levels (used)` and `obs used` are counted after dropping uninformative observations.

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| beta, ame | max relative error ≤ 1e-6 |
| se, ame_se | max relative error ≤ 1e-4 |
| nobs, n_clusters, fe_levels | exact |

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy/SciPy implementation: iteratively reweighted least squares, with the fixed effects partialled out of each weighted regression. It defines what is computed and is not a performance target.

## Not covered in this version

- Bias-corrected fixed-effects logit (`alpaca`, `bife`) and the conditional logit (`clogit`, `xtlogit, fe`).
- Probit, and Poisson with fixed effects (a separate task is planned).
- Separation caused by combinations of fixed effects or covariates (only single fixed-effect levels with an all-0 or all-1 outcome are dropped, as in Stata's and `fixest`'s defaults).
- Marginal effects of interactions or at specific values (`margins, at()`).
- Two-way clustering, sampling weights.
