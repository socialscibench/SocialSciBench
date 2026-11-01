# Task 009: Multilevel (random-intercept) logistic regression

## Computation

Two-level **random-intercept logit** (multilevel or mixed-effects logit): a 0/1 outcome for observations nested in groups, with a group-specific intercept that is treated as a normally distributed random variable rather than estimated as a fixed effect. This is the computation behind

```stata
melogit y x1 ... xk || firm:
```

with Stata's defaults (7 integration points, `intmethod(mvaghermite)`, `vce(oim)`), and, in R,

```r
lme4::glmer(y ~ x1 + ... + xk + (1 | firm), family = binomial, nAGQ = 7)
```

(`glmer`'s own default is `nAGQ = 1`, the Laplace approximation; `nAGQ = 7` uses the same number of points as Stata). Typical uses: whether an employee leaves (employees in firms), whether a firm makes an acquisition in a year (firm-years in firms), whether a person starts a business (individuals in countries), whether a patent becomes a breakthrough (patents in firms). Random intercepts are used instead of fixed effects (task 003) when there are group-level regressors (firm size, country culture) whose coefficients are of interest, or when groups are small.

The model:

> P(y_ij = 1 | u_j) = Λ(x_ij′β + u_j),  u_j ~ N(0, σ²_u),  Λ(z) = 1 / (1 + e^(−z))

The outputs are β (with a constant), their standard errors, the variance of the random intercept σ²_u (`var(_cons)`) with its standard error, and the log likelihood. Because u_j is unobserved, each group's likelihood is an integral:

> L_j(β, σ_u) = ∫ ∏_{i∈j} Λ(x_ij′β + u)^y_ij (1 − Λ(x_ij′β + u))^(1−y_ij) φ(u; 0, σ²_u) du

It has no closed form and is computed by **adaptive Gauss–Hermite quadrature**: the integrand is evaluated at Q = 7 points u_jq placed around where the group's integrand is concentrated, and a weighted sum replaces the integral. The computation has three steps:

1. **Starting values**: a logit without random effects (Stata also refines σ_u on a small grid).
2. **Adaptation** (every iteration): for each group, find where its integrand is concentrated and how wide it is. Stata's `mvaghermite` uses the posterior mean and variance of u_j; `lme4` and the reference use the mode and curvature. These are two different 7-point approximations of the same integral, so the estimates differ slightly (see Correctness).
3. **Maximization** of Σ_j log L_j over (β, ln σ_u). Each likelihood evaluation costs N × Q logistic terms. Stata uses Newton–Raphson with derivatives. `lme4` optimizes over all k + 2 parameters with derivative-free optimizers (`bobyqa`, then `Nelder_Mead`), so it needs many more likelihood evaluations as k grows. The reference maximizes the same approximation as `lme4` with Newton-type steps. It uses the exact gradient, including how the quadrature points move when the parameters change, so it needs only 4–6 iterations.

Standard errors come from the inverse of the negative Hessian at the optimum (in the reference, the numerical derivative of the exact gradient). The standard error of var(_cons) uses the delta method: SE(σ²_u) = 2σ²_u × SE(ln σ_u).

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| y | [n_obs] | int8 (0/1) |
| X | [n_obs, k] | float64 (a constant is added) |
| group | [n_obs] | int64, values 0 … n_groups − 1 |
| Q | scalar | 7 integration points |

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| beta | [k + 1] | float64 | `_b[x1]` … `_b[_cons]` |
| se | [k + 1] | float64 | `_se[x1]` … |
| var_u, se_var_u | scalar | float64 | `_b[/var(_cons[firm])]`, `_se[/var(_cons[firm])]` |
| ll | scalar | float64 | `e(ll)` |

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(y, X, group, tol=1e-10, max_iter=200) -> dict(beta, se, var_u, se_var_u, ll, ...)
# Q = 7 integration points; the constant is added as the last column of X
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Regressors are standard normal; the first is turned into a 0/1 variable (e.g. a treatment), and the last few are constant within groups (e.g. firm size, country GDP). The random intercepts have σ_u = 1, so about 23% of the latent variance is between groups. Group sizes are unbalanced (lognormal).

| # | design | n_obs | n_groups | group size (mean / max) | k (of which between-group) | share y = 1 |
|---|---|---|---|---|---|---|
| 1 | sanity check | 5,000 | 100 | 50.0 / 266 | 5 (1) | 0.309 |
| 2 | employees in firms (survey) | 50,000 | 1,000 | 50.0 / 277 | 10 (3) | 0.296 |
| 3 | firm-years in firms (panel) | 200,000 | 19,717 | 10.1 / 148 | 15 (3) | 0.298 |
| 4 | individuals in countries (cross-national survey) | 500,000 | 60 | 8,333.3 / 32,074 | 10 (3) | 0.277 |
| 5 | patents in firms | 1,000,000 | 10,000 | 100.0 / 1,109 | 15 (3) | 0.294 |
| 6 | job postings in firms | 2,000,000 | 99,821 | 20.0 / 245 | 20 (4) | 0.297 |

## Correctness

Stata's `mvaghermite` and the reference's (and `lme4`'s) mode-curvature quadrature are two different 7-point approximations of the same likelihood, and Stata stops at a looser convergence criterion. Exact agreement is therefore not expected. An implementation must match the reference within:

| Output | Tolerance |
|---|---|
| β (each coefficient, including the constant) | \|Δβ\| ≤ 0.02 × reference SE |
| SE of β | relative error ≤ 1e-3 |
| var(_cons) and its SE | relative error ≤ 1e-3 |
| log likelihood | relative difference ≤ 1e-6 |

**Two quadrature rules.** Stata's `mvaghermite` and the mode-curvature rule of `lme4` and the reference give log likelihoods that differ by up to 0.5 on workload 6 (7e-7 relative), and coefficients that differ by up to 0.01 SE. Both are correct 7-point approximations. An implementation can use either, within the tolerances above.

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy/SciPy implementation: mode-curvature adaptive Gauss–Hermite quadrature, maximized with Newton-type steps on the exact gradient. It defines what is computed and is not a performance target.

## Not covered in this version

- Random slopes (`|| firm: x1`), three or more levels (`|| industry: || firm:`) and crossed random effects. These need multidimensional integration (7² = 49 points or more per group in Stata), which is much slower.
- `vce(cluster)` / `vce(robust)`, sampling weights, `estat icc`, `predict` of random effects, and `margins` after `melogit`.
- The Laplace approximation (`intmethod(laplace)`, `glmer`'s default), `xtlogit, re` (which uses 12 points by default) and `meglm` with other families (`meprobit`, `mepoisson`).
