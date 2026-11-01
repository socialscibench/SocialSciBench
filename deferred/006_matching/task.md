# Task 006: Propensity-score and coarsened exact matching

## Computation

Two matching estimators of the **average treatment effect on the treated (ATT)**: the difference between treated units' outcomes and the outcomes of comparable untreated units. Both are standard in management research when treatment (e.g. receiving venture capital, adopting a practice, attending a training) is not randomly assigned but depends on observed characteristics.

**1. Propensity-score matching (PSM)**, the computation behind

```stata
teffects psmatch (y) (d x1 ... xk, logit), atet nneighbor(1)    // official
psmatch2 d x1 ... xk, outcome(y) logit neighbor(1)               // community (SSC)
```

and R's `MatchIt::matchit(d ~ x1 + … + xk, method = "nearest", distance = "glm", replace = TRUE, estimand = "ATT")`:

1. Fit a **logit** of the treatment `d` on the covariates; each unit's fitted probability is its propensity score.
2. Match each treated unit to the **untreated unit with the nearest propensity score**, with replacement (one untreated unit can serve several treated units). Untreated units at exactly the same distance (ties) are all used and averaged.
3. ATT = average over treated units of (own outcome − matched outcome).

`psmatch2` uses a **probit** propensity score unless `logit` is given; the benchmark uses `logit` everywhere.

**2. Coarsened exact matching (CEM)**, the computation behind

```stata
cem x1 x2 x3 x4, treatment(d)
regress y d [iweight = cem_weights]
```

and R's `MatchIt::matchit(d ~ x1 + … + x4, method = "cem", cutpoints = …, estimand = "ATT")`:

1. **Coarsen** each covariate into bins. Stata's `cem` default (Sturges' rule as `cem` implements it) uses ceil(log2(n) + 1) equally spaced cutpoints from the variable's minimum to its maximum, and a value equal to a cutpoint goes to the lower bin.
2. Units with the same bins on every covariate form a **stratum**; strata containing both treated and untreated units are **matched**, the rest are dropped.
3. **Weights:** 1 for matched treated units; (treated in stratum ÷ untreated in stratum) × (all matched untreated ÷ all matched treated) for matched untreated units; 0 otherwise.
4. ATT = weighted mean outcome of treated − weighted mean outcome of untreated.

**Where the time goes.**
- **PSM:** the logit is cheap; matching requires a nearest-neighbour search. Sorting untreated units by propensity score turns it into a fast binary search, but a naive search compares every treated unit with every untreated unit. `teffects` also computes Abadie–Imbens standard errors, which require extra matching.
- **CEM:** forming strata is a single sort. But Stata's `cem` loops over every observation and, inside that loop, updates a flag for **all** observations (`cem-mata.do`, `assignStrata`), so its time grows with the **square** of the sample size.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| y | [n] | float64 (outcome) |
| d | [n] | float64 (0/1 treatment) |
| X | [n, n_covariates] | float64 |

PSM uses all covariates; CEM coarsens the first four.

## Outputs

| Name | Content |
|---|---|
| psm | dict: `att`, `n_treated`, `n_matched_controls` |
| cem | dict: `att`, `n_strata`, `n_matched_strata`, `n_matched_treated`, `n_matched_controls` |

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(y, d, X, cem_covariates=4) -> dict(psm=dict(att, ...), cem=dict(att, n_strata, ...))
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Treatment depends on the covariates through a logit (selection on observables), the outcome depends on the same covariates, and the true treatment effect is 1.0. Half of the covariates are skewed (log-normal, like firm size).

| # | design | n | treated | covariates |
|---|---|---|---|---|
| 1 | sanity check | 1,000 | 201 | 3 |
| 2 | firms (e.g. VC-backed vs not) | 20,000 | 2,043 | 6 |
| 3 | startups (Crunchbase-scale) | 100,000 | 9,880 | 8 |
| 4 | workers or inventors | 500,000 | 24,924 | 6 |
| 5 | administrative records | 2,000,000 | 99,921 | 5 |

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| ATT (PSM, with the logit converged to 1e-10; and CEM) | relative error ≤ 1e-8 |
| CEM strata, matched strata, matched treated and untreated counts | exact |

Nearest-neighbour matching is numerically fragile at these sizes: many treated units have two nearest untreated neighbours whose propensity-score distances differ by less than 1e-9, so small differences in the fitted logit change which one is matched. Estimate the logit to tight convergence (parameter change < 1e-10); the ATT must then match the reference within 1e-8.

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy implementation: a Newton-Raphson logit, nearest-neighbour search over untreated units sorted by propensity score, and CEM strata from one sort. It defines what is computed and is not a performance target.

## Not covered in this version

- Standard errors (`teffects` reports Abadie–Imbens standard errors; `psmatch2` and `MatchIt` report different ones).
- Matching without replacement, several neighbours, calipers, Mahalanobis matching, kernel matching, and `k2k` CEM.
- User-chosen CEM cutpoints and categorical covariates.
