# Task 004: Pooled and grouped winsorization

## Computation

Winsorize variables at the 1st and 99th percentiles: every value below the 1st percentile is set to the 1st percentile, and every value above the 99th percentile is set to the 99th percentile. The percentiles are computed either over the whole sample (pooled) or separately within groups such as years, industry-years or trading days. This is the computation behind

```stata
winsor2 v1 ... vk, cuts(1 99) replace            // pooled
winsor2 v1 ... vk, cuts(1 99) by(year) replace   // within each year
```

It is a routine data-preparation step in finance, accounting and management research. Accounting ratios (return on assets, leverage, Tobin's q, growth rates) and stock returns have extreme values that would otherwise dominate regressions, so most papers using Compustat or CRSP winsorize their continuous variables, often within each year.

For each variable and each group, the computation has three steps:

1. **Sort** the non-missing values: x(1) ≤ x(2) ≤ … ≤ x(n).
2. **Compute the two cutoffs** using Stata's default percentile definition (used by `_pctile`, `summarize, detail` and `egen pctile()`). For percentile p:
   - if n·p/100 is a whole number j, the percentile is the average (x(j) + x(j+1)) / 2;
   - otherwise it is x(floor(n·p/100) + 1).
3. **Clip** every value to [lower cutoff, upper cutoff]. Missing values stay missing.

**Percentile definitions differ between tools.** Stata's definition is type 2 in Hyndman & Fan's (1996) classification: R's `quantile(type = 2)` and NumPy's `method = "averaged_inverted_cdf"`. The defaults of R's `quantile()` (type 7), pandas' `quantile()` and NumPy's `quantile()` interpolate between neighboring values instead, so they give different cutoffs. `scipy.stats.mstats.winsorize` uses yet another rule, clipping a fixed *count* of observations. A replication in R or Python that uses the defaults will not reproduce Stata's winsorized values.

**Where the time goes.** Sorting each variable within each group dominates. With `by()`, `winsor2` sorts the entire dataset once per variable (inside `egen pctile()`, twice: once per cutoff) and creates four temporary variables per variable, so its cost grows with variables × rows × log(rows).

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| X | [n_obs, n_vars] | float64 (NaN = missing) |
| group | [n_obs] or none | int64 |
| cuts | 2 | float64 (percentiles, e.g. 1 and 99) |

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| X_w | [n_obs, n_vars] | float64 | the variables after `winsor2 …, replace` |
| lo, hi | [n_groups, n_vars] | float64 | the cutoffs (`winsor2` keeps them in temporary variables) |

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(X, group, cuts) -> dict(X_w, lo, hi)
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Values are heavy-tailed (Student t with 3 degrees of freedom, scaled and shifted per variable), as accounting ratios and stock returns are, with about 5% missing values.

| # | design | n_obs | n_vars | by() | n_groups | missing | cuts |
|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1,000 | 3 | – | 1 | 6% | 1, 99 |
| 2 | firm-year, pooled (Compustat) | 150,000 | 20 | – | 1 | 5% | 1, 99 |
| 3 | firm-year, by year (Compustat) | 200,000 | 20 | year | 30 | 5% | 1, 99 |
| 4 | firm-year, by industry-year | 1,000,000 | 10 | industry#year | 1,440 | 5% | 1, 99 |
| 5 | stock-month, by month (CRSP monthly) | 3,000,000 | 5 | month | 360 | 5% | 1, 99 |
| 6 | stock-day, by day (CRSP daily) | 20,000,000 | 3 | day | 2,500 | 5% | 1, 99 |

`by()` is the grouping variable; "–" means pooled. `a#b` means each combination of `a` and `b`.

## Correctness

Correctness is checked on summaries: `checksums` in `reference.py` reports, for every winsorized variable, the number of non-missing values, their sum, sum of squares, minimum and maximum, and how many values winsorizing changed. An implementation must match the reference on every workload:

| Output | Tolerance |
|---|---|
| sums, sums of squares, min, max | max relative error ≤ 1e-10 |
| non-missing count, values changed | exact |

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy implementation (one full sort per variable). It defines what is computed and is not a performance target.

## Not covered in this version

- Trimming (`winsor2 …, trim`, which sets extreme values to missing instead of clipping them).
- Nick Cox's `winsor`, which clips a fixed number of observations (`h()`) rather than at percentiles.
- Weights, `if`/`in` subsets, and keeping the original variables (`suffix()` instead of `replace`).
