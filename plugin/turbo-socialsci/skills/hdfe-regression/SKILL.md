---
name: hdfe-regression
description: Fits OLS with absorbed high-dimensional fixed effects and one-way cluster-robust standard errors using Stata reghdfe's conventions (singletons dropped, collinear regressors dropped, reghdfe's small-sample factor and df_a, full-model R²), fast and with bounded memory on millions of rows. Use for linear regressions with fixed effects such as worker, firm, year or interactions like firm#year and clustered SEs, for example when a task names reghdfe, absorb(), feols with fixed effects, or two-way or multi-way fixed-effects regressions.
compatibility: Python 3.12 or 3.13. Installs numpy 2.3.5, numba 0.62.1, scipy, pandas and pyarrow from PyPI.
---

# Fixed-effects regression (reghdfe-style)

## Does it apply?

Use it when the regression is **linear OLS** with one outcome and numeric regressors, **at least one
absorbed fixed effect** (a column, or an interaction `a#b` meaning one effect per combination), and
standard errors **clustered by one variable** (or by an interaction).

Not covered: no fixed effects; robust, unadjusted or multi-way clustered SEs; weights; IV; Poisson
or other nonlinear models; heterogeneous slopes (`absorb(a#c.x)`). For these, do the task with your
usual tools: `feols` raises `turbo_socialsci.NotApplicable` (CLI exit code 3) before computing anything.

## Install (once per Python environment)

```bash
python3.12 -m venv .venv          # Python 3.12 or 3.13
.venv/bin/pip install -q "${CLAUDE_PLUGIN_ROOT}"/wheels/turbo_socialsci-*.whl
```

Then run your scripts with `.venv/bin/python`. Installing pulls NumPy, Numba, SciPy, pandas and pyarrow from
PyPI; `uv pip install` works too where uv runs.

## Use

```python
from turbo_socialsci import feols, warmup

warmup("hdfe")                 # loads (first time after install: compiles) the kernels; untimed
res = feols("data.parquet", y="lnwage", x=["union", "tenure"],
            absorb=["worker", "firm#year"], cluster="firm", threads=4)
print(res.summary())
res.coef        # DataFrame indexed by regressor: coef, se, t, p, ci_low, ci_high, dropped
res.vcov        # DataFrame, kept regressors
res.nobs, res.n_clusters, res.df_a, res.r2, res.r2_within, res.rss, res.dropped
```

- `data`: a pandas DataFrame or a `.parquet`, `.csv`, `.dta` or `.feather` path. Only the needed columns are read, and a DataFrame's columns are read in place.
- Subsets: filter the DataFrame first (for example `df[df["year"] >= 2000]`), then pass it.
- Dummy regressors: leave them as they are (integers, booleans or floats holding only 0 and 1). They are recognized and stored as lists of their ones, so many dummies cost little. Converting them all to one float matrix first only adds work.
- `variant`: `"auto"` (default) picks `"speed"` unless its estimated peak exceeds half of RAM, then `"memory"`.
  - With one fixed effect, both variants compute the same numbers exactly, from group means; `"speed"` also copies the sample into the groups' order, about 8 bytes per row and non-dummy column, and `"memory"` reads it where it is.
  - With several, `"memory"` forms no n × k demeaned matrix and works through batches of columns, 1.2–2.3× slower.
- `threads`: BLAS and Numba threads for the call (default: the libraries' own settings and the `OMP_NUM_THREADS`/`NUMBA_NUM_THREADS` environment variables).
- `check_only=True` returns `{"applies": True, ...}` without estimating, or raises NotApplicable.

The first call in a fresh Python process loads the compiled kernels from a cache, which takes about a
second. The first call after installation compiles them, which takes about 45 seconds (20 more for the memory variant). Call
`warmup("hdfe")`, or `warmup("hdfe", variant="memory")` for the memory variant, before timing anything.

## What it computes (reghdfe's conventions)

1. Rows with a missing outcome, regressor, fixed effect or cluster value are dropped. Then singleton groups (a level of any fixed effect with one row) are dropped repeatedly. `res.nobs` is N after both steps.
2. Regressors collinear with the fixed effects (for example a dummy that is all zero in the sample) or with regressors to their left are dropped. They are listed in `res.dropped`, with `coef` NaN and `dropped` True.
3. An intercept is included.
4. The cluster-robust sandwich is scaled by (N − 1)/(N − K) · G/(G − 1), with K = kept regressors + 1 + `df_a`. Fixed effects nested in the cluster variable contribute nothing to `df_a`.
5. t, p and the 95% interval use the t distribution with G − 1 degrees of freedom.
6. `r2` = 1 − RSS/Σ(y − ȳ)² of the full model (fixed effects included) on the estimation sample. `r2_within` uses the demeaned outcome instead.

## Command line

```bash
turbo-socialsci hdfe --data d.parquet --y y --x x1 x2 --absorb firm firm#year --cluster firm \
    --threads 4 --out coef.csv          # --check: only check coverage; exit 3 = not applicable
```
