---
name: staggered-did
description: Estimates Callaway and Sant'Anna (2021) group-time average treatment effects for staggered adoption with panel data as Stata's csdid 1.81 or 2.0.1 computes them (never- or not-yet-treated controls; dripw, drimp, reg, stdipw and ipw; varying or universal base periods; covariates fixed or varying over time; sampling weights; propensity-score trimming; unbalanced panels), with event-study, group, calendar and simple aggregations, analytic standard errors and the multiplier (wild) bootstrap with uniform critical values. Use for csdid, did or att_gt estimations, Callaway-Sant'Anna event studies, and staggered difference-in-differences.
compatibility: Python 3.12 or 3.13. Installs numpy 2.3.5, numba 0.62.1, scipy, pandas and pyarrow from PyPI.
---

# Staggered difference-in-differences (csdid-style)

## Does it apply?

Use it for the Callaway and Sant'Anna estimator on a **panel** (one row per unit and period), with
period values that are **consecutive integers** (such as years), a first-treatment period per unit
(0 = never treated within the panel), and the options listed below.

Not covered:
- repeated cross-sections;
- clustered SEs or a clustered bootstrap;
- anticipation;
- weights that vary over time;
- `bal(full)` and `bal(none)`;
- other estimators (TWFE, Sun-Abraham, de Chaisemartin-D'Haultfœuille, imputation);
- continuous treatments;
- periods with gaps.

For these, do the task with your usual tools: `att_gt` raises `turbo_socialsci.NotApplicable` (CLI
exit code 3) before computing anything.

## Install (once per Python environment)

```bash
python3.12 -m venv .venv          # Python 3.12 or 3.13
.venv/bin/pip install -q "${CLAUDE_PLUGIN_ROOT}"/wheels/turbo_socialsci-*.whl
```

Then run your scripts with `.venv/bin/python`. Installing pulls NumPy, Numba, SciPy, pandas and pyarrow from
PyPI; `uv pip install` works too where uv runs.

## Use

```python
from turbo_socialsci import att_gt, warmup

warmup("did")                  # loads (first time after install: compiles) the kernels; untimed
res = att_gt("panel.parquet", y="y", unit="county", time="year", cohort="first_treat",
             covariates=["x1", "x2"], weights="pop", control="notyet", method="dripw",
             base_period="universal", trim=0.995, rules="csdid181", n_boot=25000, seed=1,
             boot_tables=["event"], threads=4)     # bootstrap only the table you report
res.gt          # cohort, time, att, se, se_boot          (every ATT(g,t))
res.event       # e, att, se, se_boot                     (event study)
res.group, res.calendar                                   # cohort and calendar-time aggregations
res.averages    # rows simple, group_avg, calendar_avg, pre_avg, post_avg: att, se, se_boot
res.crit        # uniform critical values: gt, event, event_with_avgs, group, calendar
res.cov_event   # event-study covariance (analytic)
```

`data` is a DataFrame or a `.parquet`, `.csv`, `.dta` or `.feather` path. To use a subsample, filter
the DataFrame first.

| csdid | `att_gt` |
|---|---|
| `ivar() time() gvar()` | `unit=`, `time=`, `cohort=` (0 = never treated; otherwise one of the panel's periods) |
| controls `varlist` | `covariates=[...]`; constant within unit = fixed, otherwise varying (a cell uses its earlier period's values) |
| `[iw=w]` / sampling weights | `weights=` (constant within unit; normalized to mean 1 within each cell) |
| `notyet` | `control="notyet"` (default `"never"`) |
| `method(dripw)` (csdid's default), `drimp`, `reg`, `stdipw`, `ipw` | `method=` |
| `long2` (csdid 1.81): every cell against g − 1 | `base_period="universal"` (default `"varying"`) |
| `pscoretrim(#)` | `trim=#` (default 1.0: no trimming) |
| `wboot reps(B)`, Rademacher multipliers | `n_boot=B`, `seed=` (one multiplier per unit, shared by every parameter, drawn inside the call) |
| version | `rules="csdid181"` (csdid 1.81 with drdid 1.91) or `"csdid201"` (csdid 2.0.1) |
| `estat event, window(lo hi)` | `event_window=(lo, hi)`; None (default) reports every event time |
| bootstrap of the reported table only (`estat event`, `estat group`, ...) | `boot_tables=["event"]`, `["gt"]`, `["group"]`, `["calendar"]` or several; None (default): all |

## What it reports

- `se` is the analytic influence-function SE, √(Σᵢ ψᵢ²)/N, where N is the number of units in the estimation sample.
- `se_boot` is the bootstrap SE, the interquartile range of the draws divided by 1.349. The quantile rule follows `rules`.
- `boot_tables` limits the bootstrap to the tables you report; the others' `se_boot` and `crit` are NaN. The averages follow their tables: `pre_avg` and `post_avg` follow `"event"`, `group_avg` `"group"`, `calendar_avg` `"calendar"`. Without `"gt"`, the speed variant draws only the aggregates' own influence functions: same results, up to 1.5× faster for an event study.
- A uniform band is estimate ± crit × se_boot.
  - `crit["event"]` covers the event times only.
  - `crit["event_with_avgs"]` covers the event times together with Pre_avg and Post_avg, the rows of csdid 1.81's `estat event` table.
  - `crit["gt"]` covers every ATT(g,t).
- `pre_avg` and `post_avg` are simple averages of the event-time estimates with e < 0 and e ≥ 0 (within `event_window`). Event-time estimates weight cohorts by their weighted size, as csdid does.
- With `base_period="universal"`, the cell t = g − 1 is the base, so e = −1 is not reported.

## Speed, memory and threads

- `variant="auto"` (default) picks `"speed"`, a compiled single pass over units, unless its estimated peak exceeds half of RAM. In that case it picks `"memory"`: bounded working arrays, a third to four fifths of the speed variant's memory (least with covariates), 1.2–2.6× slower.
- Both variants compute the bootstrap's sums without forming the multipliers as numbers. Each 8 units' 256 signed sums are tabulated, and each draw adds the entry its 8 multipliers select. The multipliers, and so the results, are identical in both variants.
- `threads` sets BLAS and Numba threads for the call.
- The first call in a fresh process loads the compiled kernels from a cache, which takes about a second. The first call after installation compiles them, which takes about 45 seconds. Call `warmup("did")`, or `warmup("did", variant="memory")` for the memory variant, before timing anything.
- `check_only=True` returns `{"applies": True, ...}` without estimating, or raises NotApplicable.

## Command line

```bash
turbo-socialsci did --data panel.csv --y y --unit county --time year --cohort first_treat \
    --covariates x1 x2 --weights pop --control notyet --base-period universal --trim 0.995 \
    --n-boot 25000 --threads 4 --out results/     # writes gt.csv, event.csv, group.csv, calendar.csv, averages.csv
```
