# Task 002: Staggered-adoption difference-in-differences (`csdid`)

This is the combined task: the former tasks 002 (the core of Stata's `csdid`), 002b (its common options) and 002c (the options published replication packages add), with one reference, one loader and one set of 15 workloads.

**Target:** Stata `csdid`, in either of two versions, chosen per workload:
- **`csdid` 1.81:** the version SSC distributes; used by most existing replication packages, including the JEL DiD guide's.
- **`csdid` 2.0.1:** pinned commit `f98bcea`, released by the `csdid` authors in October 2026. It's the current version and 100–170× faster than 1.81.

The reference reproduces each exactly (section "Reference implementation").

## Computation

Notation:
- units i = 1..N and periods t = 1..T;
- G_i, the first treated period (0 = never treated);
- outcome Y_it, missing where the unit isn't observed;
- covariates X_i, fixed over time, or X_it varying over time;
- weight w_i (1 if unweighted).

**1. Sample.**
- Units with 0 < G_i ≤ 1 are dropped (treated in the first period).
- With never-treated controls, periods after the last one in which a never-treated unit is observed are not used. With not-yet-treated controls, periods after both that period and the last cohort are not used.
- Every unit still observed somewhere counts in N.

**2. Cells.** One 2×2 comparison per cohort g and period t.
- **Base period, `varying` (short gaps):** a pre-treatment cell compares t with t − 1; a post-treatment cell compares t with g − 1. Periods t = 2..T.
- **Base period, `universal` (long gaps):** every cell compares t with g − 1. Periods t = 1..T except g − 1, the reference, which isn't estimated. The outcome change is y(t) − y(g − 1), the sign of 1.81's `long2`, `csdid2`'s `long` and 2.0.1's `universal`.
- **Units in a cell:** the cohort, plus the comparison units:
  - never treated: G_i = 0;
  - not yet treated: G_i = 0, or G_i first treated after max(g, t) (1.81), or G_i untreated in both periods of the cell (2.0.1).
- **Pair balancing:** of these, only units observed in both of the cell's periods.
- **Skipped cells:** a cell without both groups is skipped. Under 2.0.1's rules, so is a propensity-score cell whose unweighted logit gives some unit a score ≥ 0.999 (R `did`'s overlap check).

**3. ATT(g,t)** from ΔY in the cell.
- **Weights** are normalized to mean 1 in the cell.
- **Covariates:** fixed ones are used as they are. Covariates that vary over time take their values in the cell's **earlier period**, as `csdid` 1.81 and `drdid` do. A unit-period with a missing covariate counts as not observed.
- **Propensity score:** a weighted logit of the cohort indicator on X with a constant. Comparison units scoring above `trim` get weight 0 in the estimator. The score is capped at 1 − 1e-6.

| Method | Estimator | csdid 2.0.1 name |
|---|---|---|
| `dripw` | doubly robust IPW (Sant'Anna and Zhao 2020) | `dr` |
| `reg` | outcome regression | `reg` |
| `stdipw` | normalized (Hájek) IPW | `ipw` |
| `ipw` | Abadie (2005) IPW, weights not normalized: csdid 1.81's `ipw` | none |

Each method gives each unit's influence function in the cell. The formulas, including the estimation effect of the regression and the propensity score, are in `reference.py::cell_rif`; they follow `drdid` 1.91's Mata.

**4. Full sample.**
- A cell's influence function is scaled by N / n_cell; units outside the cell contribute 0.
- Each cell carries a cohort-weight column: w_i for cohort members observed in the cell (1.81), or for all cohort members (2.0.1). The two differ only on unbalanced panels.

**5. Aggregates,** each a weighted average by `aggte`, which includes the estimation of the weights:

| Output | Cells | Weights |
|---|---|---|
| simple ATT | t ≥ g | cohort weights |
| ATT by cohort, g | t ≥ g | cohort weights |
| average over cohorts | the cohort ATTs | each cohort's mean weight, fixed (1.81), or its estimated share (2.0.1) |
| ATT by calendar period, t | g ≤ t | cohort weights |
| average over periods | the period ATTs | equal |
| event study, e = t − g | −5 ≤ e ≤ 5 | cohort weights |
| pre- and post-treatment averages | the event times in the window | equal |
| event-time subsets (`event_sets`) | listed event times | equal |

**6. Standard errors.**
- **Analytic:** sqrt(Σ_i (ψ_i − ψ̄)²) / N for every estimate. The event-study covariance is returned too.
- **Multiplier bootstrap,** with `boot_weights`, a +1/−1 matrix [N, B] supplied as an input:
  - **draws:** b_rk = mean_i (ψ_ik − ψ̄_k) v_ir;
  - **bootstrap SE:** the interquartile range of the draws / 1.349;
  - **simultaneous critical value:** the (1 − α) quantile, across draws, of the table's largest |b_rk / SE_k|;
  - **quantile index:** order statistic ⌈pB⌉ (2.0.1) or ⌈p(B + 1)⌉ (1.81);
  - **floor:** 2.0.1 floors the aggregate tables' critical value at the pointwise one; 1.81 doesn't.

## Inputs

| Name | Shape | Dtype | Notes |
|---|---|---|---|
| y | [n_units, n_periods] | float64 | NaN = not observed |
| X | [n_units, n_covariates] or [n_units, n_periods, n_covariates] | float64 | fixed or varying over time; may have 0 columns |
| gvar | [n_units] | int64 | 0 = never treated |
| w | [n_units] or None | float64 | |
| control | | str | `never`, `notyet` |
| method | | str | `dripw`, `reg`, `stdipw`, `ipw` (`drimp` also exists, unused) |
| event_window | (lo, hi) | int | (−5, 5) |
| base_period | | str | `varying`, `universal` |
| trim | | float | 1.0 = none |
| boot_weights | [n_units, B] or None | int8 | generated by the loader from a fixed seed (PCG64); not part of the timed call |
| alpha | | float | 0.05 |
| event_sets | list of lists, or None | | |
| rules | | str | `csdid181`, `csdid201` |

## Outputs

| Name | Shape |
|---|---|
| gt; att_gt, se_gt | [n_gt, 2]; [n_gt] |
| att_simple, se_simple | scalar |
| group_ids, att_group, se_group; att_group_avg, se_group_avg | [n_groups]; scalar |
| calendar_periods, att_calendar, se_calendar; att_calendar_avg, se_calendar_avg | [n_periods]; scalar |
| event_times, att_event, se_event, cov_event | [n_event], [n_event, n_event] |
| att_pre_avg, se_pre_avg, att_post_avg, se_post_avg | scalar |
| event_sets, att_event_sets, se_event_sets | [n_sets] |
| with `boot_weights`: se_boot_gt, se_boot_group, se_boot_calendar, se_boot_event, se_boot_simple, se_boot_group_avg, se_boot_calendar_avg, se_boot_pre_avg, se_boot_post_avg | like the estimates |
| with `boot_weights`: crit_gt, crit_group, crit_calendar, crit_event | scalar |

## Interface

```python
from reference import run
out = run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5),
          base_period="varying", trim=1.0, boot_weights=None, alpha=0.05, event_sets=None,
          rules="csdid181")
```

`realdata.load(id, data_root)` returns every argument for a workload, including its multipliers.

## Correctness

| Output | Tolerance |
|---|---|
| estimates and subset averages | \|error\| / max(\|reference\|, its SE) ≤ 1e-6 |
| analytic SEs, bootstrap SEs (given the same multipliers) | relative error ≤ 1e-4 |
| critical values (given the same multipliers) | relative error ≤ 1e-4 |
| event covariance | \|error\| / (SE_i · SE_j) ≤ 1e-4 |
| gt, group_ids, calendar_periods, event_times | exact |

**Why estimates are scaled by their SE:** dividing by the estimate alone holds estimates near zero to an impossible standard.

## Reference implementation

`reference.py` is readable NumPy, one estimation problem per cell.
- **Defaults:** it reproduces the original task-002 reference to 1.5e-13 and the task-002b reference exactly (`selfcheck.py`).
- **Against Stata:** it matches `csdid` 1.81 and 2.0.1 on the development workloads (the package README, section 6).

## Not covered

- Clustered standard errors or bootstrap
- Repeated cross-sections
- Anticipation
- Weights that vary over time
- `bal(full)` / `bal(none)`

## Shared evaluator (contract version 2)

`task.json` identifies the prompt files, generator, validator and tolerances. Inputs are read-only. Outputs must have the documented shapes and numeric types and contain only finite values; the reference's `event_sets` output (the sorted input sets, an echo of the argument) is not compared. Estimates are compared with `abs(actual - reference) / max(abs(reference), its reference SE, 1e-12)`, standard errors (analytic and bootstrap) and critical values relatively (`abs(actual - reference) / max(abs(reference), 1e-12)`), and the event-study covariance with `abs(actual - reference) / (SE_i * SE_j)`, element by element; outputs that the reference does not return for a workload (the bootstrap outputs of analytic workloads) are not expected.

The scored real workloads are 101-109 (`DATA.md`); `workloads.py` keeps the former task's six synthetic workloads (default options: never-treated controls, dripw, balanced panels, analytic standard errors) and generates the evaluator's development fixtures, which cover the options (both comparison groups, every method, weights, unbalanced panels, both base periods, trimming, both rule sets, covariates that vary over time, event-time subsets and the bootstrap). Workload IDs are integers.

Run `python -m harness evaluate --task 002 --solution solutions/002/candidate/solution.json --smoke` from the repository root for the fixtures, then select real inputs explicitly, for example `--suite real-development --ids 101 107`. Every first-call, warmup and measured output must pass. The score uses the ratio of reference and candidate median call times on the same host. Generation, imports, loading (including the bootstrap multipliers), first-call setup, validation and serialization are excluded; input-dependent preprocessing and conversion to host-ready outputs are included. See `harness/README.md` for the full protocol.
