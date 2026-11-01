# Task 001 real workloads

The estimator, reference implementation and synthetic workloads are unchanged. `realdata.py`
loads the package inputs; `real_workloads.json` pins each input file's SHA-256 and split.
Data loading and input generation are outside the timed call. Singleton removal, fixed-effect
absorption, regression and covariance computation stay inside `run`.

| Suite | IDs | Permitted use |
|---|---|---|
| `synthetic` | IDs in `workloads.py` | Development and measurement |
| `real-development` | 101–108 | Development and measurement |
| `real-smoke` | 109 | Compatibility/crash checks only; no tuning or performance score |
| `real-heldout` | 201–206, 301–304 | Frozen final evaluation only |

> **Not evaluated in this submission:** the `real-heldout` workloads are defined for a future frozen
> final evaluation and were not run. Every result reported with this repository uses the development
> and synthetic workloads.

## Development coverage

These are input sizes before singleton removal, not reference results. Start with 101, then
cover the distinct structures and regressor variants below. A staged workspace contains only
the inputs selected in its `stage.json`; this table describes the full development suite.

| ID | Rows | Regressors | Structure and purpose |
|---|---:|---:|---|
| 101 | 376,758 | 20 | Inventor × firm × year, technology areas D–E; initial real-network check |
| 102 | 1,559,804 | 20 | Inventor × firm × year, area A; another network connectivity pattern |
| 103 | 2,129,562 | 20 | Inventor × firm × year, area C; another network connectivity pattern |
| 104 | 6,757,894 | 5 / 20 / 50 | Inventor × firm × year, areas A–F; scale and regressor-width variants |
| 105 | 6,757,894 | 20 | Adds technology subclass × year fixed effects to the 104 structure |
| 106 | 575,304 | 15 | Unbalanced firm × filing-year panel with firm and year fixed effects |
| 107 | 3,176,580 | 10 | Firm dyads; citing firm × year, cited firm × year and pair fixed effects |
| 108 | 7,335,825 | 30 | Inventor-year event study; 10 real event dummies and 20 generated controls |

All eight are development workloads. Workload **109 is outside this table and this suite**:
it is generated SEC-shaped compatibility data, with no input file or file manifest.

Development workloads retain real patent fixed-effect structures and generate outcomes and
controls deterministically with seed `1000 + id`; workload 108 retains real event dummies.
Workload 104 supports 5, 20 (default) or 50 generated regressors. Choose that variant explicitly
with `--ids 104 --n-regressors 50`; it is recorded in the run identity. Other loader overrides
(including changing absorbed effects) are not benchmark variants.

Held-out patent workloads use real outcomes and controls. Workload **206 includes A–F
development-area records**: report it as a frozen full-sample evaluation, not wholly unseen
data. The patent-level split for 201/202 does not guarantee disjoint inventor/firm identities;
the panel splits use modal technology areas. SEC workloads provide a different field.

## Running and reporting

Run from the repository or staged-workspace root after installing the root requirements.
Create the candidate once, then run the checks below. See the [shared protocol](../harness/README.md)
for baseline comparisons, CPU budgets and final evaluation.

```bash
python -m harness init-solution --task 001 --out solutions/001/first
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --smoke
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --suite real-development --ids 101
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --suite real-smoke
```

The default development data root is `datasets/001/data`; override it with `--data-root`.
Final evaluation requires an external root, normally `--data-root ~/SocialSciBench-private/001/data`.
Supplied reference answers and validation logs also stay outside the repository.
Real measurements default to five repetitions and one thread. The 300-second timeout applies
to the entire worker, including all repetitions; select a larger explicit budget for large
workloads. A timeout or a passed subset is not task-wide success.

Never use held-out data/results to tune, debug or choose settings. Use `harness freeze` and
`harness final` only when development is finished; report every round, including failures.
The package's saved reference answers are not candidate inputs and are never used by the
shared evaluator, which recomputes the reference in a separate process. See
[the protocol](../harness/README.md) for staging, final evaluation and replay.
