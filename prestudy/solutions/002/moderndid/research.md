# Competitor: moderndid

## Purpose

An existing implementation wrapped as a task 002 candidate so it can be measured in the same
session, under the same evaluator protocol, as `shared-nuisance`. The adapter follows the
14 October 2026 study's (`reports/002_csdid.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to take the task's wide inputs,
avoid importing the benchmark reference at runtime and satisfy the evaluator's interface.
Package defaults for the task's specification are used; a failure is reported as a failure.

ModernDiD 0.2.0 `att_gt` (`est_method="dr"`, never-treated, varying base period, analytic
SEs, `n_jobs=1`) and `aggte`: the study's fastest native Python package.

## Evaluator runs

- `runs/20261018T194732.166248Z-17651f3e`: `--smoke` **failed** on `dev-propensity-trimming`
  (the other three fixtures passed). ModernDiD warns "Overlap condition violated for 4.0 in
  time period t" and skips every cell, returning empty ATT(g,t); its dynamic aggregation then
  fails ("need at least one array to concatenate"). The task requires trimming comparison
  units with propensity above 0.995 and estimating the cell. Because a development fixture
  failed, the evaluator scores no workloads, so this candidate has no timings in the
  same-session comparison. The 14 October study timed it on W1-W4, which have no trimmed units.
