# Competitor: csdid-py

## Purpose

An existing implementation wrapped as a task 002 candidate so it can be measured in the same
session, under the same evaluator protocol, as `shared-nuisance`. The adapter follows the
14 October 2026 study's (`reports/002_csdid.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to take the task's wide inputs,
avoid importing the benchmark reference at runtime and satisfy the evaluator's interface.
Package defaults for the task's specification are used; a failure is reported as a failure.

Python csdid 0.4.2 `ATTgt` (`faster_mode=True`, `est_method="dr"`, never-treated, varying base
period, analytic SEs, one core) and `aggte`: the study's `csdid_fast`.

## Evaluator runs

- `runs/20261018T194906.396206Z-fd98c158`: `--smoke` **failed** on `dev-propensity-trimming`
  (the other three fixtures passed): "Missing values at att_gt found", raised by `aggte` after
  the overlap check left every cell missing. The task requires trimming comparison units with
  propensity above 0.995 and estimating the cell. Because a development fixture failed, the
  evaluator scores no workloads, so this candidate has no timings in the same-session
  comparison. The 14 October study timed it on W1-W4, which have no trimmed units.
