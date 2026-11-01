# Competitor: r-did

## Purpose

An existing implementation wrapped as a task 002 candidate so it can be measured in the same
session, under the same evaluator protocol, as `shared-nuisance`. The adapter follows the
14 October 2026 study's (`reports/002_csdid.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to take the task's wide inputs,
avoid importing the benchmark reference at runtime and satisfy the evaluator's interface.
Package defaults for the task's specification are used; a failure is reported as a failure.

R did 2.5.1 `att_gt` (`faster_mode = TRUE`, `est_method = "dr"`, never-treated, varying base
period, analytic SEs, one core) and `aggte`, in-process through rpy2 3.5.17: the study's
`R_did_fast`.

## Evaluator runs

- `runs/20261018T194719.568252Z-6e9a9f63`: `--smoke`, adapter error (rpy2 `OrdDict`); fixed.
- `runs/20261018T194834.590901Z-a60386a3`: `--smoke` **failed** on `dev-propensity-trimming`
  (the other three fixtures passed). did reports "overlap condition violated" and returns
  missing ATT(g,t) for every cell, and `aggte` stops: "Missing values found in att_gt()
  estimates". The task requires trimming comparison units with propensity above 0.995 and
  estimating the cell. Because a development fixture failed, the evaluator scores no
  workloads, so this candidate has no timings in the same-session comparison. The 14 October
  study timed it on W1-W4, which have no trimmed units.
