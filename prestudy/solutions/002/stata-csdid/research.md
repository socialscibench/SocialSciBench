# Competitor: stata-csdid

## Purpose

An existing implementation wrapped as a task 002 candidate so it can be measured in the same
session, under the same evaluator protocol, as `shared-nuisance`. The adapter follows the
14 October 2026 study's (`reports/002_csdid.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to take the task's wide inputs,
avoid importing the benchmark reference at runtime and satisfy the evaluator's interface.
Package defaults for the task's specification are used; a failure is reported as a failure.

Stata csdid 2.0.0 (Mata rewrite, pinned official release commit 31a5773), `method(dr)
nevertreated base_period(varying) analytical pointwise`, then `estat simple` and
`estat event, post`, in-process through pystata (Stata 18 MP, one processor): the study's Stata
extension block. The pre-period average and SE come from the native event covariance.

## Evaluator runs

- `runs/20261018T195027.807806Z-16881323`: `--smoke` **failed** on `dev-propensity-trimming`
  (the other three fixtures passed). csdid warns "overlap condition violated for group 4 in
  time period t", reports that every ATT(g,t) cell failed to estimate, and `estat simple`
  stops with r(498). The task requires trimming comparison units with propensity above 0.995
  and estimating the cell. Because a development fixture failed, the evaluator scores no
  workloads, so this candidate has no timings in the same-session comparison. The 14 October
  study timed it on W1-W4, which have no trimmed units.
