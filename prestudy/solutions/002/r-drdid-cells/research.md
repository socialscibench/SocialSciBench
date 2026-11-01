# Competitor: r-drdid-cells

## Purpose

An existing implementation wrapped as a task 002 candidate so it can be measured in the same
session, under the same evaluator protocol, as `shared-nuisance`. The adapter follows the
14 October 2026 study's (`reports/002_csdid.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to take the task's wide inputs,
avoid importing the benchmark reference at runtime and satisfy the evaluator's interface.
Package defaults for the task's specification are used; a failure is reported as a failure.

R DRDID 1.3.0 `drdid_panel` (traditional DR, `trim.level = .995`) for every (cohort, period)
cell, with the task's rescaling and aggregation in R, called in-process through rpy2 3.5.17:
the study's `R_DRDID_adapter`, its fastest engine on W3 and W4.

## Evaluator runs

- `runs/20261018T194709.859356Z-97ec5725`: `--smoke`, every fixture failed with an adapter
  error (rpy2 returns the R list as an `OrdDict`); fixed in the adapter, not in R.
- `runs/20261018T194824.674460Z-163e29dd`: `--smoke`, all 4 development fixtures passed,
  including the propensity-trimming fixture.
- `runs/20261018T201053.813449Z-760179da`: same-session comparison run, workloads 1-6, default
  protocol (one warmup, three measured calls, one thread), `--timeout 3600`. All 4 development
  fixtures and all 6 workloads passed.

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.0281 | 0.0469 | 0.0447-0.0469 | 0.60 | 0.85 | 0.30 |
  | W2 | 0.396 | 0.453 | 0.424-0.546 | 0.87 | 1.21 | 0.43 |
  | W3 | 3.186 | 2.985 | 2.975-3.016 | 1.07 | 3.92 | 0.82 |
  | W4 | 6.228 | 7.088 | 6.990-7.920 | 0.88 | 8.27 | 2.10 |
  | W5 | 96.74 | 137.3 | 122.3-141.5 | 0.70 | 121.0 | 13.74 |
  | W6 | 115.6 | 142.6 | 141.5-143.1 | 0.81 | 143.7 | 7.64 |

  Maximum relative errors: ATT 3.3e-7 (W3), SE 1.3e-9 (W3). The 14 October study measured
  this adapter at 3.90 s (W3) and 9.10 s (W4) against 6.94 s and 12.78 s for the reference, a
  1.4-1.8x win; here it is within 0.6-1.1x of the reference on W1-W4, mainly because the
  reference measured faster (3.09 s and 6.94 s in the `shared-nuisance` run). The cause of
  that difference was not investigated.
