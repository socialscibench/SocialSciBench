# Competitor: fixest

## Purpose

An existing implementation wrapped as a task 001 candidate so it can be measured in the same
session, under the same evaluator protocol, as `eliminate-largest-fe`. The adapter follows the
14 October 2026 study's (`reports/001_reghdfe.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to avoid importing the benchmark
reference at runtime and to satisfy the evaluator's interface. Defaults are used; a failing
default is reported as a failure, and any tightened setting is a separate, labeled candidate.

R fixest 0.14.2 feols (default tolerances, fixef.rm='singleton') via rpy2 3.5.17 in-process, with the task's exact small-sample correction, as in the 14 Oct 2026 study's Round 1.

## Evaluator runs

- `runs/20261018T022430.376213Z-521df303`: `--smoke`, all 5 development fixtures passed (2026-10-18).

### Same-session comparison (2026-10-18)

Five candidates (`eliminate-largest-fe`, `xhdfe`, `within`, `julia-fem`, `fixest`) ran one after
another from `runs/compare-001.sh` in the shared environment `solutions/001/.venv`, with nothing
else benchmarking on the host. Each run measures its own reference.

- `runs/20261018T050445.888229Z-ba8122f0`: workloads 1, 3, 4, 6, 7, 8, 9, 10 and 12, default
  protocol (one thread, one warmup, three measured calls; `--timeout 3600`).
- `runs/20261018T140933.944721Z-d6f2feb9`: workload 11 with `--repeats 1` (`--timeout 21600`).

| Workload | Status | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Max beta error | Max SE error | Peak RSS (GB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | passed | 0.2225 | 0.1329 | 0.0672-0.1352 | 1.7x | 0.13 | 1.5e-14 | 1.4e-14 | 0.44 |
| 3 | passed | 22.323 | 12.838 | 12.606-12.941 | 1.7x | 13.37 | 2.6e-13 | 1.2e-13 | 10.56 |
| 4 | passed | 17.394 | 11.115 | 11.013-11.307 | 1.6x | 11.58 | 1.3e-12 | 3.0e-08 | 2.68 |
| 6 | passed | 1.716 | 0.3044 | 0.2883-0.3482 | 5.6x | 0.63 | 5.0e-14 | 1.1e-13 | 0.78 |
| 7 | passed | 23.843 | 8.178 | 8.153-8.488 | 2.9x | 8.79 | 1.5e-13 | 1.4e-09 | 5.27 |
| 8 | passed | 47.919 | 55.942 | 55.800-56.670 | 0.9x | 57.46 | 6.5e-11 | 1.4e-07 | 2.77 |
| 9 | **failed beta** | - | - | - | - | 78.36 | 1.1e-05 | 5.2e-06 | 2.78 |
| 10 | passed | 6.875 | 8.384 | 8.340-8.570 | 0.8x | 9.46 | 1.1e-13 | 8.2e-14 | 5.95 |
| 11 | **failed beta** | - | - | - | - | 1092.77 | 3.4e-06 | 3.5e-06 | 17.46 |
| 12 | passed | 66.478 | 96.465 | 95.812-97.064 | 0.7x | 92.39 | 1.9e-13 | 1.5e-13 | 24.96 |

Medians are per `run` call; the first call includes compilation or package warm-up and is not
scored. Peak RSS is the worker's lifetime high-water mark, including input generation.

Default tolerances fail the coefficient check on workloads 9 (1.1e-5) and 11 (3.4e-6); standard
errors pass. The evaluator records no timing for failed cases. This matches the 14 October
study's W9 result. A tightened setting would be a separately named candidate and has not been
run.
