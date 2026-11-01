# Competitor: julia-fem

## Purpose

An existing implementation wrapped as a task 001 candidate so it can be measured in the same
session, under the same evaluator protocol, as `eliminate-largest-fe`. The adapter follows the
14 October 2026 study's (`reports/001_reghdfe.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to avoid importing the benchmark
reference at runtime and to satisfy the evaluator's interface. Defaults are used; a failing
default is reported as a failure, and any tightened setting is a separate, labeled candidate.

Julia 1.13.1 FixedEffectModels 2.1.0 reg (default tolerance) via juliacall in-process, with the study's Julia small-sample correction, as in the 14 Oct 2026 study's Round 2.

## Evaluator runs

- `runs/20261018T023207.951812Z-6bbe0a68`: `--smoke`, all 5 development fixtures passed (2026-10-18).

### Same-session comparison (2026-10-18)

Five candidates (`eliminate-largest-fe`, `xhdfe`, `within`, `julia-fem`, `fixest`) ran one after
another from `runs/compare-001.sh` in the shared environment `solutions/001/.venv`, with nothing
else benchmarking on the host. Each run measures its own reference.

- `runs/20261018T042523.272784Z-464d5031`: workloads 1, 3, 4, 6, 7, 8, 9, 10 and 12, default
  protocol (one thread, one warmup, three measured calls; `--timeout 3600`).
- `runs/20261018T121935.984062Z-6d4c0e0a`: workload 11 with `--repeats 1` (`--timeout 21600`).

| Workload | Status | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Max beta error | Max SE error | Peak RSS (GB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | passed | 0.2210 | 0.0719 | 0.0610-0.1324 | 3.1x | 7.34 | 1.0e-14 | 1.4e-15 | 0.90 |
| 3 | passed | 22.097 | 7.078 | 6.896-7.303 | 3.1x | 14.80 | 2.0e-14 | 8.7e-15 | 5.70 |
| 4 | passed | 17.603 | 7.207 | 7.204-7.226 | 2.4x | 15.20 | 4.1e-09 | 2.4e-06 | 2.30 |
| 6 | passed | 1.899 | 0.5084 | 0.5023-0.5193 | 3.7x | 7.95 | 9.5e-13 | 4.2e-09 | 1.27 |
| 7 | passed | 23.895 | 7.253 | 7.170-7.437 | 3.3x | 15.39 | 1.1e-11 | 7.4e-09 | 2.94 |
| 8 | passed | 47.803 | 17.087 | 16.973-17.486 | 2.8x | 24.70 | 3.0e-07 | 5.7e-06 | 2.31 |
| 9 | **failed beta** | - | - | - | - | 31.42 | 2.1e-05 | 8.7e-06 | 2.29 |
| 10 | passed | 7.022 | 2.985 | 2.763-3.075 | 2.4x | 12.35 | 1.1e-14 | 8.6e-15 | 4.08 |
| 11 | **failed beta** | - | - | - | - | 311.36 | 1.1e-06 | 4.8e-06 | 9.88 |
| 12 | passed | 66.960 | 23.912 | 23.857-24.202 | 2.8x | 34.61 | 3.2e-14 | 1.8e-14 | 17.16 |

Medians are per `run` call; the first call includes compilation or package warm-up and is not
scored. Peak RSS is the worker's lifetime high-water mark, including input generation.

Default tolerance (`1e-6`) fails the coefficient check on workloads 9 (2.1e-5) and 11 (1.1e-6);
standard errors pass. The evaluator records no timing for failed cases. This matches the
14 October study's W9 result. A tightened setting (`TIGHT=True` in `solution.py`: tol 1e-10)
would be a separately named candidate and has not been run.
