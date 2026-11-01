# Competitor: within

## Purpose

An existing implementation wrapped as a task 001 candidate so it can be measured in the same
session, under the same evaluator protocol, as `eliminate-largest-fe`. The adapter follows the
14 October 2026 study's (`reports/001_reghdfe.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to avoid importing the benchmark
reference at runtime and to satisfy the evaluator's interface. Defaults are used; a failing
default is reported as a failure, and any tightened setting is a separate, labeled candidate.

within-py 0.4.0 (Rust) batched demeaning with default LSMR options, plus label encoding, singleton pruning, OLS, clustered covariance and the task's df rule, as in the 14 Oct 2026 study.

## Evaluator runs

- `runs/20261018T022209.062810Z-75e814e6`: `--smoke`, all 5 development fixtures passed (2026-10-18).

### Same-session comparison (2026-10-18)

Five candidates (`eliminate-largest-fe`, `xhdfe`, `within`, `julia-fem`, `fixest`) ran one after
another from `runs/compare-001.sh` in the shared environment `solutions/001/.venv`, with nothing
else benchmarking on the host. Each run measures its own reference.

- `runs/20261018T034924.951843Z-48246f2c`: workloads 1, 3, 4, 6, 7, 8, 9, 10 and 12, default
  protocol (one thread, one warmup, three measured calls; `--timeout 3600`).
- `runs/20261018T103900.111502Z-1d86db38`: workload 11 with `--repeats 1` (`--timeout 21600`).

| Workload | Status | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Max beta error | Max SE error | Peak RSS (GB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | passed | 0.2613 | 0.0373 | 0.0368-0.0415 | 7.0x | 0.04 | 1.2e-14 | 1.3e-16 | 0.11 |
| 3 | passed | 22.414 | 8.533 | 8.397-8.547 | 2.6x | 8.57 | 1.9e-14 | 1.8e-15 | 4.99 |
| 4 | passed | 17.740 | 10.144 | 10.068-10.440 | 1.7x | 10.29 | 1.3e-12 | 3.0e-08 | 1.42 |
| 6 | passed | 1.736 | 0.4100 | 0.3741-0.4232 | 4.2x | 0.37 | 9.3e-15 | 4.2e-16 | 0.29 |
| 7 | passed | 24.578 | 8.262 | 8.239-8.380 | 3.0x | 9.12 | 1.1e-14 | 1.0e-15 | 2.27 |
| 8 | passed | 49.620 | 10.002 | 9.910-10.032 | 5.0x | 10.01 | 6.6e-11 | 1.6e-07 | 1.32 |
| 9 | passed | 113.2 | 9.780 | 9.521-10.588 | 11.6x | 9.79 | 5.5e-08 | 2.9e-08 | 1.37 |
| 10 | passed | 7.036 | 3.916 | 3.656-4.126 | 1.8x | 4.09 | 1.1e-14 | 2.3e-15 | 2.88 |
| 11 | passed | 1853.0 | 141.0 | 1 call | 13.1x | 143.88 | 4.1e-07 | 2.1e-07 | 10.49 |
| 12 | passed | 67.839 | 33.399 | 33.084-34.398 | 2.0x | 34.56 | 3.5e-14 | 2.8e-15 | 18.44 |

Medians are per `run` call; the first call includes compilation or package warm-up and is not
scored. Peak RSS is the worker's lifetime high-water mark, including input generation.
