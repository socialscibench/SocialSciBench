# Competitor: xhdfe

## Purpose

An existing implementation wrapped as a task 001 candidate so it can be measured in the same
session, under the same evaluator protocol, as `eliminate-largest-fe`. The adapter follows the
14 October 2026 study's (`reports/001_reghdfe.md`; companion repository
`repro-socialscibench/research/2026-10-14/`), changed only to avoid importing the benchmark
reference at runtime and to satisfy the evaluator's interface. Defaults are used; a failing
default is reported as a failure, and any tightened setting is a separate, labeled candidate.

xhdfe 2.28.1.20260929 (C++) HdfeRegressor with reghdfe-comparable defaults; labels re-encoded; native outputs with df_a converted to the task convention, as in the 14 Oct 2026 study.

## Evaluator runs

- `runs/20261018T022202.786875Z-8cb8d7df`: `--smoke`, all 5 development fixtures passed (2026-10-18).

### Same-session comparison (2026-10-18)

Five candidates (`eliminate-largest-fe`, `xhdfe`, `within`, `julia-fem`, `fixest`) ran one after
another from `runs/compare-001.sh` in the shared environment `solutions/001/.venv`, with nothing
else benchmarking on the host. Each run measures its own reference.

- `runs/20261018T031236.191169Z-deeb54e3`: workloads 1, 3, 4, 6, 7, 8, 9, 10 and 12, default
  protocol (one thread, one warmup, three measured calls; `--timeout 3600`).
- `runs/20261018T073339.204026Z-d42c9e78`: workload 11 with `--repeats 1` (`--timeout 21600`).

| Workload | Status | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Max beta error | Max SE error | Peak RSS (GB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | passed | 0.2724 | 0.0342 | 0.0306-0.0415 | 8.0x | 0.05 | 1.0e-14 | 2.2e-15 | 0.15 |
| 3 | passed | 22.280 | 9.968 | 9.831-10.052 | 2.2x | 10.36 | 2.1e-14 | 2.4e-14 | 6.35 |
| 4 | passed | 17.867 | 8.435 | 8.272-8.541 | 2.1x | 8.51 | 1.3e-12 | 3.0e-08 | 1.83 |
| 6 | passed | 1.725 | 0.2240 | 0.2161-0.2275 | 7.7x | 0.26 | 1.1e-14 | 4.9e-15 | 0.38 |
| 7 | passed | 23.779 | 5.644 | 5.601-5.680 | 4.2x | 6.50 | 2.1e-14 | 1.4e-09 | 3.45 |
| 8 | passed | 48.882 | 7.758 | 7.584-7.862 | 6.3x | 7.52 | 6.6e-11 | 1.6e-07 | 1.75 |
| 9 | passed | 115.5 | 6.130 | 5.881-6.391 | 18.8x | 6.07 | 5.5e-08 | 2.9e-08 | 1.76 |
| 10 | passed | 6.860 | 3.789 | 3.701-4.038 | 1.8x | 4.55 | 1.2e-14 | 1.2e-14 | 3.74 |
| 11 | passed | 1870.9 | 1819.4 | 1 call | 1.0x | 1828.82 | 4.1e-07 | 2.1e-07 | 18.60 |
| 12 | passed | 68.001 | 49.358 | 34.150-51.754 | 1.4x | 62.37 | 3.0e-14 | 3.4e-14 | 24.02 |

Medians are per `run` call; the first call includes compilation or package warm-up and is not
scored. Peak RSS is the worker's lifetime high-water mark, including input generation.

Workload 11 took 1,819 s per call (both calls; peak RSS 18.6 GB of 32 GiB), about the
reference's time. The cause was not investigated; the study's multithreaded supplement did not
cover workload 11.
