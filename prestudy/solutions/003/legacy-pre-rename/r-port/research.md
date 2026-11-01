# Competitor: r-port

## Purpose

An existing implementation wrapped as a task 008 candidate so it can be measured in the same
session, under the same evaluator protocol, as `bigram-table-scan`. The repository's R port of
reclink2's rule (`baselines/008_recordlinkage/bench_r.R`, the "R port" column of
`reports/008_recordlinkage.md`): data.table joins for the perfect matches and the or-blocking,
`stringi` for the bigram scores (scored once per distinct pair of values), float32 rounding via
`writeBin`. There is no CRAN package with reclink's scoring rule.

The R code is the baseline's `f32`, `bigram1` and timed body, unchanged, wrapped in a function
that returns the pairs (zero-based) and the perfect count, with `setDTthreads(1)` (the baseline
used up to 12 threads). The baseline read CSV files outside its timer; here each column is copied
into R inside the timed call, as one string joined with a separator (`\x1f`) and split in R
(1.4 s per 2-million-value column, against 5-6 s for rpy2's per-string conversion). The study
measured the original R port at 62.9 s on W3 at one thread (a single run; its warm-up exceeded
60 s).

## Evaluator runs

- `runs/20261019T024213.713476Z-8598cc24`: `--smoke`, all 4 development fixtures passed.
- `runs/20261019T050131.508302Z-86e2d914`: same-session comparison run, workloads 1-5, default protocol (one
  warmup, three measured calls, one thread), `--timeout 10800`. All 4 development fixtures and all
  5 workloads passed; pairs and perfect counts exact, scores identical to the reference (maximum
  relative error 0).

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.3775 | 1.562 | 1.550-1.600 | 0.24 | 1.73 | 0.46 |
  | W2 | 1.670 | 7.725 | 7.512-7.770 | 0.22 | 7.91 | 1.24 |
  | W3 | 11.63 | 53.75 | 53.41-55.32 | 0.22 | 56.4 | 2.93 |
  | W4 | 61.77 | 280.1 | 279.7-281.7 | 0.22 | 286 | 7.55 |
  | W5 | 138.1 | 636.3 | 633.5-640.5 | 0.22 | 652 | 9.43 |

  Timings include copying the strings into R (about 1.4 s per 2-million-value column on W5, so
  about 7 s of the 636 s). The study measured 62.9 s on W3 at one thread (one run).
