# Competitor: r-rcpp

## Purpose

An existing implementation wrapped as a task 008 candidate so it can be measured in the same
session, under the same evaluator protocol, as `bigram-table-scan`. The 14 October 2026 study's
"R Rcpp comparator + original pipeline" (`repro-socialscibench/research/2026-10-14/tasks007008/
bench_r.R`, mode `rcpp`): the repository's R port (see `../r-port/research.md`) with `bigram1`
replaced by the study's compiled Rcpp comparator, built with `Rcpp::cppFunction` at import. The
study measured 25.8 s on W3 at one thread, its fastest R variant (its reclin2-blocking variant was
slower, 28.6 s, and is not included here).

The R code is identical to `r-port` except the comparator switch; the same column copies are inside
the timed call, and data.table uses one thread.

## Evaluator runs

- `runs/20261019T024225.187327Z-f15d2dfc`: `--smoke`, all 4 development fixtures passed.
- `runs/20261019T040719.483227Z-8aecb21f`: same-session comparison run, workloads 1-5, default protocol (one
  warmup, three measured calls, one thread), `--timeout 10800`. All 4 development fixtures and all
  5 workloads passed; pairs and perfect counts exact, scores identical to the reference (maximum
  relative error 0).

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.5119 | 0.6530 | 0.6281-0.6692 | 0.78 | 0.797 | 0.42 |
  | W2 | 1.703 | 2.993 | 2.890-3.044 | 0.57 | 3.27 | 1.09 |
  | W3 | 11.46 | 21.99 | 21.90-22.00 | 0.52 | 23.8 | 1.94 |
  | W4 | 61.95 | 108.9 | 108.4-111.3 | 0.57 | 116 | 5.95 |
  | W5 | 137.4 | 266.3 | 263.7-270.8 | 0.52 | 282 | 8.66 |

  Import (excluded) includes compiling the Rcpp comparator, 3.1-3.7 s. Timings include copying the
  strings into R (about 1.4 s per 2-million-value column on W5). The study measured 25.8 s on W3 at
  one thread.
