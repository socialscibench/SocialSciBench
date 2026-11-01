# Competitor: numba-comparator

## Purpose

An existing implementation wrapped as a task 008 candidate so it can be measured in the same
session, under the same evaluator protocol, as `bigram-table-scan`. The 14 October 2026 study's
"Python Numba comparator + original pipeline" (`record_numba_full` in
`repro-socialscibench/research/2026-10-14/tasks007008/implementations.py`): the reference
pipeline (pandas joins, sparse bigram matrices, float32 rounding, ties) with only `bigram1`
replaced by a compiled per-pair comparator. The study measured 4.84 s on W3 at one thread in its
native-extension round (6.83 s in its first round).

The study patched the imported reference module at runtime, which candidates may not do; this
file carries a copy of the reference pipeline (as the evaluator's starter does) with the study's
`full_bigram` kernel and `compiled_full_bigram1` wrapper, otherwise unchanged.

## Evaluator runs

- `runs/20261019T024154.604339Z-872b4d40`: `--smoke`, all 4 development fixtures passed.
- `runs/20261019T033818.357802Z-ab03784d`: same-session comparison run, workloads 1-5, default protocol (one
  warmup, three measured calls, one thread), `--timeout 10800`. All 4 development fixtures and all
  5 workloads passed; pairs and perfect counts exact, scores identical to the reference (maximum
  relative error 0).

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.3547 | 0.1346 | 0.1304-0.1374 | 2.64 | 0.977 | 0.30 |
  | W2 | 1.890 | 0.6665 | 0.6561-0.6731 | 2.84 | 1.6 | 0.73 |
  | W3 | 11.38 | 4.595 | 4.563-4.837 | 2.48 | 5.87 | 1.90 |
  | W4 | 62.18 | 27.55 | 26.77-28.09 | 2.26 | 28.8 | 5.64 |
  | W5 | 138.2 | 74.12 | 73.19-74.88 | 1.86 | 75.5 | 10.20 |

  The first call includes compiling the comparator (about 1 s). The study measured 4.84 s on W3 at
  one thread; W4 and W5 were not measured there. Peak memory follows the reference pipeline
  (10.2 GiB on W5).
