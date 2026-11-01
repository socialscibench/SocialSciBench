# Competitor: cpp-pipeline

## Purpose

An existing implementation wrapped as a task 008 candidate so it can be measured in the same
session, under the same evaluator protocol, as `bigram-table-scan`. The 14 October 2026 study's
complete C++17 pipeline (`reports/008_recordlinkage.md`, native-extension round; companion
repository `repro-socialscibench/research/2026-10-14/native-extension/task008/`): string
encoding, perfect-match removal, hash-map blocking, per-pair sorted-bigram merges, float32
rounding, ties and output, specialized to the task's five fields, weights and required(state).
The study measured it at 4.39 s on W3 at one thread (1.66 s at six), its fastest one-thread
implementation; W4 and W5 were not measured.

`linkage.cpp` is the study's file, unchanged (SHA-256 537a9294...). The bridge follows the study's
`native.py` (input packing with `astype("S")`, copied outputs, `threads=1`), changed only to
accept the evaluator's keyword arguments (it rejects other fields, weights or required() fields)
and to compile the library at import from the snapshot's source (see `build.md`).

## Evaluator runs

- `runs/20261019T024143.461496Z-c75c9573`: `--smoke`, all 4 development fixtures passed.
- `runs/20261019T031114.806653Z-b41b2449`: same-session comparison run, workloads 1-5, default protocol (one
  warmup, three measured calls, one thread), `--timeout 10800`. All 4 development fixtures and all
  5 workloads passed; pairs and perfect counts exact, scores identical to the reference (maximum
  relative error 0).

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.4140 | 0.08890 | 0.08706-0.08983 | 4.66 | 0.0916 | 0.04 |
  | W2 | 1.664 | 0.5240 | 0.5166-0.5322 | 3.17 | 0.543 | 0.09 |
  | W3 | 11.54 | 4.044 | 4.025-4.062 | 2.85 | 4.21 | 0.33 |
  | W4 | 62.93 | 23.21 | 22.93-23.76 | 2.71 | 23.1 | 1.36 |
  | W5 | 137.6 | 57.38 | 57.01-58.64 | 2.40 | 57.5 | 3.74 |

  The library is compiled at import from the snapshot's `linkage.cpp` (cached by source hash; the
  cache was already populated by the smoke run), so the first call equals the steady state. The
  study measured 4.39 s on W3 at one thread.
