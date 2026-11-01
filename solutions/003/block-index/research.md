# Candidate research: block-index

## Objective and constraints

- Task 003, contract version 2: Stata `reclink2` record linkage on name, address,
  city, ZIP and state (`wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)`, minscore and
  minbigram .6): perfect matches (all duplicate combinations), required + or-blocking with empty
  master values skipped, reclink's `bigram1` comparator with the prefix bonus, match/non-match
  weights with float32 rounding where the reference rounds, and every tied best candidate. Must
  match `reference.run`: the pair set and n_perfect exactly, scores to 1e-6 relative error.
- Target hardware: CPU; configurations `--threads 1` and `--threads 6` (physical cores of the
  Intel Core i7-8700B host, 32 GiB, macOS 15.7.3). Environment `solutions/.venv` (`build.md`).
- Research setting: the same agent built an earlier task 003 candidate (`bigram-table-scan`, now in
  `prestudy/solutions/003/legacy-pre-rename/`) earlier in this conversation. Its files were not reopened; its design (master
  dictionaries, city blocks, inverted bigram lists, exact float32 chain) was known from that work
  and is disclosed here. This candidate was written from scratch for the real-data structure
  (duplicate registers, long fixed-width strings, dense city blocks, an 8.6-million-row national
  register). The repository's reports, baselines, published records and runs were not opened.

## Method

1. **Dictionaries of the master values only.** Each distinct master value of each variable gets an
   id; every using value is hashed (eight characters per 64-bit step) and looked up, giving the id
   of the equal master value or -1. For variables wider than eight characters a one-bit filter on
   a hash of the first eight characters rejects most using values without reading the rest (in the
   national register most names and addresses are not master values).
2. **Perfect matches** compare tuples of ids (all duplicate combinations); both sides are removed.
3. **Blocking keys.** For every non-required variable, the (required values, value) pairs of the
   remaining master records index the using records with the same pair. The variable with the most
   candidate pairs (the city on every workload here) is the block variable: the using records
   sharing a master's state and city form its block; matches through name, address or ZIP outside
   the block are "extras" (de-duplicated).
4. **Blocks with several master records** (dense blocks, e.g. 104: one Chicago block of 145,108
   records for all 500 masters) are encoded once: lengths, packed first four characters, value
   ids, and an inverted list of each distinct bigram code over the block's records; a master
   value's bigram counts for all block records are sums over its bigrams' lists. **Blocks with one
   or two master records** (most of 108: 1,969 blocks for 3,265 masters) are counted directly
   against the master value's bigram table while the records are read, without inversion. The
   block's scattered using rows are prefetched (`llvm.prefetch`).
5. **Scores** follow the reference operation by operation (float64 arithmetic, float32 rounding
   where it rounds), over chunks of 128 candidates, branch-free so that LLVM vectorizes them. The
   cube-root bonus is computed with a vectorizable approximation in this pass (a float32 bit
   estimate and three Halley steps); every candidate within 1e-5 of the master's approximate best
   score is then rescored exactly with libm `pow`, and the best and ties come from the exact
   scores. (An approximate bonus moves a float32 score by at most a few units in the last place,
   so the exact best and all exact ties are always within the margin.)
6. Extras are scored exactly one by one.

Threads: Numba parallel loops over using records (lookups), keys, and tasks of master records
ordered by block and balanced by block size; a block split across tasks is encoded by each task.

## Sources (accessed 2026-10-22)

- Wasi, N. and Flaaen, A. (2015), "Record linkage using Stata: Preprocessing, linking, and
  reviewing utilities", Stata Journal 15(3): 672-697; `reclink2.ado` (dm0082).
- Blasnik, M. (2010), "RECLINK: Stata module to probabilistically match records", SSC S456876.
- FreeBSD msun `s_cbrtf.c` (B1 = 709958130, the float32 cube-root bit estimate).
- LLVM `llvm.prefetch` intrinsic, https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic;
  Numba `@intrinsic`, https://numba.readthedocs.io/en/stable/extending/high-level.html.
- MurmurHash3 `fmix64` finalizer (A. Appleby), https://github.com/aappleby/smhasher.

## Measurements and unsuccessful attempts

Single calls after a first call, development scripts (not evaluator runs), one thread; the
reference times are one call each (`solutions/.venv`).

| Version | 101 | 102 | 103 | 104 | 105 | 107 | 108 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference | 35.3 s | 5.1 s | 18.8 s | 109.2 s | 43.7 s | 18.5 s | 352.0 s |
| 1. First correct version | 2.06 s | 0.39 s | 1.42 s | - | 2.46 s | 1.08 s | 12.6 s |
| 2. Output buffer sized per task (the search ran twice) | 1.12 s | 0.18 s | 0.64 s | 3.36 s | 1.33 s | - | - |
| 3. Branch-free scoring chunks | 0.69 s | 0.19 s | 0.61 s | - | 0.85 s | 0.48 s | - |
| 4. 8-character hashing; blocks encoded in the tasks with prefetching | 0.85 s | 0.23 s | 0.68 s | - | 1.00 s | 0.48 s | 8.35 s |
| 5. Direct counting for blocks with < 3 masters | 0.85 s | 0.21 s | 0.73 s | - | 1.00 s | 0.48 s | 7.82 s |
| 6. Approximate bonus in the hot loop, exact rescoring near the best | 0.64 s | 0.16 s | 0.49 s | 1.95 s | 0.76 s | 0.38 s | 6.07 s |

(Rows 4-5 were measured while another single-threaded job ran in the background.)

- A per-task scratch scoring loop that called `_bonus` through a 4096-slot memo of the cube-root
  bonus cost as much as `pow` itself: the branchy lookup kept the loop scalar (ZIP similarities
  usually exceed .6 within a city, so most candidates need the bonus). The approximate bonus with
  exact rescoring of near-best candidates removed about two thirds of the scoring time.
- Numba's parallel transformation of a large `prange` body silently processed only some master
  records; moving the body into a separate jitted function fixed it (verified against the serial
  and pure-Python versions).
- Global block encodings (all blocks before the search) needed several GB for 108 and two passes
  over 4.6 million scattered rows; encoding inside the tasks with prefetching is faster.
- Not attempted: score-bound pruning of candidates before counting; bit-sliced bigram counting.

Correctness during development (validator: exact pairs and n_perfect, scores within 1e-6),
against one saved reference call per case in the same environment: all development fixtures
(seeds 1729, 11, 12, 13), real workloads 101-108 and synthetic 1-5 match (104, 108, s4 and s5
re-checked with the final version on 2026-10-22: 108 in 6.09 s at one thread, 10.4 GB peak
RSS of the checking process including its inputs). `fuzz.py` compares the candidate with the
reference on random small problems (4-6 variables, random required variables and weights,
short and empty strings, duplicated using records, edited master copies, thresholds .5-.7),
requiring identical pair sets, perfect-pair counts and bitwise-equal scores: 600 cases (seeds
1000-1599, 9,170 reference pairs) with no difference.

## Evaluator runs

`python -m harness evaluate` (protocol 3) on the development host, 2026-10-22/23, interpreter
`solutions/.venv`; the candidate's `solution.py` (SHA-256 c82eef69e38c...29b9) is the same in every
run's snapshot. Every invocation first passed the development fixtures at its thread count (and the
smoke check `--smoke` passed: `runs/20261022T185005.839413Z-222efeee`), then recomputed the
reference in its own process with the same thread budget. Workload 108's single reference call
takes more than 150 s, so it ran with `--warmups 0 --repeats 1` (calls "1+0+1": a first call and
one measured call, for the reference and the candidate alike); the others used the evaluator's
defaults (real: first call, one warmup, five measured calls; synthetic: first call, one warmup,
three measured calls). Speedup = reference median / candidate median. First call = the
candidate's first call in its worker, including Numba compilation (not a steady-state sample).
Peak RSS = maximum resident set of each worker process, inputs included. Max. rel. error = the
validator's largest relative score error over all candidate calls (pairs and n_perfect must match
exactly; every call matched the reference bitwise). Every development workload in scope (real
101-108, synthetic 1-5) passed at both thread counts.

### One thread (`--threads 1`)

Runs: `runs/20261022T185556.024324Z-9dbc9db2`, `runs/20261022T193346.854856Z-4d4ad68b`, `runs/20261022T195218.141722Z-2a67bacf`

Real workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 101 | 1+1+5 | 25.56 s | 0.622 s (0.609-0.636) | 41.1x | 74.77 s | 4.58 / 0.72 GiB | 0.0e+00 | PASSED |
| 102 | 1+1+5 | 3.288 s | 0.175 s (0.169-0.189) | 18.8x | 73.83 s | 1.86 / 0.63 GiB | 0.0e+00 | PASSED |
| 103 | 1+1+5 | 13.26 s | 0.522 s (0.514-0.535) | 25.4x | 76.23 s | 3.48 / 0.72 GiB | 0.0e+00 | PASSED |
| 104 | 1+1+5 | 97.65 s | 1.974 s (1.911-2.001) | 49.5x | 75.15 s | 12.73 / 1.15 GiB | 0.0e+00 | PASSED |
| 105 | 1+1+5 | 30.60 s | 0.711 s (0.708-0.741) | 43.1x | 73.43 s | 5.57 / 0.79 GiB | 0.0e+00 | PASSED |
| 106 | 1+1+5 | 10.65 s | 0.079 s (0.078-0.097) | 134.6x | 71.75 s | 1.75 / 0.65 GiB | 0.0e+00 | PASSED |
| 107 | 1+1+5 | 12.42 s | 0.344 s (0.339-0.360) | 36.1x | 70.93 s | 3.16 / 0.79 GiB | 0.0e+00 | PASSED |
| 108 | 1+0+1 | 275.8 s | 6.361 s | 43.4x | 77.61 s | 22.91 / 9.53 GiB | 0.0e+00 | PASSED |

Geometric-mean speedup (real): 41.5x over 8 cases.

Synthetic workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 1+1+3 | 0.346 s | 0.013 s (0.012-0.013) | 27.1x | 70.55 s | 0.37 / 0.57 GiB | 0.0e+00 | PASSED |
| s2 | 1+1+3 | 2.610 s | 0.057 s (0.056-0.073) | 46.0x | 74.98 s | 1.22 / 0.60 GiB | 0.0e+00 | PASSED |
| s3 | 1+1+3 | 11.32 s | 0.357 s (0.342-0.389) | 31.7x | 73.17 s | 2.63 / 0.75 GiB | 0.0e+00 | PASSED |
| s4 | 1+1+3 | 60.67 s | 1.911 s (1.895-1.915) | 31.7x | 71.42 s | 6.35 / 1.40 GiB | 0.0e+00 | PASSED |
| s5 | 1+1+3 | 134.0 s | 4.438 s (4.401-4.594) | 30.2x | 76.48 s | 11.07 / 3.75 GiB | 0.0e+00 | PASSED |

Geometric-mean speedup (synthetic): 32.8x over 5 cases.

### Six threads (`--threads 6`)

Runs: `runs/20261023T021224.552881Z-1a3901b4`, `runs/20261023T025011.664857Z-d2ece329`, `runs/20261023T030823.672227Z-6db4e013`

Real workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 101 | 1+1+5 | 26.31 s | 0.228 s (0.224-0.253) | 115.6x | 72.41 s | 4.64 / 0.84 GiB | 0.0e+00 | PASSED |
| 102 | 1+1+5 | 3.560 s | 0.085 s (0.078-0.093) | 41.7x | 73.17 s | 1.84 / 0.66 GiB | 0.0e+00 | PASSED |
| 103 | 1+1+5 | 13.55 s | 0.224 s (0.222-0.235) | 60.4x | 69.04 s | 3.57 / 0.78 GiB | 0.0e+00 | PASSED |
| 104 | 1+1+5 | 100.7 s | 0.902 s (0.884-0.909) | 111.7x | 72.34 s | 13.69 / 1.73 GiB | 0.0e+00 | PASSED |
| 105 | 1+1+5 | 30.92 s | 0.258 s (0.252-0.289) | 119.7x | 73.70 s | 5.64 / 0.92 GiB | 0.0e+00 | PASSED |
| 106 | 1+1+5 | 10.58 s | 0.042 s (0.040-0.042) | 252.2x | 71.72 s | 1.79 / 0.66 GiB | 0.0e+00 | PASSED |
| 107 | 1+1+5 | 12.34 s | 0.164 s (0.159-0.175) | 75.4x | 72.93 s | 3.10 / 0.89 GiB | 0.0e+00 | PASSED |
| 108 | 1+0+1 | 273.6 s | 2.897 s | 94.5x | 71.42 s | 22.95 / 10.02 GiB | 0.0e+00 | PASSED |

Geometric-mean speedup (real): 95.6x over 8 cases.

Synthetic workloads:

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate | Max. rel. error | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 1+1+3 | 0.347 s | 0.013 s (0.012-0.014) | 26.4x | 72.44 s | 0.38 / 0.58 GiB | 0.0e+00 | PASSED |
| s2 | 1+1+3 | 1.682 s | 0.036 s (0.034-0.036) | 46.8x | 78.36 s | 1.16 / 0.60 GiB | 0.0e+00 | PASSED |
| s3 | 1+1+3 | 11.40 s | 0.154 s (0.150-0.167) | 74.2x | 69.81 s | 2.66 / 0.74 GiB | 0.0e+00 | PASSED |
| s4 | 1+1+3 | 61.04 s | 0.999 s (0.980-1.041) | 61.1x | 70.69 s | 6.37 / 1.41 GiB | 0.0e+00 | PASSED |
| s5 | 1+1+3 | 137.1 s | 2.093 s (2.044-2.126) | 65.5x | 73.42 s | 10.89 / 3.78 GiB | 0.0e+00 | PASSED |

Geometric-mean speedup (synthetic): 51.7x over 5 cases.

### Four threads (`--threads 4`, 2026-10-25)

The multicore configuration selected for the freeze (study report, "Host"). Runs:
`runs/20261026T013348.923342Z-fa22210e` (101-107), `runs/20261026T021823.863061Z-e034ca34` (108),
`runs/20261026T024122.997042Z-086adcdf` (synthetic). The candidate is unchanged since
2026-10-22; every fixture and output passed.

| Case | Calls | Reference median | Candidate median (min-max) | Speedup | Candidate first call | Peak RSS reference / candidate |
|---|---|---:|---:|---:|---:|---:|
| 101 | 1+1+5 | 28.47 s | 0.283 s (0.265-0.289) | 100.5x | 87.18 s | 4.57 / 0.74 GiB |
| 102 | 1+1+5 | 3.46 s | 0.093 s (0.091-0.095) | 37.2x | 86.21 s | 1.80 / 0.56 GiB |
| 103 | 1+1+5 | 14.18 s | 0.322 s (0.294-0.473) | 44.1x | 85.46 s | 3.50 / 0.71 GiB |
| 104 | 1+1+5 | 108.75 s | 1.064 s (1.014-1.098) | 102.2x | 91.64 s | 13.09 / 1.44 GiB |
| 105 | 1+1+5 | 33.83 s | 0.360 s (0.349-0.506) | 94.0x | 87.25 s | 6.14 / 0.82 GiB |
| 106 | 1+1+5 | 11.17 s | 0.051 s (0.044-0.054) | 218.9x | 90.39 s | 1.70 / 0.64 GiB |
| 107 | 1+1+5 | 13.47 s | 0.188 s (0.184-0.204) | 71.6x | 86.04 s | 3.15 / 0.81 GiB |
| 108 | 1+0+1 | 313.2 s | 3.321 s | 94.3x | 89.94 s | 21.69 / 9.76 GiB |
| s1 | 1+1+3 | 0.39 s | 0.013 s (0.010-0.013) | 30.9x | 82.87 s | 0.39 / 0.52 GiB |
| s2 | 1+1+3 | 1.75 s | 0.037 s (0.035-0.038) | 47.7x | 85.35 s | 1.05 / 0.56 GiB |
| s3 | 1+1+3 | 12.55 s | 0.182 s (0.181-0.194) | 69.1x | 85.81 s | 2.65 / 0.70 GiB |
| s4 | 1+1+3 | 69.15 s | 1.008 s (0.987-1.033) | 68.6x | 74.96 s | 6.51 / 1.43 GiB |
| s5 | 1+1+3 | 137.1 s | 2.344 s (2.298-2.364) | 58.5x | 76.27 s | 10.68 / 3.77 GiB |

**Geometric-mean speedups:** 83.4x (real, 8 cases) and 52.7x (synthetic, 5 cases).
- Four threads give 70-91% of six threads' speed on the real workloads (103 and 105: 70-72%),
  and 97-100% on s1, s2 and s4.
- The first call (Numba compilation) took 75-92 s; at six threads it took 69-78 s. Most of the
  day's cases were throttled.

### Thread scaling (candidate median at one and six threads)

| Case | 1 thread | 6 threads | Ratio | Reference 1 / 6 threads |
|---|---:|---:|---:|---:|
| 101 | 0.622 s | 0.228 s | 2.73x | 25.56 s / 26.31 s |
| 102 | 0.175 s | 0.085 s | 2.05x | 3.288 s / 3.560 s |
| 103 | 0.522 s | 0.224 s | 2.33x | 13.26 s / 13.55 s |
| 104 | 1.974 s | 0.902 s | 2.19x | 97.65 s / 100.7 s |
| 105 | 0.711 s | 0.258 s | 2.75x | 30.60 s / 30.92 s |
| 106 | 0.079 s | 0.042 s | 1.89x | 10.65 s / 10.58 s |
| 107 | 0.344 s | 0.164 s | 2.10x | 12.42 s / 12.34 s |
| 108 | 6.361 s | 2.897 s | 2.20x | 275.8 s / 273.6 s |
| s1 | 0.013 s | 0.013 s | 0.97x | 0.346 s / 0.347 s |
| s2 | 0.057 s | 0.036 s | 1.58x | 2.610 s / 1.682 s |
| s3 | 0.357 s | 0.154 s | 2.32x | 11.32 s / 11.40 s |
| s4 | 1.911 s | 0.999 s | 1.91x | 60.67 s / 61.04 s |
| s5 | 4.438 s | 2.093 s | 2.12x | 134.0 s / 137.1 s |

The reference's median changes by 9% or less at six threads, except synthetic 2 (2.61 to 1.68 s).
Six threads give the candidate 1.9-2.8x on the real workloads: the parallel loops (lookups, keys,
block tasks) scale, while the dictionary build, perfect matching, output assembly and Python-level
setup are serial. Synthetic 1 (0.013 s) is too small to benefit.

### Commands

One thread:

```bash
solutions/.venv/bin/python -m harness evaluate --task 003 --solution solutions/003/block-index/solution.json --threads 1 --suite real-development --ids 101 102 103 104 105 106 107 --timeout 3600   # runs/20261022T185556.024324Z-9dbc9db2
solutions/.venv/bin/python -m harness evaluate --task 003 --solution solutions/003/block-index/solution.json --threads 1 --suite real-development --ids 108 --warmups 0 --repeats 1 --timeout 7200   # runs/20261022T193346.854856Z-4d4ad68b
solutions/.venv/bin/python -m harness evaluate --task 003 --solution solutions/003/block-index/solution.json --threads 1 --ids 1 2 3 4 5 --timeout 3600   # runs/20261022T195218.141722Z-2a67bacf
```

Six threads:

```bash
solutions/.venv/bin/python -m harness evaluate --task 003 --solution solutions/003/block-index/solution.json --threads 6 --suite real-development --ids 101 102 103 104 105 106 107 --timeout 3600   # runs/20261023T021224.552881Z-1a3901b4
solutions/.venv/bin/python -m harness evaluate --task 003 --solution solutions/003/block-index/solution.json --threads 6 --suite real-development --ids 108 --warmups 0 --repeats 1 --timeout 7200   # runs/20261023T025011.664857Z-d2ece329
solutions/.venv/bin/python -m harness evaluate --task 003 --solution solutions/003/block-index/solution.json --threads 6 --ids 1 2 3 4 5 --timeout 3600   # runs/20261023T030823.672227Z-6db4e013
```
