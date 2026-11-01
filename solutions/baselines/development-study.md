# CPU candidates for tasks 001, 002 and 003, optimized for speed and for peak memory

Development study. Times are given by study day (days 1-4). The run directories it cites under `runs/` are local records, not included in this repository.

Six candidates: for each task, one optimized for speed and one for minimal peak memory. Each was
evaluated with `python -m harness evaluate` on every scored development workload (real-development
and synthetic) and compared on the same host with existing implementations of the same
computations ("baselines").

**What changed on day 4:**
- **Task 002:** replaced, at the user's direction, by the combined task from the real-data
  package's author. It merges the former tasks 002, 002b and 002c: csdid 1.81 and 2.0.1 rules,
  every option, the multiplier bootstrap, and nine new development workloads. Task 002b was
  removed.
- **New 002 candidates:** both task 002 candidates and all its baseline adapters are new;
  everything measured for the former tasks 002 and 002b is superseded.
- **Memory candidates:** the minimal-peak-memory candidates of all three tasks are new.
- **Multicore configuration:** four threads, the user's alternative to six (see "Host").

**Held-out evaluation:** left to the maintainer. No held-out input was read or run, and no freeze
was created (the commands are at the end).

| Task | Optimized for | Candidate | Method in one line |
|---|---|---|---|
| 001 fixed-effects regression (reghdfe) | speed | `solutions/001/schur-cg` | eliminate the largest fixed effect exactly; preconditioned CG on the reduced normal equations (approximate-Cholesky block preconditioner), all columns at once |
| 001 | peak memory | `solutions/001/schur-lean` | schur-cg's solver in column batches of at most 128 MB of per-level arrays; no n x (k + 1) demeaned matrix (rows formed in chunks, twice); free pages returned to the OS; Numba kernels loaded from a prebuilt cache |
| 002 staggered DiD (csdid 1.81 / 2.0.1, all options, bootstrap) | speed | `solutions/002/cell-fused` | per-cohort Newton fits shared across cells with identical units, with hoisted sums; one fused Numba pass over units for every cell's and every aggregate's influence function; member-aware BLAS products for the bootstrap; quickselect order statistics |
| 002 | peak memory | `solutions/002/cell-lean` | the same computation in NumPy only (no JIT compiler in the process): blocks of 512 KB working matrices, a small Gram matrix of the aggregate influence functions instead of per-unit arrays, bootstrap tiles of at most 4 MB of draws, free pages returned to the OS |
| 003 record linkage (reclink2) | speed | `solutions/003/block-index` | master-only dictionaries, city blocks with inverted bigram lists, vectorized exact float32 scoring with exact rescoring near the best |
| 003 | peak memory | `solutions/003/block-lean` | block-index unchanged, with its Numba kernels loaded from a prebuilt cache and free pages returned to the OS |

## Results at a glance

All numbers below are measurements on the development host; estimates are marked as such.

**How the figures are computed:**
- Speedups are ratios of evaluator medians: the reference and the candidate are measured in the
  same evaluator run, at the same thread count, on the same inputs.
- "Geometric mean (range, cases)" summarizes the per-case speedups.
- Peak RSS ratios are the candidate worker's peak divided by the reference worker's, at one
  thread.
- Every candidate output (first call, warmup and measured) passed the task's validator.
- Four threads is the multicore configuration selected for the freeze (day 4 runs). The
  six-thread rows were measured on days 1-2, before six-thread loads were found to make the
  host sleep ("Host").

| Task | Candidate | Threads | Real workloads | Synthetic workloads | Peak RSS / reference's (1 thread, real; synthetic) |
|---|---|---:|---|---|---|
| 001 | schur-cg (speed) | 1 | 30.6x (5.7-73.1x, 10) | 24.5x (9.4-245x, 10) | 0.96 (0.76-1.22); 0.97 (0.72-1.88) |
| 001 | schur-cg (speed) | 4 | 50.9x (11.8-112x, 10) | not run | |
| 001 | schur-cg (speed) | 6 | 54.2x (13.0-115x, 10) | 34.0x (12.6-328x, 10) | |
| 001 | schur-lean (memory) | 1 | 22.6x (5.0-55.6x, 10) | 15.2x (5.2-151x, 10) | 0.63 (0.49-0.85); 0.63 (0.41-1.38) |
| 002 | cell-fused (speed) | 1 | 17.5x (1.5-230x, 9) | 29.5x (10.3-56.4x, 6) | 0.80 (0.21-2.09); 0.84 (0.07-10.1) |
| 002 | cell-fused (speed) | 4 | 40.4x (1.8-575x, 9) | 50.6x (9.4-153x, 6) | |
| 002 | cell-lean (memory) | 1 | 4.3x (0.5-27.1x, 9) | 7.1x (3.0-11.0x, 6) | 0.25 (0.09-0.42); 0.17 (0.02-0.98) |
| 003 | block-index (speed) | 1 | 41.5x (18.8-135x, 8) | 32.8x (27.1-46.0x, 5) | 0.22 (0.09-0.42); 0.44 (0.22-1.55) |
| 003 | block-index (speed) | 4 | 83.4x (37.2-219x, 8) | 52.7x (30.9-69.1x, 5) | |
| 003 | block-index (speed) | 6 | 95.6x (41.7-252x, 8) | 51.7x (26.4-74.2x, 5) | |
| 003 | block-lean (memory) | 1 | 39.3x (18.5-111x, 8) | 27.2x (16.9-33.4x, 5) | 0.10 (0.06-0.40); 0.18 (0.09-0.35) |

**Against the speed candidate on the same workloads**, the memory candidates' peaks were:
- schur-lean: 25-49% lower than schur-cg's (104 with 50 regressors: 5.78 against 11.25 GiB);
- cell-lean: 2.1-9.8x lower than cell-fused's;
- block-lean: 1.5-3.3x lower than block-index's on 101-107, and the same on 108 (where the input
  loading sets the peak) and s5 (where block-index's working arrays do).

**Their speed against the speed candidates' times:**
- schur-lean takes 1.2-2.3x schur-cg's time;
- cell-lean takes 2.5-7x cell-fused's (on 108 it is slower than the reference);
- block-lean takes 0.95-1.36x block-index's time on the real workloads and 1.0-2.0x on the synthetic
  ones (s1: 0.025 against 0.013 s). The two were measured three days apart under different
  throttling; against the reference of the same run they reach 39.3x and 41.5x (real).

**Baselines** (compare.py, one measured call; times as multiples of the speed candidate's at the
same thread count):
- **001:**
  - fixest (R): 3-76x on the real workloads at one thread, 52x on 104 at four, and 5-50x at six;
  - pyfixest's within-transformation solver: 7-21x and 6-17x;
  - Stata reghdfe: 15-375x, and it timed out at 3,600 s on 104, 105 and 108; 104k5 (75x) and
    104k50 (timed out) were run after the study (addendum at the end);
  - pyfixest's alternating projections: 3-2,338x on the cases it ran.
- **002 (one thread):**
  - Stata csdid 2.0.1 passes all five 2.0.1-rule workloads at 85-922x;
  - Stata csdid 1.82 passes 102, 106 and 109 at 2,855x, 21,471x and 3,438x. On 108 (712 s per
    call), its 29 cells of one cohort differ: Stata's logit drops a covariate that predicts
    non-treatment perfectly, and the reference does not;
  - R did 2.5.1, the fastest baseline on the 2.0.1-rule workloads, passes all five at 43-437x
    (5.6-204 s per call).
- **003 (one thread):** Stata reclink2 takes 259-398x on the four real workloads where it agrees
  with the reference, and 190-608x on the synthetic ones. It fails the checks on 102 (2 fewer
  pairs) and 105 (scores off by 3.3e-6 relative, against a 1e-6 tolerance), and stops with a
  Stata error on 103. On 108, run after the study, it gave no result within 3,600 s (addendum
  at the end).

**Weak cases:**
- 002/108 (25,000 bootstrap draws): cell-fused gains only 1.5-1.8x, and cell-lean is 2x slower
  than the reference.
- 002/s1, 003/s1 and 001/s1: tiny inputs, where Numba's fixed costs show.
- The speed candidates' peaks on small inputs (Numba's compilation at the first call) exceed the
  reference's.


## Setup

- **Host:** Intel Core i7-8700B (6 cores, 12 logical CPUs), 32 GiB, macOS 15.7.3 (Darwin 24.6.0).
  Nothing else was benchmarked on the host while a measurement ran. Every evaluator and baseline
  invocation since day 2, 18:36, ran sequentially through
  `solutions/baselines/thermal_queue.py` (a cool-down gate before each case and a thermal monitor;
  see Disclosures).
- **Environments:** `solutions/.venv` for the candidates and the reference: CPython 3.12.13, NumPy
  2.3.5 (Apple Accelerate), Numba 0.62.1 (llvmlite 0.45.1), SciPy 1.18.1, pandas 3.0.6, pyarrow
  26.0.0, threadpoolctl 3.7.0. `solutions/baselines/.venv` for the baselines (README there).
- **CPU configurations:** `--threads 1` and a multicore configuration. The multicore runs were at 6
  threads on days 1-2 (001 and 003 speed candidates) and at 4 threads on day 4 (all
  002 runs, and the 4-thread runs of the 001 and 003 speed candidates); see "Host" below. The
  evaluator sets the BLAS/OpenMP/Numba thread variables and threadpool limits.
  - The Numba candidates use the Numba pool (workqueue layer, no spinning) and BLAS in separate
    phases.
  - cell-lean uses BLAS only.
  - With several threads, the 001 candidates run the approximate Cholesky factorizations in one
    Python thread while N - 1 Numba threads assemble the operator.
- **Candidate measurements** (`harness evaluate`, protocol 3): each invocation passes the task's
  development fixtures at its thread count. Then, for each case, it runs the reference and the
  candidate in fresh processes with the same thread budget, validates every candidate output
  against the reference output of the same invocation, and reports medians.
  - Calls per case: a first call, one warmup and five measured calls (real) or three
    (synthetic).
  - Cases whose single reference call exceeds 150 s run with `--warmups 0 --repeats 1`, the
    reference and the candidate alike: 001/104 (5, 20 and 50 regressors), 001/105, 001/108,
    001/synthetic 11 and 003/108. No task 002 case needed it: the slowest reference call is 107
    at 57 s at one thread.
- **Baseline measurements** (`solutions/baselines/compare.py`): the adapter runs through the
  evaluator's own worker (fresh process per case, the same loaders and read-only inputs,
  `perf_counter` around `run`, the same thread variables and limits) with a first call and one
  measured call.
  - Every output is validated with the task's validator against a saved reference output
    (`make_refs.py`: one reference call per case, one BLAS thread).
  - The inputs, accuracy requirements, timing boundary (NumPy arrays in, NumPy arrays out;
    conversions to R or Stata are inside the timed call), hardware and thread budget are the
    candidates'. Only the number of measured calls differs.
- **Memory:** the peak resident set of each worker process (`ru_maxrss`). It includes the
  interpreter, imports, the evaluator's input loading, all calls and output serialization; R and
  Stata run inside the worker, so their memory is included.
  - The loaders' own peaks set a floor that no candidate can go below. Examples: task 001's
    regressor matrix is built from a list of columns; task 002 generates its bootstrap
    multipliers before the call (107: about 0.6 GB).
  - The tables list the loader-only peaks where they matter.

## Host: six threads, four threads

Six-thread loads on this host repeatedly triggered forced sleeps ("Low Power Sleep", despite
keep-awake assertions) and cut the CPU speed limit to 52-60% within about four minutes. At the
user's question ("If six threads cannot work, how about four threads/cores?"), a four-thread
trial on day 4, 00:30-00:42, measured fixest-r on 001/104, the six-thread case that had
slept twice (`runs/baselines-20261023/001-fixest-r-t4.jsonl`, `thermal-cases.jsonl`):
- it completed and passed every check (first call 347.2 s, measured call 350.6 s, peak RSS
  13.4 GB);
- there was no sleep and no gap in the monitor;
- but 105 of the 143 five-second samples were throttled, the speed limit fell to 56% and the
  thermal level reached 122.

Four threads is therefore the less affected multicore configuration on this host, not a clean
one. The study uses it for every multicore measurement made after the trial, records each case's
throttling (`thermal-cases.jsonl`) and selects it for the multicore freeze. One-thread loads are
throttled intermittently too: speed limits of 71-80% in many samples of long cases. All
multicore figures here are therefore conservative for a host without this limitation.

## Findings for the maintainer

No benchmark file was changed because of these.

- **Task 002's reference and Stata on separated propensity logits.** The reference documents its
  logit as Stata's `logit ... [iw = w]` but has no perfect-prediction rule.
  - On 108's cohort 25 (5 treated units), a two-valued covariate predicts non-treatment perfectly
    for 91 controls. The reference's Newton iterations send that coefficient to -37, giving those
    controls scores below 1e-8.
  - Stata omits the covariate and the 91 observations, and drdid's `predict` then gives them
    scores up to 0.28.
  - All 29 of the cohort's cells differ (cell (25, 6): -1.631 against -0.253); every other cell
    agrees within 1e-6 (`solutions/002/stata-csdid181/diagnosis-108/`).
  - The candidates reproduce the reference. Held-out workloads with small cohorts and binary
    covariates may have such cells.
- **csdid 1.82's event aggregation under `long2`** (a tool issue): pre-treatment cells are
  selected one period late for the window, while their `Tm#` labels are right
  (`estat event, window(-5 5)` returns e = -6..-2). Its `estat` after `wboot()` reports
  asymptotic SEs for the aggregations; `csdid_stats` on saved RIFs gives the bootstrap ones.
- **R did 2.5.1** runs its 0.999 overlap check and covariate conditioning check only for its
  built-in estimator names, not when `est_method` is a function.

## Benchmark changes made at the user's direction

**Task 002 replaced by the combined task (day 4).** The real-data package's author merged
the former tasks 002 (the core of `csdid`), 002b (its common options) and 002c (options of
published replication packages) into one task.
- **Package:** 15 workloads (development 101-109, held out 301, 302 and 401-404), one reference
  and one loader, published at
  https://drive.google.com/drive/folders/1xRZ3L4FU8W8ZWfgiUUlvVRU6xB8JI_JC.
- **Task files** (`002_csdid/`):
  - `reference.py` unchanged from the package;
  - `task.md`: the package's, plus a shared-evaluator section;
  - `task.json`: in the harness format, with the new tolerances;
  - `realdata.py`: the package's loader, with integer IDs and a strict check for unknown IDs;
  - `real_workloads.json`: the development files pinned by SHA-256; the held-out entries have
    empty sizes and hashes, which the maintainer fills in from the private copy, so `final`
    fails closed until then;
  - `DATA.md`: with the maintainer's step;
  - `workloads.py`: the former task's six synthetic workloads, which generate exactly their
    former inputs, plus an option generator for the evaluator's fixtures.
- **Harness:**
  - validation of the new outputs (bootstrap SEs, critical values, the event-study covariance,
    event subsets) with the new scales: estimates against max(|reference|, its SE), the
    covariance against SE_i SE_j;
  - `normalize` drops the ragged `event_sets` echo;
  - 14 task 002 fixtures (the former four, eight covering every option and both rule sets, a
    0.995-trimming case and a csdid 2.0.1 overlap case);
  - updated tests (all 64 pass).
- **Data:** the development data (eight files and their manifests) were downloaded from the
  package's development folder and checked against the manifests' SHA-256 values. No held-out
  file was downloaded. The former task's data are kept, git-ignored, in
  `datasets/002-legacy-20261023/` for replays.
- **Task 002b removed:** its directory, registry entry, data (moved to
  `datasets/002b-legacy-20261024/`), fixtures, validation branch and tests. The generic string-ID
  support of the harness is kept.
- **Superseded:** the former task 002's candidate (`cohort-fused`), the 002b candidate and all
  their adapters are kept, unchanged, in `solutions/002/legacy-pre-merge/` with the runs that
  measured them; their results describe the former tasks only.

**Earlier changes (days 2-3, also at the user's direction):** the former task 002's
trimming correction (csdid passes `pscoretrim(1)`) and the integration of task 002b, both now
superseded by the combined task, which includes them.

## Task 001: OLS with high-dimensional fixed effects and clustered SEs

### schur-cg (speed; unchanged since day 1)


Development measurements are single calls after a first call from development scripts (not
evaluator runs); the version table is in `solutions/001/schur-cg/research.md`.

- **Exact elimination of the largest fixed effect + conjugate gradients on the reduced (Schur
  complement) system**, solving y and all regressors together: the reference's per-column
  iterative demeaning becomes one block solve over the smaller fixed effects. With a Jacobi
  preconditioner (version 1): 101 27.1 to 4.06 s, 102 95.7 to 14.6 s, s9 120.2 to 10.7 s.
- **Approximate Cholesky preconditioner** for each large sparse block (Kyng-Sachdeva / AC
  sampling of the block's Laplacian), exact dense pseudo-inverses for small blocks: iterations
  at tolerance 1e-9 drop from 85-232 (Jacobi) to 18-29 on the worker/firm graphs; 101 4.06 to
  0.44 s, 102 14.6 to 2.47 s, s9 10.7 to 1.23 s (version 2).
- **Cost model** choosing an explicit or implicit reduced operator, dense small systems, no data
  copies (version 3): 108 26.2 to 14.7 s, 104 16.2 to 13.2 s, s9 1.23 to 0.59 s.
- **Version 4**: skipping fixed effects nested in another (105: 44.3 to 33.6 s with identical
  coefficients; synthetic 7: 2.41 to 1.78 s); a fused implicit operator that never stores
  C^-1 N P, in a cache-friendly level order (105's product: 487 to 276 ms at one thread, 317 to
  114 ms at six); approximate-Cholesky solves 8 columns at a time (104: 100 to 78 ms per solve at
  one thread, 67 to 30 ms at six); compact level counts in the singleton peel (104's peel 0.645 to
  0.456 s). Net: 105 44.3 to 25.3 s, 104 13.2 to 12.6 s, 108 14.7 to 14.1 s.

Approaches that failed or were rejected (measured): incomplete Cholesky IC(0) of the reduced
matrix broke down (it is not an M-matrix); an exact sparse factorization of the firm block
filled 29x on 102 (7 s with SuperLU); approximate Cholesky on 105's subclass x year block did not
reduce iterations (48 both ways) and cost 5.9 s to factorize; relabelling the factor in
elimination order alone gave nothing (the gain came with 8-column chunks); W'W with Numba kernels
was slower than Accelerate's dgemm at one thread (564 vs 255 ms) and parallel BLAS calls on row
blocks ran serially; a round-based parallel singleton peel took 7.3 s against 0.7 s for the
sequential sweep. CG tolerances 1e-8, 1e-9 and 1e-11 give the same errors against the reference
(dominated by the reference's own LSMR error), so 1e-9 is used.

Slowdowns: version 4 is slower than version 3 on 107 (3.34 to 3.59 s), synthetic 9 (0.59 to
0.68 s), 101 (0.40 to 0.42 s) and synthetic 11 (+2.7%), accepted for the gains above. Six threads
give only 1.2-1.7x on the synthetic workloads and 1.4-2.3x on the real ones: the singleton peel is
sequential and W'W runs on one BLAS thread for these shapes. The candidate's peak RSS exceeds the
reference's on small inputs (synthetic 1: 0.26 vs 0.14 GiB; synthetic 6: 0.58 vs 0.37 GiB; 101:
0.76 vs 0.62 GiB; Numba and LLVM in the process) and is lower on the large ones (104 with 50
regressors: 11.25 vs 13.48 GiB; 108: 7.69 vs 10.10 GiB; synthetic 12: 11.79 vs 15.16 GiB). The
first call, which includes Numba compilation, takes 11.5-46.1 s at one thread.


### schur-lean (peak memory)

schur-cg's computation with:
- **Column batches:** the columns of [y, X] are solved in balanced batches whose per-level arrays
  take at most 128 MB, keeping only the fixed-effect coefficients of every column. On 104 with 50
  regressors, those are 0.66 + 0.10 GiB instead of the 2.57 GiB demeaned matrix and its
  companions.
- **No demeaned matrix:** demeaned rows are formed in 32 MB chunks, once for W'W and once, by
  cluster, for the residuals and cluster scores.
- **Early releases:** index arrays are released early.
- **Returning free pages:** free pages are returned to the OS at the start of the call (the
  loader's freed columns) and after the solve.
- **Prebuilt cache:** the Numba kernels load from a prebuilt on-disk cache (`build_cache.py`, read
  from a private copy at run time). The first call takes 0.3-0.8 s instead of 11-46 s, and the
  measured process does not run the LLVM compiler.

The loader alone peaks at about twice the input (the regressors are built from a list of columns:
104 with 50 regressors, 5.79 GiB for 2.77 GiB of inputs). Every candidate's peak is at least
that. Returning the loader's free pages at the start brought 104k50 from 7.39 GiB to the loader's
own 5.82 GiB.

**Cost:** time, 10-75% at one thread in development runs. The conjugate gradients run once per
batch, and the approximate Cholesky solves traverse the whole factor per iteration whatever the
batch width.

**Not done:** int32 index arrays (about 1 GB of int64 index structures on 104).

#### Measurements

Candidates: schur-cg (speed) and schur-lean (memory)


Real workloads (evaluator medians; speedup against the reference of the same run):

| Case | Calls | Reference, 1 thread | schur-cg, 1 thread | schur-cg, 4 threads | schur-cg, 6 threads | schur-lean, 1 thread | Peak RSS, 1 thread: reference / speed / memory | Max. rel. error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | 1+1+5 | 30.83 s | 0.422 s (73.1x) | 0.332 s (93.9x) | 0.294 s (110.3x) | 0.532 s (55.6x) | 637 MB / 778 MB / 512 MB | 1.8e-06 |
| 102 | 1+1+5 | 105 s | 2.210 s (47.7x) | 1.418 s (73.0x) | 1.328 s (78.0x) | 3.265 s (31.4x) | 1.68 GiB / 1.77 GiB / 1.14 GiB | 1.6e-07 |
| 103 | 1+1+5 | 116 s | 2.690 s (43.0x) | 1.622 s (70.8x) | 1.543 s (75.6x) | 3.842 s (30.9x) | 2.30 GiB / 2.16 GiB / 1.40 GiB | 2.2e-07 |
| 104k20 | 1+0+1 | 524 s | 12.88 s (40.7x) | 6.782 s (79.9x) | 6.371 s (81.8x) | 18.25 s (29.7x) | 6.56 GiB / 5.78 GiB / 3.24 GiB | 2.4e-07 |
| 104k5 | 1+0+1 | 169 s | 8.144 s (20.7x) | 5.159 s (32.7x) | 4.702 s (35.8x) | 9.834 s (17.6x) | 3.15 GiB / 3.26 GiB / 2.46 GiB | 2.3e-08 |
| 104k50 | 1+0+1 | 1,225 s | 20.78 s (58.9x) | 11.15 s (111.8x) | 10.65 s (114.9x) | 30.85 s (44.0x) | 13.48 GiB / 11.25 GiB / 5.78 GiB | 9.1e-07 |
| 105 | 1+0+1 | 689 s | 25.12 s (27.4x) | 13.94 s (50.0x) | 12.82 s (53.3x) | 41.40 s (18.5x) | 6.85 GiB / 5.95 GiB / 3.46 GiB | 1.7e-07 |
| 106 | 1+1+5 | 3.521 s | 0.233 s (15.1x) | 0.165 s (21.7x) | 0.159 s (21.8x) | 0.418 s (8.6x) | 722 MB / 741 MB / 556 MB | 1.6e-09 |
| 107 | 1+1+5 | 20.45 s | 3.563 s (5.7x) | 1.747 s (11.8x) | 1.563 s (13.0x) | 4.171 s (5.0x) | 2.11 GiB / 2.16 GiB / 1.49 GiB | 4.2e-10 |
| 108 | 1+0+1 | 583 s | 14.37 s (40.5x) | 9.151 s (64.5x) | 8.692 s (66.5x) | 19.00 s (34.4x) | 10.10 GiB / 7.69 GiB / 5.36 GiB | 1.5e-08 |

- schur-cg, 1 thread: against the reference 30.6x (5.7-73.1x, 10 cases)
- schur-cg, 4 threads: against the reference 50.9x (11.8-111.8x, 10 cases)
- schur-cg, 6 threads: against the reference 54.2x (13.0-114.9x, 10 cases)
- schur-lean, 1 thread: against the reference 22.6x (5.0-55.6x, 10 cases)
- schur-cg, 1 thread: peak RSS / the reference's 0.96 (0.76-1.22, 10 cases)
- schur-lean, 1 thread: peak RSS / the reference's 0.63 (0.49-0.85, 10 cases)

Synthetic workloads (evaluator medians; speedup against the reference of the same run):

| Case | Calls | Reference, 1 thread | schur-cg, 1 thread | schur-cg, 4 threads | schur-cg, 6 threads | schur-lean, 1 thread | Peak RSS, 1 thread: reference / speed / memory | Max. rel. error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 1+1+3 | 0.240 s | 0.021 s (11.6x) | - | 0.017 s (12.6x) | 0.036 s (6.9x) | 140 MB / 263 MB / 167 MB | 1.1e-14 |
| s3 | 1+1+3 | 21.85 s | 1.846 s (11.8x) | - | 1.245 s (17.3x) | 3.594 s (6.4x) | 4.77 GiB / 3.42 GiB / 1.93 GiB | 6.2e-14 |
| s4 | 1+1+3 | 17.00 s | 1.126 s (15.1x) | - | 0.678 s (25.0x) | 1.580 s (11.2x) | 1.27 GiB / 1.25 GiB / 893 MB | 3.0e-08 |
| s6 | 1+1+3 | 1.711 s | 0.181 s (9.4x) | - | 0.106 s (15.9x) | 0.238 s (7.6x) | 377 MB / 598 MB / 440 MB | 8.2e-15 |
| s7 | 1+1+3 | 22.83 s | 1.794 s (12.7x) | - | 1.405 s (16.4x) | 2.896 s (8.5x) | 2.35 GiB / 2.19 GiB / 1.21 GiB | 1.4e-09 |
| s8 | 1+1+3 | 47.73 s | 0.721 s (66.2x) | - | 0.545 s (86.4x) | 1.195 s (41.5x) | 1.32 GiB / 1.20 GiB / 828 MB | 1.6e-07 |
| s9 | 1+1+3 | 109 s | 0.677 s (161.8x) | - | 0.508 s (219.8x) | 1.049 s (114.4x) | 1.28 GiB / 1.17 GiB / 766 MB | 5.5e-08 |
| s10 | 1+1+3 | 6.430 s | 0.593 s (10.8x) | - | 0.453 s (14.2x) | 1.351 s (5.2x) | 2.29 GiB / 1.67 GiB / 1.07 GiB | 2.9e-14 |
| s11 | 1+0+1 | 2,079 s | 8.486 s (245.0x) | - | 5.710 s (328.3x) | 13.23 s (150.9x) | 9.91 GiB / 8.22 GiB / 4.17 GiB | 4.1e-07 |
| s12 | 1+1+3 | 65.30 s | 6.010 s (10.9x) | - | 3.823 s (16.7x) | 12.15 s (5.7x) | 15.16 GiB / 11.79 GiB / 6.78 GiB | 1.1e-13 |

- schur-cg, 1 thread: against the reference 24.5x (9.4-245.0x, 10 cases)
- schur-cg, 6 threads: against the reference 34.0x (12.6-328.3x, 10 cases)
- schur-lean, 1 thread: against the reference 15.2x (5.2-150.9x, 10 cases)
- schur-cg, 1 thread: peak RSS / the reference's 0.97 (0.72-1.88, 10 cases)
- schur-lean, 1 thread: peak RSS / the reference's 0.63 (0.41-1.38, 10 cases)

Baselines, 1 thread (compare.py: first call + one measured call; median, and baseline / schur-cg median):

| Case | schur-cg | fixest-r | pyfixest-map | pyfixest-within | stata-reghdfe |
|---|---:|---:|---:|---:|---:|
| 101 | 0.422 s | 32.00 s (76x) | 986 s (2338x) | 6.652 s (16x) | 158 s (375x) |
| 102 | 2.210 s | 118 s (53x) |  | 28.99 s (13x) | 497 s (225x) |
| 103 | 2.690 s | 125 s (47x) |  | 35.74 s (13x) | 559 s (208x) |
| 104k20 | 12.88 s | 591 s (46x) |  | 173 s (13x) | timeout |
| 104k5 | 8.144 s |  |  | 55.09 s (7x) |  |
| 104k50 | 20.78 s |  |  | 427 s (21x) |  |
| 105 | 25.12 s | 992 s (39x) |  | 281 s (11x) | timeout |
| 106 | 0.233 s | 1.875 s (8x) | 2.114 s (9x) | 2.495 s (11x) | 12.32 s (53x) |
| 107 | 3.563 s | 9.186 s (3x) | 9.424 s (3x) | 24.17 s (7x) | 52.96 s (15x) |
| 108 | 14.37 s | 872 s (61x) |  | 260 s (18x) | timeout |
| s1 | 0.021 s | 0.075 s (4x) | 0.096 s (5x) | 0.116 s (6x) |  |
| s3 | 1.846 s | 19.66 s (11x) |  | 16.71 s (9x) |  |
| s4 | 1.126 s | 14.05 s (12x) | 82.76 s (74x) | 10.99 s (10x) |  |
| s6 | 0.181 s | 0.429 s (2x) | 0.461 s (3x) | 1.155 s (6x) |  |
| s7 | 1.794 s | 10.46 s (6x) |  | 22.31 s (12x) |  |
| s8 | 0.721 s | 79.28 s (110x) |  | 12.81 s (18x) |  |
| s9 | 0.677 s | fails checks, 881 s |  | 10.30 s (15x) |  |
| s10 | 0.593 s | 13.66 s (23x) | 4.082 s (7x) | 6.535 s (11x) |  |
| s11 | 8.486 s |  |  | 147 s (17x) |  |
| s12 | 6.010 s | 130 s (22x) |  | 82.37 s (14x) |  |
- stata-reghdfe 104k20: timed out (worker wall time 3,600 s)
- stata-reghdfe 105: timed out (worker wall time 3,600 s)
- stata-reghdfe 108: timed out (worker wall time 3,600 s)
- stata-reghdfe 104k5 and 104k50: not run in this study; run after it (104k5: 607 s, 75x;
  104k50: timed out; addendum at the end)
- fixest-r s9: first_call output: beta (max. rel. error 1.5e-06, tolerance 1e-06)

Baselines, 4 threads (compare.py: first call + one measured call; median, and baseline / schur-cg median):

| Case | schur-cg | fixest-r |
|---|---:|---:|
| 104k20 | 6.782 s | 351 s (52x) |

Baselines, 6 threads (compare.py: first call + one measured call; median, and baseline / schur-cg median):

| Case | schur-cg | fixest-r | pyfixest-within |
|---|---:|---:|---:|
| 101 | 0.294 s | 12.56 s (43x) | 2.899 s (10x) |
| 102 | 1.328 s | 66.71 s (50x) | 14.92 s (11x) |
| 103 | 1.543 s | 66.32 s (43x) | 16.55 s (11x) |
| 104k20 | 6.371 s |  | 75.56 s (12x) |
| 104k5 | 4.702 s |  | 26.92 s (6x) |
| 104k50 | 10.65 s |  | 181 s (17x) |
| 105 | 12.82 s |  | 117 s (9x) |
| 106 | 0.159 s | 1.093 s (7x) | 1.483 s (9x) |
| 107 | 1.563 s | 7.128 s (5x) | 12.25 s (8x) |
| 108 | 8.692 s |  | 103 s (12x) |
| s1 | 0.017 s |  | 0.115 s (7x) |
| s3 | 1.245 s |  | 12.39 s (10x) |
| s4 | 0.678 s |  | 5.081 s (7x) |
| s6 | 0.106 s |  | 0.737 s (7x) |
| s7 | 1.405 s |  | 11.34 s (8x) |
| s8 | 0.545 s |  | 5.556 s (10x) |
| s9 | 0.508 s |  | 4.674 s (9x) |
| s10 | 0.453 s |  | 3.935 s (9x) |
| s11 | 5.710 s |  | 61.76 s (11x) |
| s12 | 3.823 s |  | 64.43 s (17x) |



## Task 002: staggered difference-in-differences (the combined task)

**Contract:** `002_csdid/reference.py` (csdid 1.81 or 2.0.1 rules) with every option. Each
workload follows one rule set:

| Workload | Control | Method | Base period | Inference | Rules |
|---|---|---|---|---|---|
| 101 | not yet | dripw | universal | bootstrap 999 | 2.0.1 |
| 102 | never | dripw | varying | analytic, weighted, unbalanced | 1.81 |
| 103 | never | reg | universal | analytic, event subsets | 2.0.1 |
| 104 | never | dripw | varying (monthly, 3,481 cells) | analytic | 2.0.1 |
| 105 | not yet | stdipw | varying (41 cohorts) | analytic | 2.0.1 |
| 106 | never | reg | varying (no covariates) | analytic | 1.81 |
| 107 | never | dripw | varying (headline, 198,649 units) | bootstrap 999, trim 0.995 | 2.0.1 |
| 108 | not yet | ipw | universal | bootstrap 25,000 | 1.81 |
| 109 | not yet | dripw | universal (covariates vary over time) | analytic, trim 0.995 | 1.81 |

The synthetic workloads 1-6 are the former task's, with the default options.

**Tolerances:**
- estimates: error / max(|reference|, its SE) ≤ 1e-6;
- analytic and bootstrap SEs and critical values: relative 1e-4;
- the event-study covariance: error / (SE_i SE_j) ≤ 1e-4.

### cell-fused (speed)

The method is in the candidate's `research.md`:
1. **Cells and counts:** the cell list and the aggregation weights come from cell sizes, counted
   from cohort tallies on balanced panels.
2. **Estimation:**
   - per-cohort Newton fits (logit, inverse probability tilting, csdid 2.0.1's unweighted overlap
     logit, weighted least squares) with the reference's stopping rule;
   - consecutive cells with the same units and covariates share the fits and the sums that do
     not depend on the outcome change; other fits are warm-started.
3. **Pass over units:** one fused Numba pass over 64-unit blocks of one cohort for every cell's
   influence function and every aggregate's, with cached scores and block-local accumulators.
4. **Bootstrap:** member-aware BLAS products of each cohort's cell influence functions with its
   multipliers. The aggregate tables' draws are linear combinations of the cell draws plus the
   cohort weights' terms, and the order statistics come from quickselect.
5. **Safeguards:** the reference's own operations are used for the rare cells that are singular,
   lie within 1e-9 of the trimming or overlap cut, or are degenerate.

**Gains, measured in development runs at one thread** (the evaluator tables below are the
measurements):

| Change | Workload | Before | After |
|---|---|---:|---:|
| Cell counts from cohort tallies (balanced panels) | 107 | 0.33 s | 0.01 s |
| Hoisted sums for cells sharing a fit | 107 estimation | 1.11 s | 0.70 s |
| | 104 estimation | 0.38 s | 0.18 s |
| Unit pass: block-local accumulators, reused scores | 107 | 1.28 s | 0.45 s |
| | 104 | 0.81 s | 0.22 s |
| Bootstrap: aggregate draws from the cell draws (28% less product) | 107 | 2.4 s | 1.8 s |
| Bootstrap: row adds replacing a fancy-index accumulation | 108 | 2.2 s | removed |
| Logit covariance from the last Newton iteration | 105 | 1.24 s | 1.02 s |

Overall, v3.0 to v3.1: 107 went from 5.46 to 3.18 s, 104 from 1.31 to 0.44 s, 101 from 0.26 to
0.15 s.

**Unsuccessful:**
- **The method of four Russians for the +1/-1 products** (tables of the 256 signed sums of 8
  units): 45-58 ms against Accelerate dgemm's 32 ms on a 2048 x 361 x 999 chunk, because the
  lookups did not vectorize. Removed.
- **Lower-precision products:** rejected before measuring, because they would leave too little
  margin on the 1e-4 tolerance.

**Remaining costs:**
- 107: the multiplier product, 1.6-1.8 s at 45 GFLOP/s for 7.2e10 flops (the reference's dense
  product is 1.65e11 flops).
- 105: 1,681 separately warm-started logits.
- 108: 25,000 draws for 812 cells.

### cell-lean (peak memory)

The same computation in NumPy only, so no JIT compiler is in the process:
- blocks of one cohort's units with working matrices of at most 512 KB;
- the aggregate influence functions only as sums and a Gram matrix (n_aggregates^2), which give
  every aggregate's SE, the cohort average's and the event-study covariance;
- bootstrap tiles of at most 4 MB of draws (several passes over the units when a table is
  larger: 108 has 162 MB of draws in about 40 tiles);
- weighted Gram matrices accumulated over row chunks;
- free pages returned to the OS after every 64 MB of large freed arrays. macOS keeps freed blocks
  resident: NumPy's traced peak was 39 MB on 104 while the resident set grew by 190 MB.

**Working-matrix size** (104, one thread):

| Size (elements) | 2^14 | 2^16 | 2^18 | 2^20 |
|---|---:|---:|---:|---:|
| Peak | 148 MB | 164 MB | 294 MB | 762 MB |
| Time | 6.8 s | 3.5 s | 3.1 s | 3.5 s |

2^16 is used.

**Returning free pages:** after every cohort and tile, 107 went from 738 to 609 MB, but 105 took
twice as long. Counting only freed arrays of 4 MB or more, and returning pages every 64 MB, keeps
the times and gives 637 MB on 107.

**Correctness of both candidates:**
- the 14 fixtures and real workloads 101-109 pass;
- `fuzz.py` (1,000 random panels with every option, seed 7000) passes 783 of the 784 panels where
  the reference succeeds. The exception is the degenerate panel 7276 (Disclosures).

### Baselines

All at one thread, through compare.py (a first call and one measured call, validated against the
saved reference outputs):
- **Stata csdid 2.0.1** (`solutions/002/stata-csdid201`; github.com/pedrohcgs/csdid-stata, commit
  f98bcea) runs the five 2.0.1-rule workloads:
  - `bal(pair)`, an analytic run, and for the bootstrap a second run with `reps(B)`;
  - its estimators are dr, reg and ipw (the normalized one);
  - every compared output passes on 101, 103, 104, 105 and 107.
- **Stata csdid 1.82 with drdid 1.91** (`solutions/002/stata-csdid181`; SSC) runs the four
  1.81-rule workloads:
  - `long2` for the universal base, with the event window moved one period on the
    pre-treatment side to offset 1.82's event pairing under `long2`;
  - `saverif()` with `csdid_stats ..., wboot()` for the bootstrap.
  - Its first 109 row failed before that window correction (Disclosures).
  - **Results:** 106, 102 and 109 pass. On 108 it estimates the same cells, but some ATT(g,t)
    differ from the reference by up to 0.84 of max(|reference|, SE), and the aggregates by
    0.05-5%. The fixture with 108's options passes.
  - **Diagnosis** (`solutions/002/stata-csdid181/diagnosis-108/`): the 29 failing cells are
    exactly cohort 25's (5 treated units). The other 783 agree within 1e-6.
    - In cohort 25, a two-valued covariate is 0 for every treated unit and nonzero for 91
      controls, so it predicts non-treatment perfectly.
    - The reference's Newton iterations send its coefficient to -37 and those controls' scores
      below 1e-8.
    - Stata's `logit` (as drdid calls it) omits the covariate and the 91 observations, reaching
      the same log likelihood; drdid's `predict` then scores those controls up to 0.28 and
      weights them.
    - This is a genuine difference in how the two handle separation, not an adapter error; see
      "Findings for the maintainer".
- **R did 2.5.1 with DRDID 1.3.0** (`solutions/002/did-r`) runs the balanced 2.0.1-rule workloads
  (all five), since did's rules are 2.0.1's:
  - the method's DRDID estimator with the task's trimming level, behind did's own overlap and
    conditioning checks;
  - the universal base's reference cells dropped.
  - **Results:** every compared output passes on 101, 103, 104, 105 and 107, in about half of
    csdid 2.0.1's time: 6.5, 5.6, 195, 45 and 204 s per call, against csdid 2.0.1's 15.5, 8.8,
    411, 85 and 365 s. Its peak was 10.8 GB on 107 (csdid 2.0.1: 6.8 GB).
- **Bootstrap outputs:** every tool draws its own multipliers, so the bootstrap outputs are timed
  but not compared.
  - On the development bootstrap fixtures (99-199 draws; `solutions/baselines/check_adapter.py`),
    the three adapters' bootstrap SEs differ from the reference's by a median of 1-29% per output
    (single estimates up to 52%), and their critical values by 0.3-31%. That is the scale
    expected from different multipliers at that number of draws; it was not tested formally.
  - Each adapter passes every compared output on its rule set's fixtures that it supports. For
    csdid 2.0.1 this includes the weighted unbalanced stdipw and overlap fixtures (checked after
    the queue).

#### Measurements

Candidates: cell-fused (speed) and cell-lean (memory)


Real workloads (evaluator medians; speedup against the reference of the same run):

| Case | Calls | Reference, 1 thread | cell-fused, 1 thread | cell-fused, 4 threads | cell-lean, 1 thread | Peak RSS, 1 thread: reference / speed / memory | Max. rel. error |
|---|---|---:|---:|---:|---:|---:|---:|
| 101 | 1+1+5 | 2.754 s | 0.152 s (18.1x) | 0.083 s (42.6x) | 0.424 s (7.0x) | 414 MB / 561 MB / 192 MB | 7.5e-12 |
| 102 | 1+1+5 | 3.644 s | 0.236 s (15.4x) | 0.079 s (46.9x) | 1.514 s (2.7x) | 446 MB / 458 MB / 166 MB | 1.0e-11 |
| 103 | 1+1+5 | 1.306 s | 0.075 s (17.4x) | 0.029 s (60.7x) | 0.365 s (3.3x) | 428 MB / 479 MB / 136 MB | 8.8e-14 |
| 104 | 1+1+5 | 102 s | 0.445 s (229.9x) | 0.199 s (575.0x) | 3.037 s (27.1x) | 2.66 GiB / 583 MB / 232 MB | 4.1e-11 |
| 105 | 1+1+5 | 24.99 s | 0.994 s (25.1x) | 0.353 s (109.6x) | 4.158 s (5.6x) | 845 MB / 504 MB / 191 MB | 3.1e-11 |
| 106 | 1+1+5 | 0.298 s | 0.016 s (19.0x) | 0.013 s (25.1x) | 0.050 s (6.1x) | 376 MB / 455 MB / 116 MB | 3.7e-13 |
| 107 | 1+1+5 | 61.19 s | 3.306 s (18.5x) | 1.802 s (32.5x) | 8.102 s (7.3x) | 5.80 GiB / 1.27 GiB / 613 MB | 5.8e-12 |
| 108 | 1+1+5 | 2.015 s | 1.337 s (1.5x) | 0.929 s (1.8x) | 3.751 s (0.5x) | 918 MB / 1.87 GiB / 360 MB | 1.1e-10 |
| 109 | 1+1+5 | 1.419 s | 0.135 s (10.5x) | 0.073 s (25.2x) | 0.549 s (2.4x) | 408 MB / 526 MB / 131 MB | 3.4e-13 |

- cell-fused, 1 thread: against the reference 17.5x (1.5-229.9x, 9 cases)
- cell-fused, 4 threads: against the reference 40.4x (1.8-575.0x, 9 cases)
- cell-lean, 1 thread: against the reference 4.3x (0.5-27.1x, 9 cases)
- cell-fused, 1 thread: peak RSS / the reference's 0.80 (0.21-2.09, 9 cases)
- cell-lean, 1 thread: peak RSS / the reference's 0.25 (0.09-0.42, 9 cases)

Synthetic workloads (evaluator medians; speedup against the reference of the same run):

| Case | Calls | Reference, 1 thread | cell-fused, 1 thread | cell-fused, 4 threads | cell-lean, 1 thread | Peak RSS, 1 thread: reference / speed / memory | Max. rel. error |
|---|---|---:|---:|---:|---:|---:|---:|
| s1 | 1+1+3 | 0.028 s | 0.003 s (10.3x) | 0.003 s (9.4x) | 0.010 s (3.0x) | 38 MB / 382 MB / 39 MB | 1.5e-14 |
| s2 | 1+1+3 | 0.317 s | 0.010 s (32.8x) | 0.007 s (66.1x) | 0.051 s (7.7x) | 143 MB / 395 MB / 61 MB | 5.3e-14 |
| s3 | 1+1+3 | 2.901 s | 0.051 s (56.4x) | 0.021 s (152.8x) | 0.255 s (11.0x) | 303 MB / 402 MB / 89 MB | 6.7e-14 |
| s4 | 1+1+3 | 4.918 s | 0.198 s (24.8x) | 0.118 s (58.1x) | 0.546 s (8.7x) | 922 MB / 608 MB / 125 MB | 1.4e-12 |
| s5 | 1+1+3 | 92.56 s | 1.854 s (49.9x) | 1.393 s (63.9x) | 10.39 s (8.9x) | 12.31 GiB / 872 MB / 221 MB | 4.8e-12 |
| s6 | 1+1+3 | 92.25 s | 3.271 s (28.2x) | 1.925 s (47.8x) | 14.77 s (6.3x) | 5.96 GiB / 1.23 GiB / 500 MB | 1.4e-12 |

- cell-fused, 1 thread: against the reference 29.5x (10.3-56.4x, 6 cases)
- cell-fused, 4 threads: against the reference 50.6x (9.4-152.8x, 6 cases)
- cell-lean, 1 thread: against the reference 7.1x (3.0-11.0x, 6 cases)
- cell-fused, 1 thread: peak RSS / the reference's 0.84 (0.07-10.10, 6 cases)
- cell-lean, 1 thread: peak RSS / the reference's 0.17 (0.02-0.98, 6 cases)

Baselines, 1 thread (compare.py: first call + one measured call; median, and baseline / cell-fused median):

| Case | cell-fused | did-r | stata-csdid181 | stata-csdid201 |
|---|---:|---:|---:|---:|
| 101 | 0.152 s | 6.503 s (43x) * |  | 15.50 s (102x) * |
| 102 | 0.236 s |  | 674 s (2855x) |  |
| 103 | 0.075 s | 5.553 s (74x) |  | 8.848 s (118x) |
| 104 | 0.445 s | 195 s (437x) |  | 411 s (922x) |
| 105 | 0.994 s | 44.88 s (45x) |  | 84.91 s (85x) |
| 106 | 0.016 s |  | 337 s (21471x) |  |
| 107 | 3.306 s | 204 s (62x) * |  | 365 s (110x) * |
| 108 | 1.337 s |  | fails checks, 712 s * |  |
| 109 | 0.135 s |  | 464 s (3438x) |  |

\* bootstrap outputs not compared (the tool's own multipliers).
- stata-csdid181 108: first_call output: att_gt (max. rel. error 0.84, tolerance 1e-06); se_gt (max. rel. error 2.1, tolerance 0.0001); att_simple (max. rel. error 0.0078, tolerance 1e-06); se_simple (max. rel. error 0.00045, tolerance 0.0001); att_group (max. rel. error 0.77, tolerance 1e-06); se_group (max. rel. error 1.2, tolerance 0.0001); att_group_avg (max. rel. error 0.025, tolerance 1e-06); se_group_avg (max. rel. error 0.0029, tolerance 0.0001); att_calendar (max. rel. error 0.05, tolerance 1e-06); se_calendar (max. rel. error 0.0017, tolerance 0.0001); att_calendar_avg (max. rel. error 0.0047, tolerance 1e-06); se_calendar_avg (max. rel. error 0.00035, tolerance 0.0001); att_event (max. rel. error 0.026, tolerance 1e-06); se_event (max. rel. error 0.044, tolerance 0.0001); att_pre_avg (max. rel. error 0.016, tolerance 1e-06); se_pre_avg (max. rel. error 0.032, tolerance 0.0001); att_post_avg (max. rel. error 0.014, tolerance 1e-06); se_post_avg (max. rel. error 0.0087, tolerance 0.0001); cov_event (max. rel. error 0.09, tolerance 0.0001)

Earlier rows of the same cases, superseded by the rows above (kept in the results files):
- stata-csdid181 109 (day 4, 16:58): fails checks, 458 s; first_call output: event_times (shape or dtype mismatch: [11] against [10]); att_event (shape or dtype mismatch: [11] against [10]); se_event (shape or dtype mismatch: [11] against [10]); att_pre_avg (max. rel. error 0.19, tolerance 1e-06); se_pre_avg (max. rel. error 0.04, tolerance 0.0001); cov_event (shape or dtype mismatch: [11, 11] against [10, 10])



## Task 003: record linkage (reclink2)

### block-index (speed; unchanged since day 1)


Version table in `solutions/003/block-index/research.md` (development scripts, one thread).

- **Indexing instead of scanning** (version 1): dictionaries of the master values (using values
  looked up by hash), perfect matches as tuples of ids, blocking keys from (required, value)
  pairs, and per-block inverted lists of bigram codes so that a master value's bigram counts
  against all block records are sums over its bigrams' lists: 108 352.0 to 12.6 s, 101 35.3 to
  2.06 s.
- **Output buffer sized per task** (the search had run twice): 101 2.06 to 1.12 s, 104 3.36 s.
- **Branch-free scoring over chunks of 128 candidates** (the reference's float64/float32 chain
  operation by operation, vectorized by LLVM): 101 1.12 to 0.69 s, 105 1.33 to 0.85 s.
- **Block encoding inside the parallel tasks with prefetching, 8-character hashing, direct
  counting for blocks with one or two masters**: 108 12.6 to 7.82 s.
- **Approximate cube-root bonus in the hot loop, exact rescoring within 1e-5 of the best**: 101
  0.85 to 0.64 s, 104 1.95 s, 108 7.82 to 6.07 s; outputs stay bitwise equal to the reference's.

Failed or rejected: a memo of the cube-root bonus kept the scoring loop scalar and cost as much as
`pow`; global block encodings before the search needed several GB for 108 and two passes over 4.6
million scattered rows; a large Numba `prange` body silently processed only some master records
(fixed by moving the body into a separate function; verified against serial and pure-Python
versions). Not attempted: score-bound pruning of candidates, bit-sliced bigram counting.

Slowdowns: synthetic 1 does not gain from six threads (0.013 s at both). The first call takes
70.5-77.6 s at one thread (Numba compilation of the scoring kernels). Peak RSS is lower than the
reference's except on synthetic 1 (0.57 vs 0.37 GiB); on 108 it is 9.5-10.0 GiB against 22.9 GiB.

### block-lean (peak memory)

block-index's computation, unchanged, with two differences:
- **Prebuilt cache:** its Numba kernels load from a prebuilt on-disk cache, read from a private
  copy at run time. block-index compiles at the first call: about 70 s and 0.47 GB of peak memory.
- **Returning free pages:** free pages are returned to the OS at the start of the call and after
  the dictionaries.

Development runs (fresh process, one thread):

| Workload | After loading | block-index | block-lean |
|---|---:|---:|---:|
| 101 | 172 MB | 745 MB | 277 MB |
| 102 | 109 MB | 671 MB | 191 MB |
| 104 | 577 MB | 1,209 MB | 754 MB |
| 106 | 107 MB | 664 MB | 185 MB |

Steady-state times are unchanged (101: 0.59 against 0.60 s).

**Not attempted:** one-byte instead of four-byte character codes for the values. The working
arrays are not what sets the peak on these workloads.

#### Measurements

Candidates: block-index (speed) and block-lean (memory)


Real workloads (evaluator medians; speedup against the reference of the same run):

| Case | Calls | Reference, 1 thread | block-index, 1 thread | block-index, 4 threads | block-index, 6 threads | block-lean, 1 thread | Peak RSS, 1 thread: reference / speed / memory | Max. rel. error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | 1+1+5 | 25.56 s | 0.622 s (41.1x) | 0.283 s (100.5x) | 0.228 s (115.6x) | 0.609 s (42.4x) | 4.58 GiB / 738 MB / 309 MB | 0.0e+00 |
| 102 | 1+1+5 | 3.288 s | 0.175 s (18.8x) | 0.093 s (37.2x) | 0.085 s (41.7x) | 0.174 s (18.5x) | 1.86 GiB / 641 MB / 210 MB | 0.0e+00 |
| 103 | 1+1+5 | 13.26 s | 0.522 s (25.4x) | 0.322 s (44.1x) | 0.224 s (60.4x) | 0.530 s (24.9x) | 3.48 GiB / 734 MB / 316 MB | 0.0e+00 |
| 104 | 1+1+5 | 97.65 s | 1.974 s (49.5x) | 1.064 s (102.2x) | 0.902 s (111.7x) | 1.884 s (52.5x) | 12.73 GiB / 1.15 GiB / 779 MB | 0.0e+00 |
| 105 | 1+1+5 | 30.60 s | 0.711 s (43.1x) | 0.360 s (94.0x) | 0.258 s (119.7x) | 0.711 s (43.5x) | 5.57 GiB / 805 MB / 386 MB | 0.0e+00 |
| 106 | 1+1+5 | 10.65 s | 0.079 s (134.6x) | 0.051 s (218.9x) | 0.042 s (252.2x) | 0.096 s (111.4x) | 1.75 GiB / 668 MB / 205 MB | 0.0e+00 |
| 107 | 1+1+5 | 12.42 s | 0.344 s (36.1x) | 0.188 s (71.6x) | 0.164 s (75.4x) | 0.430 s (29.0x) | 3.16 GiB / 804 MB / 370 MB | 0.0e+00 |
| 108 | 1+0+1 | 276 s | 6.361 s (43.4x) | 3.321 s (94.3x) | 2.897 s (94.5x) | 8.681 s (39.3x) | 22.91 GiB / 9.53 GiB / 9.05 GiB | 0.0e+00 |

- block-index, 1 thread: against the reference 41.5x (18.8-134.6x, 8 cases)
- block-index, 4 threads: against the reference 83.4x (37.2-218.9x, 8 cases)
- block-index, 6 threads: against the reference 95.6x (41.7-252.2x, 8 cases)
- block-lean, 1 thread: against the reference 39.3x (18.5-111.4x, 8 cases)
- block-index, 1 thread: peak RSS / the reference's 0.22 (0.09-0.42, 8 cases)
- block-lean, 1 thread: peak RSS / the reference's 0.10 (0.06-0.40, 8 cases)

Synthetic workloads (evaluator medians; speedup against the reference of the same run):

| Case | Calls | Reference, 1 thread | block-index, 1 thread | block-index, 4 threads | block-index, 6 threads | block-lean, 1 thread | Peak RSS, 1 thread: reference / speed / memory | Max. rel. error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 1+1+3 | 0.346 s | 0.013 s (27.1x) | 0.013 s (30.9x) | 0.013 s (26.4x) | 0.025 s (16.9x) | 380 MB / 587 MB / 122 MB | 0.0e+00 |
| s2 | 1+1+3 | 2.610 s | 0.057 s (46.0x) | 0.037 s (47.7x) | 0.036 s (46.8x) | 0.068 s (27.6x) | 1.22 GiB / 619 MB / 148 MB | 0.0e+00 |
| s3 | 1+1+3 | 11.32 s | 0.357 s (31.7x) | 0.182 s (69.1x) | 0.154 s (74.2x) | 0.383 s (32.3x) | 2.63 GiB / 765 MB / 244 MB | 0.0e+00 |
| s4 | 1+1+3 | 60.67 s | 1.911 s (31.7x) | 1.008 s (68.6x) | 0.999 s (61.1x) | 1.927 s (33.4x) | 6.35 GiB / 1.40 GiB / 1015 MB | 0.0e+00 |
| s5 | 1+1+3 | 134 s | 4.438 s (30.2x) | 2.344 s (58.5x) | 2.093 s (65.5x) | 4.653 s (29.5x) | 11.07 GiB / 3.75 GiB / 3.77 GiB | 0.0e+00 |

- block-index, 1 thread: against the reference 32.8x (27.1-46.0x, 5 cases)
- block-index, 4 threads: against the reference 52.7x (30.9-69.1x, 5 cases)
- block-index, 6 threads: against the reference 51.7x (26.4-74.2x, 5 cases)
- block-lean, 1 thread: against the reference 27.2x (16.9-33.4x, 5 cases)
- block-index, 1 thread: peak RSS / the reference's 0.44 (0.22-1.55, 5 cases)
- block-lean, 1 thread: peak RSS / the reference's 0.18 (0.09-0.35, 5 cases)

Baselines, 1 thread (compare.py: first call + one measured call; median, and baseline / block-index median):

| Case | block-index | stata-reclink2 |
|---|---:|---:|
| 101 | 0.622 s | 170 s (274x) |
| 102 | 0.175 s | fails checks, 40.84 s |
| 103 | 0.522 s | error |
| 104 | 1.974 s | 512 s (259x) |
| 105 | 0.711 s | fails checks, 214 s |
| 106 | 0.079 s | 31.48 s (398x) |
| 107 | 0.344 s | 94.70 s (275x) |
| s1 | 0.013 s | 2.433 s (190x) |
| s2 | 0.057 s | 16.48 s (290x) |
| s3 | 0.357 s | 217 s (608x) |
- stata-reclink2 102: first_call output: i (shape or dtype mismatch: [5683] against [5685]); j (shape or dtype mismatch: [5683] against [5685]); score (shape or dtype mismatch: [5683] against [5685])
- stata-reclink2 103: SystemError: type mismatch r(109);
- stata-reclink2 105: first_call output: score (max. rel. error 3.3e-06, tolerance 1e-06)
- stata-reclink2 108: not run in this study; run after it: timed out (worker wall time
  3,600 s; addendum at the end)


## Comparisons not made

- **001:**
  - FixedEffectModels.jl (https://github.com/FixedEffects/FixedEffectModels.jl; LSMR, CPU or
    GPU): Julia is not installed on the host.
  - FastHDFE (Larch, Schönfeld & Shikher, 2026; manual
    https://www.usitc.gov/data/gravity/fasthdfe/FastHDFE%20manual.pdf): not installed. Its
    manual describes alternating projections, the algorithm family of pyfixest's default
    demeaner, reghdfe and fixest, which were measured.
- **002 (combined task):**
  - R did on the csdid 1.81-rule workloads: different rules (the not-yet-treated pre-period
    cells, the cohort average's SE). On unbalanced panels it has no pair-balanced estimator.
  - csdid 2.0.1 on 1.81's unnormalized ipw and on drimp: it has neither estimator.
  - csdid2 1.3, fastdid 1.0.6 and moderndid 0.3.1, which were measured on the former task 002
    (`solutions/002/legacy-pre-merge/`): not adapted to the combined task's options for lack of
    time. Their rules differ from both csdid versions in places documented in the package
    README.
  - DiDforBigData and DoubleML: different estimators.
  - Bootstrap outputs of every baseline: computed with the tools' own multipliers, so they can
    only agree statistically. They are timed but not validated (see the baselines README).
- **003:** probabilistic record-linkage libraries (Splink, fastLink, Python `recordlinkage`)
  estimate match weights or use other string comparators, so their outputs cannot match the
  task's pairs and scores. reclink2 is the only exact baseline found.
- **Accelerators:** no GPU or accelerator comparisons; the host has none usable.
- **Coverage:** baselines were not run on every workload (time budget); each baseline table lists
  the cases each one ran. Times for cases not run are not measurements.
- **Thread counts of the baselines:**
  - 001: fixest and pyfixest's within solver at one and six threads; fixest also on 104 at four
    threads (the four-thread trial); reghdfe and pyfixest's alternating projections at one thread;
  - 002 and 003: one thread only.

  The candidates' four-thread figures for 002 and 003 are therefore against the reference only.

## Disclosures and incidents

- **Same agent, earlier candidates.** Earlier in the same conversation this agent built
  candidates for task 001 (`eliminate-largest-fe` and others), task 002 (`shared-nuisance` and
  others) and task 003 (`bigram-table-scan`). Of these, `eliminate-largest-fe` and the task 002
  candidates in `solutions/002/legacy-pre-merge/` are kept with the study's candidates, and
  `shared-nuisance` and `bigram-table-scan` are in `prestudy/solutions/`.
  - Their files were not reopened for this study.
  - Their main ideas were known from that work, as each candidate's `research.md` states:
    eliminating the largest fixed effect exactly; sharing nuisance fits across cells; master
    dictionaries, city blocks and inverted bigram lists.
- **Overwrite incident.** A first version of the moderndid adapter was written to the tracked
  file of a prior candidate (`moderndid`, now in `prestudy/solutions/002/`), overwriting it. It
  was restored with `git checkout` without being read, and the adapter was moved to its own
  directory.
- **Byte-compilation incident (day 2).** A `py_compile` check over
  `solutions/*/*/solution.py` also compiled prior candidates' files. No output was viewed, and the
  31 new `.pyc` files were deleted.
- **Exposure incident (day 2).** After all candidate code was final and evaluated, a `grep`
  over `solutions/*/*/research.md` printed about 25 lines from prior candidates' research notes,
  mostly which development fixtures their smoke runs passed. Searching this conversation's
  transcript for this agent's own earlier notes also surfaced a few lines of earlier task
  documentation. No code was changed as a result. Since then, directories are listed by explicit
  name only (a day-4 listing of `solutions/002` showed the names of the prior candidates'
  directories, not their contents).
- **Package documents read (at the user's direction).** The combined task package's `README.md`,
  its note to the task author and its `code/` files (day 4); earlier, the same files of the 002b
  package and its `stata/run_002b.do`, `stata/export_long.py` and `stata/compare.py`.
  - The READMEs describe the held-out workloads: sizes, designs, options, and estimated reference
    and Stata times. This study used them only to size the freeze timeouts.
  - No held-out data, held-out results, Stata result files or build scripts of the packages were
    opened. The candidates were developed and tuned only on the development and synthetic
    workloads.
- **Held-out metadata.** Row and unit counts of the held-out workloads were read from the tracked
  catalogs (`real_workloads.json`, metadata only) to size the freeze timeouts. No held-out input,
  saved reference answer or final result was read, and no held-out evaluation was run.
- **Adapter fixes before measurement (day 4).** These are listed so that no failed attempt
  is hidden:
  - the csdid 2.0.1 adapter's first two development calls failed on an option-joining error
    (`invalid 'post'`) before any timing;
  - the first smoke run of `schur-lean` was invalidated by the evaluator
    (`runs/20261025T123405.686830Z-0403e0e3`). Numba wrote two new cache entries into the run's
    snapshot, because the cache had been built with writable arrays and the evaluator passes
    read-only ones. The build now passes read-only inputs, and the candidates read their caches
    from a private copy, so the snapshot cannot change.
- **Baseline adapters revised during the study (day 4).** Both were queued after checks on
  only some of the evaluator's development fixtures:
  - **csdid 1.82 (`stata-csdid181`):** its row for 109 failed: 11 event times against the
    reference's 10, and a different pre-treatment average.
    - csdid 1.82's event aggregation under `long2` selects a cohort's pre-treatment cells one
      period late for the window, while its `Tm#` labels are right: `window(-5 5)` returns
      e = -6..-2 (in `csdid_estat.ado` and `csdid_stats.ado`; reproduced on the fixture
      dev-varying-covariates-trim-181).
    - Its bootstrap path had never run. It would have stopped with Stata error r(109) on 108,
      and it took the aggregations' SEs from `estat` after `wboot()`, which reports asymptotic
      SEs.
    - The queue was stopped after 102, before 108 began. The adapter now moves the window by one
      period on the pre-treatment side, and bootstraps with `saverif()` and `csdid_stats`.
    - Every 1.81-rule fixture passes; 109 and 108 were measured again. The rows of 106 and 102
      (varying base, analytic: the same commands before and after) stand, and the failed 109 row
      stays in the results file.
  - **R did (`did-r`), before any measurement:**
    - it kept the universal base's reference cells and e = -1;
    - it skipped did's overlap check, which did 2.5.1 runs only for its built-in estimator names
      (the adapter passes functions, to set DRDID's trimming);
    - it lacked the pre/post averages' bootstrap SEs.

    All three were fixed, and every balanced 2.0.1-rule fixture passes. The planned evaluator
    smoke run of this adapter was replaced by that fixture check, because the smoke suite also
    holds fixtures the adapter refuses by design (1.81 rules, unbalanced panels).
- **Degenerate inputs the reference cannot define.** In random small panels (fuzz seed 7276), a
  cell's influence function was constant, so its bootstrap SE was rounding residue (3e-16). The
  reference's simultaneous critical value then depends on that rounding noise, and no candidate
  can reproduce it. The candidates raise the reference's own error when no draw of the maximum is
  finite. None of the task's workloads has such a cell.
- **Former task 002 finding, superseded:** the former development workload 108 had a separated
  cohort (cohort 25, 5 units, where the reference's logit hit its iteration cap with a condition
  number of 1.7e20). That workload is not part of the combined task.

- **Host sleep and throttling, days 2-3.** The pmset log shows no system sleep from the evening
  before day 1 until day 2, 13:23:41. Then forced sleeps ("Low Power Sleep",
  despite keep-awake assertions) occurred at:
  - 13:23-16:49 and 17:07-17:33, during six-thread fixest runs;
  - 18:02-18:26, with the host idle;
  - 19:57-21:33, during a six-thread fixest run whose window shows the speed limited to 52-60%
    within four minutes;
  - 22:35-22:50, during development checks (no measurement running).

  Every measurement that overlapped a sleep is excluded (`runs/baselines-20261023/exclusions.json`
  and `exclusions.jsonl`). From day 2, 18:36, all measurements ran through
  `thermal_queue.py`:
  - a cool-down gate before each case (a speed limit of 100 and a thermal level of 90 or less for
    60 s);
  - a monitor of the speed limit, the thermal level and sleeps;
  - per-case statistics (`thermal-cases.jsonl`);
  - repetition of attempts that overlap a sleep.

  Throttling is recorded, not retried; it was retried until day 3, 22:38, which cost about
  8-10 h.
- **Day 4:**
  - another forced sleep, 00:12:34-00:22:35, during the six-thread fixest 001/104 attempt; the
    agent stopped that queue at 00:26 and abandoned the remaining six-thread baseline follow-ups;
  - the four-thread trial (see "Host");
  - from 06:11, the study queue (`runs/study-20261025/plan-1.json`, then `plan-2.json`), with no
    sleep. The power log shows keyboard activity on the host at 10:37-11:12, during the lean 003
    runs (108 and synthetic); no other compute job was started.
  - Most of the day's cases were throttled, including at one thread: 12 of the first 15 queue
    steps were flagged severe, with lowest speed limits of 50-73% (`thermal-cases.jsonl`). Every
    speedup compares the reference and the candidate within the same evaluator run; absolute
    times from different hours are not directly comparable. The reference's 002/104 took 30.4 s
    in the saved-reference run (05:50) and 67-120 s in the evaluator runs.
  - The follow-up queue's nine four-thread evaluator runs (block-index and schur-cg, 18:33-22:59)
    had no sleep. But 38-90% of their five-second samples were throttled, with lowest speed
    limits of 50-76% (`thermal-cases.jsonl`). The four-thread figures are therefore
    conservative for a host without this limitation.
- **Queue interruptions** (all logged in the queue logs):
  - the user stopped the baseline queue on day 2 at 17:41;
  - the agent stopped the thermal queue at 19:31 (to change its retry policy) and at 21:37 (to
    defer the six-thread loads), on day 3 at 22:38 (task 002 changed), and on day 4 at 00:04 and
    00:26 (the combined task 002 replaced 002 and 002b; then the sleep above).
  - The runs interrupted then are listed in the logs and not used.
  - On day 4 at 17:21, the agent stopped `plan-1.json` between two cases, after csdid
    1.82's 102 and before its 108 started, to revise that adapter (Disclosures). No case was
    interrupted.
    - `plan-2.json` was then edited: csdid 1.82's 109 and 108 were put first, and the R did
      smoke evaluation was replaced by the fixture check.
    - The plan before the edit is kept at `runs/study-20261025/plan-2-before-edit-2026-10-25.json`.
    - `plan-2.json` started at 17:28.
- **Unequal call counts.** Baselines had one measured call per case; candidates five (real) or
  three (synthetic), except the heavy cases. Baseline outputs were validated against saved
  one-call reference outputs rather than against a reference recomputed in the same invocation.
- **Spot checks of earlier measurements** (day 2, evening) reproduced them within about 6%,
  except two six-thread figures: pyfixest-within on 001/102 (12.8 s against 14.9 s) and the
  former task 002's reference on 102 (9.3 s against 14.2 s). Six-thread figures measured during
  the night of day 1 may therefore be inflated by host load or throttling.

## Reproduction

From the repository root, with nothing else running on the host:

```bash
# environments: solutions/.venv (candidates, reference) and solutions/baselines/.venv (baselines)
uv venv solutions/.venv --python /usr/local/bin/python3.12
uv pip install --python solutions/.venv/bin/python -r solutions/002/cell-fused/requirements.lock
solutions/.venv/bin/python solutions/001/schur-lean/build_cache.py      # memory candidates' Numba caches
solutions/.venv/bin/python solutions/003/block-lean/build_cache.py

# candidates (one invocation per line of the queue plans; examples)
PY=solutions/.venv/bin/python
$PY -m harness evaluate --task 002 --solution solutions/002/cell-fused/solution.json --smoke --threads 1
$PY -m harness evaluate --task 002 --solution solutions/002/cell-fused/solution.json --threads 1 \
  --suite real-development --ids 101 102 103 104 105 106 107 108 109
$PY -m harness evaluate --task 002 --solution solutions/002/cell-lean/solution.json --threads 4 --ids 1 2 3 4 5 6
$PY -m harness evaluate --task 001 --solution solutions/001/schur-lean/solution.json --threads 1 \
  --suite real-development --ids 104 --n-regressors 50 --warmups 0 --repeats 1 --timeout 7200

# saved reference outputs and baselines
BPY=solutions/baselines/.venv/py/bin/python
for c in 101 102 103 104 105 106 107 108 109 s1 s2 s3 s4 s5 s6; do
  $PY solutions/baselines/make_refs.py 002 $c REFS/002; done
$BPY solutions/baselines/compare.py --task 002 --solution solutions/002/stata-csdid201/solution.json \
  --cases 103 --threads 1 --warmups 0 --repeats 1 --refdir REFS/002 --out results.jsonl \
  --python $BPY --not-comparable se_boot_ crit_

# a task 002 baseline adapter on the evaluator's development fixtures (reference and validator)
OMP_NUM_THREADS=1 $BPY solutions/baselines/check_adapter.py solutions/002/stata-csdid181/solution.py
OMP_NUM_THREADS=1 $BPY solutions/baselines/check_adapter.py solutions/002/did-r/solution.py \
  dev-notyet-universal-bootstrap-201 dev-reg-universal-event-sets-201 dev-notyet-varying-bootstrap-201 dev-overlap-201

# the whole sequence, with the thermal gate and monitor (macOS):
caffeinate -i -s $BPY solutions/baselines/thermal_queue.py runs/study-20261025/plan-1.json runs/study-20261025
caffeinate -i -s $BPY solutions/baselines/thermal_queue.py runs/study-20261025/plan-2.json runs/study-20261025

# tables of this report
$PY solutions/baselines/report_tables.py runs/study-20261025/report-config.json

# randomized checks of the task 002 candidates against the reference
$PY solutions/002/cell-fused/fuzz.py solutions/002/cell-fused/solution.py 1000 7000
$PY solutions/002/cell-lean/fuzz.py solutions/002/cell-lean/solution.py 1000 7000
```

`python -m harness replay --run RUN_DIR --out NEW_DIR` repeats any run listed below with its
recorded cases and settings (after installing the recorded environment).

## Run directories

Evaluator runs (`runs/`, git-ignored). Each holds `run.json`, `traces.jsonl` and `summary.json`,
and its snapshot of the evaluator, task and candidate.

| Task | Candidate | Threads | Real-development | Synthetic | Smoke / compatibility |
|---|---|---:|---|---|---|
| 001 | schur-cg | 1 | 101-103, 106, 107: `20261022T215434.426602Z-5729349d`; 104 (20 regressors): `20261022T223311.828997Z-d8b691c5`; 104 (5): `20261022T225246.061204Z-df2f14c7`; 104 (50): `20261022T230015.850521Z-fa445d0e`; 105: `20261022T234408.288172Z-32ec7e50`; 108: `20261023T000948.005315Z-eba532c2` | 1, 3, 4, 6-10, 12: `20261023T003150.988899Z-dee49978`; 11: `20261023T010358.482359Z-a17b6a64` | 109: `20261022T223125.550138Z-03968a2c` |
| 001 | schur-cg | 4 | 101-103, 106, 107: `20261026T031542.571084Z-d2779b4e`; 104 (20): `20261026T035408.896006Z-4531cec1`; 104 (5): `20261026T041515.937996Z-897e2553`; 104 (50): `20261026T042411.702249Z-26959f2a`; 105: `20261026T051019.926857Z-a369e3de`; 108: `20261026T053721.762823Z-ebd9b900` | not run | |
| 001 | schur-cg | 6 | 101-103, 106, 107: `20261023T052206.604804Z-6ba4680c`; 104 (20): `20261023T060022.670819Z-0e69e8e3`; 104 (5): `20261023T061949.908034Z-298fe313`; 104 (50): `20261023T062715.401175Z-cd3725b2`; 105: `20261023T071051.912648Z-43752d65`; 108: `20261023T073600.903087Z-286e661a` | 1, 3, 4, 6-10, 12: `20261023T075746.376884Z-4268a286`; 11: `20261023T082949.896955Z-c8bee552` | 109: `20261023T055833.910601Z-54235ec6` |
| 001 | schur-lean | 1 | 101-103, 106, 107: `20261025T181953.890783Z-7c2f8ea2`; 104 (20): `20261025T185642.936804Z-9e839d4a`; 104 (5): `20261025T191800.237299Z-b97583c5`; 104 (50): `20261025T192624.376201Z-94bbf764`; 105: `20261025T201428.660356Z-55c1b674`; 108: `20261025T204408.349289Z-ca408a1f` | 1, 3, 4, 6-10, 12: `20261025T210938.331957Z-5ca4a476`; 11: `20261025T214343.671021Z-325b5c8a` | smoke `20261025T171320.187251Z-6ba1e490`; 109 `20261025T185612.408476Z-9f0d2212` |
| 002 | cell-fused | 1 | `20261025T132222.994847Z-325195eb` | `20261025T140015.833287Z-603526de` | smoke `20261025T131159.843213Z-7754c15c` |
| 002 | cell-fused | 4 | `20261025T151318.614984Z-5e42de86` | `20261025T155317.049574Z-e43365d8` | |
| 002 | cell-lean | 1 | `20261025T143021.988145Z-13a88526` | `20261025T145326.956088Z-012394ea` | smoke `20261025T132212.867772Z-17799882` |
| 002 | cell-lean | 4 | `20261025T162327.654759Z-b5659a9b` | `20261025T165253.800275Z-80a438ae` | |
| 003 | block-index | 1 | 101-107: `20261022T185556.024324Z-9dbc9db2`; 108: `20261022T193346.854856Z-4d4ad68b` | `20261022T195218.141722Z-2a67bacf` | |
| 003 | block-index | 4 | 101-107: `20261026T013348.923342Z-fa22210e`; 108: `20261026T021823.863061Z-e034ca34` | `20261026T024122.997042Z-086adcdf` | |
| 003 | block-index | 6 | 101-107: `20261023T021224.552881Z-1a3901b4`; 108: `20261023T025011.664857Z-d2ece329` | `20261023T030823.672227Z-6db4e013` | |
| 003 | block-lean | 1 | 101-107: `20261025T171340.478397Z-bcb2f9c7`; 108: `20261025T174042.214822Z-00c1645d` | `20261025T175621.771892Z-c03f05a8` | smoke `20261025T171329.897429Z-8ab87674` |

**Other records:**
- **Baselines (compare.py rows):**
  - `runs/baselines-20261023/` (001 and 003, days 1-3, with `exclusions.json(l)`,
    `thermal-cases.jsonl`, `queue.log`);
  - `runs/study-20261025/` (002, day 4: `002-stata-csdid201-t1.jsonl`,
    `002-stata-csdid181-t1.jsonl`, `002-did-r-t1.jsonl`);
  - `runs/baselines-20261031-reghdfe104/` (001: reghdfe on 104k5 and 104k50, after the study; see the
    addendum);
  - `runs/baselines-20261031-reclink2-108/` (003: reclink2 on 108, after the study; see the addendum).
- **Queue plans and logs:** `runs/study-20261025/` (`plan-1.json`, `plan-2.json`, `queue.log`,
  `thermal.log`, `thermal-cases.jsonl`, `report-config.json`).
- **Saved reference outputs** for the baselines' validation (`make_refs.py`, outside the
  repository, in this session's scratch directory: `.../scratchpad/s22/t002n/ref`, `.../t001/ref`,
  `.../t003/ref`). They are not kept; `make_refs.py` regenerates them (one reference call per
  case, one BLAS thread).
- **Superseded:** the former tasks 002 and 002b's evaluator runs and baseline files
  (`runs/baselines-002v2/`, `runs/baselines-002b/`,
  `runs/baselines-20261023/task002-before-correction/`).

## Final candidates and freeze commands

**Selected configurations (three final rounds per task, within AGENTS.md's two-to-three):**

| Task | Speed, 1 thread | Speed, 4 threads | Minimal peak memory, 1 thread |
|---|---|---|---|
| 001 | `solutions/001/schur-cg` | `solutions/001/schur-cg` | `solutions/001/schur-lean` |
| 002 | `solutions/002/cell-fused` | `solutions/002/cell-fused` | `solutions/002/cell-lean` |
| 003 | `solutions/003/block-index` | `solutions/003/block-index` | `solutions/003/block-lean` |

**Why four threads for the multicore round:** on this host, six-thread loads triggered forced
sleeps and cut the CPU speed limit to 52-60% within minutes. A four-thread trial (fixest-r on
001/104, day 4) had no sleep but was throttled in 105 of 143 samples (lowest limit 56%). Four
threads is therefore the less affected multicore configuration here, not a clean one. On a host
without this limitation, use its physical core count and record it.

**Not run:** no freeze exists and no held-out input was read or run.

**Before freezing:**
- **Commit:** the `freeze` record stores the Git commit and whether the tree is dirty, and
  snapshots the declared sources and artifacts with their SHA-256 hashes. The candidate
  directories are untracked at the time of writing, so commit them first if the freeze should
  point at a clean commit.
- **Environment:** `final` runs the frozen evaluator with the interpreter that runs `harness`, so
  use the candidates' environment (`solutions/.venv`, built as in each candidate's `build.md`) for
  both commands.
- **Directories:** freeze and run directories must be outside the repository.
- **One at a time:** run the `final` commands one at a time, with nothing else running on the
  host.

**Task 002 (the combined task) needs the maintainer first:** put the package's held-out files
(301, 302, 401-404 and their manifests) under `~/SocialSciBench-private/002/data/heldout/` and fill
in their sizes and hashes in `002_csdid/real_workloads.json` (the commands are in
`002_csdid/DATA.md`). Until then, `final` stops at its file check before any round is consumed.

**Memory candidates (schur-lean, block-lean):** rebuild their Numba caches after any checkout or
copy that changes their kernel files' times, and before freezing:

```bash
solutions/.venv/bin/python solutions/001/schur-lean/build_cache.py
solutions/.venv/bin/python solutions/003/block-lean/build_cache.py
```

Each script ends with "cache complete" and updates the candidate's solution.json. The freeze copies
the cache files with their times (`shutil.copy2`). A stale cache costs the first call its compile
time and the measured peak about 0.2-0.5 GB more; the results are unchanged.

```bash
cd /path/to/SocialSciBench
PY=solutions/.venv/bin/python
F=~/SocialSciBench-private/freezes

$PY -m harness freeze --task 001 --solution solutions/001/schur-cg/solution.json \
  --out $F/001-schur-cg-t1 --rounds 1 --threads 1 --warmups 1 --repeats 5 --timeout 43200
$PY -m harness freeze --task 001 --solution solutions/001/schur-cg/solution.json \
  --out $F/001-schur-cg-t4 --rounds 1 --threads 4 --warmups 1 --repeats 5 --timeout 43200
$PY -m harness freeze --task 001 --solution solutions/001/schur-lean/solution.json \
  --out $F/001-schur-lean-t1 --rounds 1 --threads 1 --warmups 1 --repeats 5 --timeout 43200

$PY -m harness freeze --task 002 --solution solutions/002/cell-fused/solution.json \
  --out $F/002-cell-fused-t1 --rounds 1 --threads 1 --warmups 1 --repeats 5 --timeout 7200
$PY -m harness freeze --task 002 --solution solutions/002/cell-fused/solution.json \
  --out $F/002-cell-fused-t4 --rounds 1 --threads 4 --warmups 1 --repeats 5 --timeout 7200
$PY -m harness freeze --task 002 --solution solutions/002/cell-lean/solution.json \
  --out $F/002-cell-lean-t1 --rounds 1 --threads 1 --warmups 1 --repeats 5 --timeout 7200

$PY -m harness freeze --task 003 --solution solutions/003/block-index/solution.json \
  --out $F/003-block-index-t1 --rounds 1 --threads 1 --warmups 1 --repeats 5 --timeout 7200
$PY -m harness freeze --task 003 --solution solutions/003/block-index/solution.json \
  --out $F/003-block-index-t4 --rounds 1 --threads 4 --warmups 1 --repeats 5 --timeout 7200
$PY -m harness freeze --task 003 --solution solutions/003/block-lean/solution.json \
  --out $F/003-block-lean-t1 --rounds 1 --threads 1 --warmups 1 --repeats 5 --timeout 7200
```

Maintainer, one round per configuration:

```bash
P=~/SocialSciBench-private
for c in 001-schur-cg-t1 001-schur-cg-t4 001-schur-lean-t1; do
  $PY -m harness final --freeze $F/$c --data-root $P/001/data --out $P/runs/001 --note "$c"; done
for c in 002-cell-fused-t1 002-cell-fused-t4 002-cell-lean-t1; do
  $PY -m harness final --freeze $F/$c --data-root $P/002/data --out $P/runs/002 --note "$c"; done
for c in 003-block-index-t1 003-block-index-t4 003-block-lean-t1; do
  $PY -m harness final --freeze $F/$c --data-root $P/003/data --out $P/runs/003 --note "$c"; done
```

**Settings:** the evaluator's defaults for frozen final evaluation (a first call, one warmup and
five measured calls for the reference and the candidate) on every held-out workload. The
timeouts bound each worker (start-up, all seven calls and serialization). They are estimates,
sized from development reference times and the held-out sizes in the tracked catalogs and the
package README (metadata only):
- **001:** the largest held-out input (206: 14.65 million rows, twice 108's) should take its
  reference worker roughly 2-6 h; 43,200 s leaves at least a factor of 2.
- **002:** the largest held-out input is the headline 402 (88,835 units x 60 periods, unbalanced,
  999 draws). The package's author timed the reference at 12 s on an Apple M3; at the development
  host's one-thread rate (107: 56.5 s here against 20 s there) that is about 35 s per call, under
  5 minutes for the worker. 7,200 s leaves a wide margin. 401 and 404 draw 25,000 times for 1,514
  units (seconds).
- **003:** the largest held-out input (205: 5,000 master x 5.34 million using records) should take
  its reference worker about 20-25 min.
- **Memory:** 108-sized inputs needed 10.1 GiB (001) and 22.9 GiB (003) for the reference worker
  on the 32 GiB development host. 001/206 may need about twice 108's memory for the reference.

## Addendum: reghdfe on 104 with 5 and 50 regressors

The study did not run Stata reghdfe on 104k5 and 104k50: its queue (`queue-baselines.sh`) lists only the
real workloads 101-108 for it, because of the time budget. Both cases were run after the study, for Table 2
of the Stage 1 proposal, on the same host with the study's adapter and settings:
`solutions/001/stata-reghdfe` (Stata 18 MP, reghdfe 6.14.1, ftools 2.50.0), one thread, `compare.py` with a
first call and one measured call, a 3,600 s limit, and saved references from `make_refs.py` (one reference
call per case, one BLAS thread).

**One change to the queue.** `thermal_queue.py` waits before each case until the speed limit has been 100
and the thermal level at most 90 for 60 s. The idle host now reads level 100 (33 during the study), with
`/usr/libexec/audiomxd` holding one core at 100% CPU since about day 1, so that gate cannot open. The
runs used a copy of the queue (`thermal_queue_gate100.py`, in the run directory) whose gate accepts level
100. It still waits for 60 s of unthrottled samples; the monitor and the repetition of attempts that
overlap a sleep are unchanged.

**Conditions.** Both cases ran throttled nearly throughout: the speed limit was below 100 in 237 of 244 and
714 of 716 samples, with lows of 63% and 56%, and the host did not sleep. The study's long one-thread cases
mostly ran at 71-80%. The host was also slower before the cases: `make_refs.py` took 212 s on 104k5 and
1,600 s on 104k50, against the reference's 169 s and 1,225 s in the study's evaluator runs (25% and 31%
longer).

| Case | stata-reghdfe, 1 thread | schur-cg, 1 thread (study) | reghdfe / schur-cg |
|---|---:|---:|---:|
| 104k5 | 607 s (first call 608 s); passes the checks | 8.144 s | 75x |
| 104k50 | timed out (worker wall time 3,600 s) | 20.78 s | - |

- **104k5:** the ratio compares this session's reghdfe with the study's schur-cg. Adjusted for the slower
  host with the reference of the same session (reghdfe took 2.87 times the reference's time, and schur-cg
  was 20.7 times faster than the reference in the study), it is about 59x. Either value lies within the
  study's 15-375x.
- **104k50:** the reference takes 7.2 times as long as on 104k5 (study: 1,225 against 169 s). At that ratio
  a reghdfe call would take an estimated 3,300-4,400 s with or without the slowdown measured above, and
  the protocol makes two calls, so the timeout does not depend on the throttling.

**Result.** Stata reghdfe now has a measurement on all ten real cases: 15-375x on six (101, 102, 103,
104k5, 106, 107) and no result within 3,600 s on four (104k20, 104k50, 105, 108).

**Records:** `runs/baselines-20261031-reghdfe104/` (git-ignored): `plan.json`, `thermal_queue_gate100.py`,
`queue.log`, `thermal.log`, `thermal-cases.jsonl`, `001-stata-reghdfe-t1.jsonl`, and the saved references
in `refs/001/`.

## Addendum: reclink2 on 003/108

The study did not run Stata reclink2 on 108 (5,000 second-draw PPP borrowers nationwide against all
8,612,370 first-draw borrowers): its queue lists only 101-103 and 105-107, with 104 added in
`followup.sh`, because of the time budget. 108 was run after the study, following the reghdfe runs above, for
Table 2 of the Stage 1 proposal, on the same host with the study's adapter and settings:
`solutions/003/stata-reclink2` (Stata 18 MP, reclink2 from the Stata Journal, dm0082), one thread,
`compare.py` with a first call and one measured call, a 3,600 s limit, and a saved reference from
`make_refs.py`. The queue was the same copy with the thermal gate at level 100 (see the reghdfe addendum).

**Result:** timed out (worker wall time 3,600 s); block-index took 6.361 s on 108 in the study, so
reclink2 is at least 566 times slower there. The worker's files are temporary, so whether the first call
finished within the hour is not recorded. On the other real workloads reclink2 took 3.0-7.6 times the
reference's time; with the reference at 402 s in this session, that suggests about 20-51 min per call, and
the protocol makes two.

**Conditions:** throttled nearly throughout (speed limit below 100 in 707 of 716 samples, lowest 56%; no
sleep). `make_refs.py` took 402 s on 108, against the reference's 276 s in the study's evaluator run (46%
longer), with a peak of 16.4 GiB.

**Result for reclink2 across the real workloads:** 259-398x on the four where it agrees with the reference
(101, 104, 106, 107), outside the tolerance on two (102, 105), a Stata error on one (103) and no result
within 3,600 s on one (108).

**Records:** `runs/baselines-20261031-reclink2-108/` (git-ignored): `plan.json`, `thermal_queue_gate100.py`,
`queue.log`, `thermal.log`, `thermal-cases.jsonl`, `003-stata-reclink2-t1.jsonl`, and the saved reference
in `refs/003/`.
