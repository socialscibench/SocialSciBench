# Speed and peak-memory optimizations for tasks 001, 002 and 003

A summary of the optimizations in the six CPU candidates of the development study, written as
source material for an academic paper. For each task there is one candidate optimized for speed
and one for minimal peak memory. The full record is in the study report
(`solutions/baselines/development-study.md`) and in each candidate's `research.md`.

**Three kinds of figures appear below:**
- **Evaluator measurements:** `python -m harness evaluate` runs. The reference and the candidate
  run in fresh processes on the same inputs with the same thread budget, and every candidate output
  is validated. All results tables and final speedups are of this kind.
- **Development measurements:** single calls after a first call, from development scripts. They
  attribute gains to individual changes (the version tables). They were taken over several days
  on a host that throttled, so their factors are approximate.
- **Estimates:** marked as such where they occur.

## 1. Setting

| Task | Computation | Reference implementation | Accuracy required | Scored development workloads |
|---|---|---|---|---|
| 001 | OLS with several high-dimensional fixed effects and one-way clustered standard errors, as Stata `reghdfe` | singleton removal, then each column partialled out separately with SciPy's LSMR | β 1e-6 and SEs 1e-4 (relative); counts exact | 8 real (104 at 5, 20 and 50 regressors, so 10 cases); 10 synthetic |
| 002 | Callaway & Sant'Anna staggered difference-in-differences as Stata `csdid` (1.81 or 2.0.1 rules): all options, aggregations and the multiplier bootstrap with given ±1 multipliers | one small NumPy problem per (cohort, period) cell; two n_units x n_cells arrays; a dense product with the multipliers | estimates 1e-6 of max(\|reference\|, its SE); SEs, bootstrap SEs and critical values 1e-4 (relative); covariance 1e-4 of SE_i SE_j | 9 real; 6 synthetic |
| 003 | Record linkage as Stata `reclink2`: perfect matches, or-blocking, the bigram comparator, float32 rounding, every tied best candidate | pandas joins for blocking, sparse bigram matrices for the comparator | pairs and perfect-match count exact; scores 1e-6 (relative) | 8 real; 5 synthetic |

**Measurement setup:**
- **Host:** Intel Core i7-8700B (6 cores), 32 GiB, macOS 15.7.3. CPython 3.12.13, NumPy 2.3.5
  (Apple Accelerate), Numba 0.62.1, SciPy 1.18.1.
- **Calls per case:** a first call, one warmup and five measured calls (real) or three
  (synthetic). When the reference's call exceeds 150 s, the reference and the candidate both get
  one measured call.
- **Speedup:** the reference's median divided by the candidate's, within one evaluator run. Each
  summary is the geometric mean over cases (range, number of cases).
- **Memory:** the worker process's peak resident set (`ru_maxrss`). It includes the interpreter,
  imports, input loading, every call and output serialization. The evaluator's input loaders
  therefore set a floor that no candidate can go below.
- **Thread budgets:** one thread, and four threads as the multicore configuration. Six-thread
  loads made the host sleep during baseline runs; the six-thread candidate runs, measured before
  that, are also reported. The evaluator sets the BLAS, OpenMP and Numba thread counts.
- **Timing:** the first call (which includes JIT compilation) is reported separately and excluded
  from the medians.

## 2. Results

| Task | Candidate | Threads | Real workloads | Synthetic workloads | Peak RSS / reference's, 1 thread (real; synthetic) |
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

Every candidate output (first call, warmup and measured calls) passed the task's validator.
- **Largest errors:** 001: 4.1e-7 in β and 1.8e-6 relative in the SEs. 002: 1.1e-10. 003: none;
  the scores are bitwise equal to the reference's on every call.
- **Absolute times:** they range from milliseconds to minutes. Examples, one thread, reference
  against speed candidate:
  - 001/104 with 50 regressors: 1,225 s against 20.8 s;
  - 002/107 (198,649 units, 999 bootstrap draws): 61.2 s against 3.31 s;
  - 003/108 (an 8.6-million-row register): 276 s against 6.36 s.

**The memory candidates against the speed candidates** (one thread; per-case ratios,
`solutions/baselines/ratios.py`):

| Task | Peak, speed / memory candidate (real; synthetic) | Peak, reference / memory candidate (real; synthetic) | Time, memory / speed candidate (real; synthetic) |
|---|---|---|---|
| 001 | 1.32-1.94x (25-49% lower); 1.36-1.97x | 1.18-2.02x; 0.73-2.46x | 1.17-1.79x; 1.32-2.28x |
| 002 | 2.11-5.33x; 2.52-9.75x | 2.39-10.8x; 1.02-57.3x | 2.45-6.82x; 2.75-5.60x |
| 003 | 1.05-3.26x (1.51-3.26x on 101-107); 1.00-4.83x | 2.48-17.3x; 2.82-11.1x | 0.95-1.36x; 1.01-1.99x |

The speed and memory candidates of 001 and 003 were measured on different days, so their time
ratios include differences in throttling. Both 002 candidates were measured on the same day.

## 3. Speed

### 3.1 What produced the gains

The gains came from four sources. Changes to the algorithm gave by far the most in all three
tasks; low-level engineering and threads each added a factor of about 1-3:

1. **Restructuring the computation** to remove redundant work gave the first order of magnitude
   (development measurements, one thread):
   - 001: exact elimination of the largest fixed effect, then one conjugate-gradient solve of the
     reduced system for all columns, gave 4.0-11.2x;
   - 003: dictionaries, blocking keys and inverted bigram lists in place of joins and sparse
     matrices gave 13-28x;
   - 002: per-cohort fits shared across cells and one fused pass over units gave 20-37x. This was
     measured on the former task 002, the predecessor of the combined task; see 3.3.
2. **A better numerical method:** for 001, an approximate-Cholesky preconditioner cut the
   iterations from 85-232 to 18-29, a further 1.2-9.2x.
3. **Low-level engineering** gave a further 1.05-3.2x:
   - fusing passes and not storing intermediates;
   - cache-friendly orders and blocking;
   - making inner loops vectorize while keeping the reference's floating-point semantics.
4. **Threads** came last: four threads gave 1.2-3.0x over one on the real workloads. Serial phases
   and memory-bound kernels limit the gain.

**Exactness shaped several choices:**
- an approximation is used only where an exact step corrects it (003's rescoring near the best);
- rare edge cases fall back to the reference's own operations (002);
- tolerances were set where the remaining error is the reference's own (001's conjugate gradients);
- lower precision was rejected where it would leave too little margin (002's bootstrap).

### 3.2 Task 001: schur-cg

**Method:**
1. **Singletons and levels:** rows alone in a level of a fixed effect are peeled to a fixed point,
   and levels are recoded densely.
2. **Redundant fixed effects:** a fixed effect whose levels are unions of another's (year, given
   subclass x year) is skipped. Its indicators are sums of the other's, so the projection is
   unchanged, but it still counts in df_a.
3. **Exact elimination:** the fixed effect with the most levels (the worker, inventor or pair) is
   eliminated exactly. Given the other effects, its coefficients are group means.
4. **The reduced system** is the Schur complement S = A_oo - N_ob C_b^-1 N_bo over the remaining
   effects' levels. Preconditioned conjugate gradients solve it for y and all regressors at once.
5. **Operator:** S is assembled when a bound on its entries is at most twice the cost of applying
   it implicitly. Otherwise it is applied as one fused pass over the eliminated levels:
   - C_b^-1 N_bo P is never stored;
   - the levels run in order of their smallest other level, for locality;
   - each thread scatters into its own buffer.

   Systems of at most 4,000 levels are solved densely, and effects of at most 512 levels go
   through BLAS.
6. **Preconditioner,** block-diagonal by fixed effect:
   - an exact pseudo-inverse for blocks of at most 2,000 levels;
   - a randomized approximate Cholesky factorization of the block's Laplacian for large sparse
     blocks (Kyng & Sachdeva 2016; the AC variant of Gao, Kyng & Spielman 2023), with vertices
     eliminated by increasing degree and solves 8 columns at a time (one cache line per row);
   - otherwise the diagonal.
7. **Stopping rule:** conjugate gradients stop at a relative residual of 1e-9 per column.
8. **The rest** follows the reference's definitions: W'W by BLAS, the cluster scores, the CRV1
   sandwich and reghdfe's df_a rules.

**Attribution** (development measurements, one thread, seconds; factor against the previous row):

| Version | 101 | 102 | 103 | 107 | s9 |
|---|---:|---:|---:|---:|---:|
| Reference | 27.1 | 95.7 | 115.0 | 21.0 | 120.2 |
| 1. Exact elimination + conjugate gradients on S, Jacobi preconditioner | 4.06 (6.7x) | 14.6 (6.6x) | 13.1 (8.8x) | 5.24 (4.0x) | 10.7 (11.2x) |
| 2. + approximate-Cholesky block preconditioner, assembled S | 0.44 (9.2x) | 2.47 (5.9x) | 2.97 (4.4x) | 4.22 (1.2x) | 1.23 (8.7x) |
| 3-4. + cost model, dense small blocks, no copies; redundant effects skipped, fused operator, 8-column solves, compact peel counts | 0.42 (1.05x) | 2.20 (1.12x) | 2.65 (1.12x) | 3.59 (1.18x) | 0.68 (1.81x) |
| 4 at six threads | 0.31 (1.35x) | 1.29 (1.71x) | 1.54 (1.72x) | 1.69 (2.12x) | 0.54 (1.26x) |

On the largest workloads version 1 was not run:
- 104: reference 714 s; version 2 16.2 s; version 4 12.6 s;
- 108: reference 613 s; version 2 26.2 s; version 4 14.1 s;
- 105: reference 811 s; version 3 44.3 s; version 4 25.3 s.

**Component measurements** (development):
- **Redundant fixed effect skipped:** 105 (year nested in subclass x year): 44.3 to 33.6 s, with
  the same coefficients (2.4e-14 relative). Synthetic 7: 2.41 to 1.78 s.
- **Fused implicit operator** (105, one product with 21 columns):
  - one thread: 487 ms with three sparse products through a stored 892,132 x 21 intermediate,
    386 ms fused, and 276 ms with the locality order;
  - six threads: 317 to 114 ms.
- **Approximate-Cholesky solves in 8-column chunks** (104's firm block: 136,563 levels, 1.70
  million factor entries):
  - 21 columns: 100 to 78 ms per solve at one thread, and 67 to 30 ms at six;
  - 51 columns: 206 to 189 ms at one thread, and 131 to 74 ms at six;
  - synthetic 11, with a very sparse factor, became 2.7% slower; this was accepted.
- **Compact int32 level counts** in the singleton peel: 104's peel went from 0.645 to 0.456 s.
- **Iterations** at tolerance 1e-9 on the worker/firm graphs: Jacobi 85-232, approximate Cholesky
  18-29, and exact block solves 8-20 (too costly to factorize; see section 5).

**Accuracy:** tolerances of 1e-8, 1e-9 and 1e-11 give the same errors against the reference. The
errors (up to 4.1e-7 in β) are therefore the reference's own LSMR error, not the candidate's.

**Threads:** six threads give 1.4-2.3x over one on the real workloads and 1.2-1.7x on the
synthetic ones; four threads reach 89-96% of six threads' speed. Two phases stay serial:
- the singleton peel: a round-based parallel version took 7.3 s against 0.7 s;
- W'W: Accelerate's dgemm runs on one thread for 6.76 million x 21.

**First call:** 11.5-46.1 s, which includes the Numba compilation.

### 3.3 Task 002: cell-fused

**Method,** in four passes:
1. **Cells and sizes:** the cells in csdid's order, under either rule set. Cells observed for
   every unit are counted from cohort tallies, without a pass over units.
2. **Estimation,** in parallel over cohorts:
   - Newton fits with the reference's stopping rule (logit, inverse probability tilting, and
     csdid 2.0.1's unweighted overlap logit), then weighted least squares;
   - consecutive cells with the same units and covariates share the fits and the sums that do
     not depend on the outcome change, so each further cell costs one pass for 2 + 2k sums;
   - other fits start from the previous cell's solution.
3. **One fused pass over units,** in parallel over 64-unit blocks of one cohort. Each unit adds its
   influence-function value to every cell it belongs to and, through the aggregation weights, to
   every aggregate. This includes the terms for the estimated cohort weights.
   - Accumulators are block-local and column-major.
   - The scores of consecutive cells with the same fit are reused.
4. **Bootstrap:**
   - each cohort's cell influence functions times its multipliers, with BLAS, over the cells the
     cohort belongs to only (member-aware products);
   - the aggregate tables' draws are linear combinations of the cell draws plus the cohort-weight
     terms, so no aggregate column enters the product;
   - order statistics by quickselect instead of sorting.

**Exactness safeguards:** a cell is recomputed with the reference's own operations when:
- its fit is singular or does not converge;
- a score lies within 1e-9 of the trimming or overlap cut;
- a warm-started overlap fit is extreme.

A cell whose SE is at rounding level relative to its influence function is also computed with
the reference's own operations.

**Attribution:**
- **Restructuring** was measured on the former task 002, the predecessor of the combined task,
  with different workloads (candidate `cohort-fused`, one thread). Per-cohort logit and least
  squares with a fused pass took:
  - 105: 113.9 to 3.07 s (37x);
  - 103: 26.8 to 1.37 s (20x);
  - 107: 45.6 to 1.54 s (30x).
- **Making the Numba kernels vectorize** (same lineage, former tasks):
  - Numba's `error_model="numpy"` stops checking every division, which had kept the exp and
    division loop scalar (25 to 12 ns per unit and iteration). Former 105: 3.07 to 1.06 s (2.9x).
  - One-dimensional row views in the Newton kernels: the Hessian block kernel went from 60 to
    3 µs (20x) and one fit from 114 to 15 ms (former task 002b).
  - Vectorizing across 64-unit blocks: the fused pass went from about 4 to about 1 ns per
    unit-cell.
  - A branch-free exp (Cody-Waite reduction and a degree-13 polynomial, within 1 ulp of libm)
    vectorizes.
- **The combined task, version 3.0 to 3.1** (development, one thread):

| Change | Workload | Before | After |
|---|---|---:|---:|
| Cell counts from cohort tallies | 107 | 0.33 s | 0.01 s |
| Hoisted sums for cells sharing a fit | 107 estimation | 1.11 s | 0.70 s |
| | 104 estimation | 0.38 s | 0.18 s |
| Block-local accumulators, reused scores | 107 pass over units | 1.28 s | 0.45 s |
| | 104 pass over units | 0.81 s | 0.22 s |
| Aggregate draws from the cell draws (28% fewer product columns), row additions, quickselect | 107 bootstrap | 2.4 s | 1.8 s |
| | 108 bootstrap | 1.6 s | 1.0 s |
| Logit covariance from the last Newton iteration (one pass fewer per fit) | 105 | 1.24 s | 1.02 s |

  The whole call went from 5.46 to 3.18 s on 107, 1.31 to 0.44 s on 104, 1.34 to 1.02 s on 105,
  1.93 to 1.27 s on 108, and 0.26 to 0.15 s on 101.

**Threads:** four threads give 1.2-3.0x over one on the real workloads (102: 3.0x, 105: 2.8x,
104: 2.2x, 108: 1.4x) and 1.0-2.4x on the synthetic ones. Both were measured on the same day.

**Remaining costs** (development, one thread):
- **107:** the multiplier product, 1.6-1.8 s for 7.2e10 flops at Accelerate's 45 GFLOP/s. The
  reference's dense product is 1.65e11 flops.
- **105:** 1,681 separately warm-started logits, 0.84 s.
- **108:** 25,000 draws for 812 cells, 0.55 s for the product and 0.42 s for the order statistics.
  This is the weak case: 1.5x at one thread and 1.8x at four.

**Randomized checks:** 783 of the 784 random panels on which the reference succeeds pass (1,000
drawn). The exception is degenerate: a cell's influence function is constant in exact arithmetic,
so the reference's critical value is set by rounding noise.

### 3.4 Task 003: block-index

**Method:**
1. **Dictionaries of the master values only.** Each using value is hashed eight characters per
   64-bit step and looked up. For values longer than eight characters, a one-bit filter on the
   hash of the first eight rejects most of them without reading the rest.
2. **Perfect matches** compare tuples of value ids.
3. **Blocking keys** come from (required values, value) pairs. The variable with the most
   candidate pairs (city) defines the blocks; matches through the other variables are de-duplicated
   extras.
4. **Blocks with several masters** are encoded once: lengths, the first four characters packed,
   value ids, and an inverted list per distinct bigram. A master value's bigram counts against
   every record of the block are then sums over its bigrams' lists. Blocks with one or two masters
   are counted directly while the records are read. Scattered rows are prefetched
   (`llvm.prefetch`).
5. **Scores** follow the reference operation by operation, in float64 with float32 rounding where
   the reference rounds:
   - branch-free over chunks of 128 candidates, so that LLVM vectorizes them;
   - the cube-root bonus is approximated in the hot loop (a float32 bit estimate and three Halley
     steps);
   - every candidate within 1e-5 of the master's approximate best is rescored exactly with libm
     `pow`, and the best and the ties come from the exact scores.
6. **Threads** run over using records (lookups), keys, and tasks of master records balanced by
   block size.

**Attribution** (development measurements, one thread, seconds; factor against the previous row):

| Version | 101 | 102 | 103 | 105 | 107 | 108 |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 35.3 | 5.1 | 18.8 | 43.7 | 18.5 | 352.0 |
| 1. Indexing: master dictionaries, blocking keys, inverted bigram lists | 2.06 (17.1x) | 0.39 (13.1x) | 1.42 (13.2x) | 2.46 (17.8x) | 1.08 (17.1x) | 12.6 (27.9x) |
| 2-6. Engineering (below) | 0.64 (3.2x) | 0.16 (2.4x) | 0.49 (2.9x) | 0.76 (3.2x) | 0.38 (2.8x) | 6.07 (2.1x) |

The engineering steps, in order:
- **Output buffer sized per task:** the search had run twice. 101 went from 2.06 to 1.12 s.
- **Branch-free scoring chunks:** 101 1.12 to 0.69 s; 105 1.33 to 0.85 s.
- **Blocks encoded inside the parallel tasks** with prefetching, 8-character hashing, and direct
  counting for blocks with one or two masters: 108 12.6 to 7.82 s.
- **Approximate bonus with exact rescoring:** 101 0.85 to 0.64 s; 108 7.82 to 6.07 s. This removed
  about two thirds of the scoring time, and the outputs stay bitwise equal.

**Threads:** six threads give 1.9-2.8x over one on the real workloads (evaluator
runs). Four threads reach 70-91% of six threads' speed. The dictionary build, perfect
matching and output assembly are serial.

**First call:** 70-78 s at one thread (up to 92 s at four), which includes the Numba compilation.

**Randomized checks:** 600 random problems (9,170 reference pairs) give identical pair sets and
bitwise-equal scores.

## 4. Peak memory

### 4.1 What produced the gains

1. **Measuring what sets the peak first.** Three components were separated: the loader's floor,
   the JIT compiler and materialized intermediates. Each memory candidate attacks the ones above
   the floor.
2. **Keeping the JIT compiler out of the measured process.** Numba's compilation at the first call
   adds 0.3-0.47 GB. schur-lean and block-lean load prebuilt, data-independent machine code from
   Numba's on-disk cache. cell-lean uses NumPy only. This also removes the compilation from the
   first call (block-lean: 0.4-9.8 s against block-index's 70-78 s; schur-lean: 0.8-32 s against
   schur-cg's 11.5-46 s; each first call includes one call's work).
3. **Not materializing large intermediates:**
   - recomputing instead of storing (schur-lean forms the demeaned rows twice; cell-lean
     recomputes the influence functions per block);
   - reducing per-unit arrays to sufficient statistics (cell-lean's Gram matrix of the aggregate
     influence functions).
4. **Fixed byte budgets for working sets:** column batches of at most 128 MB, row chunks of 32 MB,
   working matrices of 512 KB, and bootstrap tiles of 4 MB. The budgets were tuned by measurement.
5. **Returning freed pages to the operating system.** On macOS the allocator keeps freed blocks
   resident, so the peak grows although the live arrays are small. The candidates call
   `malloc_zone_pressure_relief` (glibc: `malloc_trim`): schur-lean and block-lean at two fixed
   points of the call, cell-lean after every 64 MB of large freed arrays. Returning pages more
   often made the next allocations fault them in again (2x slower on 002/105).

**Cost in time:**
- none for block-lean (the same algorithm);
- 1.2-2.3x for schur-lean (a conjugate-gradient solve per batch);
- 2.5-6.8x for cell-lean (NumPy-only kernels and tiled passes; their shares were not measured
  separately).

### 4.2 What sets the peak

| Task | Loader floor (before the call) | JIT compilation (speed candidate) | Largest working arrays of the speed candidate |
|---|---|---|---|
| 001 | regressors built from a list of columns: about twice the input (104 with 50 regressors: 5.79 GiB for 2.77 GiB of input) | about 0.3 GB | the demeaned n x (k + 1) matrix (2.57 GiB on 104 with 50 regressors), several (levels x (k + 1)) arrays (0.66 GiB each), conjugate-gradient vectors for all columns, int64 index arrays (about 1 GB on 104) |
| 002 | bootstrap multipliers generated with about three n_units x B int8 temporaries (107: 200 MB each; about 0.58 GiB) | about 0.33 GB, plus 40 MB for importing Numba | the n_units x n_aggregates influence functions (111 MB on 107); 108's 812 x 25,000 table of draws (162 MB) with temporaries |
| 003 | string arrays built from Python lists (0.11-0.58 GiB on 101-107; 8.89 GiB on 108) | about 0.47 GB (and about 70 s) | small: UCS4 codes, int32 ids, bigram lists per block |

The references' own peaks are larger: 13.5 GiB on 001's 104 with 50 regressors, 12.3 GiB on
002's synthetic 5, and 22.9 GiB on 003's 108.

### 4.3 Task 001: schur-lean

schur-cg's computation with:
- **Column batches:** the columns of [y, X] are solved in balanced batches whose per-level arrays
  (sums, conjugate-gradient vectors, operator buffers) take at most 128 MB. Only the fixed-effect
  coefficients of every column are kept: 0.66 + 0.10 GiB on 104 with 50 regressors.
- **No demeaned matrix:** a demeaned row is [y, X] minus its fixed-effect coefficients. Rows are
  formed in 32 MB chunks, twice: once for W'W (BLAS), and once, cluster by cluster, for the
  residuals and cluster scores. The products are schur-cg's; only the order of summation differs.
- **Early releases** of the first codes, the singleton mask and the stacked-level index.
- **Free pages returned to the OS** at the start of the call (the loader's freed columns) and
  after the solve. On 104 with 50 regressors this took the peak from 7.39 GiB to 5.82 GiB, which
  is the loader's own peak.
- **Prebuilt Numba cache:**
  - the first call has no compilation: 0.8-32 s, one call's work, against 11.5-46.1 s for
    schur-cg;
  - the peak is about 0.2 GB lower on small workloads (development: 106 0.66 to 0.49 GiB; 101
    0.71 to 0.50 GiB).
  - For the cache to work inside the evaluator:
    - the kernels' module has a fixed name;
    - kernels take the thread count as an argument, since `numba.get_num_threads()` inside
      compiled code is not cacheable;
    - the cache is built with read-only inputs, as the evaluator passes them;
    - the cache is read from a private copy, so a compilation can never write into the
      evaluator's snapshot.

**Results:**
- **Against the references:** peaks at 0.63 times the reference's (real and synthetic); 1.18-2.02x
  lower on the real workloads.
- **104 with 50 regressors:** 5.78 GiB against schur-cg's 11.25 GiB and the reference's 11.7-13.5
  GiB. That is the loader's own floor (5.79 GiB).
- **108:** 5.36 GiB, also the loader's floor (5.39 GiB).
- **Synthetic 11:** 4.17 GiB, against 8.22 (schur-cg) and 9.92 GiB (reference).
- **Synthetic 1 and 6:** above the reference's peak. These inputs are a few MB, and Numba's import
  and cached code (about 0.1 GB) outweigh SciPy.

**Cost:** 1.17-1.79x schur-cg's time on the real workloads. The conjugate gradients run once per
batch, and every approximate-Cholesky solve traverses the whole factor whatever the batch width.
105 (four fixed effects, 21 columns in three batches) costs the most relative to schur-cg (1.65x).

**Not done:** int32 index arrays (about 1 GB of int64 index structures on 104).

### 4.4 Task 002: cell-lean

The same computation as cell-fused, organized for small working memory, with NumPy only:
- **No JIT compiler:** this avoids Numba's 0.33 GB compilation peak and 40 MB import.
- **Estimation one cohort at a time.** Newton Hessians and regression matrices are accumulated
  over chunks of 2^16 elements, so no n_cell x k temporaries are formed.
- **The pass over units** runs in blocks of one cohort's units. Working matrices hold at most 2^16
  elements (512 KB), and a block's influence functions for all its cells form one units x cells
  matrix.
- **Sufficient statistics:** the aggregate influence functions are accumulated only as sums and a
  cross-product (Gram) matrix, n_aggregates^2 in size. That gives:
  - every aggregate's SE;
  - the cohort average's SE, with csdid 2.0.1's unit-level cohort weights;
  - the event-study covariance.

  cell-fused keeps an n_units x n_aggregates array instead (111 MB on 107).
- **Bootstrap tiles** of at most 2^19 elements (4 MB) of cells x draws. The int8 multipliers are
  converted to float64 in pieces, and each tile adds to the critical value's maximum and to the
  aggregate draws, then is freed. 108 (162 MB of draws) runs in about 40 tiles.
- **Rate-limited page return:** free pages are returned after every 64 MB of freed arrays of 4 MB
  or more.
  - **Why** (development): on 104, NumPy's traced peak was 39 MB while the resident set grew from
    100 to 286 MB. An allocator setting (`MallocMediumZone=0`, which a candidate cannot set in the
    evaluator's process) took 108 from 279 to 222 MB.
  - **Returning pages after every cohort and tile** took 107 from 738 to 609 MB, but made 105 2x
    slower (3.3-4.1 to 6.5-7.4 s) and 104 1.8x slower.
  - **The rate-limited policy** keeps 104's and 105's times and gives 637 MB on 107 (7.1-7.6 s,
    against 6.1 s without returning pages).
- **Working-matrix size,** tuned on 104 (development, one thread):

| Size (elements) | 2^14 | 2^16 | 2^18 | 2^20 | 2^22 |
|---|---:|---:|---:|---:|---:|
| Peak | 148 MB | 164 MB | 294 MB | 762 MB | 652 MB |
| Time | 6.8 s | 3.5 s | 3.1 s | 3.5 s | 3.4 s |

**Results:**
- **Against the reference:** peaks of 0.04-0.60 GiB, 2.4-10.8x lower on the real workloads and up
  to 57x lower on the synthetic ones (synthetic 5: 12.3 GiB against 0.22 GiB).
- **Against cell-fused:** 2.1-9.8x lower.
- **107:** the 0.60 GiB is almost all the loader's multiplier generation (about 0.58 GiB).

**Cost:**
- 2.45-6.82x cell-fused's time, but still 4.3x faster than the reference on the real workloads
  (geometric mean).
- **108** is 2x slower than the reference: 40 tiles, each a NumPy pass over the units, against
  the reference's single table, which costs it 0.91 GiB of peak.

**Rejected:** caching each shared fit's outcome matrix. It was slower on 105, where no two cells
share their units, and took 16 MB per cohort on 107.

### 4.5 Task 003: block-lean

block-index's computation unchanged, plus:
- **Prebuilt Numba cache,** set up as in schur-lean. It removes the 0.47 GB compilation peak and
  cuts the first call from 70-78 s to 0.4-9.8 s (one call's work).
- **Free pages returned to the OS** at the start of the call (pages the loader freed) and after
  the dictionaries are built.

**Results:**
- **Development, fresh process:** 101 went from 745 to 277 MB, 102 671 to 191 MB, 104 1,209 to
  754 MB, and 106 664 to 185 MB.
- **Evaluator:** 0.10 times the reference's peak on the real workloads (2.5-17.3x lower) and 0.18
  on the synthetic ones. That is 1.5-3.3x below block-index on 101-107.
- **Where the input sets the peak,** there is no gain: on 108, the loader's 8.89 GiB plus 0.16 GiB;
  on synthetic 5, block-index's own working arrays.

**Cost:** none attributable to the computation, which is the same code path.
- In same-day development runs the times matched: 101 0.59 against 0.60 s, 102 0.17 s for both,
  104 1.86 against 1.92 s.
- 106 took 0.10 s against 0.08 s. The two page-return calls are the likely cause; this is an
  inference, not measured separately.
- The evaluator ratios (0.95-1.36x on the real workloads) come from runs three days apart.

**Not attempted:** one-byte character codes instead of UCS4. The working arrays do not set the
peak.

## 5. Approaches that did not pay off

| Task | Approach | Outcome |
|---|---|---|
| 001 | Incomplete Cholesky IC(0) of the reduced matrix | Broke down: S is not an M-matrix (the firm-year coupling has positive entries) |
| 001 | Exact sparse factorization of the firm block | Fill of 3.6x on 101, but 29x on 102 (7 s with SuperLU) |
| 001 | Approximate Cholesky on 105's subclass x year block (6.6 million edges) | Iterations unchanged (48 both ways); 5.9 s to factorize |
| 001 | Relabelling the approximate-Cholesky factor in elimination order alone | No gain (2.95 s of solves on 104 before and after); the gain came with 8-column chunks |
| 001 | W'W by Numba kernels | 564 ms at one thread (146 ms at six) against dgemm's 255 ms (one thread for this shape); BLAS was kept |
| 001 | W'W by BLAS calls on row blocks from parallel Python threads | Ran serially (CPU time = wall time) |
| 001 | Round-based parallel singleton peel | 7.3 s against 0.7 s for the sequential sweep (singleton chains need many rounds) |
| 002 | Method of four Russians for the ±1 multiplier products (tables of 256 signed sums of 8 units) | 45-58 ms against dgemm's 32.5 ms on a 2,048 x 361 x 999 chunk, because the table lookups did not vectorize; inside the solution, 107's product went from 2.24 to 3.55 s, plus 0.86 s for the bit patterns |
| 002 | float32 products for the bootstrap | Rejected before measuring: float32 accumulation over 10^5 units gives draw errors of about 1e-5 relative (estimate), too close to the 1e-4 tolerance |
| 002 | Per-cohort features for never-treated controls | Rejected by operation count (estimate): T + 2k + 1 features per cohort against T - 1 cells |
| 002 | Warm-starting Newton at the intercept-only fit (former task) | Diverged for rare-treatment cohorts; a damped Newton saved Hessian evaluations but not likelihood evaluations |
| 002 | Returning free pages after every cohort and tile | 2x slower on 105 (pages faulted again); replaced by a rate-limited policy |
| 003 | A 4,096-slot memo of the cube-root bonus | As costly as `pow`: the branchy lookup kept the scoring loop scalar |
| 003 | Encoding every block before the search | Several GB on 108 and two passes over 4.6 million scattered rows; encoding inside the parallel tasks with prefetching was faster |

**A correctness pitfall:** in Numba 0.62.1, the parallel transformation of a large `prange` body
silently processed only some of 003's master records. Moving the body into a separate jitted
function fixed it. The fix was verified against serial and pure-Python versions.

## 6. Existing implementations

Measured on the same host, with the same inputs, thread budget and validation. The adapter's
conversions are inside the timed call, and each tool had one measured call per case. Times are
multiples of the speed candidate's at the same thread count.

| Task | Implementation | Time against the speed candidate | Notes |
|---|---|---|---|
| 001 | R fixest 0.14.2 | 3-76x (one thread); 52x on 104 (four); 5-50x (six) | fails 1e-6 on synthetic 9 (1.5e-6 in β) |
| 001 | pyfixest 0.60.0, LSMR on the `within` backend | 7-21x (one thread); 6-17x (six) | |
| 001 | pyfixest 0.60.0, alternating projections | 3-2,338x | run on some cases only |
| 001 | Stata 18 MP reghdfe 6.14.1 | 15-375x | timed out at 3,600 s on 104, 105 and 108; run after the study, 75x on 104k5 and timed out on 104k50 (study report, addendum) |
| 002 | R did 2.5.1 with DRDID 1.3.0 | 43-437x | csdid 2.0.1-rule workloads (balanced) |
| 002 | Stata csdid 2.0.1 | 85-922x | csdid 2.0.1-rule workloads |
| 002 | Stata csdid 1.82 with drdid 1.91 | 2,855-21,471x | passes 102, 106, 109; on 108, 29 cells of one cohort differ (handling of a perfectly predicting covariate) |
| 003 | Stata reclink2 | 259-398x | agrees on four real workloads; fails 102 and 105, errors on 103; run after the study, timed out on 108 (study report, addendum) |

Bootstrap outputs of the baselines use the tools' own multipliers, so they are timed but not
compared.

## 7. Limitations

- **Host state.** There was a single host, and it throttled during most runs.
  - The four-thread runs had no sleep, but 38-90% of their 5-second samples were throttled, with
    the CPU speed limited to 50-76% at worst.
  - Speedups are computed within one evaluator run to cancel drift, but absolute times from
    different days are not directly comparable. The reference's 002/104 took 30-120 s depending on
    the hour.
  - The multicore figures are therefore conservative.
- **Development data only.** The candidates were tuned on development and synthetic workloads;
  the held-out evaluation is left to the benchmark's maintainer.
- **The memory metric** counts the interpreter and the input loading. On large inputs the
  loader's own peak bounds what any candidate can show (001/108, 001/104 with 50 regressors,
  003/108). On tiny inputs the imports dominate.
- **Baselines** had one measured call per case (the candidates had five or three). They were
  validated against saved reference outputs, not a reference recomputed in the same run. Not
  every baseline ran on every workload or thread count.
- **Independence.** The same agent had built earlier candidates for these tasks,
  so the core ideas were known from that work rather than found independently:
  - eliminating the largest fixed effect;
  - sharing nuisance fits across cells;
  - master dictionaries with inverted bigram lists.

  The study report lists the incidents in which prior candidates' material was exposed.
- **Attribution tables** come from single-call development measurements. Some of 003's rows were
  taken with another single-threaded job running in the background.

## 8. Where the figures come from

- **Candidates:** `solutions/001/schur-cg`, `solutions/001/schur-lean`, `solutions/002/cell-fused`,
  `solutions/002/cell-lean`, `solutions/003/block-index` and `solutions/003/block-lean`. Each
  `research.md` holds the version tables, component measurements, runs and commands.
- **Lineage of the 002 kernels:** `solutions/002/legacy-pre-merge/cohort-fused/research.md` and
  `solutions/002/legacy-pre-merge/002b/cell-fused/research.md`.
- **Study report:** `solutions/baselines/development-study.md`. It has the results tables, the
  baselines, the run directories, disclosures and reproduction commands.
- **Ratios in section 2:** `solutions/.venv/bin/python solutions/baselines/ratios.py
  runs/study-20261025/report-config.json`.
- **Run directories:** `runs/` (git-ignored), listed in the study report's "Run directories"
  section.
