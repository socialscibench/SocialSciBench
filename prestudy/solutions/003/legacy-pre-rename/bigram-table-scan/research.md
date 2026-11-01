# Candidate research: bigram-table-scan

## Objective and constraints

- Task 008, contract version 2: Stata `reclink2` record linkage on name, address, city, ZIP and
  state (`wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)`, minscore and minbigram .6):
  perfect matches, state-constrained or-blocking, reclink's `bigram1` comparator with the
  Winkler-type prefix bonus, match/non-match weights with float32 rounding after every
  operation, and every tied best candidate. Must match `reference.run`: the complete pair set and
  the perfect-pair count exactly, every score to 1e-6 relative error.
- Target hardware: CPU, one thread (the evaluator default). Host: Intel Core i7-8700B (6 cores),
  32 GiB RAM, macOS 15.7.3. Environment: the shared task 008 environment (`build.md`): CPython
  3.12.13, NumPy 2.3.5, Numba 0.62.1 (llvmlite 0.45.1, LLVM 20).
- Informed research session: the user authorized consulting the repository's prior results
  (`reports/008_recordlinkage.md`) and the 14 October 2026 study, including its C++ pipeline,
  which is also a competitor here. The development fixtures and the reference were used for
  validation only.

## Method

The reference finds candidates with pandas joins and scores every candidate pair with sparse
bigram matrices. About 300 register firms share a city, so a master record has about 550
candidates, nearly all of them the firms in its own city and state. The candidate does the same
computation, organized around those city blocks:

1. **Dictionaries of the master values only.** For each field, the distinct master values get ids
   (open-addressing hash table); every using value is hashed once and looked up, and a value
   absent from the master file gets id -1 (a one-bit filter of the master hashes rejects most of
   those without touching the table). Perfect matches (all five ids equal, all duplicates kept),
   the blocking joins and "equal strings score 1" only ever compare a using value with a master
   value, so these ids are enough. Hashing scans each fixed-width UCS4 row over its full width
   without branches (the zero padding adds nothing), with two 32-bit multiply-add accumulators.
2. **City blocks.** The eligible using records (not perfectly matched, state present in the master
   file) are sorted by (state, layout field) with a two-key counting sort. The layout field is the
   or-block field whose agreeing records form the most candidate pairs (counted ignoring the
   state), which is the city. A master record's candidates are then one block of records that
   agree with it on state and city, found by binary search, plus the few records found through ZIP,
   name or address in other cities (de-duplicated: a record found through field v is skipped if it
   agrees on a field processed before v). Master records are processed in block order.
3. **Encode each block once, invert it.** When the first master record of a block arrives, the
   block's values are encoded from the input strings (length, first four characters, distinct
   bigram codes in a per-field alphabet of the characters of the master values, plus one code for
   all others), with software prefetching of the scattered rows, and an inverted list is built per
   bigram. The block stays in L1/L2 for its about 11 master records.
4. **Bigram counts by inverted lists.** `bigram1` needs, for master value m and using value s, the
   number of m's bigrams (with multiplicity) that occur in s. For the block, that is a sum over the
   inverted lists of m's own bigrams, about 2 updates per candidate per field instead of about 16
   lookups. Candidates outside the block use a table of m's bigram counts, summed over s's
   distinct bigrams.
5. **The comparator and weights, vectorized.** The rest follows the reference operation by
   operation, so scores are bitwise identical: g = f32(num / den), then f32(g + k (1 - g) / 10)
   with k the common-prefix count (from XORed packed prefixes), then the M and NM terms and their
   float32 accumulation in field order. Equal strings and empty using values are encoded as
   (num, den, k) = (1, 1, 0) and (0, 1, 0), which give exactly 1 and 0, so the arithmetic has no
   branches and LLVM vectorizes it over chunks of 128 candidates. (bi - minbigram)^(1/3) is
   memoized by value. Strings shorter than 3 characters take a separate exact path.

Inputs must be ASCII without embedded NUL characters (the reference itself fails on non-ASCII
strings); other inputs raise `ValueError`. Variable lists, weights and required() fields other
than the task's are supported (tested in `fuzz.py`).

## Sources (accessed 2026-10-18 and 2026-10-19)

- Wasi, N. and Flaaen, A. (2015), "Record linkage using Stata: Preprocessing, linking, and
  reviewing utilities", Stata Journal 15(3): 672-697; `reclink2.ado` from the dm0082 package,
  https://www.stata-journal.com/software/sj15-3/dm0082/reclink2.ado (read for the blocking,
  perfect-match and output semantics; SHA-256 5ed4aae0...).
- Blasnik, M. (2010), "RECLINK: Stata module to probabilistically match records", SSC S456876.
- Numba performance tips, https://numba.readthedocs.io/en/stable/user/performance-tips.html;
  high-level extension API (`@intrinsic`),
  https://numba.readthedocs.io/en/stable/extending/high-level.html.
- LLVM `llvm.prefetch` intrinsic, https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic.
- MurmurHash3 `fmix64` finalizer (A. Appleby), https://github.com/aappleby/smhasher.
- `reports/008_recordlinkage.md`, "Implementation study (14 October 2026)".

## Hypotheses and bottlenecks

- **Joins and sparse products dominate the reference.** A compiled scan over city blocks should
  do the same pair scoring in tens of nanoseconds per pair. Confirmed: version 1 (below) was
  13x faster on W3.
- **After that, per-candidate overheads in the compiled loops.** Measured, in order of impact:
  - Numba checks signed array indices for negative values (wraparound); with computed indices
    such as `a[p + b]` this blocked vectorization (microbenchmark: 4.0 ns per element against
    0.54 ns with a slice indexed by the loop variable). All hot loops use slices or unsigned
    indices.
  - Conditional expressions on boolean and integer arrays kept LLVM from vectorizing the float32
    chain (6.0 against 2.5 ns per element); the exact (num, den, k) encoding removes them.
  - AVX2 has no packed int64-to-double conversion; 32-bit integers in the per-candidate step cut
    it from 0.31 to 0.21 s on W5.
  - `pow` for the cube-root bonus was not a bottleneck (about 11 ns per call; memoized anyway).
- **The block layout field matters.** A first heuristic (the field with the fewest distinct master
  values) picked ZIP, because master city typos make more distinct city values: only 47% of W3's
  candidates fell in the block. Counting candidate pairs per field picks the city (99%).
- **At W5 scale, random access to 2 million using rows.** Permuting the using file into block order
  (gather, then scatter) cost 1.25 s and then 0.84 s on W5; encoding each block when it is first
  needed, with prefetching, costs about 0.4 s plus 0.25 s for the inverted lists. The bit filter
  on dictionary lookups did not help (the time was in the character scan); the branch-free
  full-width scan did (0.63 to about 0.37 s for five fields on W5).
- **Remaining time (W5, one instrumented call, about 2.6 s):** the scoring arithmetic about 1.0 s
  (the float32 divisions run at the divider's throughput), block encoding and inversion about
  0.65 s, dictionary lookups 0.37 s, bigram counting 0.3 s, candidates outside the blocks 0.2 s,
  sorting 0.1 s. The number of candidate pairs (42.8 million on W5) is inherent to the rule;
  pruning candidates with score bounds was not attempted.

## Experiments

Prototype measurements are single `run` calls after a first call, timed in the development
script (not evaluator runs), with the reference timed once in the same session:

| Version | W1 (s) | W2 (s) | W3 (s) | W4 (s) | W5 (s) |
|---|---:|---:|---:|---:|---:|
| Reference | 0.40 | 1.91 | 12.17 | 65.5 | 146.2 |
| 1. master dictionaries, block order, per-candidate table lookups | 0.024 | 0.127 | 0.94 | | |
| 2. scoring per master record over arrays | 0.030 | 0.115 | 0.78-0.86 | | |
| 3. codes stored field by field, running sums over the block; layout field by candidate pairs (city) | 0.017 | 0.092 | 0.57 | | |
| 4. chunks of 128, inverted lists per block, slice and unsigned indexing | 0.009 | 0.046 | 0.28 | 1.32-1.54 | 5.50-5.61 |
| 5. two-key radix sort, encoding written in block order | 0.007 | 0.044 | 0.28-0.30 | 1.41-1.51 | 3.96-4.01 |
| 6. block-local encoding, filtered dictionary lookups | 0.008 | 0.037-0.044 | 0.25 | 1.41-1.43 | 3.80 |
| 7. branch-free full-width hashing, prefetching | 0.007-0.008 | 0.031-0.036 | 0.19-0.21 | 1.03-1.07 | 2.68-2.79 |
| 8. 32-bit integers in the per-candidate step (final) | 0.007 | 0.032-0.034 | 0.18-0.19 | 0.99-1.05 | 2.61-3.35 |

The host had a background process (a repeatedly crashing helper and its crash reporter) using
15-30% of one core throughout, so single-call times vary by 10-30%; development comparisons used
repeated in-process timings of the search kernel on saved inputs. Every version above returned
bitwise-identical pairs, perfect counts and scores on all measured workloads.

- Random small problems against the reference (`fuzz.py`, 2,000 cases: 1-40 master and 1-80
  using records, alphabets of 1-5 characters, values of 0-11 characters, up to 30% empty values
  in every field including the required one, duplicated records in both files; 70% with the task's
  settings, 30% with 2-5 variables, random required() fields and weights): no failures; pairs,
  perfect counts and scores compared bitwise (22,201 reference pairs; 45,449 master values shorter
  than 3 characters). It found one bug during development (all fields required, so no field is
  scored), since fixed.

## Evaluator runs

- `runs/20261019T023320.928531Z-a715f349`: `--smoke`, all 4 development fixtures passed.
- `--smoke --validation-seed` 11, 12, 13: `runs/20261019T024359.362148Z-1c6d5364`,
  `runs/20261019T024503.386714Z-5362232c`, `runs/20261019T024608.131421Z-3752f7ab`, all passed.
- `runs/20261019T024724.672285Z-63751f80`: same-session comparison run, workloads 1-5, default protocol (one
  warmup, three measured calls, one thread), `--timeout 10800`. All 4 development fixtures and all
  5 workloads passed; pairs and perfect counts exact, scores identical to the reference (maximum
  relative error 0).

  | Workload | Reference median (s) | Candidate median (s) | Candidate min-max (s) | Speedup | First call (s) | Peak RSS (GiB) |
  |---|---:|---:|---:|---:|---:|---:|
  | W1 | 0.3993 | 0.005510 | 0.005419-0.006129 | 72.46 | 13.4 | 0.33 |
  | W2 | 1.874 | 0.04065 | 0.03805-0.05504 | 46.11 | 14.5 | 0.34 |
  | W3 | 12.64 | 0.1845 | 0.1815-0.1879 | 68.52 | 13.3 | 0.49 |
  | W4 | 68.23 | 1.016 | 1.005-1.017 | 67.15 | 14 | 1.24 |
  | W5 | 149.3 | 2.572 | 2.470-2.817 | 58.05 | 15.2 | 3.76 |

  The first call includes Numba compilation (about 13-15 s). This run measured the reference 8-11%
  slower on W3-W5 than the four competitor runs that followed (11.4-11.6, 61.8-62.9 and 137.4-138.2 s
  there); the speedups above use this run's own reference. The reference's peak RSS is 2.6, 6.3 and
  10.8 GiB on W3-W5. Every first-call, warmup and measured output passed.
