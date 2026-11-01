# 003: Record linkage — baselines and results

`CPU` · `data linking` · `L1` · `Stata reclink2 (Wasi & Flaaen, 2014 version) · reclink 1.7 · R data.table 1.18 + stringi 1.8 port · Python NumPy/SciPy/pandas`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](../../003_reclink/task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. Record linkage was numbered 008 when these measurements were taken; the measurements are unchanged. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools, research context and the October 2026 implementation study.

## Baseline timings (Apple M3 Pro)

| # | design | n_master | n_using | cities | true links | candidate pairs scored | perfect matches | pairs kept | true links found | Stata `reclink2` (s) | Stata `reclink` (s) | R port (s) | Python reference (s) | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 500 | 2,000 | 6 | 300 | 149,610 | 106 | 508 | 297 | 1.48200 | 1.43800 | 0.66642 | 0.13584 | – |
| 2 | survey employers to business register | 2,000 | 20,000 | 66 | 1,200 | 691,342 | 426 | 2,008 | 1,196 | 9.47000 | 9.32800 | 3.21361 | 0.65703 | – |
| 3 | startups to state business registrations | 10,000 | 100,000 | 333 | 5,000 | 4,592,859 | 1,666 | 10,085 | 4,981 | 134.22800 | 134.01100 | 22.43952 | 4.48421 | – |
| 4 | loan recipients to establishment database | 50,000 | 500,000 | 1,666 | 35,000 | 22,719,376 | 11,811 | 50,139 | 34,863 | 30+ min (est. ~46 min‡) | not run (est. ~45 min‡) | 120.76089 | 27.17095 | Stata `reclink2`: all outputs (stopped at 30 min); Stata `reclink`: all outputs (not run in full) |
| 5 | establishments to establishment database (national) | 100,000 | 2,000,000 | 6,666 | 70,000 | 42,778,502 | 23,564 | 100,269 | 69,773 | 30+ min (est. ~5.4 h‡) | not run (est. ~5.4 h‡) | 326.90281 | 69.82897 | Stata `reclink2`: all outputs (stopped at 30 min); Stata `reclink`: all outputs (not run in full) |

How to read the timing columns (the appendix defines every column):

- **Stata `reclink2` (s)** and **Stata `reclink` (s)**: the command above (`reclink` with the same options), timed with `timer` after the master file is loaded.
  - Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM. Stata/MP runs the file reads and `replace` commands on several cores.
  - `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Each time is a single run.
  - Stopped after 30 minutes. `reclink` is not run in full on workloads where `reclink2` was stopped, because both run the same loop.
- ‡**Stata estimates**: the command is timed on a random sample of 200 master records with the same using file, and the total is (sample time) × (n_master ÷ 200). The cost per master record is roughly constant within a workload, because each record reads the whole using file. Sample times: workload 4: `reclink2` 11.0 s for 200 records, `reclink` 10.8 s for 200 records; workload 5: `reclink2` 38.8 s for 200 records, `reclink` 38.7 s for 200 records.
- **R port (s)**: `bench_r.R`, which applies `reclink2`'s rule with `data.table` joins for the candidates and `stringi` for the bigram scores, on up to 12 threads. R 4.6.1, data.table 1.18.6.1, stringi 1.8.9. Timed from the loaded data to the final pairs. Stopped after 30 minutes.
- **Python reference (s)**: `reference.py`, timed with `time.perf_counter()`.

## Agreement with Stata and R

Agreement with the reference:

- **Stata `reclink2`** (workloads 1, 2, 3): identical pairs: yes; score sums within 1e-16.
- **Stata `reclink`** (workloads 1, 2, 3): identical pairs: yes; score sums within 1e-16.
- **R port** (workloads 1, 2, 3, 4, 5): identical pairs: yes; identical score sums.

Results are in `../baselines/003_reclink/results/results.jsonl`.

## Notes on existing tools

There is no R or Python package with `reclink`'s scoring rule. R's `reclin2` and `fastLink` and Python's `recordlinkage` use other comparators (e.g. Jaro–Winkler) and weights (e.g. Fellegi–Sunter estimated by EM), so they give different matches. The R column is therefore a port of the same rule (`bench_r.R`, data.table joins plus `stringi`), which reproduces `reclink2` exactly. Kranker's `dtalink` (SSC) is a faster Stata alternative with its own scoring, not benchmarked here.

- **The help file's example** uses the option `wmnomatch()`, which fails with "option wmnomatch() not allowed" (r(198)): the option is `wnomatch()`. In addition, a typo in the source (`wmnomatch` in the check that weights are ≥ 1) means non-match weights below 1, such as `wnomatch(0.5 4 4)`, are accepted although the help file says weights must be ≥ 1.
- **Many matches are false positives** with these weights. A firm that is not in the register still gets a best candidate above .6 when it shares the state (weight 5) and city (weight 5) with a register firm (compare "pairs kept" with "true links"). Applied work raises `minscore`, adds weights on the name, and reviews matches by hand (`clrevmatch`). The benchmark keeps the help-file weights as a pure computation.

## Appendix: workload table columns

The examples use one running case: linking startups from a venture database (master) to state business registrations (using).

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–5 from smallest to largest. | `3` is startups to state business registrations. |
| design | The linking task the workload imitates (see the research context below). | startups to state business registrations |
| n_master | Records in the master file (in memory in Stata), each to be linked. | 10,000 startups |
| n_using | Records in the using file, the register searched for each master record. | 100,000 registered firms |
| cities | Number of distinct cities in the register. Candidates are mostly the register firms in the master record's city, so n_using ÷ cities sets the work per master record. | 333 cities, about 300 firms each |
| true links | Master records generated from a register record, so the correct match is known. The others are firms not in the register. | 5,000 |
| candidate pairs scored | Master–using pairs that pass the blocking rule (same state and the same name, address, city or ZIP code) and get a score. | 4.6 million |
| perfect matches | Master records equal to a register record on all five fields (score 1). | "ACME LABS / 12 MAIN ST / SPRINGFIELD / 62701 / IL" in both files |
| pairs kept | Best-scoring candidates with score ≥ .6, including perfect matches and ties. | |
| true links found | Pairs kept that are the true link. | |
| Stata `reclink2` (s) | Seconds for `reclink2`. Stopped after 30 minutes. | `30+ min (est. …‡)`: stopped, total estimated |
| Stata `reclink` (s) | Seconds for `reclink` (Blasnik's original command, same options). | `not run (est. …‡)`: only timed on the sample |
| R port (s) | Seconds for the R port of the same rule. | |
| Python reference (s) | Seconds for the Python reference. | |
| Could not compute | Outputs a tool did not produce on that workload, and why. | |

## Research context: what each workload represents

| # | Design | What it represents | Related settings |
|---|---|---|---|
| 1 | sanity check | No real setting: 500 records against 2,000, to check correctness in seconds. | – |
| 2 | survey employers to business register | Linking the employers named by 2,000 survey respondents (name and address of the workplace) to a business register of 20,000 firms in their areas. | `reclink2` and its companion commands were developed with support from the University of Michigan's Census-Enhanced Health and Retirement Study project, which links survey data to Census business data (Wasi & Flaaen 2015; ICPSR record). |
| 3 | startups to state business registrations | Linking 10,000 startups from a venture database (e.g. Crunchbase) to 100,000 state business registrations by name and address, to get founding dates and legal form. | Guzman & Stern (2020) build entrepreneurship measures from the business registration records of 32 US states. |
| 4 | loan recipients to establishment database | Linking 50,000 loan recipients (e.g. Paycheck Protection Program borrowers, with name and address from the loan record) to an establishment database of 500,000 firms. | The Census Bureau's BDS-SBA COVID data link PPP loans to the Census Business Register by tax identifier, and otherwise by probabilistic matching on business name and street address. |
| 5 | establishments to establishment database (national) | Linking 100,000 establishments to a national database of 2 million (e.g. NETS, Dun & Bradstreet or a census business register). | Kerr & Fu (2008) link patent assignees to the Census Bureau's business register (SSEL) by name. |

Sources:
- Wasi, N. & Flaaen, A. (2015), "Record linkage using Stata: Preprocessing, linking, and reviewing utilities," *Stata Journal* 15(3): 672–697. [ICPSR record](https://linkagelibrary.icpsr.umich.edu/linkagelibrary/project/107948/view)
- Blasnik, M. (2010), "RECLINK: Stata module to probabilistically match records," Statistical Software Components S456876, Boston College.
- Guzman, J. & Stern, S. (2020), "The State of American Entrepreneurship: New Estimates of the Quantity and Quality of Entrepreneurship for 32 US States, 1988–2014," *American Economic Journal: Economic Policy* 12(4): 212–243. [Data and code](https://www.openicpsr.org/openicpsr/project/111705/version/V1/view)
- U.S. Census Bureau, "BDS-SBA COVID Methodology." [census.gov](https://www.census.gov/programs-surveys/ces/data/public-use-data/experimental-bds/bds-sba-covid/methodology.html)
- Kerr, W. & Fu, S. (2008), "The survey of industrial R&D—patent database link project," *Journal of Technology Transfer* 33(2): 173–186. [RePEc](https://ideas.repec.org/a/kap/jtecht/v33y2008i2p173-186.html)
- Kranker, K. (2018), "dtalink: Faster probabilistic record linking and deduplication methods in Stata for large data files," Stata Conference, Columbus. [RePEc](https://econpapers.repec.org/RePEc:boc:scon18:31)

## Implementation study (14 October 2026)

Compiling the existing comparator improves the R and Python pipelines. A complete C++ pipeline adds further gains: on W3 it is **1.10× faster than the fresh Numba adapter at one thread**, or **2.92× faster at six threads**, and uses substantially less process memory.

Measurements: **2026-10-14**, repository baseline **`25f279d`**, Intel Core i7-8700B (6 cores), 32 GiB RAM, macOS 15.7.3. These are bounded comparisons on generated data, not general linkage-quality or language rankings.

The exact contract includes global perfect-match removal, state-constrained OR blocking, asymmetric bigram comparison, intermediate float32 rounding, and every tied best match. W1 is **500 × 2,000**, W2 **2,000 × 20,000**, and W3 **10,000 × 100,000** records; W4–W5 were not measured. The byte-based adapters target the supplied **ASCII** workloads; arbitrary Unicode equivalence is not established.

### R/Python round

Warm full-pipeline seconds below use Python 3.12.13 / Numba 0.62.1 and R 4.1.2 / data.table 1.18.6.1 / stringi 1.8.9 / Rcpp 1.1.2 / reclin2 0.6.0. The compiled-comparator and reclin2-blocking adapters preserve the repository score; they are experimental integrations, not package defaults.

| Implementation, one thread | W1 seconds | W2 seconds | W3 seconds | W3 measured runs |
|---|---:|---:|---:|---|
| Python reference | 0.57352 | 2.84730 | 16.00814 | 6 across two blocks |
| Python Numba comparator + original pipeline | 0.22146 | 1.20794 | 6.82521 | 6 across two blocks |
| Original R data.table/stringi | 2.44600 | 13.18000 | 62.91400 | **1** |
| R Rcpp comparator + original pipeline | 0.98900 | 4.67600 | 25.79800 | 3 |
| R reclin2 blocking + Rcpp comparator | 0.96300 | 4.66600 | 28.60200 | 3 |

W1/W2 have three measured runs after full warm-up. W3 Python medians pool **all six observations from two blocks**, with no outlier removal: reference **13.23–28.94 s**, Numba **5.24–13.72 s**. Original R's warm-up exceeded 60 seconds, triggering the predeclared one-measurement rule; its W3 value is not a three-run median.

Numba improves on this round's Python reference by **2.59× / 2.36× / 2.35×**. Rcpp improves on original R by **2.47× / 2.82× / 2.44×**; the last ratio uses the single R observation. Reclin2 blocking does not improve W3 over the simpler Rcpp adapter. No comparable isolated peak-memory measurements were collected in this round.

Every measured configuration reproduces the complete pairs, perfect-match count and bitwise scores: **508 / 2,008 / 10,085 pairs**, including **106 / 426 / 1,666 perfect pairs**. Python edge checks cover 121 comparator cases plus tie/global-exclusion fixtures; Rcpp separately passes 121 comparator cases, not an additional R pipeline tie test. Repeated bigrams can make the executable comparator exceed one; adapters preserve that behavior.

### Native-extension round

The following medians use a **fresh reference and three measured full-pipeline runs after full warm-up**. No measurements or speedup denominators are pooled with the preceding round.

| Implementation | Threads | W1 seconds | W2 seconds | W3 seconds | W3 process peak MiB |
|---|---:|---:|---:|---:|---:|
| Python reference | 1 | 0.57900 | 1.76709 | 12.23069 | 2530.3 |
| Numba comparator + original pipeline | 1 | 0.14446 | 0.70000 | 4.84210 | 1709.2 |
| Complete C++ pipeline | 1 | 0.07930 | 0.51449 | 4.39161 | 206.2 |
| Complete C++ pipeline | 6 | 0.03108 | 0.25266 | 1.65649 | 208.8 |

Against the fresh Python reference, C++ gains **7.30× / 3.43× / 2.79×** at one thread and **18.63× / 6.99× / 7.38×** at six. W3 ranges are **12.04–12.27 s** for the reference, **4.82–4.92 s** for Numba, **4.39–4.46 s** for C++ at one thread and **1.65–1.75 s** at six.

The **final measured output of each configuration** passes complete pair/perfect-count and bitwise-score checks; earlier timed repetitions are not individually compared. Separate C++ validation passes 121 comparator cases and 64 pipeline configurations at one/six threads, including duplicate perfect records, global exclusion, ties and missing values.

The new C++17 implementation handles feature encoding, exact blocking, deduplication, scoring, tie selection and output construction. It specializes the five fields, fixed weights and required state. Six `std::thread` workers partition the search by master row; feature/index construction remains serial. Intermediate float32 rounding is retained, fast-math is disabled, and embedded NULs are rejected. This is task-specific research code, not a general linkage library.

Peak RSS includes the fresh worker's runtime/imports, inputs, full warm-up and repetitions, sampled **before output serialization**. The C++ bridge still includes Python/NumPy. These are absolute process peaks, not incremental allocations or standalone-C++ memory, and differ in scope from task 007's first-fit peaks.

Both rounds time the complete pipeline, including native input packing, copied outputs and deallocation where applicable. Data generation, startup, imports, compilation, file I/O and validation are excluded. Batches run sequentially; fixed order and uncontrolled desktop activity, temperature and frequency limit precise speedup claims.

### Different-estimator experiment

Splink 5.0.0 learns match probabilities and does not implement the fixed score or best-tie policy. At posterior ≥0.95, the frozen W1 configuration returns 272 links with **100% precision / 90.67% recall**, versus the original task's **58.46% / 99.00%**; its three-run median training-plus-inference time is **1.807 s**.

On W2 it returns **zero links** in warm-up and all three measured fits (median **12.194 s**): precision is undefined and recall zero. This failed exploratory configuration was not retuned afterward. No learned-parameter or unthresholded-score diagnostics were saved, so its cause is unresolved; pilots had no held-out model-selection set. These results do not rank Splink generally or constitute an exact-output speedup.

### Evidence and implementation sources

The full reproduction bundle is in the local sibling repository `../repro-socialscibench` (relative to the SocialSciBench root), retaining archive commit **`af01b9c`**. Its `research/2026-10-14/tasks007008/` and `research/2026-10-14/native-extension/task008/` preserve reports, raw observations, code, checks and provenance. This is a local path, not a public download URL.

Primary sources: [Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html), [reclin2](https://github.com/djvanderlaan/reclin2), and [Splink](https://moj-analytical-services.github.io/splink/index.html).

## Same-session comparison (19 October 2026)

A new candidate, `solutions/003/legacy-pre-rename/bigram-table-scan`, and five existing implementations wrapped as candidates were run through the shared evaluator (protocol v2) one after another, on the same Intel Core i7-8700B Mac (32 GiB, macOS 15.7.3) and in one Python environment. The five are `cpp-pipeline`, `numba-comparator` and `r-rcpp` (adapters following the 14 October study), `r-port` (the repository's R baseline) and `stata-reclink2` (the original command). All five workloads are covered. Each evaluator run times its own reference.

`bigram-table-scan` hashes every using value once against dictionaries of the master values and sorts the eligible register records into (state, city) blocks. When the first master record of a block arrives, the block's bigrams are encoded once, with an inverted list per bigram. A master record's bigram counts against its whole block are then sums over the lists of its own bigrams. The rest of `reclink2`'s comparator, its weights and the float32 rounding after every operation follow the reference operation by operation, vectorized over candidates. Method, sources and failed attempts are in its `research.md`.

**Scope.** One thread; one warmup and three measured calls per workload; medians of complete `run` calls, including string conversion and, for R, copying the strings into R. Generation, imports (including compiling the C++ library and the Rcpp comparator) and the first call (for `bigram-table-scan`, about 14 s of Numba compilation) are excluded. These timings are not comparable with the M3 Pro table or the 14 October study.

| Workload | Reference | bigram-table-scan | C++ pipeline | Numba comparator | R + Rcpp comparator | R port | Best other / bigram-table-scan |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 0.399 | **0.00551** | 0.0889 | 0.135 | 0.653 | 1.562 | 16.1 (C++) |
| W2 | 1.874 | **0.0407** | 0.524 | 0.667 | 2.993 | 7.725 | 12.9 (C++) |
| W3 | 12.64 | **0.185** | 4.044 | 4.595 | 21.99 | 53.75 | 21.9 (C++) |
| W4 | 68.23 | **1.016** | 23.21 | 27.55 | 108.9 | 280.1 | 22.8 (C++) |
| W5 | 149.3 | **2.572** | 57.38 | 74.12 | 266.3 | 636.3 | 22.3 (C++) |

Seconds. The reference column is from the `bigram-table-scan` run. The other runs measured the reference 7–10% faster on W3–W5 (11.4–11.6, 61.8–62.9 and 137.4–138.2 s), and at 0.35–0.51 s on W1 and 1.66–1.89 s on W2. The last column is the fastest competitor's median divided by `bigram-table-scan`'s, so it does not depend on the reference. Ranges, first calls and memory are in each candidate's `research.md`.

- **`bigram-table-scan`:** passes every workload with the reference's exact pairs, perfect counts and scores (maximum score error 0), 46–72× faster than its run's reference. Its peak memory is 3.8 GiB on W5. The C++ pipeline's worker peaks at 3.7 GiB, so most of that is the worker's inputs and their generation. The reference peaks at 10.8 GiB.
- **Timed competitors:** all four also return exactly the reference's output.
  - The C++ pipeline is the fastest of them on every workload, at 2.4–4.7× the reference.
  - The Numba comparator runs at 1.9–2.8× the reference.
  - The two R implementations are slower than the Python reference here, at 0.52–0.78× (Rcpp comparator) and 0.22–0.24× (R port).
- **Stata `reclink2`:** fails `dev-ties-missing-perfect`, so the evaluator timed no workloads for it.
  - **The case:** the fixture's using file holds two identical records that are the best candidate of two master records.
  - **What goes wrong:** `reclink2` collapses identical using records before matching. It maps matches back to the original ids with an old-style `merge` that pairs the copies one to one, so each master record keeps one copy instead of both (7 pairs instead of the reference's 9).
  - **Why the earlier baseline agreed:** the standard workloads' using records are unique, which is why that baseline found identical pairs on W1–W3. There, on the M3 Pro with 12 cores, `reclink2` took 134 s on W3, with an estimated 46 min on W4 and 5.4 h on W5.
- **Comparison with the 14 October study** (same host, earlier session):

  | Implementation (W3) | This session | 14 October study |
  |---|---:|---:|
  | C++ pipeline | 4.04 s | 4.39 s |
  | Numba comparator | 4.60 s | 4.84 s |
  | R + Rcpp comparator | 22.0 s | 25.8 s |
  | R port | 53.8 s | 62.9 s (one run) |

**Interpretation.** The table measures this task's five workloads at one thread; it does not rank libraries.

- **No package implements this rule.** No R or Python package implements `reclink`'s scoring rule (see "Notes on existing tools"). The competitors are this project's own implementations of it: the R port and the 14 October study's Numba, Rcpp and C++ variants. The original command, Stata `reclink2`, could not be timed under the evaluator's rules.
- **The candidate is tuned to these workloads.**
  - **Informed design:** `bigram-table-scan` was designed in an informed session, after reading the 14 October study and inspecting these workloads.
  - **Where the speed comes from:** mostly the city structure. On W3, 99% of a master record's candidates agree with it on the city, so each block is encoded once and reused by its master records (about 11 per city on W5).
  - **Where it helps less:** where or-blocking finds candidates through many small lists, for example mostly by ZIP code or name, that reuse shrinks. The results do not depend on it.
  - **The C++ contrast:** the C++ pipeline compares each pair on its own, with a sorted bigram merge per field, so its cost stays proportional to the number of pairs on any structure.
- **Specialization differs.** Both `bigram-table-scan` and the C++ pipeline compute only this specification.
  - **C++ pipeline:** specialized further, with fixed fields, weights and `required(state)`.
  - **`bigram-table-scan`:** accepts any variable list, weights and `required()` fields.
  - **pandas and data.table pipelines:** they materialize every candidate pair as a table row, which is why their memory grows with the number of pairs.
- **One thread hides parallelism.** The C++ pipeline was 2.65× faster with six threads on W3 in the 14 October study. `bigram-table-scan` is not multithreaded either; its blocks are independent, so it would parallelize the same way.
- **R timings include copying the strings into R**, about 7 s per call on W5.
- **Same author:** the same author wrote the candidate and the competitor adapters. The adapters follow the 14 October study's code and the repository's R baseline.
- **Background load:** a crash-looping desktop process used 15–30% of one core during the session. That likely explains part of the reference's run-to-run variation.

Fairer follow-ups:
- **Held-out workload designs** where blocking is dominated by ZIP codes or names. This is a benchmark change, so it needs a separately scoped task.
- **Timing Stata `reclink2` on the standard workloads.** This needs a separately scoped decision on the duplicate-records fixture.
- **A multithreaded round.**

Run IDs, archive hashes and the batch commands are in `published/003/same-session-2026-10-19.md`.

## Reproduce

```bash
python baselines/003_reclink/bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp
```

Requires Stata/MP 19.5 with `reclink2` (`net install dm0082, from(http://www.stata-journal.com/software/sj15-3)`) and `reclink` (SSC), R with `data.table` and `stringi`, and Python with NumPy, SciPy and pandas.

Run the reproduction commands from the repository root. New baseline runs default to unique directories under `runs/`. Historical raw measurements are preserved under `baselines/003_reclink/results/`.
