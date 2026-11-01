# 007: Fuzzy name matching — baselines and results

`CPU` · `data linking` · `L1` · `Stata matchit 1.5.2 · R stringdist 0.9.17 · Python NumPy/SciPy`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools, research context and an early implementation study.

## Baseline timings (Apple M3 Pro)

| # | design | n_master | n_using | pairs compared | true links | pairs found (≥ 0.5) | true links found | Stata `matchit` (s) | R `stringdist` (s) | Python reference (s) | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 500 | 500 | 250,000 | 250 | 5,591 | 245 | 1.88700 | 0.37465 | 0.00697 | – |
| 2 | firm database to firm database (e.g. Compustat to ExecuComp) | 5,000 | 5,000 | 25,000,000 | 3,000 | 575,479 | 2,899 | 189.40600 | 36.62698 | 0.37199 | – |
| 3 | patent assignees to public firms | 50,000 | 10,000 | 500,000,000 | 3,000 | 11,310,772 | 2,886 | 30+ min (est. ~1.1 h†) | 740.25052 | 7.08651 | Stata: all outputs (stopped at 30 min) |
| 4 | startups to startups (e.g. Crunchbase to PitchBook) | 50,000 | 50,000 | 2,500,000,000 | 25,000 | 56,714,674 | 24,128 | 30+ min (est. ~6.0 h†) | 30+ min (est. ~1.0 h‡) | 36.47021 | Stata: all outputs (stopped at 30 min); R: all outputs (stopped at 30 min) |
| 5 | job-posting employers to firm database | 200,000 | 20,000 | 4,000,000,000 | 10,000 | 82,220,121 | 9,711 | 30+ min (est. ~9.3 h†) | 30+ min (est. ~1.6 h‡) | 58.25436 | Stata: all outputs (stopped at 30 min); R: all outputs (stopped at 30 min) |

How to read the timing columns (the appendix defines every column):

- **Stata `matchit` (s)**: `matchit idm namem using using.dta, idusing(idu) txtusing(nameu) threshold(.5) flag(1)`, timed after the master list is loaded (`flag(1)` only makes `matchit` report its progress every 1% instead of every 20%). Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Stopped after 30 minutes. Each time is a single run.
- †**Stata estimates** extend `matchit`'s own progress report: (elapsed time ÷ share of master names done). Progress at the stop: workload 3: 45% of master names done after 29.9 min; workload 4: 8% of master names done after 28.8 min; workload 5: 5% of master names done after 28.0 min.
- **R `stringdist` (s)**: `stringsimmatrix(a, b, method = "cosine", q = 2, nthread = 12)` on blocks of master names, keeping pairs ≥ 0.5; R 4.6.1 with stringdist 0.9.17. Stopped after 30 minutes. ‡Because both Stata and R score every pair, a stopped run's total is estimated as its time per pair on the largest completed workload × the number of pairs.
- **Python reference (s)**: `reference.py`, timed with `time.perf_counter()`.

## Agreement with Stata and R

Agreement with the reference:

- **Stata** (workloads 1, 2): identical pairs: yes; score sums within 2e-16.
- **R** (workloads 1, 2, 3): finds slightly fewer pairs (workload 1: 5,571 vs 5,591; workload 2: 571,721 vs 575,479; workload 3: 11,256,169 vs 11,310,772). The missing pairs are pairs whose score is **exactly 0.5**, which is common because scores are ratios of small integers (e.g. 4 ÷ √(8 × 8)). `stringdist` reports similarity as 1 − distance, which turns some of these into 0.4999999999999999, just below the threshold. Lowering R's threshold by 1e-12 recovers every pair on workload 1. This is a reproducibility trap: the same threshold yields different candidate lists in Stata and R. Implementations must compute the score as matchit's formula does (dot product ÷ √(product of sums of squares)) so that exact 0.5 scores are kept.

Results are in `results/results.jsonl`.

## Note: match quality

Bigram cosine scores are dominated by shared suffixes and short words, so many unrelated names pass a 0.5 threshold (see "pairs found" vs "true links"). Applied work usually strips legal suffixes and common words first, or uses token weights (`matchit`'s `weights()` option). The benchmark keeps the defaults, as a pure computation.

## Appendix: workload table columns

The examples use one running case: linking patent assignees (master) to Compustat company names (using).

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–5 from smallest to largest. | `3` is patent assignees to public firms. |
| design | The linking task the workload imitates (see the research context below). | patent assignees to public firms |
| n_master | Names in the master list (the data in memory in Stata). | 50,000 assignee names |
| n_using | Names in the using list. | 10,000 company names |
| pairs compared | n_master × n_using: every possible pair. | 500 million |
| true links | Using names generated as noisy versions of a master name, so the correct match is known. | "ACME HOLDINGS INC" in one list, "ACME HOLDINGS CORP" in the other |
| pairs found (≥ 0.5) | Pairs whose score is at least 0.5, the candidates a researcher would review. Many are false positives that share only a suffix. | |
| true links found | True links among the pairs found. | |
| Stata `matchit` (s) | Seconds for `matchit`. Stopped after 30 minutes. | `30+ min (est. …†)`: stopped, total estimated |
| R `stringdist` (s) | Seconds for the R computation. | |
| Python reference (s) | Seconds for the Python reference. | |
| Could not compute | Outputs a tool did not produce on that workload, and why. | |

## Research context: what each workload represents

| # | Design | What it represents | Related settings |
|---|---|---|---|
| 1 | sanity check | No real setting: two lists of 500 names, to check correctness in seconds. | – |
| 2 | firm database to firm database | Linking two databases of about 5,000 public firms each (e.g. a firm list to an executive database) that use different identifiers or name spellings. | – |
| 3 | patent assignees to public firms | Linking 50,000 assignee names from patent data to 10,000 public companies, a classic step in innovation research. | Arora, Belenzon & Sheer (2021) match patents to Compustat firms for 1980–2015, handling name changes and ownership changes (the DISCERN data). |
| 4 | startups to startups | Linking 50,000 startups between two venture databases (e.g. Crunchbase and PitchBook), as when combining funding and outcome data. | – |
| 5 | job-posting employers to firm database | Linking 200,000 employer names from job postings to 20,000 firms. | Acemoglu, Autor, Hazell & Restrepo (NBER WP 28257) work with employers in Burning Glass job postings; Hosseini & Lichtinger (2025) link Revelio résumé and posting data to about 285,000 firms. |

Sources:
- Arora, A., Belenzon, S. & Sheer, L. (2021), "Matching Patents to Compustat Firms, 1980–2015: Dynamic Reassignment, Name Changes, and Ownership Structures," *Research Policy*. [Duke Scholars](https://scholars.duke.edu/publication/1476732)
- Acemoglu, D., Autor, D., Hazell, J. & Restrepo, P., "AI and Jobs: Evidence from Online Vacancies," NBER Working Paper 28257. [NBER conference version](https://conference.nber.org/conf_papers/f143876.pdf)
- Hosseini, S.M. & Lichtinger, G. (2025), "Generative AI as Seniority-Biased Technological Change: Evidence from U.S. Résumé and Job Posting Data," SSRN 5425555. [Stanford DEL seminar](https://digitaleconomy.stanford.edu/event/seyed-m-hosseini-and-guy-lichtinger-generative-ai-as-seniority-biased-technological-change-evidence-from-u-s-resume-and-job-posting-data)

## Implementation study

The existing Python sparse implementation is a strong baseline. The largest measured improvement came from an adapted C AllPairs join: **1.65× faster on W3**, with a **6.70× smaller first-fit process peak**, against the reference measured in the same native-extension round.

Measurements: Intel Core i7-8700B (6 cores), 32 GiB RAM, macOS 15.7.3. These are local synthetic-workload comparisons, not rankings of languages or universal fastest implementations.

The contract returns every pair whose character-bigram-count cosine is at least 0.5, preserving repeated bigrams, case and punctuation. Top-k truncation, TF-IDF and edit-distance scorers change the task. W1 is **500 × 500**, W2 **5,000 × 5,000**, and W3 **50,000 × 10,000** names; W4–W5 were not measured.

### R/Python round

Seconds below are medians of **three measured full-pipeline runs after a full warm-up**. Python 3.12.13 used SciPy 1.18.1 and sparse_dot_topn 1.2.0; R 4.1.2 used stringdist 0.9.17, Matrix 1.3-4 and proxyC 0.5.2. Adapters are experimental research code.

| Implementation | Threads | W1 seconds | W2 seconds | W3 seconds |
|---|---:|---:|---:|---:|
| Python/SciPy reference | 1 | 0.01699 | 0.83349 | 17.32495 |
| Python sparse_dot_topn, raw-count adapter | 1 | 0.01625 | 0.84045 | 20.52653 |
| Original R stringdist | 1 | 0.49300 | 50.92100 | — |
| R Matrix, raw-count adapter | 1 | 0.02200 | 1.79700 | 49.12500 |
| R proxyC, raw-count adapter | 1 | 0.04100 | 2.93600 | — |
| R proxyC, native cosine | 1 | 0.03100 | 1.69600 | 47.40600 |
| Python sparse_dot_topn, multicore supplement | 6 | — | 0.85663 | 13.70396 |

Original R **fails complete pair equality**: it returns 20 fewer pairs on W1 and **3,758 fewer on W2**. Saved evidence establishes net deficits and unequal pair arrays, not separate missing/extra counts. Ratios against this failing baseline are runtime reductions, not exact-output speedups. Original R and raw proxyC were not run on W3.

All raw-product adapters reproduce complete pairs and bitwise scores. Native proxyC reproduces complete pairs, with maximum individual-score difference **5.22e−15**; this is measured tolerance agreement, not a universal boundary guarantee. The reference returns **5,591 / 575,479 / 11,310,772 pairs**. Raw-product adapters retain the reference's normalization order at the cutoff.

Single-thread sparse_dot_topn does not improve W2/W3. Six threads reduce W3 time by **1.50× versus the same package at one thread**, or **1.26× versus this round's reference**. The installed proxyC build lacks TBB, so its multicore ceiling was not tested. This round provides no comparable isolated peak-memory measurements.

### Native-extension round

These are **fresh reference measurements**, with three warm measured runs per configuration. Do not pool the two rounds or calculate native speedups using the earlier reference times.

| Implementation | Threads | W1 seconds | W2 seconds | W3 seconds | W3 first-fit peak MiB |
|---|---:|---:|---:|---:|---:|
| Python/SciPy reference | 1 | 0.01591 | 0.74229 | 13.92602 | 3597.8 |
| Adapted author AllPairs, C | 1 | 0.01214 | 0.37828 | 8.46255 | 537.3 |
| Adapted author L2AP, C | 1 | 0.01302 | 0.45077 | 9.81247 | 564.2 |
| SuiteSparse:GraphBLAS 9.4.5 | 1 | 0.01932 | 1.04491 | 21.43227 | 6616.7 |
| SuiteSparse:GraphBLAS 9.4.5 | 6 | 0.01540 | 0.65843 | 14.69988 | 7661.4 |

AllPairs improves on the fresh reference by **1.31× / 1.96× / 1.65×**. On W3 its range is **8.29–8.67 s**, versus **13.88–14.43 s** for the reference. Six-thread GraphBLAS takes **13.83–15.69 s**; the overlapping ranges establish no useful speed advantage. L2AP performs fewer full-dot verifications than AllPairs but has a slower complete pipeline here.

All 15 configurations pass full pair and bitwise-score equality on **every warm-up and measured repetition**; 117 small pipeline fixtures also pass. The author code is adapted from L2AP 0.2.1 for bipartite joins and binary64 arithmetic, with conservative candidate-bound handling and final raw-count rescoring. This is not the untouched upstream executable or a proof for arbitrary floating-point inputs. The shared feature builder preserves Python code-point semantics; two Unicode fixtures do not establish exhaustive Unicode coverage.

GraphBLAS uses a Python adapter over compiled C integer sparse multiplication. The measured python-graphblas 2025.2.0 requires GraphBLAS below version 10; **no 10.x engine was measured**. AllPairs/L2AP are serial; six-thread GraphBLAS is a separate multicore comparison.

Memory is each fresh process's high-water mark immediately after its **first full invocation, before correctness allocations**. It includes imports, generated/loaded input and computation, not just algorithm allocation or the peak of the three timed repetitions. Compare these memory figures only within this round.

Both rounds time feature construction through materialized pair/score outputs. Generation, startup, imports, compilation, file I/O and correctness checks are excluded. Runs were sequential; fixed method order and uncontrolled desktop activity, temperature and frequency limit precision. Correctness is established on the measured inputs, not all possible names.

### Evidence and implementation sources

The raw observations and code of this early study are not included in this repository.

Primary implementation sources: [sparse_dot_topn](https://github.com/ing-bank/sparse_dot_topn), [proxyC sparse similarity](https://koheiw.github.io/proxyC/reference/simil.html), and [the author's AllPairs/L2AP distribution](https://www.davidanastasiu.net/software/l2ap/index.php).

## Reproduce

```bash
python deferred/007_fuzzymatch/bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp
```

Requires Stata/MP 19.5 with `matchit` and `freqindex` (SSC), R with `stringdist`, and Python with NumPy, SciPy and pandas.

Run the reproduction commands from the repository root. New baseline runs default to unique directories under `runs/`. Historical raw measurements are preserved under `deferred/007_fuzzymatch/results/`.
