# 010: Data larger than memory — baselines and results

> Planned task, not yet in the shared evaluator. The timings and results in this file and in `results/` were recorded early in the project with this directory's scripts and have not been re-checked under the evaluator.

`CPU + disk` · `data preparation` · `L1 (I/O and memory bound)` · `Stata 19.5 · R haven + data.table · pandas · DuckDB · Python streaming reference`

The task itself (recipes, inputs, outputs, workloads, budgets, checksums) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py`, `dta.py`, `generate.py` and `checksums.py`. This file holds what is kept out of the prompt: the motivating real-data workload, recorded results for existing tools, measurement details and improvement areas.

Prepare analysis datasets from yearly files whose total size exceeds the allowed memory.
The motivating workload contains 55.7 GB of commercial workforce position data: seven
2020–2026 Stata files, 121.9 million rows and 24 variables, on an 18 GB laptop.

**Main finding:** the streaming reference and DuckDB completed every measured recipe and met
every imposed budget. On the real files, DuckDB extracted in 8.3 s and aggregated in 1.7 s after a
separate 224 s conversion to Parquet. In-memory approaches depended on how much the recipe
reduced the data; all tools completed the aggregate. Correct Stata missing-value and coding
semantics are part of the task, not optional adjustments for a faster engine.

## Real-data workload

Workload 7 runs the recipes on the real files: 55.7 GB of commercial workforce position data, seven 2020–2026 Stata files, 121,931,548 rows, 3,216 real companies. It requires licensed data and has no memory budget; it records what happens on the 18 GB host.

## Implementations measured

| Tool | Pipeline |
|---|---|
| Stata 19.5 | Per-year `use varlist if … using`, then `append`/`encode`; per-year and final `collapse` for aggregate; `use` + `append` + `save` for append. |
| R | [bench_r.R](bench_r.R): `haven::read_dta(col_select = …)`, `data.table`, `haven::write_dta`; reduce each year first for aggregate. |
| pandas | [bench_python.py](bench_python.py): chunked filtering or aggregation, then concatenation and `to_stata`; append loads and concatenates the files. |
| DuckDB | SQL on Parquet copies, 12 requested threads, budget-derived `memory_limit` set to 60%, and streamed `.dta` output. Conversion is timed separately. |
| Python reference | [reference.py](reference.py) with NumPy and [dta.py](dta.py): chunked reading/writing, defining the target result. |

## Measurement

Recorded host: **Apple M3 Pro, 12 cores, 18 GB RAM, internal SSD**. Software: Stata/MP 19.5
(12-core license), R 4.6.1 with haven/data.table 1.18, Python 3.14 with pandas 3.0,
DuckDB 1.5.6 and pyarrow 26. These results are separate from other tasks' Intel-Mac studies.

- **Time:** each displayed value is a single recorded run, timed inside the tool with Stata
  `timer`, R `Sys.time()` or Python `time.perf_counter()`. Output writing is included; process
  startup is excluded. Top-level imports precede timing, but some Python helper imports occur
  inside the timer. Stata's timer has 1 ms resolution, despite five displayed decimals.
- **Peak memory:** the process's OS-reported peak footprint from `/usr/bin/time -l`, combined
  with `footprint` sampling approximately every second. It is not simply resident RAM;
  compression/swap affect runs. The budget covers the tool, not the system's file cache.
- **Units:** published tables retain their original “GB” labels. Input sizes use decimal GB;
  the runner enforces budgets in GiB (`2**30` bytes) and reports binary-unit memory peaks.
  The displayed data/budget ratios use the nominal labels, not byte-exact ratios.
- **Invalid/stopped runs:** “over budget” means stopped at a sampled excess or rejected from
  its final peak; “30+ min” means the time limit; “stopped: swap filled the disk” denotes the
  free-space guard (15 GB). These are not valid solutions. A ✗ denotes a checksum mismatch.
- **Cache preparation:** input files are read once before every tool run; DuckDB reads its
  Parquet inputs. Inputs up to about 8 GB can remain cached. This is a repeated-read setup,
  not a cold-disk comparison or a guarantee of identical cache residency.
- **SOL (“speed of light”):** measured once per workload/recipe as reading the `.dta` inputs
  once plus writing as many bytes as the reference output. **Efficiency = SOL / solution time**;
  each table reports the fastest valid tool. SOL is an approximate I/O baseline, not a strict bound.
  Buffered writes can yield reference append efficiencies of 1–1.5. DuckDB reads smaller,
  columnar Parquet inputs and reaches higher values (up to 8); conversion is excluded.
  With conversion included, its efficiencies are below 1 on all recorded workloads.

## Recorded results

Cells give **seconds · peak memory**.
The append-only [raw records](results/results.jsonl) also retain earlier W1/W2 runs;
these tables do not pool them into repeated-run medians.

### extract (US positions since 2015, 9 variables, coded)

| # | data ÷ budget | output rows | SOL (s) | Stata 19.5 | R | pandas | DuckDB | Python reference | best efficiency | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 614,770 | 0.07000 | 0.72500 s · 0.15 GB | 2.42035 s · 0.29 GB | 1.80697 s · 0.42 GB | 0.40850 s · 0.14 GB | 0.40710 s · 0.31 GB | 0.17 (Python reference) | – |
| 2 | 2 | 2,456,957 | 0.26000 | 2.85200 s · 0.40 GB | 8.52664 s · 0.76 GB | over budget (stopped at 1.27 GB) | 1.58107 s · 0.19 GB | 1.52138 s · 0.31 GB | 0.17 (Python reference) | pandas: over the memory budget |
| 3 | 8 | 9,834,176 | 1.68300 | over budget (stopped at 1.20 GB) | over budget (stopped at 1.00 GB) | over budget (stopped at 1.56 GB) | 1.88026 s · 0.21 GB | 7.13718 s · 0.32 GB | 0.90 (DuckDB) | Stata 19.5: over the memory budget; R: over the memory budget; pandas: over the memory budget |
| 4 | 0.5 | 2,456,957 | 0.43200 | 2.88700 s · 0.40 GB | 8.83928 s · 0.76 GB | 6.75054 s · 1.25 GB | 0.75834 s · 0.19 GB | 2.10211 s · 0.31 GB | 0.57 (DuckDB) | – |
| 5 | 2 | 9,834,176 | 1.77300 | 14.11400 s · 1.47 GB | 34.96447 s · 2.62 GB | 25.85609 s · 3.48 GB | 1.81434 s · 0.22 GB | 7.30509 s · 0.32 GB | 0.98 (DuckDB) | – |
| 6 | 8 | 39,344,490 | 7.05000 | over budget (stopped at 4.55 GB) | over budget (stopped at 4.71 GB) | over budget (stopped at 5.51 GB) | 6.65870 s · 0.34 GB | 26.70404 s · 0.27 GB | 1.06 (DuckDB) | Stata 19.5: over the memory budget; R: over the memory budget; pandas: over the memory budget |
| 7 | – (no budget) | 57,110,633 | 14.72400 | 1267.12000 s · 8.31 GB | 289.69754 s · 11.00 GB | 210.42079 s · 18.19 GB | 8.28926 s · 0.47 GB | 42.87151 s · 0.48 GB | 1.78 (DuckDB) | – |

### aggregate (firm-quarter panel)

| # | data ÷ budget | output rows | SOL (s) | Stata 19.5 | R | pandas | DuckDB | Python reference | best efficiency | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 90,528 | 0.06300 | 0.48500 s · 0.11 GB | 1.33681 s · 0.17 GB | 1.38258 s · 0.26 GB | 0.28641 s · 0.13 GB | 0.45311 s · 0.27 GB | 0.22 (DuckDB) | – |
| 2 | 2 | 125,495 | 0.24600 | 0.93100 s · 0.14 GB | 4.13784 s · 0.28 GB | 4.79253 s · 0.28 GB | 0.51497 s · 0.17 GB | 1.69385 s · 0.24 GB | 0.48 (DuckDB) | – |
| 3 | 8 | 144,586 | 1.61200 | 3.40800 s · 0.34 GB | 15.38260 s · 0.58 GB | 21.56402 s · 0.30 GB | 0.71279 s · 0.21 GB | 6.94777 s · 0.25 GB | 2.26 (DuckDB) | – |
| 4 | 0.5 | 125,495 | 0.42000 | 0.99200 s · 0.14 GB | 4.40574 s · 0.26 GB | 5.32814 s · 0.28 GB | 0.52137 s · 0.17 GB | 1.77364 s · 0.27 GB | 0.81 (DuckDB) | – |
| 5 | 2 | 144,586 | 1.71800 | 3.32600 s · 0.32 GB | 14.99086 s · 0.49 GB | 20.02025 s · 0.30 GB | 0.67283 s · 0.20 GB | 6.64674 s · 0.23 GB | 2.55 (DuckDB) | – |
| 6 | 8 | 151,355 | 6.57900 | 13.49600 s · 1.14 GB | 61.85055 s · 1.76 GB | 83.11123 s · 0.34 GB | 1.50046 s · 0.29 GB | 28.18063 s · 0.39 GB | 4.38 (DuckDB) | – |
| 7 | – (no budget) | 114,152 | 13.86100 | 21.76500 s · 1.59 GB | 109.56281 s · 2.62 GB | 136.00143 s · 0.37 GB | 1.72842 s · 0.35 GB | 43.57263 s · 0.44 GB | 8.02 (DuckDB) | – |

### append (all files, all variables)

| # | data ÷ budget | output rows | SOL (s) | Stata 19.5 | R | pandas | DuckDB | Python reference | best efficiency | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 1,094,089 | 0.30800 | 0.57900 s · 0.52 GB | 7.41036 s · 0.50 GB | over budget (stopped at 1.47 GB) | 1.60570 s · 0.54 GB | 0.39315 s · 0.18 GB | 0.78 (Python reference) | pandas: over the memory budget |
| 2 | 2 | 4,376,365 | 0.89500 | over budget (stopped at 1.30 GB) | over budget (stopped at 1.67 GB) | over budget (stopped at 1.09 GB) | 5.14639 s · 0.73 GB | 1.35158 s · 0.16 GB | 0.66 (Python reference) | Stata 19.5: over the memory budget; R: over the memory budget; pandas: over the memory budget |
| 3 | 8 | 17,505,467 | 4.33800 | over budget (stopped at 1.74 GB) | over budget (stopped at 1.16 GB) | over budget (stopped at 1.07 GB) | 19.31365 s · 0.86 GB | 5.34398 s · 0.14 GB | 0.81 (Python reference) | Stata 19.5: over the memory budget; R: over the memory budget; pandas: over the memory budget |
| 4 | 0.5 | 4,376,365 | 1.07600 | 3.63500 s · 2.13 GB | 27.58097 s · 1.67 GB | over budget (stopped at 4.78 GB) | 5.17161 s · 0.61 GB | 1.30582 s · 0.16 GB | 0.82 (Python reference) | pandas: over the memory budget |
| 5 | 2 | 17,505,467 | 7.97400 | over budget (stopped at 5.09 GB) | over budget (stopped at 4.72 GB) | over budget (stopped at 4.11 GB) | 18.94742 s · 0.76 GB | 5.47597 s · 0.14 GB | 1.46 (Python reference) | Stata 19.5: over the memory budget; R: over the memory budget; pandas: over the memory budget |
| 6 | 8 | 70,021,878 | 91.21100 | over budget (stopped at 4.99 GB) | over budget (stopped at 4.25 GB) | over budget (stopped at 4.00 GB) | 81.15040 s · 1.18 GB | 71.90424 s · 0.17 GB | 1.27 (Python reference) | Stata 19.5: over the memory budget; R: over the memory budget; pandas: over the memory budget |
| 7 | – (no budget) | 121,931,548 | 126.67900 | 415.13200 s · 57.00 GB | failed after 23 min: R's memory limit (18 GB) reached (peak 17.00 GB) | killed by macOS after 20 min: out of memory (peak 55.31 GB) | 149.85286 s · 1.55 GB | 90.34144 s · 0.15 GB | 1.40 (Python reference) | R: ran out of memory; pandas: ran out of memory |

### One-time conversion to Parquet

[arrow_io.py](arrow_io.py) streams `.dta` files into Parquet because this DuckDB baseline does
not read `.dta` directly. The following cost is **excluded** from the recipe tables. The real
files shrink from 55.7 GB to 8.04 GB; repeated queries can amortize the 224.46 s conversion.

| # | data | conversion time (s) | peak memory | Parquet size | `.dta` size |
|---|---|---|---|---|---|
| 1 | 0.5 | 2.93331 | 0.64 GB | 0.08 GB | 0.5 GB |
| 2 | 2 | 8.04020 | 0.59 GB | 0.32 GB | 2 GB |
| 3 | 8 | 31.34786 | 0.54 GB | 1.28 GB | 8 GB |
| 4 | 2 | 8.13130 | 0.54 GB | 0.32 GB | 2 GB |
| 5 | 8 | 31.87371 | 0.59 GB | 1.28 GB | 8 GB |
| 6 | 32 | 123.65018 | 0.54 GB | 5.12 GB | 32 GB |
| 7 | real | 224.45566 | 0.56 GB | 8.04 GB | 55.7 GB |

### Interpretation and limits

- Streaming approaches pass every measured workload. In-memory append fails above budget
  (W2/W3/W5/W6); extract fits after sufficient filtering but fails at the largest ratios.
  Aggregate succeeds for every tool because per-year/chunk reduction keeps the combined panel small.
- Representation matters: the W4 2 GB append uses 2.13 GB in Stata and 1.67 GB in R;
  pandas exceeds 4.78 GB, while the streaming reference uses 0.16 GB. These are measured
  footprints, not evidence that every data representation needs the same bytes per row.
- On real W7, all tools finish extract. Stata also finishes append in 415 s with a 57 GB
  footprint on the 18 GB machine; macOS compression, especially of padded strings, helps.
  pandas is killed after 20 minutes; R stops after 23 minutes at its default memory limit
  (`mem.maxVSize()`, the machine's RAM). Success without a budget is not an in-budget result.
- **Stata's extract bottleneck is `encode` in a diagnostic rerun:** of 565 s total, reading
  took 24 s, append 7 s, role encoding 359 s, state encoding 169 s, and saving 1.5 s.
  Encoding accounts for 93%. The main table records 1,267 s for the same recipe, illustrating
  substantial variation with memory pressure; the diagnostic rerun is not substituted for it.
- DuckDB is fastest for real-data extract/aggregate once converted. The reference is faster
  for append, where DuckDB must reconstruct the full `.dta` output. Conversion cost and
  single-run timing variability must accompany any speed comparison.

## Agreement and tool notes

All **79 completed candidate
comparisons** in the saved records pass this gate. This is checksum agreement, not a proof of
identical rows or label mappings: different values/text can have the same totals. Aggregate
sorting is required by the contract but not checked by these order-independent summaries.
The implementation tests `abs(actual-reference) <= 1e-10 * max(1, abs(reference))` for every
`sum_` key, including string lengths; the tolerance table in task.md retains the stricter exact-string-length
contract. Counts and label totals are checked exactly.

- Stata batch commands around 1,100 characters caused exit 133 here. The runner passes long
  file lists through a text file instead of the command line.
- pandas `DataFrame.to_stata` defaults to format 114; request `version=118`.

## Improvement areas

These opportunities follow from the measured bottlenecks; they are not implemented results.

| Area | Required improvement |
|---|---|
| **1. Stata-compatible disk engine** | Stream `use if`, `keep`, `merge`, `collapse`, `encode` and `by` operations into labelled `.dta` outputs. Preserve missing-value comparisons and sort order, empty-string missingness, byte-order codes, and Stata's zero sum for all-missing groups. Flag `merge m:m`: Stata pairs rows in order, whereas SQL joins form combinations. |
| **2. Faithful columnar companion** | Convert once and retain Stata types, labels, 1960-based dates and extended missing codes. Row-oriented `.dta` forces the current readers to scan padded rows; columnar storage enables projected reads. The present Parquet adapter is not fully faithful for every Stata feature. |
| **3. Faster bottleneck commands** | Target measured costs such as `encode`; check existing `ftools`/`gtools` coverage before replacing commands such as `collapse`. |
| **4. Preflight planner** | Estimate memory, output/temp-disk needs and likely cost from file headers and a sample; warn before a run swaps heavily or exceeds its budget. |
| **5. Verified migration** | Automate counts/sums/label comparisons between old and new pipelines, with stronger row/value checks when equivalence matters. Pair speed with explicit Stata-semantic validation. |
| **6. Disk-backed estimation** | Extend to OLS with fixed effects and clustered SEs using disk-backed processing or sufficient summaries where valid. This is unmeasured here; task 001 already exposes difficulty with poorly connected worker–firm panels. |

## Scope notes

Not covered in the recorded runs: strL; preserving `.a`–`.z` in DuckDB's single-null Parquet path; large merges,
reshape, deduplication or rolling windows; regressions; other engines such as Polars, Arrow
in R or Spark. Calling DuckDB from Stata's `python:` block is also unmeasured and would add
result-transfer costs. Existing Parquet metadata does not retain full value-label tables.

## Reproduce

Run from this task's directory, using a work folder outside Dropbox:

```bash
python bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp --work /path/outside/dropbox --out results/
```

Requires Stata/MP 19.5, R with haven/data.table, Python with NumPy/pandas/pyarrow/duckdb, and
macOS (`/usr/bin/time -l`, `footprint`). Allow about three times the largest input size as
free work space. Use `--ids 1 2 3 4 5 6` without the licensed real files; W7's paths are
currently configured in `bench.py`. `--out` appends records; choose a fresh directory to
separate a new run. The runner deletes generated data and temporary outputs after use.

**Current checkout gap:** `bench.py` unconditionally copies `bench_stata.do`, but that script
is absent from the committed folder. Restore it before using the command, even for a
non-Stata `--tools` selection.

## Research context

The real workload represents résumé/job-position panels used for firm-quarter research;
related work also uses tens of millions of online job postings:

- Hosseini & Lichtinger (2025), “Generative AI as Seniority-Biased Technological Change,”
  SSRN 5425555: [Stanford DEL seminar](https://digitaleconomy.stanford.edu/event/seyed-m-hosseini-and-guy-lichtinger-generative-ai-as-seniority-biased-technological-change-evidence-from-u-s-resume-and-job-posting-data).
- Acemoglu, Autor, Hazell & Restrepo, “AI and Jobs: Evidence from Online Vacancies,”
  NBER Working Paper 28257: [conference version](https://conference.nber.org/conf_papers/f143876.pdf).
