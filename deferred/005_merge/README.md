# 005: Table joins — baselines and results

> Planned task, not yet in the shared evaluator. The timings and results in this file and in `results/` were recorded early in the project with this directory's scripts and have not been re-checked under the evaluator.

`CPU` · `data preparation` · `L1` · `Stata merge · R data.table 1.18 · Python pandas 3.0`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools and research context.

## Baseline timings (Apple M3 Pro)

| # | design | merge | keys | key type | n_master | n_using | rows out | matched | master only | using only | Stata `merge` (s) | of which reading using (s) | R `data.table` (s) | Python `pandas` (s) | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1:1 | `id` | numeric | 1,000 | 1,000 | 1,174 | 826 | 174 | 174 | 0.00600 | 0.00000 | 0.00266 | 0.00330 | – |
| 2 | firm-year 1:1 (Compustat + ExecuComp) | 1:1 | `firm year` | numeric | 200,000 | 44,000 | 204,700 | 39,300 | 160,700 | 4,700 | 0.06000 | 0.00200 | 0.06193 | 0.04335 | – |
| 3 | firm-year m:1 firm data, string key (CUSIP) | m:1 | `cusip` | string | 200,000 | 10,000 | 200,500 | 190,000 | 10,000 | 500 | 0.08500 | 0.00100 | 0.09766 | 0.04271 | – |
| 4 | stock-month m:1 firm-year (CRSP + Compustat) | m:1 | `permno year` | numeric | 3,000,000 | 200,000 | 3,017,890 | 2,185,320 | 814,680 | 17,890 | 0.75900 | 0.00700 | 0.38591 | 0.33689 | – |
| 5 | job postings m:1 employers, string key | m:1 | `employer` | string | 10,000,000 | 475,000 | 10,027,869 | 9,112,457 | 887,543 | 27,869 | 3.86500 | 0.00900 | 8.88257 | 4.58615 | – |
| 6 | stock-day m:1 stock-month (CRSP daily + monthly) | m:1 | `permno month` | numeric | 20,160,000 | 1,088,296 | 20,317,131 | 19,554,465 | 605,535 | 157,131 | 2.37800 | 0.01600 | 2.82612 | 1.30368 | – |

How to read the timing columns (the appendix defines every column):

- **Stata `merge` (s)**: `merge {1:1|m:1} keys using using.dta`, timed after the master table is loaded. Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM. `merge` reads the using table from disk inside the command; **of which reading using (s)** is a separate timing of `use using.dta` on its own, which shows how much of the merge time is file reading. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Stopped after 30 minutes. Each time is a single run.
- **R `data.table` (s)**: `merge(x, y, by = keys, all = TRUE)` on two tables already in memory (loaded with `fread` before timing), R 4.6.1 with data.table 1.18 on 12 threads. It does not check key uniqueness.
- **Python `pandas` (s)**: `pandas.merge(x, y, on = keys, how = "outer", indicator = True, validate = …)` on two DataFrames in memory, pandas 3.0, timed with `time.perf_counter()`. `validate` checks key uniqueness as Stata does. pandas also serves as the reference: it defines the expected output.
- **Merging is not a bottleneck in Stata at these sizes.** Every tool finishes every merge in under 10 seconds, Stata's `merge` is within about 2× of the fastest tool everywhere, and it is the fastest of the three on the 10-million-row string-key merge (workload 5). Reading the using file takes Stata at most 0.02 s.
- Fastest tool by workload: 1: R, 2: pandas, 3: pandas, 4: pandas, 5: Stata, 6: pandas. On the largest workload (20,160,000 master rows), Stata takes 2.4 s, R 2.8 s and pandas 1.3 s.

## Agreement with Stata and R

Agreement with the reference (pandas):

- **Stata** (workloads 1, 2, 3, 4, 5, 6): rows and counts identical: yes; column sums within 3e-14.
- **R** (workloads 1, 2, 3, 4, 5, 6): rows and counts identical: yes; column sums within 2e-11.

Column sums differ only in the last digits because the tools add the same numbers in different orders.

Results are in `results/results.jsonl`.

## Appendix: workload table columns

The examples use one running case: adding CEO characteristics (using table) to a Compustat firm-year panel (master table).

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–6 from smallest to largest. | `2` is the Compustat + ExecuComp merge. |
| design | The merge the workload imitates (see the research context below). | firm-year 1:1 (Compustat + ExecuComp) |
| merge | `1:1`: each key combination appears at most once in both tables. `m:1`: it may repeat in the master table but appears at most once in the using table. | `1:1`: one row per firm-year in both tables |
| keys | The variables the tables are matched on. | `firm year` |
| key type | `numeric` (integer codes) or `string` (text codes, slower to compare). | `string`: 8-character CUSIP-style codes |
| n_master | Rows in the master table (the one in memory in Stata). | 200,000 firm-years |
| n_using | Rows in the using table. | 44,000 CEO-years |
| rows out | Rows after the merge: matched + master only + using only. | 204,700 |
| matched | Rows with `_merge` = 3: the key is in both tables. | 39,300 firm-years have CEO data |
| master only | Rows with `_merge` = 1: in master, not in using. | 160,700 firm-years without CEO data |
| using only | Rows with `_merge` = 2: in using, not in master. | 4,700 CEO-years for firms or years not in the panel |
| Stata `merge` (s) | Seconds for the `merge` command, including reading the using table from disk. Stopped after 30 minutes. | |
| of which reading using (s) | Seconds for `use using.dta` on its own: the part of Stata's time spent reading the file. | |
| R `data.table` (s) | Seconds for `merge(…, all = TRUE)` on tables in memory. | |
| Python `pandas` (s) | Seconds for `pandas.merge(…, how = "outer")` on tables in memory; also the reference. | |
| Could not compute | Outputs a tool did not produce on that workload, and why. | "–" means every tool produced every output. |

## Research context: what each workload represents

| # | Design | What it represents | Related papers |
|---|---|---|---|
| 1 | sanity check | No real setting: two 1,000-row tables, to check correctness in milliseconds. | – |
| 2 | firm-year 1:1 (Compustat + ExecuComp) | Adding executive data for about 2,000 large firms to a 200,000 firm-year panel on firm and year: a 1:1 merge where most panel rows have no match. | Jenter & Kanaan (2015) link CEO turnovers to firm and industry performance in U.S. public firms. |
| 3 | firm-year m:1 firm data, string key (CUSIP) | Adding time-invariant firm information (e.g. founding year, headquarters) to a firm-year panel on an 8-character CUSIP-style identifier. | – |
| 4 | stock-month m:1 firm-year (CRSP + Compustat) | Adding annual accounting data (20 variables) to 3 million stock-months on stock id and year, the core step of linking CRSP returns to Compustat fundamentals. | – |
| 5 | job postings m:1 employers, string key | Adding employer characteristics to 10 million job postings on a 10-character employer id; a few employers post a large share of all postings. | Acemoglu, Autor, Hazell & Restrepo (NBER WP 28257) use establishment-level Burning Glass job postings; Hosseini & Lichtinger (2025) link Revelio résumés and postings to about 285,000 firms. |
| 6 | stock-day m:1 stock-month (CRSP daily + monthly) | Adding monthly characteristics to 20 million daily stock returns on stock id and month. | – |

Sources:
- Jenter, D. & Kanaan, F. (2015), "CEO Turnover and Relative Performance Evaluation," *Journal of Finance* 70(5), 2155–2184. [LSE Research Online](https://researchonline.lse.ac.uk/id/eprint/64421)
- Acemoglu, D., Autor, D., Hazell, J. & Restrepo, P., "AI and Jobs: Evidence from Online Vacancies," NBER Working Paper 28257. [NBER conference version](https://conference.nber.org/conf_papers/f143876.pdf)
- Hosseini, S.M. & Lichtinger, G. (2025), "Generative AI as Seniority-Biased Technological Change: Evidence from U.S. Résumé and Job Posting Data," SSRN 5425555. [Stanford DEL seminar](https://digitaleconomy.stanford.edu/event/seyed-m-hosseini-and-guy-lichtinger-generative-ai-as-seniority-biased-technological-change-evidence-from-u-s-resume-and-job-posting-data)

## Reproduce

```bash
python bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp --out results/
```

Requires Stata/MP 19.5, R with `data.table`, and Python with pandas.
