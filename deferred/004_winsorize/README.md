# 004: Winsorization — baselines and results

> Planned task, not yet in the shared evaluator. The timings and results in this file and in `results/` were recorded early in the project with this directory's scripts and have not been re-checked under the evaluator.

`CPU` · `data preparation` · `L1` · `Stata winsor2 1.1 · R data.table 1.18 · Python NumPy`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools and research context.

## Baseline timings (Apple M3 Pro)

| # | design | n_obs | n_vars | by() | n_groups | missing | cuts | values changed | Stata `winsor2` (s) | R `data.table` (s) | Python reference (s) | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1,000 | 3 | – | 1 | 6% | 1, 99 | 1.9% | 0.00400 | 0.00754 | 0.00129 | – |
| 2 | firm-year, pooled (Compustat) | 150,000 | 20 | – | 1 | 5% | 1, 99 | 2.0% | 0.58800 | 0.06295 | 0.31424 | – |
| 3 | firm-year, by year (Compustat) | 200,000 | 20 | year | 30 | 5% | 1, 99 | 2.0% | 4.91500 | 0.10660 | 0.67353 | – |
| 4 | firm-year, by industry-year | 1,000,000 | 10 | industry#year | 1,440 | 5% | 1, 99 | 1.8% | 10.84600 | 0.63892 | 2.15158 | – |
| 5 | stock-month, by month (CRSP monthly) | 3,000,000 | 5 | month | 360 | 5% | 1, 99 | 2.0% | 10.06800 | 0.32786 | 4.30639 | – |
| 6 | stock-day, by day (CRSP daily) | 20,000,000 | 3 | day | 2,500 | 5% | 1, 99 | 2.0% | 31.02800 | 1.22826 | 31.25233 | – |

How to read the timing columns (the appendix defines every column):

- **Stata `winsor2` (s)**: `winsor2 v1 … vk, cuts(1 99) replace [by(g)]`, timed after the data are loaded. Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Stopped after 30 minutes. Each time is a single run.
- **R `data.table` (s)**: `dt[, (vars) := lapply(.SD, winsorize), by = g, .SDcols = vars]`, where `winsorize` clips at `quantile(x, c(.01, .99), type = 2, na.rm = TRUE)`. R 4.6.1 with data.table 1.18 on 12 threads, data loaded with `fread` before timing. This is the common R idiom with the percentile definition set to match Stata; packages such as `DescTools::Winsorize` default to type 7.
- **Python reference (s)**: `reference.py`, timed with `time.perf_counter()`.
- R is faster than Stata on workloads 2, 3, 4, 5, 6 (9–46×); on the 1,000-row sanity check both take milliseconds. On the largest workload (20,000,000 rows), Stata takes 31.0 s, R 1.2 s and the Python reference 31.3 s; the reference is unoptimized (one full sort per variable) and is not a speed target.

## Agreement with Stata and R

Agreement with the reference:

- **R** (workloads 1, 2, 3, 4, 5, 6): sums, min and max within 3e-13; counts and values changed identical: yes.
- **Stata with `by()`** (workloads 3, 4, 5, 6): sums, min and max within 4e-16; counts and values changed identical: yes.
- **Stata pooled** (workloads 1, 2): sums, min and max within 2e-14; non-missing counts identical; "values changed" differs by up to 2 per variable. These are the values that equal a cutoff exactly: pooled `winsor2` rounds the cutoff through a macro (see the note below) and so moves them by about 1e-15. This is accepted as a known Stata quirk; the tolerance on "values changed" applies to new implementations, not to pooled `winsor2`.

Results are in `results/results.jsonl`.

## Note: pooled `winsor2` cutoffs

**A Stata quirk.** Without `by()`, `winsor2` stores each cutoff in a local macro (`local qleft = r(r1)`), which rounds it to about 16 significant digits. Values are then clipped to a cutoff that is off by about 1e-15 relative, and the values that equal a cutoff exactly are replaced by a number that differs from them by about 1e-15. This is numerically harmless, but it means pooled `winsor2` output is not bit-for-bit equal to the exact cutoffs. With `by()`, the cutoffs are kept in variables and match exactly.

## Appendix: workload table columns

The examples use one running case: winsorizing return on assets (ROA) in a Compustat firm-year panel.

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–6 from smallest to largest. | `3` is Compustat-style data winsorized within each year. |
| design | The data-preparation step the workload imitates (see the research context below). | firm-year, by year (Compustat) |
| n_obs | Rows in the data. | 200,000 firm-years |
| n_vars | Number of variables winsorized in the same call. | 20: ROA, leverage, Tobin's q, sales growth, … |
| by() | The grouping variable: percentiles are computed separately within each group. "–" means pooled (one set of cutoffs for the whole sample). `a#b` means each combination of `a` and `b`. | `year`: ROA is clipped at each year's own 1st and 99th percentiles, so a crisis year with many losses is not clipped by the cutoffs of normal years. |
| n_groups | Number of groups. | 30 years |
| missing | Share of values that are missing. They stay missing and are not used to compute the cutoffs. | 5% |
| cuts | The lower and upper percentiles, as in `winsor2`'s `cuts()`. | 1, 99 |
| values changed | Share of non-missing values that winsorizing changed (about 2% for cuts of 1 and 99). | 2.0% |
| Stata `winsor2` (s) | Seconds for the `winsor2` call. Stopped after 30 minutes. | |
| R `data.table` (s) | Seconds for the R winsorizing step. | |
| Python reference (s) | Seconds for the Python reference. | |
| Could not compute | Outputs a tool did not produce on that workload, and why. | "–" means every tool produced every output. |

## Research context: what each workload represents

| # | Design | What it represents | Related papers |
|---|---|---|---|
| 1 | sanity check | No real setting: 1,000 rows and 3 variables, to check correctness in milliseconds. | – |
| 2 | firm-year, pooled (Compustat) | A Compustat-style firm-year panel (150,000 firm-years) with 20 accounting variables, winsorized once over the whole sample: the most common version of this step. | Adams, Hayunga, Mansi, Reeb & Verardi (2019) review how finance studies identify and treat outliers, including winsorizing and trimming. |
| 3 | firm-year, by year (Compustat) | The same kind of panel (200,000 firm-years over 30 years), winsorized within each year so that cutoffs follow changes in the cross-section over time. | Kothari, Sabino & Zach (2005) examine the implications of data trimming for tests of market efficiency. |
| 4 | firm-year, by industry-year | A larger firm panel (1,000,000 rows, 10 variables) winsorized within each of 1,440 industry-years (48 industries × 30 years), for ratios whose scale differs by industry. | – |
| 5 | stock-month, by month (CRSP monthly) | 3,000,000 stock-months (about 30 years of monthly returns and characteristics), 5 variables, winsorized within each of 360 months, as in cross-sectional asset-pricing tests. | – |
| 6 | stock-day, by day (CRSP daily) | 20,000,000 stock-days (about 8,000 stocks over 2,500 trading days), 3 variables, winsorized within each trading day. | – |

Sources:
- Hyndman, R.J. & Fan, Y. (1996), "Sample Quantiles in Statistical Packages," *The American Statistician* 50(4), 361–365. [R documentation of the nine definitions](https://rdocumentation.org/packages/stats/topics/quantile)
- Adams, J., Hayunga, D., Mansi, S., Reeb, D. & Verardi, V. (2019), "Identifying and Treating Outliers in Finance," *Financial Management* 48(2), 345–384. [RePEc](https://ideas.repec.org/a/bla/finmgt/v48y2019i2p345-384.html)
- Kothari, S.P., Sabino, J.S. & Zach, T. (2005), "Implications of Survival and Data Trimming for Tests of Market Efficiency," *Journal of Accounting and Economics* 39(1), 129–161. [RePEc](https://ideas.repec.org/a/eee/jaecon/v39y2005i1p129-161.html)

## Reproduce

```bash
python bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp --out results/
```

Requires Stata/MP 19.5 with `winsor2` (`ssc install winsor2`), R with `data.table`, and Python with NumPy and pandas.
