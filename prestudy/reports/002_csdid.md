# 002: Staggered-adoption difference-in-differences — baselines and results

`CPU` · `estimator` · `L2 (many small models + aggregation)` · `Stata csdid 1.81 / drdid 1.91 · csdid2 1.3` · `workload set v2`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](../../002_csdid/task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools, research context and the October 2026 implementation study.

## Baseline timings (Apple M3 Pro)

**Workload set v2** replaces the v1 workloads (synthetic one-factor-at-a-time scaling cases) with recognizable research settings, numbered 1–6 from small to large, and re-times all baselines on Stata 19.5. v1 workload numbers do not carry over.

| # | design | n_units | n_periods | n_cohorts | share treated | n_covariates | n_gt | n_obs (long) | Stata csdid (s) | Stata csdid2 (s) | Reference (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1,000 | 10 | 4 | 71% | 2 | 36 | 10,000 | 1.90300 | 0.54200 | 0.00897 |
| 2 | state/county policy adoption | 3,000 | 20 | 10 | 51% | 3 | 190 | 60,000 | 39.17700 | 2.36000 | 0.13404 |
| 3 | firm-year study | 5,000 | 30 | 20 | 25% | 5 | 580 | 150,000 | 457.23900 | 8.78200 | 0.90349 |
| 4 | private firms / establishments, annual | 50,000 | 20 | 10 | 69% | 10 | 190 | 1,000,000 | 825.20000 | 11.29900 | 1.87195 |
| 5 | monthly platform rollout | 100,000 | 60 | 40 | 69% | 5 | 2,360 | 6,000,000 | 30+ min (est. ~14.9 h†) | 388.37700 | 40.95132 |
| 6 | GenAI adoption, firm-quarter | 285,000 | 42 | 10 | 10% | 5 | 410 | 11,970,000 | 30+ min (est. > 4.2 h‡) | 236.95000 | 39.66621 |

How to read the timing columns (the appendix defines every column):

- **Stata csdid (s)** and **Stata csdid2 (s)** are the two baselines. Each is the estimation command plus post-estimation (`estat simple` and `estat event`; for `csdid2` also `estat attgt`, which retrieves the ATT(g,t)), timed after the data are loaded. The split between estimation and post-estimation is in `../baselines/002_csdid/results/results.jsonl`. `csdid2` is the faster Stata option in this historical baseline block, so speedup claims within that block use it as the comparator. Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM, with `set max_memory 12g` so that memory errors are reported instead of the machine swapping. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Each time is a single run.
- **`csdid` is stopped after 30 minutes.** †The estimate scales the elapsed time to all cells: (elapsed time ÷ cells finished) × n_gt. Since later cells are no faster than earlier ones, it is a lower bound. ‡`csdid` prints one progress line per 50 cells; when no line had appeared, fewer than 50 cells were done, and the estimate assumes 49 (a strict lower bound). Progress at the stop: workload 5: `csdid` had finished 50 of 2,360 cells after 18.9 min; workload 6: `csdid` finished fewer than 50 of 410 cells in 30 min.
- **`csdid` would also run out of memory on the largest workloads.** It keeps every cell's influence function as a double-precision variable over all rows of the long dataset until it finishes, which needs n_gt × n_obs × 8 bytes: workload 5: 113 GB; workload 6: 39 GB. That is more than the 18 GB of RAM in this machine, so these runs could not finish even without the time limit.
- **`csdid2` and the reference have no time limit** and always run to completion.
- **Reference (s)** is the entire Python computation, including the aggregations, timed with `time.perf_counter()`. It is comparable to the Stata columns.

## Agreement with Stata

The reference itself matches Stata on every completed run (`csdid`: workloads 1–4; `csdid2 ..., short`: workloads 1–6):

- All ATT estimates (ATT(g,t), simple, event study, pre/post averages) agree to within 1e-09 relative error.
- All standard errors agree to within 4e-09 relative error.

Results are in `../baselines/002_csdid/results/results.jsonl`.

## Appendix: workload table columns

The examples use one running case: firms adopting a remote-work policy, observed every year.

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–6 from smallest to largest. | `6` is the GenAI firm-quarter workload. |
| design | The research setting the workload imitates (see the research context below). | firm-year study |
| n_units | Number of distinct units followed over time: distinct values of `ivar()`. | 5,000 firms |
| n_periods | Number of time points per unit (years, quarters or months). | 30 years (1991–2020) |
| n_cohorts | Number of distinct adoption periods. Units first treated in the same period form one cohort. | Firms adopting in 2013 are the 2013 cohort. 20 cohorts = firms adopted in 20 different years. |
| share treated | Share of units ever treated. The rest are never treated and form the comparison group. | 25% of firms adopt at some point; 75% never do. |
| n_covariates | Number of baseline (pre-treatment) control variables in `X`, used for both the propensity score and the outcome model. | 5: firm size, age, industry, R&D intensity, leverage |
| n_gt | Number of cohort × period cells, n_cohorts × (n_periods − 1). `csdid` estimates a separate small DiD, ATT(g,t), for each. | ATT(2013, 2015) = effect on the 2013 cohort in 2015. 20 cohorts × 29 periods = 580 cells. |
| n_obs (long) | Rows in Stata's long-format data: n_units × n_periods. | 5,000 firms × 30 years = 150,000 rows |
| Stata csdid (s) | Seconds for `csdid` + `estat simple` + `estat event`. Stopped after 30 minutes. | `30+ min (est. ~5 h†)` means stopped, with the total estimated from its progress. |
| Stata csdid2 (s) | Seconds for `csdid2 ..., short` + `estat attgt` + `estat simple` + `estat event`. No time limit. | |
| Reference (s) | Seconds for the Python reference to produce all outputs. | |

## Research context: what each workload represents

| # | Design | What it represents | Similar published settings |
|---|---|---|---|
| 1 | sanity check | No real setting: a tiny panel that checks correctness in seconds. | – |
| 2 | state/county policy adoption | Local units (counties, cities, markets) exposed to a policy or entry event that starts in different years in different places. About half the units are eventually treated. | Callaway & Sant'Anna (2021): county-level teen employment and state minimum-wage increases, 2001–2007, the paper that introduced this estimator. Greenwood & Wattal (2017): staggered Uber entry across California markets, 2009–2014, and alcohol-related traffic fatalities. |
| 3 | firm-year study | A firm-level annual panel (Compustat-type public firms, or startups) over a long window, where firms adopt a practice or are hit by a state law in different years. A minority of firms are treated. | Png (2017): staggered state adoption of the Uniform Trade Secrets Act and firm R&D, 1979–1998. Babina, Fedyk, He & Hodson (NBER WP 31325): firm-level AI investment measured from résumé and job-posting data (firm-level AI adoption, though not estimated with this estimator). |
| 4 | private firms / establishments, annual | A large annual panel of private firms or establishments (e.g. Orbis, NETS, job-posting establishments) with 10 baseline controls. | Acemoglu, Autor, Hazell & Restrepo (NBER WP 28257): establishment-level AI vacancies from Burning Glass job postings, 2007 and 2010–2018 (similar data scale; the design uses AI exposure rather than adoption timing). |
| 5 | monthly platform rollout | Users, sellers, stores or workers on a platform who receive a feature in different months. Five years of monthly data with 40 rollout waves. | Brynjolfsson, Li & Raymond (2025): staggered rollout of a generative-AI assistant to 5,172 customer-support agents, with high-frequency productivity data (fewer units, same staggered-rollout design). |
| 6 | GenAI adoption, firm-quarter | Employer-wide data built from LinkedIn-type résumés and job postings (Revelio, Lightcast): about 285,000 firms, quarterly 2015–2025, with about 10% of firms adopting generative AI in the 10 quarters from 2023 onward. | Hosseini & Lichtinger (2025): about 285,000 U.S. firms and 62 million workers from Revelio résumé and job-posting data, 2015–2025; firms count as adopters once they post "GenAI integrator" roles; difference-in-differences of junior vs senior employment. |

Sources:
- Callaway, B. & Sant'Anna, P. (2021), "Difference-in-Differences with Multiple Time Periods," *Journal of Econometrics* 225(2), 200–230. [arXiv](https://arxiv.org/abs/1803.09015)
- Greenwood, B. & Wattal, S. (2017), "Show Me the Way to Go Home: An Empirical Investigation of Ride-Sharing and Alcohol Related Motor Vehicle Fatalities," *MIS Quarterly* 41(1), 163–187. [MISQ](https://misq.org/show-me-the-way-to-go-home-an-empirical-investigation-of-ride-sharing-and-alcohol-related-motor-vehicle-fatalities.html)
- Png, I.P.L. (2017), "Law and Innovation: Evidence from State Trade Secrets Laws," *Review of Economics and Statistics* 99(1), 167–179. [RePEc](https://ideas.repec.org/a/tpr/restat/v99y2017i1p167-179.html)
- Babina, T., Fedyk, A., He, A. & Hodson, J., "Firm Investments in Artificial Intelligence Technologies and Changes in Workforce Composition," NBER Working Paper 31325. [NBER](https://www.nber.org/papers/w31325)
- Acemoglu, D., Autor, D., Hazell, J. & Restrepo, P., "AI and Jobs: Evidence from Online Vacancies," NBER Working Paper 28257. [NBER conference version](https://conference.nber.org/conf_papers/f143876.pdf)
- Brynjolfsson, E., Li, D. & Raymond, L. (2025), "Generative AI at Work," *Quarterly Journal of Economics* 140(2), 889–942. [arXiv](https://arxiv.org/abs/2304.11771)
- Hosseini, S.M. & Lichtinger, G. (2025), "Generative AI as Seniority-Biased Technological Change: Evidence from U.S. Résumé and Job Posting Data," SSRN 5425555. [Stanford DEL seminar](https://digitaleconomy.stanford.edu/event/seyed-m-hosseini-and-guy-lichtinger-generative-ai-as-seniority-biased-technological-change-evidence-from-u-s-resume-and-job-posting-data)

## Implementation study (14 October 2026)

On this Intel Mac, the task-specific R DRDID adapter wins the larger measured cases in the R/Python block. Stata csdid 2.0.0 reproduces the outputs but is slower than its freshly measured reference. These are separate timing blocks; do not combine them into one ranking.

Measured on an Intel Core i7-8700B, six cores, 32 GiB RAM, macOS 15.7.3, against benchmark baseline `25f279d`. These results are separate from the repository's historical Apple M3 Pro measurements.

The target is balanced-panel traditional doubly robust DiD: logit-MLE propensity, control OLS, never-treated controls, varying pre-treatment bases, analytic influence-function SEs and the task's aggregation conventions. All 12 fields are checked, including exact group/time/event labels, ATT relative error ≤1e-6 and SE relative error ≤1e-4.

| Workload | Units × periods | Cohorts | Covariates | Long-panel rows |
|---|---:|---:|---:|---:|
| W1 | 1,000 × 10 | 4 | 2 | 10,000 |
| W2 | 3,000 × 20 | 10 | 3 | 60,000 |
| W3 | 5,000 × 30 | 20 | 5 | 150,000 |
| W4 | 50,000 × 20 | 10 | 10 | 1,000,000 |

Every reported timing is a median of three measured fits after one full-workload warm-up. Generation, imports, file transport and external format conversion are excluded unless added explicitly below; internal preprocessing, estimation, SEs and all required aggregates are included. Bootstrap and simultaneous confidence bands are disabled.

One compute thread is requested through BLAS/OpenMP/pool settings; R uses cores/jobs=1 with native parallel modes off, and Stata confirms one processor. Apple Accelerate's effective pool size is not enumerated by threadpoolctl. Runs are sequential, but desktop activity and CPU frequency are uncontrolled; small differences are not statistically established.

**R/Python block: loaded-input seconds.** Python 3.12.13 and R 4.1.2; every engine/workload result passes all 12 checks.

| Implementation | W1 | W2 | W3 | W4 |
|---|---:|---:|---:|---:|
| Repository NumPy reference | **0.0359** | **0.4325** | 6.9426 | 12.7758 |
| ModernDiD 0.2.0 + pre-average adapter | 0.2326 | 2.4127 | 15.6415 | 29.3524 |
| differences 0.3.0 + pre-average adapter | 0.4514 | 4.0236 | 25.7454 | 49.3046 |
| Python csdid 0.4.2, faster mode + pre-average adapter | 0.3424 | 3.2559 | 15.5295 | 25.6817 |
| R did 2.5.1, faster mode | 0.1190 | 2.6240 | 9.6920 | 12.5220 |
| R DRDID 1.3.0 cells + task aggregation | 0.0530 | 0.7860 | **3.8990** | **9.0950** |
| R fastdid 1.0.6 + task adapter | 0.8320 | 4.3270 | 17.6830 | 21.2920 |

The DRDID adapter is custom orchestration of native panel cells and task-specific aggregation, avoiding unnecessary full-cell covariance/Wald diagnostics. Its W3/W4 speedups are **1.78×/1.40×**. Native R did's W4 range, 12.274–12.546 s, overlaps the reference's 11.885–13.430 s: treat that small median difference as a tie. Fastdid uses one fit with native cell/event influence functions, followed by adapters. [DRDID panel estimator](https://psantanna.com/DRDID/reference/drdid_panel.html).

differences and fastdid omit 0.995 control-score trimming; all four workloads have zero trimmed controls. Several packages also differ at the exact 0.995 boundary or impose additional overlap guards. Passing here establishes workload-specific agreement, not unrestricted equivalence. Worst errors across this block are 3.27e-7 ATT and 2.39e-6 SE.

**Stata extension block: fresh reference and separate timings.** This is Stata csdid 2.0.0's Mata rewrite, not the archived package named `csdid2`. All 12 outputs pass on W1–W4.

| Workload | Fresh NumPy reference | Stata loaded-long fit + summaries | Stata preparation + fit, composed |
|---|---:|---:|---:|
| W1 | 0.027745 s | 0.082 s | 0.085493 s |
| W2 | 0.377031 s | 0.881 s | 0.888054 s |
| W3 | 3.345385 s | 9.223 s | 9.245359 s |
| W4 | 7.483492 s | 15.728 s | 15.957944 s |

Stata 18 MP uses `method(dr) nevertreated base_period(varying) analytical pointwise`; its changed defaults must be overridden. A small adapter obtains the pre-event average/SE from native event covariance. The composed column adds one separately observed array-to-long conversion to the native median; it is not a repeated end-to-end measurement. Stata timer resolution is milliseconds. W4 ranges are 7.074–10.245 s for the reference and 15.469–15.930 s for Stata. Nuisance caches reset between fits. [Pinned official September 27 release](https://github.com/pedrohcgs/csdid-stata/commit/31a577382d7c09757ed3baa095659f45cc63c107).

**Excluded candidate:** diff-diff 3.12.0's DR method divides control-odds augmentation by treated count, whereas this task normalizes by the sum of control odds. A source-only W1 probe fails ATT/SE tolerances (maximum relative errors 0.00690/0.0001547), with clipping verified inactive. This is an estimator mismatch, not a Rust performance result; its Rust backend was not built or ranked. [Pinned estimator source](https://github.com/igerber/diff-diff/blob/ff03d08c6762306b7b69ad388c3ea7bb4cb11c4a/diff_diff/staggered.py#L3721-L3778).

The strongest repository optimization opportunities are caching identical cohort propensity fits, reusing the never-treated OLS factorization, and reducing dense influence-function/cohort-weight storage. Preserve nuisance-estimation and aggregation-weight uncertainty when doing so. W5/W6, at 6M/11.97M rows, remain unmeasured.

Full scripts, raw observations, hashes and reproduction instructions are retained locally in the sibling `repro-socialscibench` bundle, archive commit `af01b9c`; that companion has not been published. This report retains the main comparison results.

## Same-session comparison (18 October 2026)

A new candidate, `solutions/002/shared-nuisance`, and five existing implementations wrapped as candidates (`r-drdid-cells`, `r-did`, `moderndid`, `csdid-py`, `stata-csdid`, adapters following the 14 October study) were run through the shared evaluator (protocol v2) one after another, on the same Intel Core i7-8700B Mac (32 GiB, macOS 15.7.3) and in one Python environment. All six workloads are covered. Each evaluator run times its own reference.

`shared-nuisance` uses the fact that a cohort's cells share their sample: it fits the propensity logit once per cohort instead of once per cell, fits one never-treated outcome regression for all periods, and evaluates each cell's doubly robust influence function in closed form in one compiled pass over units. That pass accumulates the cell variances and the simple and event-time aggregates without forming the units × cells influence-function matrix. Nuisance-estimation and cohort-weight terms are kept. Method, sources and failed attempts are in its `research.md`.

**Scope.** One thread; one warmup and three measured calls per workload; medians of complete `run` calls, including, for R DRDID, copying the data into R. Generation, imports and the first call (for `shared-nuisance`, about 3 s of Numba compilation) are excluded. These timings are not comparable with the M3 Pro table or the 14 October study.

| Workload | Reference | shared-nuisance | R DRDID cells | R DRDID cells / shared-nuisance |
|---|---:|---:|---:|---:|
| W1 | 0.0242 | **0.00187** | 0.0469 | 25.1 |
| W2 | 0.345 | **0.00708** | 0.453 | 64.1 |
| W3 | 3.087 | **0.0394** | 2.985 | 75.9 |
| W4 | 6.936 | **0.104** | 7.088 | 68.3 |
| W5 | 96.72 | **0.807** | 137.3 | 170.1 |
| W6 | 116.1 | **1.351** | 142.6 | 105.6 |

Seconds. The reference column is from the `shared-nuisance` run; the R DRDID run measured it within 15% (0.0281, 0.396, 3.186, 6.228, 96.74 and 115.6 s). The last column does not depend on the reference. Ranges, first calls, errors and memory are in each candidate's `research.md`.

- `shared-nuisance` passes every workload, 13–120× faster than the reference. Its worst errors are 1.0e-9 (ATT) and 2.2e-14 (SE); peak memory is 0.76 GiB on W6, against 12.5 GiB for the reference on W5.
- R DRDID cells passes every workload but is 0.6–1.1× the reference's speed here. The 14 October study measured it 1.4–1.8× faster than the reference on W3 and W4; in this session the reference measured faster (3.09 s against 6.94 s on W3) and the cause was not investigated.
- R did 2.5.1, ModernDiD 0.2.0, Python csdid 0.4.2 and Stata csdid 2.0.0 fail the `dev-propensity-trimming` fixture, so the evaluator timed no workloads for them. That fixture has one comparison unit with propensity above 0.995, which the task drops from the comparison group. These packages instead flag the cells as overlap violations and return no estimate, and their aggregation then stops on the missing cells. No comparison unit is trimmed on W1–W6 (largest propensity 0.68), so on the workloads alone they might pass, as on W1–W4 in the 14 October study; that was not measured here.

**Interpretation.** The table measures this task's six workloads at one thread; it does not rank general-purpose libraries.

- The speedup comes from exact algebra specific to this specification: never-treated controls, time-invariant covariates and a balanced panel make every cell of a cohort share one propensity fit and every cell share one outcome regression. The packages support not-yet-treated controls, unbalanced panels and repeated cross-sections, where those fits differ by cell, and they estimate cells independently.
- `shared-nuisance` was designed in an informed session after reading the 14 October study's recommendations and the workloads. It computes only the task's outputs; the library times include data handling, checks and extra results.
- The overlap guards are a deliberate design choice of those packages, not an error. Their failures follow the task's trimming rule, which R DRDID implements and they do not.
- One thread hides any multithreading. The candidate is not multithreaded either.
- The same author wrote the candidate and the competitor adapters; the adapters follow the 14 October study's.

Fairer follow-ups: competitor runs on W1–W6 without the trimming fixture (a separately scoped benchmark change); held-out designs evaluated blind; and a multithreaded round.

Run IDs, archive hashes and the batch commands are in `published/002/same-session-2026-10-18.md`.

## Reproduce

```bash
python baselines/002_csdid/bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp
```

Run the reproduction commands from the repository root. New baseline runs default to unique directories under `runs/`. Historical raw measurements are preserved under `baselines/002_csdid/results/`.
