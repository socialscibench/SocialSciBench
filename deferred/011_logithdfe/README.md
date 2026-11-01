# 011: Fixed-effects logit and average marginal effects — baselines and results

> Planned task, not yet in the shared evaluator. The timings and results in this file and in `results/` were recorded early in the project with this directory's scripts and have not been re-checked under the evaluator.

`CPU` · `estimator` · `L2 (nonlinear model + post-estimation)` · `Stata logit + margins · R fixest 0.14.2 + marginaleffects 1.0.0`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools and research context.

## Baseline timings (Apple M3 Pro)

### Total time

| # | design | absorb() | cluster | n_obs | n_covariates | n_fe | fe_levels (used) | n_clusters (used) | mobility | event rate | obs used | Stata (s) | R (s) | Reference (s) | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | `firm year` | firm | 3,000 | 4 | 2 | 250 × 10 | 250 | – | 29% | 2,500 | 1.57500 | 0.22030 | 0.01748 | R: AME standard errors |
| 2 | small multi-way check | `firm industry#year state#year` | firm | 6,000 | 4 | 3 | 516 × 100 × 100 | 516 | – | 31% | 5,160 | 7.32600 | 0.17562 | 0.06522 | R: AME standard errors |
| 3 | large public firms (S&P 1500) | `firm year` | firm | 22,500 | 6 | 2 | 1,192 × 15 | 1,192 | – | 20% | 17,880 | 83.27100 | 0.32397 | 0.08771 | R: AME standard errors |
| 4 | public firms + industry shocks (Compustat) | `firm industry#year` | firm | 75,000 | 11 | 2 | 3,404 × 720 | 3,404 | – | 15% | 51,060 | 30+ min (est. ~1.3 h‡) | 1.40666 | 0.37757 | Stata: all outputs (stopped at 30 min); R: AME standard errors |
| 5 | firms + industry and local shocks | `firm industry#year state#year` | state | 200,000 | 11 | 3 | 9,284 × 480 × 500 | 50 | – | 10% | 92,840 | 30+ min (est. ~13.5 h‡) | 3.06774 | 1.04277 | Stata: all outputs (stopped at 30 min); R: AME standard errors |
| 6 | workers + employers (résumé data) | `worker employer year` | employer | 500,000 | 6 | 3 | 24,212 × 1,000 × 10 | 1,000 | 5% | 10% | 242,120 | cannot run: r(916) unable to allocate matrix | 4.28698 | 5.35185 | Stata: all outputs (cannot run); R: AME standard errors |
| 7 | large private firms / establishments | `firm industry#year` | firm | 1,000,000 | 11 | 2 | 46,783 × 480 | 46,783 | – | 10% | 467,830 | cannot run: r(915) unable to allocate matrix | 15.25905 | 2.60906 | Stata: all outputs (cannot run); R: AME standard errors |

### Time by step

| # | design | Stata `logit` (s) | Stata `margins` (s) | R `feglm` (s) | R `avg_slopes` (s) | Reference: estimation (s) | Reference: AMEs (s) |
|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1.05800 | 0.51700 | 0.06647 | 0.15383 | 0.01480 | 0.00267 |
| 2 | small multi-way check | 5.63800 | 1.68800 | 0.06387 | 0.11175 | 0.05893 | 0.00624 |
| 3 | large public firms (S&P 1500) | 70.23100 | 13.04000 | 0.07247 | 0.25150 | 0.07511 | 0.01252 |
| 4 | public firms + industry shocks (Compustat) | 30+ min (est. ~1.2 h‡) | not reached (est. ~11 min‡) | 0.11901 | 1.28765 | 0.30793 | 0.06955 |
| 5 | firms + industry and local shocks | 30+ min (est. ~11.7 h‡) | not reached (est. ~1.8 h‡) | 0.21459 | 2.85315 | 0.88129 | 0.16136 |
| 6 | workers + employers (résumé data) | cannot run | cannot run | 0.95656 | 3.33042 | 4.71937 | 0.63248 |
| 7 | large private firms / establishments | cannot run | cannot run | 0.52919 | 14.72986 | 2.09258 | 0.51648 |

How to read the timing columns (the appendix defines every column):

- **Stata**: `logit y i.d x1 … i.fe1 i.fe2 …, vce(cluster cl)` (step 1) plus `margins, dydx(d x1 …)` (step 2), timed after the data are loaded. Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM, with `set maxvar 120000` (the default 5,000 is too small for one dummy per fixed-effect level) and `set max_memory 12g` so that memory errors are reported instead of the machine swapping. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Each time is a single run.
- **Stata is stopped after 30 minutes.** Progress at the stop: workload 4: 0 `logit` iteration lines printed; workload 5: 0 `logit` iteration lines printed.
- ‡**Estimates** extrapolate from three complete Stata calibration runs on a firm + year design (table below). Total time grew as about (number of parameters)^2.9, consistent with Stata building dense matrices with one row and column per fixed-effect dummy over data that also grow with the number of units. Each estimate scales the largest calibration run by obs used × parameters^1.9, where parameters = estimated fixed-effect levels + covariates, and splits it into `logit` and `margins` at the calibration ratio (87% / 13%). It assumes the same growth beyond the measured range and is approximate.

  | Firms used | Obs used | `logit` (s) | `margins` (s) | Total (s) |
  |---|---|---|---|---|
  | 713 | 10,695 | 28.31600 | 11.47100 | 39.78700 |
  | 1,374 | 20,610 | 218.69800 | 47.48600 | 266.18400 |
  | 2,081 | 31,215 | 707.67200 | 108.35000 | 816.02200 |
- **Stata cannot run** workload 6 (25,228 parameters): r(916) "unable to allocate matrix"; workload 7 (47,274 parameters): r(915) "unable to allocate matrix". r(915) means the dense matrices do not fit in memory; r(907) means the model needs more variables than the maximum Stata/MP allows (120,000), which `set maxvar` cannot raise.
- **Workload 6 uses 1,000 employers, not 5,000.** A first version with 5,000 small employers (about 10 workers each) ran into *separation*: combinations of worker and employer effects predicted the outcome perfectly, so some fixed-effect estimates head to ±∞ and the maximum-likelihood estimate does not exist. The reference's iterations then slowed without converging (more than 2.5 hours before it was stopped). Single fixed-effect levels with an all-0 or all-1 outcome are dropped by every tool, but separation by *combinations* of levels is not detected by Stata's or `fixest`'s defaults either. With larger employers the problem disappears and the model converges in 7 iterations. Small cells are therefore a practical hazard of worker–employer logits in real data.
- **R**: `fixest::feglm(..., family = binomial, cluster = ~cl)` (step 1) plus `marginaleffects::avg_slopes(m, vcov = FALSE)` (step 2), R 4.6.1 with fixest 0.14.2 on 12 threads, data loaded with `data.table::fread` before timing. `d` is passed as a logical so its AME is a discrete change, as in Stata. `marginaleffects` 1.0.0 stops with an error when asked for standard errors after `feglm` with fixed effects ("cannot take into account the uncertainty in fixed-effects parameters"), so R reports AME point estimates only. `feglm` is run with `ssc(K.adj = FALSE, G.adj = TRUE)` so that its clustered standard errors use Stata's `G/(G−1)` correction; other options are fixest defaults.
- **Reference**: the entire Python computation, timed with `time.perf_counter()`. "Estimation" covers dropping, IRLS and the coefficient standard errors; "AMEs" covers the marginal effects and their standard errors.

## Agreement with Stata and R

Agreement with the reference:

- **Stata** (workloads 1, 2, 3): coefficients within 2e-08, standard errors within 3e-08, AMEs within 1e-08, AME standard errors within 9e-09 (relative error), including the three-way design of workload 2. The small differences come from Stata's looser convergence tolerance.
- **R** (workloads 1, 2, 3, 4, 5, 6, 7): coefficients within 6e-10, AMEs within 1e-09, standard errors within 1e-05. R's standard errors are the least precise because of fixest's default convergence tolerances (`fixef.tol = 1e-6`, `glm.tol = 1e-8`); tightening them to 1e-10 and 1e-12 brings them within about 3e-8 of the reference. The timings use the defaults, since that is what researchers run.
- Wherever they ran, Stata, R and the reference drop the same observations and keep the same clusters.

Results are in `results/results.jsonl`.

## Appendix: workload table columns

The examples use one running case: whether a firm adopts AI in a given year (1 = adopts, 0 = does not), with a treatment `d` = 1 if the firm's state introduced an AI tax credit.

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–7 from smallest to largest. | `4` is the Compustat workload with industry shocks. |
| design | The research design the workload imitates (see the research context below). | public firms + industry shocks (Compustat) |
| absorb() | The fixed effects, written as in reghdfe's `absorb()`. `a#b` is an interaction: one fixed effect per combination of `a` and `b`. In Stata each becomes `i.fe1 i.fe2 …`; in R, `y ~ x \| fe1 + fe2`. | `firm industry#year`: one intercept per firm, and one per industry in each year (absorbing industry-wide shocks such as a sector downturn). |
| cluster | The variable standard errors are clustered on: observations in the same cluster may have correlated errors. | `state`: firms in the same state share state-level shocks. |
| n_obs | Rows in the data before dropping. | 5,000 firms × 15 years = 75,000 |
| n_covariates | Number of explanatory variables: the binary treatment `d` plus the continuous covariates. | 11 = `d` + 10 controls (size, R&D intensity, leverage, …) |
| n_fe | Number of fixed-effect dimensions. | `firm industry#year` → 2 |
| fe_levels (used) | Levels of each fixed effect estimated, in absorb() order, after dropping. Stata creates one dummy variable per level, so their sum (plus the covariates) is the number of parameters Stata estimates. | `3,404 × 720` = 3,404 firm effects and 720 industry-year effects |
| n_clusters (used) | Number of clusters left after dropping. | 3,404 firms |
| mobility | Worker–employer designs only: the yearly probability that a worker switches employer. Lower mobility connects the worker and employer effects more weakly, which slows fixed-effect algorithms. | 5% means 1 worker in 20 changes employer each year |
| event rate | Share of observations with y = 1. | 15% of firm-years are adoption events |
| obs used | Rows left after dropping fixed-effect levels whose outcome never varies: `e(N)` in Stata. | 51,060 |
| Stata (s) | Seconds for `logit` + `margins`. Stopped after 30 minutes. | `30+ min (est. …‡)`: stopped at 30 minutes, total estimated |
| R (s) | Seconds for `feglm` + `avg_slopes` (AME point estimates only). | |
| Reference (s) | Seconds for the Python reference to compute every output. | |
| Could not compute | Outputs a tool did not produce on that workload, and why. | "R: AME standard errors" |
| Time by step | The same runs split into estimation (`logit` / `feglm` / reference estimation) and marginal effects (`margins` / `avg_slopes` / reference AMEs). § marks a `logit` time measured to the second when `margins` was then stopped. | |

## Research context: what each workload represents

| # | Design | What it represents | Similar published settings |
|---|---|---|---|
| 1 | sanity check | No real setting: a tiny firm + year panel that checks correctness in seconds. | – |
| 2 | small multi-way check | A small firm panel with firm, industry-by-year and state-by-year effects. Stata can still run it, so the three-way computation is checked against Stata, not only against R. | – |
| 3 | large public firms (S&P 1500) | About 1,500 large listed firms over 15 years with firm and year effects, and a fairly common binary event (20% of firm-years) such as a CEO dismissal or an acquisition. | Jenter & Kanaan (2015): probability of forced CEO turnover after poor industry performance, 3,365 turnovers in U.S. public firms, 1993–2009. |
| 4 | public firms + industry shocks (Compustat) | All U.S. listed firms over 15 years with firm and industry-by-year effects (48 industries), a less common event (15%), and 10 firm controls. Industry-by-year effects are the most common extension of firm + year in management and finance papers. | Babina, Fedyk, He & Hodson (NBER WP 31325): firm-level AI investment measured from résumé and job-posting data for U.S. firms. |
| 5 | firms + industry and local shocks | 20,000 firms over 10 years with firm, industry-by-year and state-by-year effects, clustered by state: the design for a state-level policy or local labor-market shock. | Png (2017): staggered state adoption of the Uniform Trade Secrets Act and firm R&D, 1979–1998 (firm outcomes, state-level variation). |
| 6 | workers + employers (résumé data) | 50,000 workers over 10 years moving among 1,000 employers (about 50 workers each; 5% switch employer each year), with worker, employer and year effects, clustered by employer, and a binary event such as acquiring an AI skill. Separating worker and employer effects is the hardest fixed-effect structure here. | Hosseini & Lichtinger (2025): about 62 million workers at 285,000 U.S. firms from Revelio résumé and job-posting data, 2015–2025. |
| 7 | large private firms / establishments | 100,000 private firms or establishments over 10 years with firm and industry-by-year effects (10% events). | Acemoglu, Autor, Hazell & Restrepo (NBER WP 28257): establishment-level AI hiring from Burning Glass job postings, 2007 and 2010–2018 (similar data scale). |

Sources:
- Fernández-Val, I. & Weidner, M. (2016), "Individual and Time Effects in Nonlinear Panel Models with Large N, T," *Journal of Econometrics* 192(1), 291–312. [arXiv](https://arxiv.org/pdf/1311.7065)
- Stammann, A., Heiss, F. & McFadden, D. (2016), "Estimating Fixed Effects Logit Models with Large Panel Data," working paper. [RePEc](https://ideas.repec.org/p/zbw/vfsc16/145837.html)
- Ai, C. & Norton, E.C. (2003), "Interaction Terms in Logit and Probit Models," *Economics Letters* 80(1), 123–129. [RePEc](https://ideas.repec.org/a/eee/ecolet/v80y2003i1p123-129.html)
- Jenter, D. & Kanaan, F. (2015), "CEO Turnover and Relative Performance Evaluation," *Journal of Finance* 70(5), 2155–2184. [LSE Research Online](https://researchonline.lse.ac.uk/id/eprint/64421)
- Babina, T., Fedyk, A., He, A. & Hodson, J., "Firm Investments in Artificial Intelligence Technologies and Changes in Workforce Composition," NBER Working Paper 31325. [NBER](https://www.nber.org/papers/w31325)
- Png, I.P.L. (2017), "Law and Innovation: Evidence from State Trade Secrets Laws," *Review of Economics and Statistics* 99(1), 167–179. [RePEc](https://ideas.repec.org/a/tpr/restat/v99y2017i1p167-179.html)
- Hosseini, S.M. & Lichtinger, G. (2025), "Generative AI as Seniority-Biased Technological Change: Evidence from U.S. Résumé and Job Posting Data," SSRN 5425555. [Stanford DEL seminar](https://digitaleconomy.stanford.edu/event/seyed-m-hosseini-and-guy-lichtinger-generative-ai-as-seniority-biased-technological-change-evidence-from-u-s-resume-and-job-posting-data)
- Acemoglu, D., Autor, D., Hazell, J. & Restrepo, P., "AI and Jobs: Evidence from Online Vacancies," NBER Working Paper 28257. [NBER conference version](https://conference.nber.org/conf_papers/f143876.pdf)

## Reproduce

```bash
python bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp --out results/
```

Requires Stata/MP 19.5, R with `fixest`, `marginaleffects` and `data.table`, and Python with NumPy, SciPy and pandas.
