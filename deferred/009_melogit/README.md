# 009: Multilevel logistic regression — baselines and results

> Planned task, not yet in the shared evaluator. The timings and results in this file and in `results/` were recorded early in the project with this directory's scripts and have not been re-checked under the evaluator.

`CPU` · `estimator` · `L2 (nonlinear model + numerical integration)` · `Stata melogit · R lme4 2.0.6 · Python NumPy/SciPy`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools and research context.

## Baseline timings (Apple M3 Pro)

| # | design | n_obs | n_groups | group size (mean / max) | k (of which between-group) | share y = 1 | Stata `melogit` (s) | R `glmer`, nAGQ = 7 (s) | Python reference (s) | Could not compute |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 5,000 | 100 | 50.0 / 266 | 5 (1) | 0.309 | 0.96000 | 0.76444 | 0.03850 | R: SE of var(_cons) (`lme4` does not report standard errors of variance components) |
| 2 | employees in firms (survey) | 50,000 | 1,000 | 50.0 / 277 | 10 (3) | 0.296 | 1.39000 | 46.93064 | 0.61880 | R: SE of var(_cons) (`lme4` does not report standard errors of variance components) |
| 3 | firm-years in firms (panel) | 200,000 | 19,717 | 10.1 / 148 | 15 (3) | 0.298 | 2.72200 | 969.50939 | 3.82045 | R: SE of var(_cons) (`lme4` does not report standard errors of variance components) |
| 4 | individuals in countries (cross-national survey) | 500,000 | 60 | 8,333.3 / 32,074 | 10 (3) | 0.277 | error: r(1400) | 89.84983 | 5.94265 | Stata: all outputs (`initial values not feasible`, r(1400)); R: SE of var(_cons) (`lme4` does not report standard errors of variance components) |
| 5 | patents in firms | 1,000,000 | 10,000 | 100.0 / 1,109 | 15 (3) | 0.294 | 7.96800 | 30+ min (est. ~31 min‡) | 16.71041 | R: all outputs (stopped at 30 min) |
| 6 | job postings in firms | 2,000,000 | 99,821 | 20.0 / 245 | 20 (4) | 0.297 | 19.34800 | 32.10588§ | 48.03217 | R: valid estimates (the 7-point stage failed; the reported estimates are the nAGQ = 0 ones, log likelihood +Inf); R: SE of var(_cons) (`lme4` does not report standard errors of variance components) |

How to read the timing columns (the appendix defines every column):

- **Stata `melogit` (s)**: `melogit y x1 ... xk || g:` with the defaults, timed with `timer` after the data are loaded. Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM. `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Stopped after 30 minutes. Each time is a single run.
- **R `glmer`, nAGQ = 7 (s)**: `lme4::glmer(y ~ x1 + ... + xk + (1 | g), family = binomial, nAGQ = 7)` with the default optimizers. R 4.6.1, lme4 2.0.6. Timed from the loaded data to the fitted model (`vcov()` is computed afterwards and not timed). Stopped after 30 minutes. `glmer` printed no convergence warning on any workload (see the notes on existing tools below). § marks a run whose 7-point stage failed.
- ‡**Estimates for stopped runs**: the same command is timed on a random 10% of the groups (all their observations), and the total is (sample time) × (n_obs ÷ sample n_obs). Sample times: workload 5 (97,183 observations): R 182.8 s.
- **Python reference (s)**: `reference.py`, timed with `time.perf_counter()`.

## Agreement with Stata and R

Stata passes every tolerance on all the workloads it completes; its log likelihood is slightly *higher* than the reference's because its quadrature rule is different, not because it found a better optimum.

Agreement with the reference:

| # | Stata: max \|Δβ\| ÷ SE | Stata: max rel. error SE | Stata: rel. error var(_cons) | Stata: rel. error SE of var(_cons) | Stata: Δ log likelihood (relative) | R: max \|Δβ\| ÷ SE | R: max rel. error SE | R: rel. error var(_cons) | R: Δ log likelihood (relative) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.0e-04 | 2.6e-04 | 1.5e-04 | 1.4e-04 | +1.5e-07 | 3.7e-05 | 4.6e-06 | 7.8e-06 | -4.4e-11 |
| 2 | 8.6e-04 | 6.1e-04 | 2.6e-04 | 4.0e-04 | +2.7e-07 | 5.5e-03 | 7.8e-03 | 3.4e-05 | -2.0e-09 |
| 3 | 4.9e-03 | 1.7e-04 | 2.7e-04 | 2.6e-04 | +7.3e-07 | 8.2e-01 | 4.0e-02 | 2.5e-03 | -1.0e-05 |
| 4 | – | – | – | – | – | 9.3e-05 | 6.1e-05 | 8.0e-06 | -3.8e-14 |
| 5 | 1.4e-03 | 6.3e-04 | 1.7e-04 | 2.4e-04 | +9.4e-08 | – | – | – | – |
| 6 | 9.7e-03 | 2.8e-04 | 2.7e-04 | 4.0e-04 | +4.9e-07 | 1.6e+01 | 2.1e-02 | 9.9e-03 | +inf |

Results are in `results/results.jsonl`.

## Notes on existing tools

- **`glmer` can stop early without a warning.** On workload 3 it ran 970 s and returned coefficients up to 0.82 SE from the reference with a log likelihood 1.14 lower (the reference's is higher, so the reference is the better optimum). It gave no convergence warning. On workload 2 it stops slightly early too (SEs off by up to 0.8%).
- **`glmer` can return the wrong model without a warning.** On workload 6 the 7-point stage stopped after 2 evaluations with the internal message "objective function went below allowed minimum" (in `m@optinfo$message`, not printed). The returned fixed effects are exactly those of `nAGQ = 0` (a cruder approximation, 28 s), and `logLik()` is +Inf. The slopes are about 3% too small, up to 15.7 standard errors away. Its time (§) is therefore not the time of a 7-point fit. Always check `logLik()` and `m@optinfo` after `glmer(..., nAGQ > 1)`.
- **Stata fails with few, large groups.** On workload 4 (60 groups of about 8,300) `melogit` stops with "initial values not feasible" (r(1400)) after its starting-value step. `startvalues(iv)`, `startgrid(0.5 1 2)`, `intmethod(mcaghermite)` and `intpoints(12)` fail the same way. `intmethod(laplace)` works (19.24000 s) and matches the reference's var(_cons) to 1e-5, because with groups this large the Laplace approximation is very accurate. R and the reference fit this workload with 7 points without trouble.

## Appendix: workload table columns

The examples use one running case: whether a firm's job posting asks for AI skills (1 = yes), for postings nested in firms.

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–6 from smallest to largest. | `6` is job postings in firms. |
| design | The research setting the workload imitates (see the research context below). | job postings in firms |
| n_obs | Observations (level 1). | 2 million postings |
| n_groups | Groups (level 2), each with its own random intercept. | 100,000 firms |
| group size (mean / max) | Observations per group. The likelihood of each group is one integral, whatever its size. | 20 postings per firm on average, the largest firm 245 |
| k (of which between-group) | Regressors, not counting the constant; in brackets, how many are constant within groups. | 20 (4): e.g. posting's occupation and wage are within-firm, firm size and age are between-firm |
| share y = 1 | Share of observations with y = 1. | 0.30 |
| Stata `melogit` (s) | Seconds for `melogit`. Stopped after 30 minutes. | `30+ min (est. …‡)`: stopped, total estimated |
| R `glmer`, nAGQ = 7 (s) | Seconds for `glmer` with 7 quadrature points. | |
| Python reference (s) | Seconds for the Python reference. | |
| Could not compute | Outputs a tool did not produce on that workload, and why. | R: SE of var(_cons) |

## Research context: what each workload represents

| # | Design | What it represents | Related settings |
|---|---|---|---|
| 1 | sanity check | No real setting: 5,000 observations in 100 groups, to check correctness in seconds. | – |
| 2 | employees in firms (survey) | An employee survey: 50,000 employees in 1,000 organizations, outcome e.g. intention to quit (0/1), with 3 organization-level regressors (size, HR practices, industry). | – |
| 3 | firm-years in firms (panel) | 20,000 firms observed for about 10 years each, outcome e.g. whether the firm makes an acquisition or adopts a technology in a year, with a firm random intercept in place of firm fixed effects. The same model as `xtlogit, re`. | – |
| 4 | individuals in countries (cross-national survey) | 500,000 adults in 60 countries, outcome whether the person is starting a business, with country-level regressors (culture, institutions). Few, large groups. | Autio, Pathak & Wennberg (2013) relate GEM entrepreneurship data on individuals in 42 countries to national cultural practices with multilevel models. |
| 5 | patents in firms | 1 million patents of 10,000 firms, outcome e.g. whether a patent becomes highly cited, with firm-level regressors. | – |
| 6 | job postings in firms | 2 million job postings of 100,000 firms, outcome whether the posting asks for AI skills, with 4 firm-level regressors. | Acemoglu, Autor, Hazell & Restrepo use Burning Glass postings to measure establishments' AI hiring (they aggregate to the establishment level rather than fitting a multilevel model). |

Sources:
- Autio, E., Pathak, S. & Wennberg, K. (2013), "Consequences of cultural practices for entrepreneurial behaviors," *Journal of International Business Studies* 44(4): 334–362. [RePEc](https://ideas.repec.org/a/pal/jintbs/v44y2013i4p334-362.html)
- Acemoglu, D., Autor, D., Hazell, J. & Restrepo, P., "AI and Jobs: Evidence from Online Vacancies," NBER Working Paper 28257. [NBER conference version](https://conference.nber.org/conf_papers/f143876.pdf)

## Reproduce

```bash
python bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp --out results/
```

Requires Stata/MP 19.5, R with `lme4` and `data.table`, and Python with NumPy, SciPy and pandas.
