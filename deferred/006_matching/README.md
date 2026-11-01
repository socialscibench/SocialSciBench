# 006: Matching estimators — baselines and results

> Planned task, not yet in the shared evaluator. The timings and results in this file and in `results/` were recorded early in the project with this directory's scripts and have not been re-checked under the evaluator.

`CPU` · `estimator` · `L1` · `Stata teffects psmatch, psmatch2 4.0.12, cem · R MatchIt 4.8.1 · Python NumPy`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](task.md), which is used as the model prompt together with `reference.py` and `workloads.py`. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools and research context.

## Baseline timings (Apple M3 Pro)

### Propensity-score matching

| # | design | n | treated | covariates | Stata `teffects psmatch` (s) | Stata `psmatch2` (s) | R `MatchIt` (s) | Python reference (s) | ATT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1,000 | 201 | 3 | 0.46600 | 0.41600 | 0.05524 | 0.00169 | 1.0486 |
| 2 | firms (e.g. VC-backed vs not) | 20,000 | 2,043 | 6 | 1.00900 | 0.47300 | 0.07193 | 0.01008 | 1.0243 |
| 3 | startups (Crunchbase-scale) | 100,000 | 9,880 | 8 | 10.10900 | 0.66600 | 0.25179 | 0.03411 | 1.0354 |
| 4 | workers or inventors | 500,000 | 24,924 | 6 | 203.97400 | 1.17900 | 0.85627 | 0.21519 | 0.9801 |
| 5 | administrative records | 2,000,000 | 99,921 | 5 | 30+ min (est. ~45 min‡) | 3.19900 | 2.71727 | 0.95546 | 0.9980 |

### Coarsened exact matching

| # | design | n | treated | CEM covariates | strata | matched strata | matched treated | matched untreated | Stata `cem` (s) | R `MatchIt` (s) | Python reference (s) | ATT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1,000 | 201 | 4 | 243 | 92 | 174 | 547 | 0.05700 | 0.00289 | 0.00065 | 1.0175 |
| 2 | firms (e.g. VC-backed vs not) | 20,000 | 2,043 | 4 | 2,972 | 804 | 1,842 | 12,069 | 6.20900 | 0.02643 | 0.02011 | 1.1207 |
| 3 | startups (Crunchbase-scale) | 100,000 | 9,880 | 4 | 4,325 | 1,636 | 9,587 | 80,907 | 111.38700 | 0.05222 | 0.07234 | 0.9706 |
| 4 | workers or inventors | 500,000 | 24,924 | 4 | 7,965 | 2,908 | 24,626 | 443,628 | 30+ min (est. ~33 min‡) | 0.27285 | 0.43311 | 1.1948 |
| 5 | administrative records | 2,000,000 | 99,921 | 4 | 9,000 | 3,871 | 99,501 | 1,861,652 | 30+ min (est. ~6.7 h‡) | 0.99046 | 1.97865 | 0.9253 |

### Could not compute

| # | Could not compute |
|---|---|
| 1 | – |
| 2 | – |
| 3 | – |
| 4 | Stata `cem`: all outputs (stopped at 30 min) |
| 5 | Stata `teffects`: all outputs (stopped at 30 min); Stata `cem`: all outputs (stopped at 30 min) |

How to read the timing columns (the appendix defines every column):

- **Stata**: each command is timed on its own, in its own Stata session, after the data are loaded: `teffects psmatch (y) (d x…, logit), atet nneighbor(1)`, `psmatch2 d x…, outcome(y) logit neighbor(1)`, and `cem x1 x2 x3 x4, treatment(d)` (the ATT regression on the CEM weights is not timed). `teffects`' time includes its Abadie–Imbens standard errors, which it always computes. Stata/MP 19.5 (StataNow) with a 12-core license, on an Apple M3 Pro with 18 GB of RAM. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. Each command is stopped after 30 minutes. Each time is a single run.
- ‡**Estimates** extrapolate each stopped command from its own last two completed runs: time grew as about n^1.8 for `cem` (consistent with the quadratic loop in its Mata code) and n^1.9 for `teffects`. They assume the same growth beyond the measured range and are approximate.
- **R `MatchIt` (s)**: `matchit()` alone, for PSM and for CEM separately (CEM with Stata's cutpoints passed in `cutpoints`), R 4.6.1 with MatchIt 4.8.1; data loaded with `fread` before timing.
- **Python reference (s)**: `reference.py`, timed with `time.perf_counter()`.

## Agreement with Stata and R

Agreement with the reference:

- **PSM:** on workloads 1–2 every tool gives the reference's ATT (within 3e-14). On larger workloads the ATTs differ in the 4th–5th significant digit (largest relative difference: 8e-04). This is not a different method but a numerical fragility of nearest-neighbour matching: with hundreds of thousands of untreated units, many treated units have two nearest neighbours whose distances differ by less than 1e-9, so the tiny differences between the tools' propensity scores (each stops its logit at a different convergence tolerance) change which one is matched. On workload 4, 99 treated units are in that situation, and perturbing the reference's propensity scores by one part in a billion moves its ATT from 0.98010 to 0.97978, as much as the tools differ. **For new implementations the rule is therefore: estimate the logit to tight convergence (parameter change < 1e-10), then the ATT must match the reference within 1e-8.** Stata's two commands agree with each other wherever both finish.
- **CEM:** Stata `cem` (workloads 1, 2, 3): identical strata and counts, ATT within 1e-14. R `MatchIt` with Stata's cutpoints: identical counts and ATT within 3e-12 on workloads 2, 3, 4, 5; on workload(s) 1 it matches 173 treated vs 174, because it places values on a bin edge differently (see the note below).

Results are in `results/results.jsonl`.

## Note: default CEM in Stata and R

**Default CEM differs between Stata and R.** R's `MatchIt` and `cem` packages choose different default cutpoints (and R's `cem` package uses "pretty" cutpoints, as the Stata source notes), so the same command gives different strata and a different ATT in each language. The R baseline is given Stata's cutpoints explicitly. Even then, `MatchIt` assigns values that sit exactly on a bin edge (such as a variable's minimum) differently, which can move a unit in or out of a stratum on small samples.

## Appendix: workload table columns

The examples use one running case: the effect of venture-capital (VC) backing on firm growth, comparing VC-backed firms (treated) with similar firms without VC (untreated).

| Column | Meaning | Example |
|---|---|---|
| # | Workload number, 1–5 from smallest to largest. | `2` is the firm-level workload. |
| design | The kind of study the workload imitates (see the research context below). | firms (e.g. VC-backed vs not) |
| n | Number of units (treated + untreated). | 20,000 firms |
| treated | Number of treated units. | about 2,000 VC-backed firms |
| covariates | Number of covariates in the propensity score. | 6: age, size, industry R&D intensity, … |
| Stata `teffects psmatch` (s) | Seconds for `teffects psmatch` (official; includes its standard errors). | |
| Stata `psmatch2` (s) | Seconds for `psmatch2` (community, SSC). | |
| R `MatchIt` (s) | Seconds for `matchit()`. | |
| Python reference (s) | Seconds for the Python reference. | |
| ATT | The estimated average treatment effect on the treated (true value 1.0). | 1.024: VC-backed firms grow 1.02 units more than their matched counterparts |
| CEM covariates | Covariates coarsened and matched exactly. | 4: age, size, R&D intensity, leverage |
| strata | Distinct combinations of bins (cells) across the CEM covariates. | |
| matched strata | Strata containing both treated and untreated units; the others are dropped. | |
| matched treated / untreated | Units in matched strata. Treated units in unmatched strata are dropped from the estimate. | |
| Stata `cem` (s) | Seconds for `cem`. Stopped after 30 minutes. | `30+ min (est. …‡)`: stopped at 30 minutes, total estimated |
| Could not compute | Outputs a tool did not produce on that workload, and why. | |

## Research context: what each workload represents

| # | Design | What it represents | Related papers |
|---|---|---|---|
| 1 | sanity check | No real setting: 1,000 units, to check correctness in seconds. | – |
| 2 | firms (e.g. VC-backed vs not) | 20,000 firms, 10% treated, 6 firm characteristics: the classic firm-level matching design. | Puri & Zarutskie (2012) compare VC-financed firms with matched non-VC-financed firms over their life cycle. |
| 3 | startups (Crunchbase-scale) | 100,000 startups, 10% treated (e.g. accelerator participation), 8 characteristics. | – |
| 4 | workers or inventors | 500,000 individuals, 5% treated (e.g. a training program or moving to a new employer). | – |
| 5 | administrative records | 2,000,000 individuals from administrative data, 5% treated. | – |

Methods: Iacus, King & Porro (2012) introduce coarsened exact matching; Abadie & Imbens (2016) derive the standard errors `teffects psmatch` uses, which account for the estimated propensity score.

Sources:
- Puri, M. & Zarutskie, R. (2012), "On the Life Cycle Dynamics of Venture-Capital- and Non-Venture-Capital-Financed Firms," *Journal of Finance* 67(6), 2247–2293. [NBER working paper](https://www.nber.org/papers/w14250)
- Iacus, S.M., King, G. & Porro, G. (2012), "Causal Inference without Balance Checking: Coarsened Exact Matching," *Political Analysis* 20(1), 1–24. [RePEc](https://ideas.repec.org/a/cup/polals/v20y2012i01p1-24_01.html)
- Abadie, A. & Imbens, G.W. (2016), "Matching on the Estimated Propensity Score," *Econometrica* 84(2), 781–807. [Econometric Society](https://www.econometricsociety.org/publications/econometrica/2016/03/01/matching-estimated-propensity-score)

## Reproduce

```bash
python bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp --out results/
```

Requires Stata/MP 19.5 with `psmatch2` and `cem` (SSC), R with `MatchIt` and `data.table`, and Python with NumPy and pandas.
