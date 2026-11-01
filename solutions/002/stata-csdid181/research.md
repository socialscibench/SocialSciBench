# Baseline: Stata csdid 1.82 with drdid 1.91 (task 002, csdid181-rule workloads)

- **Source:** csdid 1.82 and drdid 1.91 (Rios-Avila, Sant'Anna & Callaway; SSC). 1.82 is the SSC
  release of 1.81's rules, which the task's csdid181 rule set reproduces.
- **Commands:** in `solution.py`'s docstring.
- **Mapping of the task's options:**
  - **methods:** all five (dripw, reg, ipw, stdipw, drimp).
  - **comparison group and base period:** `notyet`; the universal base period is `long2`.
  - **trimming:** `pscoretrim(#)`.
  - **covariates that vary over time:** given per unit-period row; drdid uses each cell's earlier
    period.
  - **event subsets and covariance:** 1.82 has no event-subset command, so these are formed from
    estat event's covariance.
  - **bootstrap:** with multipliers, the estimation also saves its influence functions
    (`saverif()`), and `csdid_stats attgt|simple|group|calendar|event,
    wboot(reps(B) wbtype(rademacher))` gives the bootstrap standard errors and the uniform bands'
    critical values, drawn from Stata's own multipliers (not comparable).

## Revision of 2026-10-25 (after the first queue measurements)

The first version was checked only on 106 before it was queued. Its measured rows (106 passed,
109 failed, 102 passed) are kept in `runs/study-20261025/002-stata-csdid181-t1.jsonl`.

**109 failed** with 11 event times (the reference has 10) and a different pre-treatment average.
Every other output passed.
- **The cause (csdid 1.82's source, `csdid.ado` and `csdid_estat.ado`):** under `long2`, a
  cohort's cells are stored in period order without g - 1, but the event aggregation pairs them
  with `e(tlev)`, the periods from the second on.
  - A pre-treatment cell of period t is then taken as period t + 1. Its `Tm#` label is right,
    but the window selects it as e + 1, so `estat event, window(-5 5)` returns e = -6..-2 before
    treatment.
  - The same pairing is in `csdid_stats` (the RIF file saves `tlev`).
  - The cells, the group, calendar and simple aggregations, and the post-treatment event times
    are unaffected.
- **The fix:** the adapter moves the window by one period on the pre-treatment side
  (`window(lo+1 hi)` for lo < 0). Then the event times, `Pre_avg` and the covariance are the
  task's.

**The bootstrap path had never run in a measurement.** It would have failed on 108:
- it read a matrix into a scalar (Stata error r(109), "type mismatch"), reproduced on the
  bootstrap fixture below;
- it took the aggregations' bootstrap standard errors from `estat` after a second estimation with
  `wboot()`, but 1.82 reports asymptotic standard errors there ("Test will be based on asymptotic
  VCoV"; its help points to `saverif()` and `csdid_stats`);
- it left the critical values missing.

It now saves the RIFs in the one estimation and runs `csdid_stats` with `wboot()` for the cells
and each aggregation.

**The change:** `revision.diff` (`patch -R` on `solution.py` gives the first version,
which measured 106, 102 and the failed 109 row).

**Unaffected measurements:** the revision changes nothing on varying-base analytic workloads (106
and 102 run the same commands), so their rows stand. 109 and 108 are measured again with the
revised adapter (appended to the same file).

**Check of the revision** (`solutions/baselines/check_adapter.py`; the evaluator's
development fixtures, reference and validator; bootstrap outputs reported, not compared). Every
fixture under 1.81's rules passes:
- dev-panel-0/1, dev-late-adoption, dev-propensity-trimming;
- dev-notyet-ipw-universal-bootstrap-181;
- dev-varying-covariates-trim-181;
- dev-weighted-unbalanced-dripw-181, dev-notyet-drimp-weighted-unbalanced;
- dev-high-propensity-trim.

On the bootstrap fixture (199 draws), every bootstrap output is finite. Against the reference's
(computed from its own multipliers):
- standard errors differ by 2-24% (median about 11%);
- critical values differ by 2-6%.

The first version gives e = -6..-2 on dev-varying-covariates-trim-181 and stops with r(109) on
the bootstrap fixture.

## Measurements

**Development check (compare.py, one thread, 2026-10-25):** 106 passes every check, 299 s for
one call against the reference's 0.30 s.

**Queue measurements** (compare.py, one thread, a first call and one measured call;
`runs/study-20261025/002-stata-csdid181-t1.jsonl`, in order):

| Workload | Adapter | First call | Measured call | Peak RSS | Result |
|---|---|---:|---:|---:|---|
| 106 | first version | 314 s | 337 s | 1.30 GB | passed |
| 109 | first version | 433 s | 458 s | 1.34 GB | failed: the event window (above) |
| 102 | first version | 604 s | 674 s | 1.83 GB | passed |
| 109 | revised | 436 s | 464 s | 1.37 GB | passed |
| 108 | revised | 690 s | 712 s | 1.51 GB | fails: some ATT(g,t) cells (below) |

The reference takes 0.30 s (106), 0.90 s (109), 2.09 s (102) and 2.18 s (108) per call at one
thread. The study report compares the times with the candidates'.

**108** (not yet treated controls, ipw, universal base, 25,000 draws):
- csdid 1.82 estimates the same cells as the reference, but some ATT(g,t) differ: up to 0.84 of
  max(|reference|, its SE), and up to 2.1 relative for their SEs;
- the aggregates differ by 0.05-5%.

The fixture with the same options (dev-notyet-ipw-universal-bootstrap-181) passes, so the mapping
of the options holds.

**Diagnosis** (`diagnosis-108/`: scripts and their outputs; `diag108.py` saves the per-cell
results to an .npz named as its second argument, kept outside the repository because it holds
reference outputs):
- **Where:** the 29 failing cells are exactly cohort 25's 29 cells (5 treated units against 185
  not-yet-treated controls). The other 783 cells agree within 1e-6.
- **Why:** in cohort 25 the two-valued covariate x4 is 0 for all 5 treated units and nonzero for
  91 of the controls, so it predicts non-treatment perfectly (quasi-complete separation; the propensity
  logit has no finite maximum).
  - **The reference** documents its logit as Stata's `logit ... [iw = w]`. It iterates Newton
    steps until they are below 1e-10, which sends x4's coefficient to -37.3 (Hessian condition
    number about 1e62) and those 91 controls' scores below 1e-8. They get no weight.
  - **Stata's logit,** as drdid 1.91 calls it, reports "x4 != 0 predicts failure perfectly; x4
    omitted and 91 obs not used". It fits the other 99 units to the same log likelihood
    (-18.023691) and the same other coefficients. drdid's `predict` then scores all 190 units
    without x4, so those controls get scores up to 0.28 and enter the IPW weights.
  - The ATTs differ accordingly: cell (25, 6) is -1.631 in the reference and -0.253 in csdid.
- **Conclusion:** the difference is a genuine difference in separation handling between the task's
  reference and Stata, not an adapter error. The candidates reproduce the reference's iteration.
  The maintainer may want to know that the reference departs from Stata's `logit` on perfectly
  predicted controls.

**csdid 1.82's bootstrap cost:** csdid_stats' interquartile-range standard error re-sorts the
whole B x k matrix of draws once per column (`iqrse` in `csdid_stats.ado`). On 108 (25,000 draws,
812 cells with their weight columns) that is the bulk of the bootstrap's time.
