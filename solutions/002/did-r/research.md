# Baseline: R did 2.5.1 (task 002, balanced csdid201-rule workloads)

- **Source:** did 2.5.1 (Callaway & Sant'Anna), https://github.com/bcallaway11/did, with DRDID
  1.3.0 (Sant'Anna & Zhao).
- **Rules:** did's rules are the ones csdid 2.0.1 adopted:
  - not-yet-treated units untreated in both periods of a cell;
  - estimated cohort shares in the cohort average's SE;
  - the 0.999 overlap check.

  The adapter therefore runs the csdid201-rule workloads that are balanced, with fixed
  covariates.
- **Mapping:** in `solution.py`'s docstring. The task's trimming level is passed to the DRDID
  estimator (`trim.level`). The pre/post averages, event subsets and covariance are formed from
  the dynamic aggregation's influence functions.
- **Bootstrap:** with multipliers, the one estimation runs did's multiplier bootstrap and
  simultaneous band (`bstrap = TRUE, cband = TRUE`), the aggregations are bootstrapped by aggte,
  and the pre/post averages by did's `mboot()` on their influence functions. The draws are
  did's own multipliers (not comparable).
- **Not supported:**
  - unbalanced panels: did's unbalanced estimator is not csdid's pair balancing;
  - covariates that vary over time;
  - 1.81's ipw and drimp;
  - the csdid181-rule workloads.

## Revision of 2026-10-25 (before any measurement)

The first version had not been run. On the evaluator's development fixtures
(`solutions/baselines/check_adapter.py`, reference and validator) it failed three of the
four balanced csdid201-rule fixtures:
- **Universal base** (dev-notyet-universal-bootstrap-201, dev-reg-universal-event-sets-201): did
  returns the reference cells (g, g - 1) and event time -1 at 0, which the task does not
  estimate. That gave 3 extra cells, an 11th event time, and a pre-treatment average diluted by
  exactly 1/5. Both are now dropped.
- **dev-overlap-201:** did estimated cohort 4's cells, which the 0.999 overlap rule leaves out.
  - did 2.5.1 runs `overlap_check_fail` (and the covariate conditioning check
    `rcond_check_fail`) only when `est_method` is the name of a built-in estimator. The adapter
    passes functions, to set DRDID's `trim.level`.
  - The functions now call did's own checks and stop with did's conditions ("overlap",
    "singular"). did then leaves the cells unestimated, as with its built-ins: "The ATT for this
    cell will be set to NA".
- **Bootstrap:** the pre/post averages' bootstrap SEs were missing. A second full estimation with
  `bstrap = TRUE` repeated the first.
  - Now one estimation does both. att_gt's bootstrap replaces the cells' SEs, so their analytic
    SEs are recomputed with att_gt's own formula on the stored influence functions
    (sqrt(diag(IF'IF/n)/n)).

**After the revision**, every balanced csdid201-rule fixture passes every compared output:
- dev-notyet-universal-bootstrap-201;
- dev-reg-universal-event-sets-201;
- dev-notyet-varying-bootstrap-201;
- dev-overlap-201.

dev-weighted-unbalanced-stdipw-201 is refused (unbalanced). The bootstrap outputs are finite.
Against the reference's, which come from its own multipliers, with 199 and 99 draws:
- standard errors differ by a median of 5-20% (up to 52% for single estimates);
- critical values differ by 0.3-31%.

The evaluator smoke run planned for this adapter was replaced by this check. The smoke suite
also contains fixtures outside the adapter's scope (1.81 rules, unbalanced panels), which it
refuses by design.

**Queue measurements** (compare.py, one thread, a first call and one measured call;
`runs/study-20261025/002-did-r-t1.jsonl`). Every compared output passes:

| Workload | First call | Measured call | Peak RSS | csdid 2.0.1 (measured call) |
|---|---:|---:|---:|---:|
| 103 | 6.3 s | 5.6 s | 1.64 GB | 8.8 s |
| 101 (999 draws) | 7.3 s | 6.5 s | 1.08 GB | 15.5 s |
| 105 | 45.8 s | 44.9 s | 2.34 GB | 84.9 s |
| 104 | 190 s | 195 s | 4.80 GB | 411 s |
| 107 (999 draws) | 201 s | 204 s | 10.78 GB | 365 s |

The study report compares them with the candidates.
