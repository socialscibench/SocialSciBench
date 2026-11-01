# Baseline: Stata csdid 2.0.1 (task 002, csdid201-rule workloads)

- **Source:** csdid 2.0.1 (Callaway, Rios-Avila & Sant'Anna), commit f98bcea of
  https://github.com/pedrohcgs/csdid-stata. It is the version the task's csdid201 rules
  reproduce, and the comparison the package's author names for the headline timing.
- **Commands:** in `solution.py`'s docstring.
- **Mapping of the task's options:**
  - **methods:** dripw, reg and stdipw are `method(dr)`, `method(reg)` and `method(ipw)`;
    2.0.1's ipw is the normalized estimator.
  - **comparison group and base period:** `notyet` / `never`, `base_period(varying|universal)`.
  - **trimming:** `pscoretrim(#)`.
  - **balancing:** pair balancing is `bal(pair)` (2.0.1's default is `bal(full)`).
  - **aggregations:** `estat ..., dropmissing` aggregates over the estimated cells, as the
    reference does after the overlap check.
  - **bootstrap:** `pointwise` keeps the analytic run from drawing the bootstrap it otherwise
    uses for simultaneous bands. With multipliers, a second estimation with `reps(B)` gives the
    bootstrap outputs, drawn from Stata's own Rademacher multipliers (not comparable).
- **Development checks (compare.py, one thread, 2026-10-25):** 101, 103 and 105 pass every
  comparable check (estimates within the 1e-6 scaled tolerance; analytic SEs, event subsets and
  the event-study covariance within 1e-4). Times: 14.6 s, 8.6 s and 82.4 s. The queue
  measurements are in the study report.
- **Not supported:** 1.81's unnormalized ipw and drimp (2.0.1 has neither); the csdid181-rule
  workloads (`stata-csdid181`).

**Queue measurements** (compare.py, one thread, a first call and one measured call;
`runs/study-20261025/002-stata-csdid201-t1.jsonl`). Every compared output passes:

| Workload | First call | Measured call | Peak RSS |
|---|---:|---:|---:|
| 103 | 14.0 s | 8.8 s | 0.52 GB |
| 101 (999 draws) | 16.2 s | 15.5 s | 0.60 GB |
| 105 | 86.0 s | 84.9 s | 1.05 GB |
| 107 (999 draws) | 359 s | 365 s | 6.79 GB |
| 104 | 403 s | 411 s | 2.95 GB |

**Fixture check after the queue** (`solutions/baselines/check_adapter.py`, 2026-10-25):
- every csdid201-rule fixture passes every compared output:
  - dev-notyet-universal-bootstrap-201;
  - dev-weighted-unbalanced-stdipw-201;
  - dev-reg-universal-event-sets-201;
  - dev-notyet-varying-bootstrap-201;
  - dev-overlap-201 (2.0.1's overlap warnings for cohort 4).
- The bootstrap outputs are finite. Against the reference's, which come from its own multipliers
  (199 and 99 draws), SEs differ by a median of 1-29% per output and critical values by 1-10%.
