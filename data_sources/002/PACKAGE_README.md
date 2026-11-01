# Real-data workloads for csdid (task 002)

This folder is the complete package for the real-data workloads for task 002, staggered difference-in-differences with Callaway & Sant'Anna's doubly robust estimator (`csdid ..., method(dripw)`). It contains the workload design, the data, a loader, the task's reference implementation and the build scripts.

The task itself (what is computed, inputs, outputs, tolerances) was written by the task author and is reproduced unchanged in **Appendix A** and in `code/task.md`. **Appendix B** is the full data lineage. This guide adds real-data workloads to it.

## 1. Goal

Speed up `csdid`-type staggered DiD and show reviewers that a faster method:

1. gives the **same answers** as the reference, and
2. is faster on **the staggered-adoption designs management researchers actually run, on real data**, not only on the six synthetic workloads in `code/workloads.py`.

## 2. What the computation is (short version)

For every cohort `g` (units first treated in period `g`) and every period `t`, csdid runs a small **2×2 DiD** comparing the cohort with the never-treated units. Each cell fits a **logit** (propensity score) and an **OLS** (outcome model), then combines them into a doubly robust `ATT(g,t)` with an influence function. The cells are then aggregated into an overall ATT, an event study, and pre/post averages.

**Cost grows with cells × units per cell**, roughly number of cohorts × periods × never-treated units, because every cell refits both models. That's why long panels with many cohorts are slow. Full details are in the Appendix.

**Input format** (one row per unit):
- `y` [n_units, n_periods]: a balanced panel of outcomes;
- `X` [n_units, k]: covariates fixed over time (baseline values);
- `gvar` [n_units]: first treated period, or 0 if never treated. Cohorts start at period 2 or later.

## 3. The groups of workloads

Think of it like studying for an exam.

| Group | Like… | What it is | When to use it |
|---|---|---|---|
| **Synthetic 1–6** (in `code/workloads.py`) | worked examples | made-up data of six recognizable research shapes | any time |
| **100s: development** | practice problems | real patent data, technology areas A–F (pharma, chemistry, mechanical, …) | any time, while building and improving the method |
| **200s: held out, same field** | the final exam | real patent data, technology areas G–H (computing, electronics), never practiced on | only on a frozen method |
| **300s, 400s: held out, different fields** | a final exam in other subjects | SEC filings (corporate governance) and county employment with state minimum wages (labor policy) | only on a frozen method |

**Rules:**
- **Develop and tune only** on the 100s and the synthetic workloads 1–6.
- **The 200s, 300s and 400s are for final evaluation only.** Don't use them for tuning, debugging or choosing settings, and don't look at their results while developing.
- **Freeze the method first:** make a dated git commit (or a dated copy of the code and settings) before running the held-out workloads. Keep to two or three rounds, and report every round, including failures.
- **Disclosure:** the held-out data are in this folder, so the evidence rests on this protocol. The paper will state that the held-out workloads were available to the developers but, by protocol, used only to evaluate frozen versions.

## 4. Common staggered-DiD designs in management, and where each is covered

| Design | Typical management example | Development | Held out |
|---|---|---|---|
| State-level shock studied at the firm or local level | state laws applied to firms in the state | 101 | 401, 402, 403 |
| Firm-level event (alliances, diversification, executive change) | first alliance; CEO turnover | 102, 103 | 203, 301 |
| Individual-level panel (employees, inventors) | job moves | 104, 107 | 201 |
| Rare, late technology adoption | AI adoption | 105 | 202 |
| Nearly universal adoption (few never-treated) | practices most firms eventually adopt | 108 | 301 (83% treated) |
| No covariates (unconditional DiD) | many published papers | 106 | – |
| Binary outcome | any activity, exit | 107 | – |
| Many covariates | rich controls | 103 (10) | 401–403 (7) |
| Monthly or quarterly data, many cohorts | platform rollouts, quarterly firm data | 104 (monthly), 105 (quarterly) | 202, 403 (quarterly) |

## 5. Workloads

All 15 workloads are built and pass the data checks in `data/data_checks.jsonl`: values are finite, `gvar` codes are valid, every workload has never-treated units, and panels are balanced.

**How to read the time columns:**
- These are **estimates, not measurements** (see section 6).
- "Python reference" is `code/reference.py`; 7 of these were measured before runs were stopped, and are marked.
- "Cells" is `n_cohorts × (n_periods − 1)`.

### Development: PatentsView, technology areas A–F

| # | One row is… | Treatment (first period of…) | Outcome | Covariates | Units × periods | Cohorts | Treated | Smallest cohort | Cells | Stata `csdid` | Stata `csdid2` | Python reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 101 | a firm in a year, 2000–2019 | the firm's state reaching ≥ 10 AI patent filings in a year | log(1 + patents) | 5 | 9,175 × 20 | 13 | 83% | 27 | 247 | ~3 min | ~1–3 s | 0.2 s (measured) |
| 102 | a firm in a year, 1990–2019 | first co-patent with another company (alliance proxy) | log(1 + patents) | 5 | 6,813 × 30 | 29 | 13% | 5 | 841 | ~12 min | ~10–25 s | 2.1 s (measured) |
| 103 | a firm in a year, 2000–2019 | first patent in a technology section new to the firm | log(1 + patents) | **10** | 50,466 × 20 | 19 | 16% | 124 | 361 | ~24 min | ~0.5–1.3 min | 8.3 s (measured) |
| **104** | **an inventor in a month, 2015–2019** | first filing for a new firm | patents filed in month | 5 | **340,441 × 60** | **59** | 32% | 931 | **3,481** | **~3 days** | **~0.5–1 h** | ~5.6 min |
| 105 | a firm in a quarter, 2015Q1–2025Q2 | first granted AI patent | log(1 + patents) | 5 | 57,786 × 42 | 41 | 1% | 3 | 1,681 | ~4.5 h | ~3–8 min | 40 s (measured) |
| 106 | same as 101 | same as 101 | same as 101 | **0** | 9,175 × 20 | 13 | 83% | 27 | 247 | ~3 min | ~1–3 s | 0.1 s (measured) |
| 107 | an inventor in a year, 2000–2019 | first filing for a new firm | **1{any patent}** | 5 | 198,649 × 20 | 19 | 53% | 1,530 | 361 | ~1.6 h | ~1–3 min | 14 s (measured) |
| 108 | one of the largest firms in a year, 1990–2019 | first patent with inventors in two or more countries | log(1 + patents) | 5 | 775 × 30 | 28 | 78% | 2 | 812 | ~1.3 min | under 1 s | 0.2 s (measured) |

### Held out: PatentsView, technology areas G–H

| # | Mirrors | One row is… | Treatment | Outcome | Covariates | Units × periods | Cohorts | Treated | Smallest cohort | Cells | Stata `csdid` | Stata `csdid2` | Python reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 201 | 107 | an inventor in a year, 2000–2019 | first filing for a new firm | log(1 + patents) | 5 | 166,971 × 20 | 19 | 58% | 1,344 | 361 | ~1.3 h | ~1–2 min | ~11 s |
| 202 | 105 | a firm in a quarter, 2015Q1–2025Q2 | first granted AI patent | log(1 + patents) | 5 | 33,526 × 42 | 41 | 4% | 15 | 1,681 | ~2.6 h | ~2–5 min | ~22 s |
| 203 | 102 | a firm in a year, 1990–2019 | first co-patent | log(1 + patents) | 5 | 2,344 × 30 | 29 | 13% | **1** | 841 | ~4 min | ~3–9 s | ~0.7 s |

### Held out: SEC insider filings (corporate governance)

| # | One row is… | Treatment | Outcome | Covariates | Units × periods | Cohorts | Treated | Smallest cohort | Cells | Stata `csdid` | Stata `csdid2` | Python reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 301 | a firm in a year, 2006–2025 | first CEO turnover | net insider purchase ratio | 5 | 1,946 × 20 | 19 | 83% | 23 | 361 | ~1 min | under 1 s | ~0.1 s |

### Held out: QCEW county employment and state minimum wages (labor policy)

| # | One row is… | Treatment | Outcome | Covariates | Units × periods | Cohorts | Treated | Smallest cohort | Cells | Stata `csdid` | Stata `csdid2` | Python reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 401 | a county in a year, 2010–2024 (restaurants, NAICS 722) | state minimum wage first above $7.25 | log employment | 7 | 1,514 × 15 | 5 | 36% | 72 | 70 | ~6 s | under 1 s | under 0.1 s |
| 402 | a county × 3-digit industry in a year | same | log employment | 7 | 47,994 × 15 | 5 | 38% | 2,330 | 70 | ~3.4 min | ~5–12 s | ~1 s |
| **403** | **a county × 3-digit industry in a quarter, 2010Q1–2024Q4** | same, by quarter | log employment | 7 | **42,874 × 60** | 9 | 37% | 54 | 531 | **~1.5 h** | ~0.5–1.3 min | ~6 s |

**What the estimates suggest:**
- **The original Stata `csdid` takes hours on half the workloads,** and about three days on the monthly inventor panel (104).
- **`csdid2` is much faster, but still takes minutes,** and up to an hour on 104.
- **The slow cases are long panels with many cohorts** (104, 105, 107, 201, 202, 403). That's where a faster method matters.

## 6. How the times were estimated

Times are scaled from the measurements already made for the six synthetic workloads, on the same machine (Apple M3 Pro, Stata/MP 19.5 with 12 cores; see the study report in the shared repository):

| Tool | Rule (fitted to synthetic workloads 1–6) |
|---|---|
| Stata `csdid` 1.81 | about 4 µs × (n_units × n_periods) × cells: each cell rescans the whole long-format dataset |
| Stata `csdid2` | about 2–5 µs per unit per cell, where units per cell = cohort size + never-treated |
| Python reference | about 0.41 µs per unit per cell. This is the median over the 7 workloads measured here, which ranged 0.26–0.54, plus one outlier: 1.5 for the tiny workload 108. |

Expect the estimates to be off by up to 2–3×. Measured runs should replace them.

## 7. Data sources

| Source | What | Access |
|---|---|---|
| **PatentsView**, release 2026-04-10 (data.uspto.gov, dataset PVGPATDIS) | patents granted 1976–2025; disambiguated inventors and firms; CPC technology classes; citations | free USPTO.gov account with multi-factor authentication; scripted downloads need an ODP API key |
| **SEC Insider Transactions Data Sets** (Forms 3, 4, 5) | officer and director filings and trades, 2006 Q1–2026 Q3 | free; scripts must send a contact email in the `User-Agent` header |
| **BLS QCEW** single files, 2010–2024, annual and quarterly | county × industry employment, wages, establishments | free (bls.gov/cew); scripts must send a contact header |
| **State minimum wages:** Vaghul & Zipperer v1.4.0 (1974–2022) and the U.S. Department of Labor's state minimum-wage history table (through 2024) | state minimum wage by year and quarter | free (GitHub release; dol.gov) |

Raw data are not included. The build scripts record checksums of every raw file.

## 8. How the workloads were built

**Rules for all workloads:**
- **Balanced panels.** Count outcomes are zero-filled: a period with no patents counts as zero. That's what most csdid users effectively get, since common software drops incomplete units by default.
- **Always-treated units are dropped:** units already treated in the first period, which csdid can't use. The number dropped is in each manifest.
- **Covariates are baseline values** measured before the panel starts, so they're fixed over time as the task requires.
- **Treatment after the window counts as never treated** within the window.

**PatentsView:**
- Utility patents only, withdrawn patents excluded.
- A patent's firm is its first company assignee.
- **Development and held-out are split by unit:** each firm or inventor goes to the technology area (A–F or G–H) where most of its patents are, so the two sets share no units. This split is reused from the reghdfe (task 001) build.
- Outcomes use **filing year** (patents filed in the year and later granted), except 105 and 202, which use grant dates. AI patents are CPC subclass **G06N**; software is **G06F**.
- **Treatments:**
  - **101:** a firm's state is its most common assignee state, and the state is treated in the first year it has at least 10 AI patent filings.
  - **102 and 203:** a co-patent is a patent with two or more company assignees.
  - **103:** the first patent whose main CPC section the firm hadn't patented in through 1999.
  - **104, 107 and 201:** a move is the first filing for a firm other than the inventor's firm at the baseline date.
  - **105 and 202:** the first granted AI patent.
  - **108:** the first patent whose inventors are in two or more countries.
- **Samples:**
  - 101: US firms with at least 3 patents filed 1990–1999.
  - 102 and 203: firms with at least 5 patents filed 1980–1989.
  - 103: firms with a patent filed 1995–1999.
  - 105 and 202: firms with a patent granted 2010–2014.
  - 108: the 3,000 largest firms with a patent filed 1985–1989.
  - Inventor panels: inventors with at least 3 patents, one of them filed in the five years before the panel.

**SEC (301):**
- Original Forms 3/4/5 only; amendments are excluded.
- Insiders are directors and officers. An insider counts as present at a firm from their first to their last filing year there.
- The sample is firms with insider filings in every year 2006–2025.
- A CEO turnover is the first year a new CEO (by title) appears after a different CEO was observed.
- The outcome is (shares bought − shares sold) / (shares bought + shares sold) in open-market trades, and 0 if there were none.

**QCEW (401–403):**
- Private sector, county level, units with disclosed and positive employment in every period.
- A state is treated in the first year (or quarter) its minimum wage is above the federal $7.25. States already above it in 2010 are dropped (13 states plus DC, including Alaska, California and Massachusetts; listed in each manifest).
- For 2023–2024, minimum wages come from the Labor Department's table, applied from the year's first quarter.

**Covariates:**

| Workloads | Covariates |
|---|---|
| 101 | log patent stock, log inventors (last 5 years), technology breadth (1 − HHI over CPC subclasses), share of patents with foreign inventors, years since first patent |
| 102, 203 | log patent stock, log inventors, technology breadth, mean team size, foreign-firm indicator |
| 103 | log patent stock, log patents in last 5 years, log inventors, technology breadth, mean team size, mean log claims, share with foreign inventors, log citations received, years since first patent, foreign-firm indicator |
| 104 | log prior patents, career years, prior firms, prior technology sections, mean team size |
| 105, 202 | log patent stock, log inventors, technology breadth, share of software (G06F) patents, years since first patent |
| 106 | none |
| 107, 201 | log prior patents, career years, prior firms, prior technology sections, female indicator (PatentsView's name-based coding) |
| 108 | log patent stock, log inventors, technology breadth, foreign-firm indicator, years since first patent |
| 301 | log insiders, share directors, share officers, log(1 + trades), mean boards per director (all in 2006) |
| 401–403 | log employment, log average weekly wage, log establishments, log total private employment of the county, three Census-region dummies (West omitted), all from 2010 |

**Known limitations:**
- **Workload 203 has a cohort of one firm,** and 105 and 108 have cohorts of 2–3 units. Tiny cohorts are realistic but can strain the propensity-score logit. The reference solved 105 and 108; 203 hasn't been run.
- **The patent treatments are events, not policy experiments.** That's fine for testing computation, but they aren't causal claims.
- **Connecticut is absent from 401–403.** Connecticut's QCEW county codes changed to planning regions in recent years, so its units aren't balanced over 2010–2024.
- **2023–2024 minimum wages** use the Labor Department's yearly rate, applied from the first quarter. Mid-year changes in those two years are approximated.
- **QCEW hides small cells for confidentiality,** so 402 and 403 keep only units disclosed in every period.

### Data lineage: summary

How each workload's units are selected from the raw downloads. The full step-by-step lineage, with every variable's formula and treatment definition, is in Appendix B (also in `build/DATA_LINEAGE.md`).

**Shared starting point (PatentsView):**
- 8,512,858 utility patents; 7,587,013 with a firm and a filing date.
- Firms and inventors are assigned by their main technology area: 236,761 development and 111,267 held-out firms; 1,846,175 development and 1,378,328 held-out inventors.

| # | From raw data to the workload (units) | Always-treated dropped |
|---|---|---|
| 101 | 236,761 development firms → 118,548 with a US state → 14,281 with ≥ 3 patents filed 1990–1999 → **9,175** | 5,106 (firms in the 6 states treated by 2000) |
| 102 | 236,761 development firms → 8,250 with ≥ 5 patents filed 1980–1989 → **6,813** | 1,437 |
| 103 | 236,761 development firms → 52,804 with a patent filed 1995–1999 → **50,466** | 2,338 |
| 104 | 1,846,175 development inventors → 344,214 with ≥ 3 firm patents and one filed 2010–2014 → **340,441** | 3,773 |
| 105 | 236,761 development firms → 58,019 with a patent granted 2010–2014 → **57,786** | 233 |
| 106 | same units as 101; covariates removed | – |
| 107 | 1,846,175 development inventors → 226,091 with ≥ 3 firm patents and one filed 1995–1999 → **198,649** | 27,442 |
| 108 | 236,761 development firms → 3,000 largest → 1,759 with a patent filed 1985–1989 → **775** | 984 |
| 201 | 1,378,328 held-out inventors → 193,336 eligible (as in 107) → **166,971** | 26,365 |
| 202 | 111,267 held-out firms → 34,361 with a patent granted 2010–2014 → **33,526** | 835 |
| 203 | 111,267 held-out firms → 2,810 with ≥ 5 patents filed 1980–1989 → **2,344** | 466 |
| 301 | 4,496,561 SEC filings → 4,217,464 original Forms 3/4/5, 2006–2025 → 4,004,481 director or officer filings (19,120 firms) → **1,946** firms with filings in every year | 0 |
| 401 | 3,229 counties with restaurant (NAICS 722) employment → 1,921 disclosed in all 15 years → 1,906 with a 2010 baseline → **1,514** | 392 (13 states plus DC above $7.25 in 2010) |
| 402 | 251,378 county × industries → 62,293 disclosed in all 15 years → 61,590 with a baseline → **47,994** | 13,596 |
| 403 | 251,378 county × industries → 55,830 disclosed in all 60 quarters → 55,258 with a baseline → **42,874** | 12,384 |

## 9. Common cases this task doesn't cover yet

From `task.md` ("Not covered in this version"): wild-bootstrap standard errors and uniform bands (`wboot`), the not-yet-treated comparison group (`notyet`), the long base period, **unbalanced panels**, sampling weights, and group or calendar aggregations.

Several are common in management research. Ranked by priority for a future task variant:
1. **Clustered standard errors plus multiplier bootstrap**, e.g. clustering by state for 101 and 401–403. This is the slowest part in practice.
2. **Not-yet-treated controls,** for adoption studies like 105 and 202.
3. **Unbalanced panels,** with one specified rule.

## 10. The folder

```
csdid_real_data_package/
├── README.md                this guide (task.md is Appendix A; the data lineage is Appendix B)
├── requirements.txt
├── code/
│   ├── task.md              task definition (by the task author, unchanged)
│   ├── task.json            machine-readable contract (by the task author, unchanged)
│   ├── reference.py         reference implementation: run(y, X, gvar) (by the task author, unchanged)
│   ├── workloads.py         synthetic workloads 1-6 (by the task author, unchanged)
│   ├── realdata.py          loader: load(id, data_root) -> dict(y, X, gvar)
│   └── example.py           load a workload and solve it with the reference
├── data/
│   ├── development/         workloads 101-105, 107, 108 (Parquet) + one manifest per file; 106 reuses 101
│   ├── heldout/             workloads 201-203, 301, 401-403: FINAL EVALUATION ONLY
│   └── data_checks.jsonl    size, cohorts and checks for every workload
└── build/                   scripts that built everything from the raw downloads, and DATA_LINEAGE.md
```

**Getting started:**

```bash
cd csdid_real_data_package
python -m pip install -r requirements.txt
python code/example.py 101          # small: well under a second
```

```python
import sys; sys.path.insert(0, "code")
from realdata import load
d = load(104, data_root="data")      # dict(y [340441, 60], X [340441, 5], gvar [340441])
```

A faster method must accept `run(y, X, gvar)` and return the same outputs as `code/reference.py`, within the tolerances in Appendix A.

## 11. Next steps

1. Measure Stata `csdid`, `csdid2` and the reference on the development workloads, replacing the estimates.
2. Develop the faster method on 101–108 and synthetic 1–6.
3. Freeze it, then run the 200s, 300s and 400s and report all results.
4. Plan a task variant for clustering plus bootstrap, which is the most common feature not yet covered.

---

## Appendix A: task definition (`task.md`, written by the task author, reproduced unchanged)

# Task 002: Staggered-adoption difference-in-differences (doubly robust)

## Computation

Staggered-adoption difference-in-differences (Callaway & Sant'Anna 2021) using the doubly robust inverse-probability-weighting estimator (Sant'Anna & Zhao 2020). This is the computation behind

```stata
csdid y x1 ... xk, ivar(id) time(t) gvar(g) method(dripw)
estat simple
estat event
```

or, equivalently, with the rewrite `csdid2`:

```stata
csdid2 y x1 ... xk, ivar(id) time(t) gvar(g) method(dripw) short
estat simple
estat event
```

`csdid2` defaults to the **long** base period (every pre-treatment cell compared with `g − 1`). It needs `short` to match `csdid`'s default. This changes only the pre-treatment cells and the `Pre_avg` and `Tm*` event-time effects.

Units first treated in period `g` form **cohort g**; `g = 0` marks never-treated units. The computation has four steps:

1. **One 2×2 DiD per (cohort, period) cell.** For every cohort `g` and period `t = 2..T`, take the cohort-g units plus all never-treated units, and compare period `t` with a base period:
   - post-treatment cells (`t ≥ g`): base period `g − 1`;
   - pre-treatment cells (`t < g`): base period `t − 1` (csdid's default "varying" base period).
2. **Doubly robust ATT in each cell.** Using the outcome change `Δy = y_t − y_base`:
   - fit a **logit** of cohort membership on `X` (the propensity score);
   - fit an **OLS** of `Δy` on `X` among never-treated units (the outcome model);
   - combine them into `ATT(g,t)`, plus a per-unit influence function that accounts for the estimation error of both models.
   - Comparison units with a propensity score above 0.995 get zero weight (csdid's trimming rule).
3. **Rescale to the full sample.** Each cell's influence function is rescaled to all `n_units` units, so cells can be combined. The standard error of `ATT(g,t)` is `sqrt(Σ_i (ψ_i − ψ̄)²) / n_units`.
4. **Aggregate.**
   - **Simple ATT:** average of the post-treatment `ATT(g,t)`, weighted by cohort size. The influence function includes the uncertainty from estimating those weights.
   - **Event study:** for each event time `e = t − g`, the cohort-size-weighted average of `ATT(g, g+e)`.
   - **Pre/post averages:** equal-weighted averages of the pre-treatment (`e < 0`) and post-treatment (`e ≥ 0`) event-time effects.

**Where the time goes.** There are `n_cohorts × (n_periods − 1)` cells, each running its own logit and regression on the cohort's units plus all never-treated units, so cost grows with cells × units per cell. When few units are ever treated (workload 6), almost every unit is in every cell. Stata's `csdid` calls `drdid` once per cell, and each call filters and re-sorts the full long-format dataset, which adds several passes over all rows for every cell. It also keeps every cell's influence function as a full-length variable, so its memory grows with cells × rows. `csdid2` computes the cells in Mata without that per-cell data handling. The cells are independent of each other, so they can be computed in parallel.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| y | [n_units, n_periods] | float64 |
| X | [n_units, n_covariates] | float64 |
| gvar | [n_units] | int64 |

- `y` is a balanced panel; periods are numbered `1..n_periods`.
- `X` holds time-invariant covariates; a constant is added internally.
- `gvar[i]` is the first treated period of unit `i`, or `0` if never treated. Every cohort has `g ≥ 2`.

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| gt | [n_gt, 2] | int64 | (cohort, period) of each cell, in csdid's order |
| att_gt | [n_gt] | float64 | `e(b)`, `g*:t_*` columns |
| se_gt | [n_gt] | float64 | square root of `e(V)` diagonal |
| att_simple, se_simple | scalar | float64 | `estat simple` |
| event_times | [n_event] | int64 | `Tm*`, `Tp*` labels of `estat event` |
| att_event, se_event | [n_event] | float64 | `estat event` |
| att_pre_avg, se_pre_avg | scalar | float64 | `estat event`, `Pre_avg` |
| att_post_avg, se_post_avg | scalar | float64 | `estat event`, `Post_avg` |

`n_gt = n_cohorts × (n_periods − 1)`.

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(y, X, gvar) -> dict(gt, att_gt, se_gt, att_simple, se_simple, event_times, att_event, se_event,
                        att_pre_avg, se_pre_avg, att_post_avg, se_post_avg)
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Treatment timing depends on the covariates, and the covariates have time-varying effects on the outcome, so both the propensity score and the outcome model matter.

| # | design | n_units | n_periods | n_cohorts | share treated | n_covariates | n_gt | n_obs (long) |
|---|---|---|---|---|---|---|---|---|
| 1 | sanity check | 1,000 | 10 | 4 | 71% | 2 | 36 | 10,000 |
| 2 | state/county policy adoption | 3,000 | 20 | 10 | 51% | 3 | 190 | 60,000 |
| 3 | firm-year study | 5,000 | 30 | 20 | 25% | 5 | 580 | 150,000 |
| 4 | private firms / establishments, annual | 50,000 | 20 | 10 | 69% | 10 | 190 | 1,000,000 |
| 5 | monthly platform rollout | 100,000 | 60 | 40 | 69% | 5 | 2,360 | 6,000,000 |
| 6 | GenAI adoption, firm-quarter | 285,000 | 42 | 10 | 10% | 5 | 410 | 11,970,000 |

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| att_gt, att_simple, att_event, pre/post averages | max relative error ≤ 1e-6 |
| all standard errors | max relative error ≤ 1e-4 |
| gt, event_times | exact |

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy implementation: one small regression problem per (cohort, period) cell. It defines what is computed and is not a performance target.

## Not covered in this version

- Wild-bootstrap standard errors and uniform confidence bands (`wboot`). These are random, so a deterministic version would take the bootstrap multiplier weights as an input.
- The not-yet-treated comparison group (`notyet`), the `long` base period, unbalanced panels, sampling weights, and the group and calendar aggregations.

## Shared evaluator (contract version 2)

`task.json` identifies the prompt files, generator, validator and tolerances. Inputs are read-only. Outputs must have the documented shapes and numeric types and contain only finite values. Relative errors use `abs(actual - reference) / max(abs(reference), 1e-12)` element by element, including near zero.

Run `python -m harness evaluate --task 002 --solution solutions/002/candidate/solution.json --smoke` from the repository root for public development fixtures. Omit `--smoke` to validate those fixtures and time official workload 1; use `--ids` or `--all-workloads` for wider coverage. Every first-call, warmup and measured output must pass. The score uses the ratio of reference and candidate median call times on the same host. Generation, imports, first-call setup, validation and serialization are excluded; input-dependent preprocessing and conversion to host-ready outputs are included. See `harness/README.md` for the full protocol. Historical baseline results predate this evaluator and are not automatically certified by it.

---

## Appendix B: full data lineage (`build/DATA_LINEAGE.md`)

### Data lineage: from raw downloads to workloads (task 002, csdid)

This document traces every workload back to the raw files: which tables and columns are used, each transformation step, how many units remain after it, and how every variable is computed. The authoritative code is in this folder (`stage_qcew.py`, `build_patentsview.py`, `build_sec.py`, `build_qcew.py`). Counts come from read-only queries on the build's intermediate tables.

Every workload has the same shape:
- one row per unit;
- `gvar`, the first treated period, or 0 if never treated within the window;
- baseline covariates;
- outcomes `y001` … `yT` for a balanced panel.

**Rules for all workloads:**
- **Zero-filling:** count outcomes are zero-filled, so a period with no patents is 0.
- **Always-treated units are dropped:** units treated in or before the first period. csdid can't use them.
- **Treatment after the window counts as never treated.**
- **Covariates are measured before the panel starts.**

#### 1. Raw files

| Source | Files | Used for |
|---|---|---|
| PatentsView, release 2026-04-10 (PVGPATDIS) | `g_patent`, `g_application`, `g_inventor_disambiguated`, `g_assignee_disambiguated`, `g_location_disambiguated`, `g_cpc_current`, `g_us_patent_citation` | 101–108, 201–203 |
| SEC Insider Transactions Data Sets, 2006q1–2026q3 (83 files) | `SUBMISSION`, `REPORTINGOWNER`, `NONDERIV_TRANS` | 301 |
| BLS QCEW single files, 2010–2024 | `YYYY_annual_singlefile.zip`, `YYYY_qtrly_singlefile.zip` (30 files) | 401–403 |
| State minimum wages | Vaghul & Zipperer v1.4.0 `mw_state_annual.xlsx`, `mw_state_quarterly.xlsx` (1974–2022); U.S. Department of Labor state minimum-wage history page (through 2024) | 401–403 |

**Shared base tables.** PatentsView and SEC are staged, and the PatentsView base tables built, by the task-001 (reghdfe) package. That package's `build/DATA_LINEAGE.md` covers raw file sizes, the patent, firm and inventor definitions, and the development/held-out split. The main points:
- **Patents:** 8,512,858 utility, not-withdrawn patents.
- **Firm:** a patent's firm is its first company assignee.
- **Split:** each firm or inventor is assigned to development (A–F) or held out (G–H) by the main CPC section of most of its 1980–2019 patents. That gives 236,761 development and 111,267 held-out firms, and 1,846,175 development and 1,378,328 held-out inventors.

#### 2. PatentsView additions for task 002 (`build_patentsview.py`, base step)

| Table | Rule | Size |
|---|---|---|
| AI patents | any `cpc_subclass = 'G06N'` (machine learning and AI) in `g_cpc_current` | 98,815 patents |
| Software patents | any `cpc_subclass = 'G06F'` | 927,825 patents |
| Co-patents | two or more distinct company assignees in `g_assignee_disambiguated` | 251,256 patents |
| Firm patents | utility patents with a firm and a valid filing date: filing year, grant date, main section and subclass, team size, foreign-inventor flag, international-team flag (inventors in two or more countries), AI, software and co-patent flags | 7,587,013 patents; 444,642 firms |
| Inventor patents | inventor × patent with filing date, firm, main section, team size, gender code | 22,256,328 rows; 4,090,537 inventors |

**Firm baseline covariates, at year B** (patents filed up to B; citations by citing grant year up to B):

| Name | Formula |
|---|---|
| `log_stock` | log(1 + patents filed ≤ B) |
| `log_recent5` | log(1 + patents filed B − 4 … B) |
| `log_inventors` | log(1 + distinct inventors on patents filed B − 4 … B) |
| `breadth` | 1 − Σ (share of patents in each main CPC subclass)², patents filed ≤ B |
| `mean_team` | mean team size, patents filed ≤ B |
| `mean_log_claims` | mean log claims, patents filed ≤ B |
| `share_foreign_inventors` | share of patents with a non-US inventor, filed ≤ B |
| `log_cites_received` | log(1 + citations to the firm's patents from patents granted ≤ B) |
| `years_since_first` | B − first filing year |
| `foreign_firm` | 1 if the firm's most common assignee type is foreign company |
| `share_software` | share of patents filed ≤ B that are software (G06F) |

**Inventor baseline covariates, at date B** (patents filed on or before B):

| Name | Formula |
|---|---|
| `log_prior_patents` | log(1 + patents) |
| `career_years` | (B − first filing date) / 365.25 |
| `prior_firms` | distinct firms |
| `prior_sections` | distinct main CPC sections |
| `mean_team` | mean team size |
| `female` | 1 if the inventor's most common `gender_code` is F (PatentsView's name-based coding) |

#### 3. Workload-by-workload lineage

##### 101: firms × year, 2000–2019; state AI shock (development)

| Step | Rule | Units |
|---|---|---|
| Development firms | modal section A–F | 236,761 |
| → with a US state | the firm's most common assignee region is a US state | 118,548 |
| → active before the panel | at least 3 patents filed 1990–1999 | 14,281 |
| → drop always-treated | firms in states already treated by 2000 (6 states reached the threshold in 1988–2000) | **9,175** (5,106 dropped) |

- **Treatment:** the first year in which the firm's state has at least 10 AI patent filings, counting all firms located in the state.
  - Cohorts: 2001, 2007, 2008 and 2010–2019, which is **13 cohorts**.
  - States first reaching the threshold in 2020–2021, or never, are never treated in the window.
- **Outcome:** log(1 + patents filed in year t, later granted).
- **Covariates (B = 1999):** `log_stock`, `log_inventors`, `breadth`, `share_foreign_inventors`, `years_since_first`.

##### 102 (development) and 203 (held out): firms × year, 1990–2019; first co-patent

| Step | Rule | 102 | 203 |
|---|---|---|---|
| Firms in group | modal section A–F (102) or G–H (203) | 236,761 | 111,267 |
| → active before the panel | at least 5 patents filed 1980–1989 | 8,250 | 2,810 |
| → drop always-treated | first co-patent filed in or before 1990 | **6,813** (1,437 dropped) | **2,344** (466 dropped) |

- **Treatment:** filing year of the firm's first co-patent (alliance proxy).
- **Outcome:** log(1 + patents filed).
- **Covariates (B = 1989):** `log_stock`, `log_inventors`, `breadth`, `mean_team`, `foreign_firm`.

##### 103: firms × year, 2000–2019; diversification (development)

| Step | Rule | Units |
|---|---|---|
| Development firms | | 236,761 |
| → active before the panel | at least one patent filed 1995–1999 | 52,804 |
| → drop always-treated | first new-section patent filed in 2000 | **50,466** (2,338 dropped) |

- **Treatment:** filing year of the first patent (2000 or later) whose main CPC section the firm had never patented in through 1999.
- **Outcome:** log(1 + patents filed).
- **Covariates (B = 1999), 10:** `log_stock`, `log_recent5`, `log_inventors`, `breadth`, `mean_team`, `mean_log_claims`, `share_foreign_inventors`, `log_cites_received`, `years_since_first`, `foreign_firm`.

##### 104: inventors × month, 2015–2019; inventor move (development)

| Step | Rule | Units |
|---|---|---|
| Development inventors | modal section A–F | 1,846,175 |
| → eligible | at least 3 patents with a firm, at least one filed 2010–2014 | 344,214 |
| → drop always-treated | move in January 2015 (period 1) | **340,441** (3,773 dropped) |

- **Baseline firm:** the firm of the inventor's latest patent filed on or before 2014-12-31.
- **Treatment:** the month of the first filing (2015–2019) for a firm other than the baseline firm. **59 cohorts.**
- **Outcome:** number of patents filed in the month (60 months).
- **Covariates (B = 2014-12-31):** `log_prior_patents`, `career_years`, `prior_firms`, `prior_sections`, `mean_team`.

##### 105 (development) and 202 (held out): firms × quarter, 2015Q1–2025Q2; AI adoption

| Step | Rule | 105 | 202 |
|---|---|---|---|
| Firms in group | | 236,761 | 111,267 |
| → active before the panel | at least one patent granted 2010–2014 | 58,019 | 34,361 |
| → drop always-treated | first AI patent granted before 2015 or in 2015Q1 | **57,786** (233 dropped) | **33,526** (835 dropped) |

- **Treatment:** the quarter of the firm's first granted AI patent (G06N). **41 cohorts.** AI patents granted after 2025Q2, or never, count as never treated.
- **Outcome:** log(1 + patents granted in the quarter), 42 quarters.
  - Grant dates are used here, not filing dates, because recent filings are not yet granted.
- **Covariates (B = 2014):** `log_stock`, `log_inventors`, `breadth`, `share_software`, `years_since_first`.

##### 106: same as 101 without covariates (development)

No separate file. The loader reads 101 and returns `X` with zero columns.

##### 107 (development) and 201 (held out): inventors × year, 2000–2019; inventor move

| Step | Rule | 107 | 201 |
|---|---|---|---|
| Inventors in group | modal section A–F (107) or G–H (201) | 1,846,175 | 1,378,328 |
| → eligible | at least 3 patents with a firm, at least one filed 1995–1999 | 226,091 | 193,336 |
| → drop always-treated | move in 2000 (period 1) | **198,649** (27,442 dropped) | **166,971** (26,365 dropped) |

- **Baseline firm:** the firm of the latest patent filed on or before 1999-12-31.
- **Treatment:** the year of the first filing (2000–2019) for another firm. **19 cohorts.**
- **Outcome:**
  - 107: 1{any patent filed in the year}, the binary-outcome case;
  - 201: log(1 + patents filed).
- **Covariates (B = 1999-12-31):** `log_prior_patents`, `career_years`, `prior_firms`, `prior_sections`, `female`.

##### 108: largest firms × year, 1990–2019; international R&D teams (development)

| Step | Rule | Units |
|---|---|---|
| Development firms | | 236,761 |
| → largest | the 3,000 with the most patents granted 1980–2019 | 3,000 |
| → active before the panel | at least one patent filed 1985–1989 | 1,759 |
| → drop always-treated | first international-team patent filed in or before 1990 | **775** (984 dropped) |

- **Treatment:** filing year of the firm's first patent with inventors in two or more countries. 78% of the remaining firms are eventually treated (the "nearly universal adoption" case).
- **Outcome:** log(1 + patents filed).
- **Covariates (B = 1989):** `log_stock`, `log_inventors`, `breadth`, `foreign_firm`, `years_since_first`.

##### 301: SEC firms × year, 2006–2025; CEO turnover (held out)

| Step | Rule | Count |
|---|---|---|
| Filings | `SUBMISSION` | 4,496,561 |
| → original forms 3, 4, 5, filed 2006–2025 | amendments dropped | 4,217,464 |
| Owner × filing, directors and officers | join `REPORTINGOWNER` | 4,004,481 (19,120 firms) |
| Firm × year | firms with at least one insider present (insiders present from first to last filing year at the firm) | 136,729 firm-years |
| → firms present in every year 2006–2025 | balanced panel | 1,946 firms (after the next step) |
| → drop always-treated | none: a turnover needs an earlier CEO, so the earliest possible is 2007 | **1,946** |

- **Treatment:** the first year a CEO-titled insider appears after a different CEO-titled insider was observed. **19 cohorts**; 83% of firms are treated.
- **Outcome:** net purchase ratio, (shares bought − shares sold) / (shares bought + shares sold) in open-market trades (`TRANS_CODE` P and S). 0 if there were none.
- **Covariates (2006):** log insiders, share directors, share officers, log(1 + trades), mean boards per director.

##### 401–403: QCEW with state minimum wages (held out)

**QCEW staging (`stage_qcew.py`):** the 30 single files are filtered to private ownership (`own_code` 5) and county level (`agglvl_code` 71 = county total, 75 = county × 3-digit NAICS). Real counties only: statewide (xx000) and unknown-county (xx999) codes are excluded. That leaves 3,043,664 annual and 12,174,656 quarterly rows.

| Step | 401 (NAICS 722, year) | 402 (all 3-digit, year) | 403 (all 3-digit, quarter) |
|---|---|---|---|
| Units appearing at all | 3,229 counties | 251,378 county × industries | 251,378 |
| → disclosed and positive employment in every period (15 years or 60 quarters) | 1,921 | 62,293 | 55,830 |
| → 2010 baseline available and in the 50 states or DC | 1,906 | 61,590 | 55,258 |
| → drop always-treated states (above $7.25 already in 2010) | **1,514** (392 dropped) | **47,994** (13,596 dropped) | **42,874** (12,384 dropped) |

- **Minimum wage by state and period:**
  - **2010–2022:** Vaghul & Zipperer's state *maximum* within the year (annual) or quarter (quarterly).
  - **2023–2024:** the Labor Department's yearly rate, applied from the first quarter.
  - States with no state minimum get the federal $7.25.
- **Treatment:** the first year (401, 402) or quarter (403) the state minimum wage is above $7.25. **5 cohorts annually, 9 quarterly.**
  - Dropped as always-treated: 13 states plus DC (AK, CA, DC, IL, ME, MA, MI, NV, NM, OH, OR, RI, VT, WA).
  - Connecticut is absent because its county codes changed to planning regions, so its units aren't balanced over 2010–2024.
- **Outcome:**
  - 401 and 402: log annual average employment;
  - 403: log of the quarter's average monthly employment (`month1_emplvl` to `month3_emplvl`).
- **Covariates (2010 annual), 7:** log employment, log average weekly wage, log establishments of the unit; log total private employment of the county (`agglvl_code` 71, `industry_code` 10); Northeast, Midwest and South dummies (West omitted).

#### 4. Summary

| # | Units × periods | Cohorts | Never treated | Smallest cohort | Cells |
|---|---|---|---|---|---|
| 101 | 9,175 × 20 | 13 | 1,566 | 27 | 247 |
| 102 | 6,813 × 30 | 29 | 5,938 | 5 | 841 |
| 103 | 50,466 × 20 | 19 | 42,406 | 124 | 361 |
| 104 | 340,441 × 60 | 59 | 230,089 | 931 | 3,481 |
| 105 | 57,786 × 42 | 41 | 57,021 | 3 | 1,681 |
| 106 | 9,175 × 20 (no covariates) | 13 | 1,566 | 27 | 247 |
| 107 | 198,649 × 20 | 19 | 94,050 | 1,530 | 361 |
| 108 | 775 × 30 | 28 | 171 | 2 | 812 |
| 201 | 166,971 × 20 | 19 | 69,811 | 1,344 | 361 |
| 202 | 33,526 × 42 | 41 | 32,207 | 15 | 1,681 |
| 203 | 2,344 × 30 | 29 | 2,037 | 1 | 841 |
| 301 | 1,946 × 20 | 19 | 325 | 23 | 361 |
| 401 | 1,514 × 15 | 5 | 962 | 72 | 70 |
| 402 | 47,994 × 15 | 5 | 29,997 | 2,330 | 70 |
| 403 | 42,874 × 60 | 9 | 26,825 | 54 | 531 |

Each workload's manifest (`data/*/manifest_<id>.json`) records the same definitions, the always-treated units dropped, the source, the SHA-256 of the file and the build date.
