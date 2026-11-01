# Real-data workloads for reghdfe

This page is the complete guide to the real-data workloads for speeding up `reghdfe`. It covers what we're doing, the workloads, how they were built, how to use them, and the rules for evaluating a faster method. The shared folder `reghdfe_real_data_package` contains everything needed to start.

## 1. Goal

We want a method that runs `reghdfe`-type regressions much faster and gives the same answers. To convince reviewers, we show it on **real regressions that management researchers run**, not only on made-up data.

The computation is Stata's

```stata
reghdfe y x1 ... xk, absorb(fe1 fe2 ... feG) vce(cluster cl)
```

It has four steps:

1. **Drop singletons.** Remove rows that are the only member of some fixed-effect level, and repeat until none remain.
2. **Partial out the fixed effects** from `y` and every `x`. **This step takes almost all the time.** It's slow when the fixed effects are poorly connected, for example when few inventors move between firms.
3. **Run OLS** on the demeaned data.
4. **Compute cluster-robust standard errors** with reghdfe's small-sample correction.

The exact definitions (inputs, outputs, tolerances, degrees-of-freedom rules) are in `code/task.md`. A readable reference implementation that defines the correct answer is in `code/reference.py`.

## 2. The three groups of workloads

Think of it like studying for an exam.

| Group | Like… | What it is | When to use it |
|---|---|---|---|
| **100s: development** | practice problems | Patent data from technology areas A–F (pharma, chemistry, mechanical, …). The **"who works where" structure is real**; the outcome and controls are made up. | Any time, while building and improving the method |
| **200s: held out** | the final exam | Patent data from technology areas G–H (computing, electronics), never practiced on. Real outcome and controls. | Only when the method is finished |
| **300s: held out, different field** | a final exam in another subject | SEC filings on company executives and boards (corporate governance). Real outcome and controls. | Only when the method is finished |

Plus **synthetic workloads 1–12**, which are fully made up and change one thing at a time (job mobility, size, number of regressors). They help explain *why* a method is fast.

**Why this matters:** if a method is tuned and tested on the same data, reviewers can say it just fits that data. Doing well on the 200s and 300s, which were never used for tuning, shows it really works.

**Rules:**
- Develop only on the **100s** and the **synthetic workloads 1–12**.
- The **200s and 300s are included in the folder, but they are for final evaluation only.** Don't use them for tuning, debugging or choosing settings, and don't look at their results while developing.
- Before running the 200s and 300s, **freeze the method:** make a dated git commit (or a dated copy of the code and settings) and note it. Run them only on frozen versions, keep the number of rounds small (two or three), and report every round, including failures.
- **Why the note matters:** the held-out data are available to the developer, so the evidence rests on this protocol. In the paper, we'll state that the held-out workloads were available to the developers but, by protocol, used only for final evaluation of frozen versions.
- **Workload 109** is made-up data shaped like the SEC data. Use it only to check that the method *runs* on that kind of data, such as fixing crashes. **Don't tune speed or accuracy on it.**

## 3. Five kinds of regression, each tested in both fields

| Kind of regression | Example | Development | Held out: patents | Held out: SEC |
|---|---|---|---|---|
| Person × organization network | inventor or executive × firm × year | 101–104 | 201, 206 | 301 |
| Panel with extra interacted fixed effects | firm × year plus technology × year | 105, 106 | 202, 203 | 302 |
| Pairs of organizations (dyads) | firm i × firm j × year | 107 | 204 | 303 |
| Event study | years around a job move or CEO change | 108 | 205 | 304 |
| Smoke test (SEC-shaped, made up) | | 109 | | |

## 4. Workloads

All workloads have been built, checked, and solved with the Python reference implementation.

How to read the columns:
- **Rows dropped:** singletons removed in step 1.
- **Mobility:** share of people linked to two or more organizations.
- **Components:** separate groups of connected people and organizations.
- **Reference time:** how long `code/reference.py` takes on an Apple M3 Pro.

### Development (in the folder)

| # | One row is… | absorb() | Cluster | Rows | Rows dropped | Regressors | Mobility | Components | Reference time |
|---|---|---|---|---|---|---|---|---|---|
| 101 | an inventor on a patent (areas D–E; small, quick test) | inventor, firm, year | firm | 376,758 | 28.0% | 20 | 31% | 6,442 | 13 s |
| 102 | an inventor on a patent (area A) | inventor, firm, year | firm | 1,559,804 | 17.7% | 20 | 41% | 12,253 | 41 s |
| 103 | an inventor on a patent (area C) | inventor, firm, year | firm | 2,129,562 | 13.3% | 20 | 47% | 4,383 | 46 s |
| 104 | an inventor on a patent (areas A–F) | inventor, firm, year | firm | 6,757,894 | 13.2% | 20 (or 5 or 50) | 48% | 24,339 | 3.2 min |
| 105 | same as 104 | adds technology subclass × year | firm | 6,757,894 | 13.2% | 20 | 48% | 24,317 | 3.9 min |
| 106 | a firm in a filing year (unbalanced) | firm, year | firm | 575,304 | 0.0% | 15 | – | – | 1.6 s |
| 107 | a pair of firms in a year (top 1,000 firms) | citing × year, cited × year, pair | pair | 3,176,580 | 0.0% | 10 | – | – | 8.6 s |
| 108 | an inventor in a year (event study around job moves) | inventor, year, firm | inventor | 7,335,825 | 0.5% | 30 (10 event dummies + 20) | 60% | 14,177 | 4.3 min |
| 109 | made-up executive × firm × year with board ties | person, firm, year | firm | 551,020 | 2.4% | 12 | 11% | 174 | 7.6 s |

### Held out: patents (in the folder; for final evaluation only)

| # | One row is… | Outcome | absorb() | Cluster | Rows | Rows dropped | Regressors | Mobility | Reference time |
|---|---|---|---|---|---|---|---|---|---|
| 201 | an inventor on a patent (areas G–H) | log(1 + citations in 5 years) | inventor, firm, year | firm | 7,893,490 | 9.1% | 20 | 47% | 3.7 min |
| 202 | same as 201 | same | adds technology subclass × year | firm | 7,893,490 | 9.1% | 20 | 47% | 4.5 min |
| 203 | a firm in a filing year | log(1 + patents filed) | firm, year, technology class × year, region × year | firm | 272,461 | 0.5% | 15 | – | 2.5 s |
| 204 | a pair of firms in a year (top 1,000) | log(1 + citations from i to j) | citing × year, cited × year, pair | pair | 4,828,140 | 0.0% | 10 | – | 13 s |
| 205 | an inventor in a year | log(1 + patents) | inventor, year, firm | inventor | 6,805,841 | 0.4% | 30 (10 event dummies + 20) | 58% | 3.8 min |
| **206** | **an inventor on a patent, all areas (headline)** | log(1 + citations in 5 years) | inventor, firm, year | firm | **14,651,384** | 9.0% | 20 | 51% | **8.3 min** |

### Held out: SEC (in the folder; for final evaluation only)

| # | One row is… | Outcome | absorb() | Cluster | Rows | Rows dropped | Regressors | Reference time |
|---|---|---|---|---|---|---|---|---|
| 301 | an executive or director at a firm in a year | 1 if they bought shares on the open market | person, firm, year | firm | 1,482,380 | 3.6% | 12 | 27 s |
| 302 | a firm in a year | share of insiders who are new | firm, year, state × year | firm | 117,609 | 2.2% | 10 | 0.5 s |
| 303 | a pair of firms in a year | 1 if they share a director or officer | firm i × year, firm j × year, pair | pair | 997,243 | 4.9% | 10 | 14 s |
| 304 | a firm in a year (event study around CEO turnover; 7,261 events) | net insider buying | firm, year | firm | 136,729 | 2.6% | 20 (10 event dummies + 10) | 0.3 s |

**The reference times aren't a speed baseline.** The reference is written to be readable and correct, not fast. Stata `reghdfe` and R `fixest` timings on these workloads are the next step.

## 5. What real data adds

The synthetic workloads miss several features of real research data:
- **Many singletons:** 9–28% of rows are dropped in the inventor workloads. Synthetic data had essentially none.
- **Fragmented networks:** thousands to tens of thousands of separate connected groups.
- **Realistic structure:** unbalanced panels, very unequal firm sizes, job moves clustered by field, and a director network shaped by people sitting on many boards.
- **Real outcomes and controls**, including binary outcomes in 301 and 303.

## 6. Data sources

| Source | What | Access |
|---|---|---|
| **PatentsView**, release 2026-04-10 (US patent office, data.uspto.gov, dataset PVGPATDIS) | patents granted 1976–2025; disambiguated inventors and firms; technology classes; citations | free account with multi-factor authentication; downloads need a CAPTCHA or an API key |
| **SEC Insider Transactions Data Sets** (Forms 3, 4, 5), sec.gov | officer, director and 10%-owner filings and trades, 2006 Q1–2026 Q3 | free; scripts must send a contact email in the `User-Agent` header |

Raw data are not in the folder. The build scripts record checksums, so anyone who downloads the same releases can rebuild identical files.

## 7. How the workloads were built

**Common rules:**
- **Patents:** utility patents only, withdrawn patents excluded, grant years 1980–2019. That last limit means every patent has a full 5-year citation window by 2025.
- **A patent's firm:** its first company assignee (US or foreign company). Patents with no company assignee are excluded from firm-based workloads.
- **Technology area:** the patent's main CPC class. Development uses areas A–F, held-out uses G–H.
- **Panels split by modal area:** inventors and firms in panels are assigned to the area where most of their patents are, so the development and held-out panels don't share units.
- **No look-ahead:** every control uses only information from before the observation.
- **SEC:** original Forms 3/4/5 only; amendments are excluded. Insiders are directors or officers. An insider counts as present at a firm in every year from their first to their last filing there. Years 2006–2025.

**Controls in the inventor × patent workloads** (201, 202, 206; development workloads use made-up controls instead):

| Group | Variables |
|---|---|
| Patent | log claims; log(1 + backward citations); share of citations added by the examiner; number of technology subclasses; years from filing to grant |
| Team | team size; average prior patents of co-inventors; share of first-time inventors; number of inventor regions; international team; share of inventors coded female (PatentsView's name-based coding) |
| Inventor | log(1 + prior patents); career length; number of prior firms; moved firms since last patent; number of prior technology areas |
| Firm | log(1 + patent stock); log(1 + patents in the last 5 years); technology breadth; log(1 + citations received) |

The other workloads use similar firm-, pair- and inventor-year controls; the variable names are in each file.

**Controls the fixed effects would absorb were replaced.** Firm age and career length (year minus a first year) are exact combinations of the unit and year fixed effects, and firm-by-year variables are absorbed by `firm#year` in the dyad workloads. Stata would silently drop them, so they were replaced with controls that vary within the fixed effects.

**Development workloads with made-up values:** the real ID structure is kept, and `y` and `X` are generated the same way as in the synthetic workloads, seeded with 1000 + workload number. The 10 event-time dummies in 108 are real.

**Checks performed:**
- **Independent recomputation:** the outcome and all 20 controls were recomputed in plain pandas for 500 random rows and matched exactly.
- **Reference solve:** every workload was solved with `reference.py`, and all coefficients and standard errors are finite.
- **A DuckDB bug was found and avoided:** one kind of query (a correlated `EXISTS`) returned wrong results, so all such queries were rewritten as joins.

**Known limitations:**
- **Workload 302** uses state × year rather than industry × year, because the SEC insider data have no industry codes.
- **The SEC network includes mutual-fund directors** who sit on many fund boards. That's real, but governance papers often exclude funds.
- **PatentsView's inventor and firm matching isn't perfect.** That's standard in this literature.

### Data lineage: summary

How each workload is derived from the raw downloads, with the rows remaining after each main step. The full step-by-step lineage, with every variable's formula, is in the appendix (also in `build/DATA_LINEAGE.md`).

**Shared starting point (PatentsView):**
- 9,454,161 patents → 8,512,858 utility, not withdrawn → 7,587,677 with a company assignee (the firm).
- 24,037,380 inventor rows → 22,256,621 inventor × utility patent.

| Workloads | From raw data to the workload (rows) | After singleton removal |
|---|---|---|
| 101–105, 201, 202, 206 (inventor × patent) | 22,256,621 inventor × patent → 20,877,138 with a firm → 14,651,724 granted 1980–2019 → **14,651,384** with a valid filing date and class (= 206). Split by the patent's main area: 101 (D–E) 376,758 · 102 (A) 1,559,804 · 103 (C) 2,129,562 · 104/105 (A–F) 6,757,894 · 201/202 (G–H) 7,893,490 | 101: 271,130 · 102: 1,284,231 · 103: 1,846,487 · 104: 5,867,713 · 105: 5,867,255 · 201: 7,175,392 · 202: 7,175,324 · 206: 13,332,465 |
| 108, 205 (inventor × year) | 1,145,776 inventors with ≥ 3 firm patents 1980–2019 → 14,141,666 inventor-years (first to last patent year) → split by the inventor's main area: **108** 7,335,825 · **205** 6,805,841 | 108: 7,298,772 · 205: 6,775,609 |
| 106, 203 (firm × filing year) | 7,587,013 firm patents → 5,080,636 filed 1980–2015 → 66,838 firms with ≥ 5 filings → 847,971 firm-years → split by the firm's main area: **106** 575,304 · **203** 272,461 | 106: 575,073 · 203: 270,986 |
| 107, 204 (firm-pair dyads) | top 1,000 firms per group → 158,829 (107) / 241,407 (204) ordered pairs with a citation in 2000–2019 → × 20 years: **107** 3,176,580 · **204** 4,828,140 | no singletons |
| 301–304 (SEC) | 4,496,561 filings → 4,217,464 original Forms 3/4/5 filed 2006–2025 → 4,004,481 director or officer filings (220,786 insiders, 19,120 firms) → **301** 1,482,380 insider-firm-years · **304** 136,729 firm-years · **302** 117,609 (each firm's first year excluded) · **303** 997,243 pair-years (123,618 firm pairs) | 301: 1,428,958 · 302: 115,030 · 303: 948,312 · 304: 133,135 |
| 109 | made up (`code/synthetic_extra.py`): 551,020 rows | 537,702 |

## 8. Correctness requirements for a faster method

A faster method must match `reference.py` on every workload:

| Output | Tolerance |
|---|---|
| coefficients | relative error ≤ 1e-6 |
| standard errors | relative error ≤ 1e-4 |
| observations after dropping singletons, absorbed degrees of freedom, number of clusters | exact |

The reference answers for every workload are in `data/reference_outputs/<id>.npz`. Use them to check results **after** a run. A method must never read them, or infer anything from them, while it runs.

Speed counts only if the result is correct. On poorly connected data, a loose convergence tolerance can miss the coefficient tolerance. R `fixest` with default settings already does on two synthetic low-mobility workloads.

## 9. How the comparison will be run

- **Platforms:** Stata `reghdfe` (the standard), R `fixest` (the fastest existing tool), the Python reference, and the new method.
- **Same everything:** identical data, specification and hardware for every platform.
- **Two accuracy settings:**
  - *Default settings*: what users get out of the box.
  - *Matched accuracy* (the main comparison): each tool's tolerance is tightened until it passes the correctness checks.
- **Timing:** median of 5 runs, with thread count and peak memory recorded. A single-thread run is included, so speedups aren't just more cores.
- **Reporting:** every workload, including failures. Speedups are reported by kind of regression and by difficulty, not only as one average.

## 10. The shared folder: `reghdfe_real_data_package` (1.1 GB)

```
reghdfe_real_data_package/
├── README.md                  this guide
├── requirements.txt           numpy, scipy, pandas, pyarrow, duckdb
├── code/
│   ├── example.py             load a workload and solve it with the reference
│   ├── realdata.py            loader: load(id, data_root) -> y, X, fe_ids, cluster_ids
│   ├── synthetic_extra.py     made-up workload 109
│   ├── reference.py           task-001 reference implementation (defines the correct answer)
│   ├── workloads.py           task-001 synthetic workloads 1-12
│   └── task.md                task-001 definition: inputs, outputs, tolerances
├── data/
│   ├── development/           workloads 101-108 (Parquet) + one manifest per file
│   ├── heldout/               workloads 201-206 and 301-304: FINAL EVALUATION ONLY
│   ├── reference_outputs/     reference answers (beta, vcov, nobs, df_a, n_clusters) for every workload
│   └── validation.jsonl       descriptors and reference times for every workload
└── build/                     scripts that built everything from the raw downloads, with a README
                               and DATA_LINEAGE.md (the full lineage, also the appendix below)
```

**Getting started:**

```bash
cd reghdfe_real_data_package
python -m pip install -r requirements.txt
python code/example.py 101        # small: about 14 seconds
```

```python
import sys; sys.path.insert(0, "code")
from realdata import load
inputs = load(104, data_root="data")                    # 20 made-up regressors
inputs = load(104, data_root="data", n_regressors=50)   # same structure, 50 regressors
inputs = load(109, data_root="data")                    # made-up SEC-shaped smoke test
```

A method should accept the same inputs as `reference.run(y, X, fe_ids, cluster_ids, tol, maxiter)` and return the same outputs.

## 11. Next steps

1. Time Stata `reghdfe` and R `fixest` on the development workloads, at matched accuracy.
2. Develop the faster method on 101–109 and 1–12.
3. Freeze the method (dated commit), run it on the 200s and 300s, compare with `data/reference_outputs/`, and report all results.

---

## Appendix: full data lineage (`build/DATA_LINEAGE.md`)

### Data lineage: from raw downloads to workloads (task 001, reghdfe)

This document traces every workload back to the raw files: which tables and columns are used, each transformation step, how many rows remain after it, and how every variable is computed. The authoritative code is in this folder (`stage_*.py`, `build_*.py`). Row counts come from read-only queries on the build's intermediate tables.

Workload numbers: development **101–109**, held-out patents **201–206**, held-out SEC **301–304**.

#### 1. Raw files

##### PatentsView, release 2026-04-10 (USPTO Open Data Portal, product PVGPATDIS)

| File | Rows | Columns used |
|---|---|---|
| `g_patent.tsv.zip` | 9,454,161 | `patent_id`, `patent_type`, `patent_date` (grant date), `num_claims`, `withdrawn` |
| `g_application.tsv.zip` | 9,451,902 | `patent_id`, `filing_date` |
| `g_inventor_disambiguated.tsv.zip` | 24,037,380 | `patent_id`, `inventor_sequence`, `inventor_id`, `gender_code`, `location_id` |
| `g_assignee_disambiguated.tsv.zip` | 8,751,310 | `patent_id`, `assignee_sequence`, `assignee_id`, `assignee_type`, `location_id` |
| `g_location_disambiguated.tsv.zip` | 100,452 | `location_id`, `disambig_state`, `disambig_country` |
| `g_cpc_current.tsv.zip` | 59,805,669 | `patent_id`, `cpc_sequence`, `cpc_section`, `cpc_class`, `cpc_subclass` |
| `g_us_patent_citation.tsv.zip` | 152,631,929 | `patent_id` (citing), `citation_patent_id` (cited), `citation_category` |

##### SEC Insider Transactions Data Sets, 2006q1–2026q3 (83 quarterly files)

| Table | Rows | Columns used |
|---|---|---|
| `SUBMISSION` | 4,496,561 | `ACCESSION_NUMBER`, `FILING_DATE`, `DOCUMENT_TYPE`, `ISSUERCIK` |
| `REPORTINGOWNER` | 4,874,337 | `ACCESSION_NUMBER`, `RPTOWNERCIK`, `RPTOWNER_RELATIONSHIP`, `RPTOWNER_TITLE`, `RPTOWNER_STATE` |
| `NONDERIV_TRANS` | 7,353,719 | `ACCESSION_NUMBER`, `TRANS_DATE`, `TRANS_CODE`, `TRANS_SHARES`, `SHRS_OWND_FOLWNG_TRANS` |

**Staging** (`stage_patentsview.py`, `stage_sec.py`) converts each file to typed Parquet, keeping the columns above. No rows are dropped. SHA-256 checksums of the raw files are saved in `staging/*/checksums.json`.

#### 2. PatentsView base tables (`build_patentsview.py`, steps `base`, `hist`, `firmyear`)

| Step | Rule | Rows remaining |
|---|---|---|
| Patents | all patents in `g_patent` | 9,454,161 |
| → utility only | `patent_type = 'utility'` (drops 864k design, 33k plant, 22k reissue and others) | 8,531,961 |
| → not withdrawn | `withdrawn = 0` | **8,512,858** (table `P`) |
| Filing date | earliest `filing_date` per patent; kept only if between 1900-01-01 and the grant date (some raw dates are invalid, e.g. year 1074) | 8,512,705 with a valid date |
| Main technology class | the row with `cpc_sequence = 0` (exactly one per patent) gives section, class and subclass | 8,512,704 with a class |
| Firm | first company assignee: lowest `assignee_sequence` among `assignee_type` 2 (US company) or 3 (foreign company) | 7,587,677 utility patents with a firm; 494,143 distinct firms overall |
| Firm location | the firm assignee's `location_id` → `US-<state>` for US locations, otherwise the country code | |
| Inventor × patent | `g_inventor_disambiguated` joined to `P`; duplicate (inventor, patent) pairs removed (lowest `inventor_sequence` kept) | **22,256,621** rows; 4,090,591 inventors (table `IP`) |
| Citations between utility patents | both citing and cited patent in `P` | 118,732,706 of 152,631,929 (the rest cite design, older or other patents) |
| 5-year forward citations | per cited patent: citing patents granted on or after its grant date and within 5 years | 4,730,761 patents with at least one |

**Per-patent attributes:**

| Name | Formula (raw columns) |
|---|---|
| backward citations | number of rows in `g_us_patent_citation` with this `patent_id` |
| examiner share | share of those rows with `citation_category = 'cited by examiner'` |
| number of subclasses | distinct `cpc_subclass` in `g_cpc_current` for the patent |
| team | from the inventor rows: team size; number of distinct inventor regions; international team = more than one inventor country; share female = share of `gender_code = 'F'` among inventors coded F or M |

**Inventor history before each patent** (computed for every inventor × patent row):

| Name | Formula |
|---|---|
| prior patents | the inventor's patents with an earlier grant date |
| career years | days from the inventor's first grant date to this grant date, ÷ 365.25 |
| prior firms | distinct firms on the inventor's patents with an earlier grant date |
| prior sections | distinct main CPC sections on the inventor's patents with an earlier grant date |
| moved | 1 if the previous patent (by grant date, then patent number) was granted earlier and has a different firm |

**Firm history before year t** (using patents granted before year t):

| Name | Formula |
|---|---|
| patent stock | patents granted before t |
| patents in last 5 years | patents granted in t − 5 … t − 1 |
| technology breadth | 1 − Σ (share of the firm's patents in each main CPC subclass)², over patents with a subclass |
| citations received | citations to the firm's patents from patents granted before t |

**Development/held-out split by technology area:**
- **Patent level:** a patent's area is its main CPC section. A–F is development; G–H is held out.
- **Unit level** (inventors, firms): each inventor or firm is assigned to the section where most of its 1980–2019 patents are (ties go to the alphabetically first section). That gives 1,846,175 development and 1,378,328 held-out inventors, and 236,761 development and 111,267 held-out firms.

#### 3. Inventor × patent workloads: 101–105, 201, 202, 206

| Step | Rule | Rows remaining |
|---|---|---|
| Inventor × patent rows | table `IP` | 22,256,621 |
| → with a firm | the patent has a company assignee | 20,877,138 |
| → granted 1980–2019 | so every patent has a complete 5-year citation window by the end of 2025 | 14,651,724 |
| → valid filing date and main class | | **14,651,384** (2,855,024 inventors; 347,993 firms) |

| Workload | Main section of the patent | Rows | Rows after singleton removal |
|---|---|---|---|
| 101 | D, E | 376,758 | 271,130 |
| 102 | A | 1,559,804 | 1,284,231 |
| 103 | C | 2,129,562 | 1,846,487 |
| 104, 105 | A–F | 6,757,894 | 5,867,713 (104); 5,867,255 (105) |
| 201, 202 | G–H | 7,893,490 | 7,175,392 (201); 7,175,324 (202) |
| 206 | all (A–H) | 14,651,384 | 13,332,465 |

Singleton removal is done by the reference or the method at run time, not by the build.

- **Fixed effects:** inventor, firm, grant year (as integer codes); 105 and 202 add main CPC subclass × grant year. **Cluster:** firm.
- **Outcome (201, 202, 206):** log(1 + 5-year forward citations).
- **20 controls (201, 202, 206):**
  1. log claims (`num_claims`, at least 1)
  2. log(1 + backward citations)
  3. examiner share
  4. number of subclasses
  5. grant lag in years (grant date − filing date)
  6. team size
  7. mean prior patents of co-inventors (excluding this inventor; 0 if solo)
  8. share of first-time inventors on the team
  9. number of inventor regions
  10. international team
  11. share female
  12. log(1 + prior patents)
  13. career years
  14. prior firms
  15. moved
  16. prior sections
  17. log(1 + firm patent stock)
  18. log(1 + firm patents in last 5 years)
  19. firm technology breadth
  20. log(1 + firm citations received)
- **Development (101–105):** only the fixed-effect and cluster codes are stored. The loader generates `y` and `X` (20 regressors by default) the same way as `workloads.py`, seeded with 1000 + workload number.
- **Verified:** the outcome and all 20 controls were recomputed independently in pandas for 500 random rows and matched exactly.

#### 4. Inventor × year event studies: 108, 205

| Step | Rule | Rows remaining |
|---|---|---|
| Eligible inventors | at least 3 patents with a firm granted 1980–2019 | 1,145,776 inventors |
| Inventor × year grid | every year from the inventor's first to last patent year | **14,141,666** rows |
| Split by the inventor's modal section | development / held out | **108:** 7,335,825 rows (576,636 inventors) · **205:** 6,805,841 rows (569,140 inventors) |

- **Current firm in year t:** the firm of the inventor's most recent patent granted in or before t (the latest grant in a year wins).
- **Move year:** the first year the current firm differs from the previous year's (678,914 inventors move).
- **Event-time dummies (10):** relative time = year − move year, for −5 or less, −4, −3, −2, 0, 1, 2, 3, 4, and 5 or more (−1 omitted). All zero for inventors who never move.
- **Fixed effects:** inventor, year, current firm. **Cluster:** inventor.
- **Outcome (205):** log(1 + patents granted in the year).
- **20 controls (205):**
  1. years since the last patent
  2. career² (years since first patent, squared)
  3. log(1 + prior patents)
  4. log(1 + patents in t − 1)
  5. log(1 + patents in t − 2)
  6. log(1 + collaboration volume, the sum of earlier team sizes)
  7. prior sections
  8. prior firms
  9. log(1 + citations received before t)
  10. mean prior team size
  11. share of prior patents that are solo
  12. mean prior log claims
  13. log(1 + current firm's stock)
  14. log(1 + current firm's patents in last 5 years)
  15. firm breadth
  16. log(1 + firm citations received)
  17. log(1 + firm inventors in t − 1)
  18. years at the current firm
  19. foreign firm
  20. log(1 + firm citations received in t − 1)

  Career length and firm age are *not* used, because they're exact combinations of the inventor, firm and year fixed effects.
- **Development (108):** the real event dummies are stored; the loader adds `y` and 20 generated controls.

#### 5. Firm × filing-year panels: 106, 203

| Step | Rule | Rows remaining |
|---|---|---|
| Firm patents with a filing date | | 7,587,013 patents; 444,642 firms |
| → filed 1980–2015 | 2015 is the cutoff to limit grant-lag truncation | 5,080,636 patents; 327,952 firms |
| Eligible firms | at least 5 filings in 1980–2015 | 66,838 firms |
| Firm × year grid | from first to last filing year in 1980–2015, zero-filled | **847,971** rows |
| Split by the firm's modal section | development / held out (206 rows have no modal section) | **106:** 575,304 rows (42,886 firms) · **203:** 272,461 rows (23,906 firms) |

- **Fixed effects:** firm, year; 203 adds the firm's most common CPC class × year and its most common assignee region × year. **Cluster:** firm.
- **Outcome (203):** log(1 + patents filed in the year and later granted).
- **15 controls (203), all from year t − 1 unless noted:**
  1. log(1 + filings before t)
  2. log(1 + filings)
  3. log(1 + inventors)
  4. share of inventors new to the firm
  5. share of filings with an inventor who moved in from another firm
  6. diversification (1 − HHI of subclasses, filings up to t − 1)
  7. mean team size
  8. share of filings with foreign inventors
  9. log(1 + citations received)
  10. self-citation share
  11. log(1 + cumulative distinct inventors)
  12. number of inventor regions
  13. mean log claims
  14. examiner share
  15. no filings indicator

  Firm age isn't used, because it's collinear with the firm and year fixed effects.

#### 6. Firm-pair citation dyads: 107, 204

| Step | Rule | Development (107) | Held out (204) |
|---|---|---|---|
| Top firms | the 1,000 firms with the most patents granted 1980–2019, within the group | 1,000 | 1,000 |
| Citing firm × cited firm × year | citations between two different top firms, citing patents granted up to 2019 | 849,052 rows | 1,293,920 rows |
| Pairs | ordered pairs with at least one citation in 2000–2019 | 158,829 | 241,407 |
| Grid | each pair × 2000–2019 (20 years), zero-filled | **3,176,580** | **4,828,140** |

- **Fixed effects:** citing firm × year, cited firm × year, pair. **Cluster:** pair.
- **Outcome (204):** log(1 + citations from the citing firm's patents granted in year t to the cited firm's patents).
- **10 controls (204):**
  1. technology proximity: cosine similarity of the two firms' cumulative subclass counts before t
  2. geographic proximity: the same, using inventor regions
  3. log(1 + shared inventors before t)
  4. log(1 + co-assigned patents before t)
  5. log(1 + cumulative citations i → j through t − 2)
  6. log(1 + cumulative citations j → i through t − 2)
  7. log(1 + inventors linking the pair in t − 3 … t − 1)
  8. product of the two firms' log patent stocks
  9. log(1 + citations i → j in t − 1)
  10. log(1 + citations j → i in t − 1)

  Single-firm-by-year variables are absorbed by the citing × year and cited × year fixed effects, so they're not used.

#### 7. SEC workloads: 301–304 (`build_sec.py`)

| Step | Rule | Rows remaining |
|---|---|---|
| Filings | `SUBMISSION` | 4,496,561 |
| → original forms | `DOCUMENT_TYPE` in 3, 4, 5 (amendments 3/A, 4/A, 5/A dropped) | 4,378,160 |
| → filed 2006–2025 | 2026 is incomplete | 4,217,464 |
| Owner × filing | join `REPORTINGOWNER`; keep owners whose `RPTOWNER_RELATIONSHIP` includes Director or Officer | **4,004,481**; 220,786 insiders, 19,120 firms, 351,179 insider–firm pairs |
| Insider × firm × year | every year from the pair's first to last filing year (presence) | **1,482,380** (= 301) |
| Firm × year | firms with at least one insider present | **136,729** (= 304) |
| → excluding each firm's first year | in its first year, all insiders are new by construction | **117,609** (= 302) |
| Firm pairs | pairs of firms sharing an insider in the same year at least once (123,618 pairs), in years both firms are present | **997,243** (= 303) |

- **Roles in a year:** from that year's filings (director, officer, 10% owner; CEO or CFO if the title matches "chief executive"/"CEO" or "chief financial"/"CFO"), carried forward to filled years.
- **Trades:** `NONDERIV_TRANS` rows of kept filings (6,340,067; 595,503 purchases `P`, 2,285,361 sales `S`), dated by `TRANS_DATE` when it falls in 2003–2026, otherwise by the filing year.
- **CEO turnover (304):** the first year a CEO-titled insider appears at a firm after a different CEO-titled insider was observed earlier. 7,261 firms have one.

| # | Outcome | Fixed effects | Cluster | Controls |
|---|---|---|---|---|
| 301 | 1{open-market purchase in the year} | insider, firm, year | firm | 12: director, officer, 10% owner, CEO, CFO, tenure at firm, number of boards, log(1 + holdings in t − 1), log(1 + trades in t − 1), sold in t − 1, log firm insiders, firm turnover |
| 302 | share of the firm's insiders who are new that year | firm, year, state × year (state = the firm's most common insider state) | firm | 10: log insiders, share directors, share officers, share 10% owners, log(1 + trades in t − 1), share buying in t − 1, share selling in t − 1, mean tenure, mean boards per director, log(1 + filings in t − 1) |
| 303 | 1{the two firms share a director or officer in the year} | firm i × year, firm j × year, pair | pair | 10: tie in t − 2, log(1 + shared insiders in t − 1), ever tied before, and products of the two firms' insiders, turnover, director shares, lagged trades and board busyness; plus tie in t − 1 and log(1 + insiders shared before t) |
| 304 | net purchase ratio (P − S shares) / (P + S shares), 0 if no trades | firm, year | firm | 10 event-time dummies + 10: log insiders, share directors, share officers, mean tenure, log(1 + trades in t − 1), share buying in t − 1, share 10% owners, mean boards per director, log(1 + filings), turnover in t − 1 |

#### 8. Synthetic workload 109

Generated by `code/synthetic_extra.py`, with no data file. 5,000 firms × 10 years; board seats Poisson(11), at least 3; 10% yearly turnover; 20% of replacements come from people already on other boards. The outcome is a binary indicator for the top 12% of a latent index. Parameters were fixed before any SEC file was read.

#### 9. Files written

`OUT/development/<id>.parquet`, `OUT/heldout/<id>.parquet` and `manifest_<id>.json`. Each manifest records the unit, sample, fixed effects, cluster, row count, source release, SHA-256 of the file and build date.
