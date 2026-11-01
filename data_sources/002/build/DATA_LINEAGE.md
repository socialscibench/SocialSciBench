# Data lineage: from raw downloads to workloads (task 002, csdid)

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

## 1. Raw files

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

## 2. PatentsView additions for task 002 (`build_patentsview.py`, base step)

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

## 3. Workload-by-workload lineage

### 101: firms × year, 2000–2019; state AI shock (development)

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

### 102 (development) and 203 (held out): firms × year, 1990–2019; first co-patent

| Step | Rule | 102 | 203 |
|---|---|---|---|
| Firms in group | modal section A–F (102) or G–H (203) | 236,761 | 111,267 |
| → active before the panel | at least 5 patents filed 1980–1989 | 8,250 | 2,810 |
| → drop always-treated | first co-patent filed in or before 1990 | **6,813** (1,437 dropped) | **2,344** (466 dropped) |

- **Treatment:** filing year of the firm's first co-patent (alliance proxy).
- **Outcome:** log(1 + patents filed).
- **Covariates (B = 1989):** `log_stock`, `log_inventors`, `breadth`, `mean_team`, `foreign_firm`.

### 103: firms × year, 2000–2019; diversification (development)

| Step | Rule | Units |
|---|---|---|
| Development firms | | 236,761 |
| → active before the panel | at least one patent filed 1995–1999 | 52,804 |
| → drop always-treated | first new-section patent filed in 2000 | **50,466** (2,338 dropped) |

- **Treatment:** filing year of the first patent (2000 or later) whose main CPC section the firm had never patented in through 1999.
- **Outcome:** log(1 + patents filed).
- **Covariates (B = 1999), 10:** `log_stock`, `log_recent5`, `log_inventors`, `breadth`, `mean_team`, `mean_log_claims`, `share_foreign_inventors`, `log_cites_received`, `years_since_first`, `foreign_firm`.

### 104: inventors × month, 2015–2019; inventor move (development)

| Step | Rule | Units |
|---|---|---|
| Development inventors | modal section A–F | 1,846,175 |
| → eligible | at least 3 patents with a firm, at least one filed 2010–2014 | 344,214 |
| → drop always-treated | move in January 2015 (period 1) | **340,441** (3,773 dropped) |

- **Baseline firm:** the firm of the inventor's latest patent filed on or before 2014-12-31.
- **Treatment:** the month of the first filing (2015–2019) for a firm other than the baseline firm. **59 cohorts.**
- **Outcome:** number of patents filed in the month (60 months).
- **Covariates (B = 2014-12-31):** `log_prior_patents`, `career_years`, `prior_firms`, `prior_sections`, `mean_team`.

### 105 (development) and 202 (held out): firms × quarter, 2015Q1–2025Q2; AI adoption

| Step | Rule | 105 | 202 |
|---|---|---|---|
| Firms in group | | 236,761 | 111,267 |
| → active before the panel | at least one patent granted 2010–2014 | 58,019 | 34,361 |
| → drop always-treated | first AI patent granted before 2015 or in 2015Q1 | **57,786** (233 dropped) | **33,526** (835 dropped) |

- **Treatment:** the quarter of the firm's first granted AI patent (G06N). **41 cohorts.** AI patents granted after 2025Q2, or never, count as never treated.
- **Outcome:** log(1 + patents granted in the quarter), 42 quarters.
  - Grant dates are used here, not filing dates, because recent filings are not yet granted.
- **Covariates (B = 2014):** `log_stock`, `log_inventors`, `breadth`, `share_software`, `years_since_first`.

### 106: same as 101 without covariates (development)

No separate file. The loader reads 101 and returns `X` with zero columns.

### 107 (development) and 201 (held out): inventors × year, 2000–2019; inventor move

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

### 108: largest firms × year, 1990–2019; international R&D teams (development)

| Step | Rule | Units |
|---|---|---|
| Development firms | | 236,761 |
| → largest | the 3,000 with the most patents granted 1980–2019 | 3,000 |
| → active before the panel | at least one patent filed 1985–1989 | 1,759 |
| → drop always-treated | first international-team patent filed in or before 1990 | **775** (984 dropped) |

- **Treatment:** filing year of the firm's first patent with inventors in two or more countries. 78% of the remaining firms are eventually treated (the "nearly universal adoption" case).
- **Outcome:** log(1 + patents filed).
- **Covariates (B = 1989):** `log_stock`, `log_inventors`, `breadth`, `foreign_firm`, `years_since_first`.

### 301: SEC firms × year, 2006–2025; CEO turnover (held out)

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

### 401–403: QCEW with state minimum wages (held out)

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

## 4. Summary

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
