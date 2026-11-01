# Data lineage: from raw downloads to workloads (task 001, reghdfe)

This document traces every workload back to the raw files: which tables and columns are used, each transformation step, how many rows remain after it, and how every variable is computed. The authoritative code is in this folder (`stage_*.py`, `build_*.py`). Row counts come from read-only queries on the build's intermediate tables.

Workload numbers: development **101–109**, held-out patents **201–206**, held-out SEC **301–304**.

## 1. Raw files

### PatentsView, release 2026-04-10 (USPTO Open Data Portal, product PVGPATDIS)

| File | Rows | Columns used |
|---|---|---|
| `g_patent.tsv.zip` | 9,454,161 | `patent_id`, `patent_type`, `patent_date` (grant date), `num_claims`, `withdrawn` |
| `g_application.tsv.zip` | 9,451,902 | `patent_id`, `filing_date` |
| `g_inventor_disambiguated.tsv.zip` | 24,037,380 | `patent_id`, `inventor_sequence`, `inventor_id`, `gender_code`, `location_id` |
| `g_assignee_disambiguated.tsv.zip` | 8,751,310 | `patent_id`, `assignee_sequence`, `assignee_id`, `assignee_type`, `location_id` |
| `g_location_disambiguated.tsv.zip` | 100,452 | `location_id`, `disambig_state`, `disambig_country` |
| `g_cpc_current.tsv.zip` | 59,805,669 | `patent_id`, `cpc_sequence`, `cpc_section`, `cpc_class`, `cpc_subclass` |
| `g_us_patent_citation.tsv.zip` | 152,631,929 | `patent_id` (citing), `citation_patent_id` (cited), `citation_category` |

### SEC Insider Transactions Data Sets, 2006q1–2026q3 (83 quarterly files)

| Table | Rows | Columns used |
|---|---|---|
| `SUBMISSION` | 4,496,561 | `ACCESSION_NUMBER`, `FILING_DATE`, `DOCUMENT_TYPE`, `ISSUERCIK` |
| `REPORTINGOWNER` | 4,874,337 | `ACCESSION_NUMBER`, `RPTOWNERCIK`, `RPTOWNER_RELATIONSHIP`, `RPTOWNER_TITLE`, `RPTOWNER_STATE` |
| `NONDERIV_TRANS` | 7,353,719 | `ACCESSION_NUMBER`, `TRANS_DATE`, `TRANS_CODE`, `TRANS_SHARES`, `SHRS_OWND_FOLWNG_TRANS` |

**Staging** (`stage_patentsview.py`, `stage_sec.py`) converts each file to typed Parquet, keeping the columns above. No rows are dropped. SHA-256 checksums of the raw files are saved in `staging/*/checksums.json`.

## 2. PatentsView base tables (`build_patentsview.py`, steps `base`, `hist`, `firmyear`)

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

## 3. Inventor × patent workloads: 101–105, 201, 202, 206

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

## 4. Inventor × year event studies: 108, 205

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

## 5. Firm × filing-year panels: 106, 203

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

## 6. Firm-pair citation dyads: 107, 204

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

## 7. SEC workloads: 301–304 (`build_sec.py`)

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

## 8. Synthetic workload 109

Generated by `code/synthetic_extra.py`, with no data file. 5,000 firms × 10 years; board seats Poisson(11), at least 3; 10% yearly turnover; 20% of replacements come from people already on other boards. The outcome is a binary indicator for the top 12% of a latent index. Parameters were fixed before any SEC file was read.

## 9. Files written

`OUT/development/<id>.parquet`, `OUT/heldout/<id>.parquet` and `manifest_<id>.json`. Each manifest records the unit, sample, fixed effects, cluster, row count, source release, SHA-256 of the file and build date.
