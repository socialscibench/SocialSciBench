# Data lineage: from raw downloads to workloads (task 003, reclink)

This document traces every workload back to the raw files: which tables and columns are used, each transformation step, how many records remain after it, and how the five linkage fields are formed. The authoritative code is in this folder: `stage_sources.py`, `build_workloads.py` and `candidates.py`. Counts come from read-only queries on the staged tables. No linkage method was run.

Every workload is a **master** file (records to link) and a **using** file (the register). Both have the task's five fields, `name`, `addr`, `city`, `zip`, `state`, plus the source's record identifier in `id`.

## 1. Raw files

| Source | Files | Records | Columns used |
|---|---|---|---|
| **SBA Paycheck Protection Program (PPP) loans**, FOIA release of 2024-09-30 (data.sba.gov, dataset `ppp-foia`) | `public_150k_plus_240930.csv`, `public_up_to_150k_1…12_240930.csv`, `ppp-data-dictionary.xlsx` | 11,468,210 loans | `LoanNumber`, `ProcessingMethod` (PPP = first draw, PPS = second draw), `BorrowerName`, `BorrowerAddress`, `BorrowerCity`, `BorrowerState`, `BorrowerZip`, `ProjectState` (fallback state), `BusinessType`, `NAICSCode`, `FranchiseName`, `NonProfit` |
| **IRS Exempt Organizations Business Master File** (irs.gov, posting of 03/10/2026) | `eo1.csv`–`eo4.csv` (all regions), `eo-info.pdf` | 1,975,975 organizations | `EIN`, `NAME`, `STREET`, `CITY`, `STATE`, `ZIP`, `NTEE_CD`, `SUBSECTION` |
| **OSHA inspections** (U.S. Department of Labor Open Data Portal, "Download Complete Dataset", 2026-10-21) | `OSHA_inspection.zip` (6 CSV chunks) | 5,202,096 inspections | `activity_nr`, `estab_name`, `site_address`, `site_city`, `site_state`, `site_zip`, `naics_code`, `sic_code`, `owner_type`, `open_date` |
| **EPA Facility Registry Service**, national single file (epa.gov/frs, file dated 2026-10-12) | `national_single.zip` (`NATIONAL_SINGLE.CSV` + documentation PDF) | 5,359,938 facilities | `REGISTRY_ID`, `PRIMARY_NAME`, `LOCATION_ADDRESS`, `CITY_NAME`, `STATE_CODE`, `POSTAL_CODE`, `NAICS_CODES` |

SHA-256 checksums of every raw file are written to `staging/checksums.json`.

**Reading notes:**
- **PPP:** some files are not valid UTF-8, so all are read as Latin-1, and accented letters still become plain ASCII.
- **EPA:** a small number of malformed CSV lines are skipped. The file has 5,361,876 data lines including embedded line breaks, and 5,359,938 records were read, so at most about 1,900 (under 0.04%) were skipped.
- **OSHA:** every data line was read (5,202,096).

## 2. Turning raw fields into the task's five fields (`stage_sources.py`)

Applied to every source:

| Field | Rule |
|---|---|
| `name`, `addr`, `city` | accents removed (é → E), uppercase, non-ASCII characters dropped, whitespace collapsed to single spaces, trimmed. Placeholders become empty: `N/A`, `NA`, `NOT AVAILABLE`, `EXEMPTION 6` (PPP's privacy redaction), `NONE`, `UNKNOWN`, and EPA's `(UNKNOWN ...)` values. |
| `zip` | the first 5 digits; `00000` and anything without 5 leading digits become empty |
| `state` | a 2-letter code (PPP falls back to `ProjectState` if `BorrowerState` is empty); anything else becomes empty |

**No name or address standardization** is applied. `task.md` puts `stnd_compname` and `stnd_address` out of scope, so the real spelling differences between sources are kept on purpose. Punctuation is kept.

**Records kept for linkage:** non-empty `name` and `state`, since `state` is the required blocking field.

| Source | Raw | With name and state |
|---|---|---|
| PPP | 11,468,210 | 11,468,115 (95 redacted or missing names removed): **8,612,370 first-draw** and **2,855,745 second-draw** loans |
| IRS EO BMF | 1,975,975 | 1,974,631 |
| OSHA | 5,202,096 | 5,195,057 |
| EPA FRS | 5,359,938 | 5,342,266 |

## 3. Development workloads (101–108): PPP and the IRS register

**Samples** are deterministic: the pool's rows are ordered by `hash(id, seed)` and the first *n* kept. The seed is the workload number. **Registers** are kept whole for the stated states.

| # | Case | Master pool → master | Using (register) |
|---|---|---|---|
| 101 | A: small sample → large register | second-draw borrowers in Oregon: 31,213 → **2,000** | Oregon first-draw borrowers: **84,657** |
| 102 | B: panel linkage across rounds | second-draw borrowers in Vermont: **5,680** (all) | Vermont first-draw borrowers: **16,016** |
| 103 | C + H: cross-source, nonprofits | first-draw nonprofit borrowers (`NonProfit = Y`) in ME, NH, VT, MA, RI, CT: **12,033** (all) | IRS EO BMF organizations in those six states: **101,179** |
| 104 | D: dense big-city blocks | second-draw borrowers in Chicago: 42,141 → **500** | Illinois first-draw borrowers: **469,711** (Chicago alone: 145,108) |
| 105 | E: chains and franchises | second-draw borrowers in Tennessee with a `FranchiseName` or NAICS 72 (accommodation and food services): 5,248 → **2,000** | Tennessee first-draw borrowers: **180,402** |
| 106 | F: sparse fields | the 102 master with `addr` and `zip` emptied: **5,680** | as 102: **16,016** |
| 107 | G: person-named businesses | second-draw borrowers in Louisiana with `BusinessType` Sole Proprietorship, Self-Employed Individuals or Independent Contractors: 18,440 → **2,000** | Louisiana first-draw borrowers: **169,475** |
| 108 | I: national scale | second-draw borrowers nationwide: 2,855,745 → **5,000** | all first-draw borrowers: **8,612,370** |

**Why first-draw versus second-draw:** many businesses received both loans, often through different lenders, so their records were entered twice, independently. That's a natural, realistic linkage between two recordings of the same firms.

## 4. Held-out workloads (201–205): OSHA establishments → EPA FRS

**From OSHA inspections to establishments:**

| Step | Rule | Records |
|---|---|---|
| Inspections with name and state | | 5,195,057 |
| → private sector | `owner_type = A` | 3,953,314 |
| → inspected 2015–2025 | by `open_date`, to align with today's EPA register | 749,414 |
| → one row per establishment | distinct (name, addr, city, zip, state), keeping the latest inspection (its `activity_nr` is the `id`) | **694,568** |

| # | Case | Master pool → master | Using (register) |
|---|---|---|---|
| 201 | A + C: small sample → another agency's register | establishments in Minnesota: 17,465 → **2,000** | EPA FRS facilities in Minnesota: **249,328** |
| 202 | D: dense big-city blocks | establishments in Houston: 3,382 → **500** | EPA FRS facilities in Texas: **340,776** |
| 203 | E: multi-site firms | manufacturing establishments (NAICS 31–33) in Ohio: 5,142 → **2,000** | EPA FRS facilities in Ohio: **104,369** |
| 204 | F: sparse fields | establishments in Washington: 52,603 → **2,000**, with `addr` and `zip` emptied | EPA FRS facilities in Washington: **101,419** |
| 205 | I: national scale | all establishments: 694,568 → **5,000** | all EPA FRS facilities: **5,342,266** |

## 5. Candidate pairs (`candidates.py`)

The number of candidate pairs is what drives the time, so it's counted for every workload, **without running any linkage**:
- **Rule:** a candidate is a (master, using) pair in the same `state` that agrees exactly on at least one of `name`, `addr`, `city` or `zip`, with empty master values never blocking (task.md, step 2).
- **How it's counted:** pairs in the same city are counted from city sizes, then the distinct pairs agreeing on name, address or ZIP in a different city are added.
- **Exact matches,** which agree on all five fields and are removed before blocking, are counted separately.

| # | Master × using | Candidate pairs | Exact matches | Exact duplicates in the register | Largest register city |
|---|---|---|---|---|---|
| 101 | 2,000 × 84,657 | 16,506,014 | 555 | 45 | Portland, OR (23,445) |
| 102 | 5,680 × 16,016 | 1,885,532 | 1,024 | 5 | Burlington, VT (1,135) |
| 103 | 12,033 × 101,179 | 5,933,906 | 903 | 1,881 | Boston, MA (4,572) |
| 104 | 500 × 469,711 | 72,563,052 | 237 | 1,412 | Chicago, IL (145,108) |
| 105 | 2,000 × 180,402 | 21,066,664 | 616 | 463 | Memphis, TN (32,216) |
| 106 | 5,680 × 16,016 | 1,748,704 | 0 | 5 | Burlington, VT (1,135) |
| 107 | 2,000 × 169,475 | 13,981,623 | 1,206 | 467 | New Orleans, LA (20,620) |
| 108 | 5,000 × 8,612,370 | 68,820,815 | 1,744 | 15,585 | Chicago, IL (145,108) |
| 201 | 2,000 × 249,328 | 7,380,370 | 282 | 3,590 | Minneapolis, MN (10,149) |
| 202 | 500 × 340,776 | 14,241,883 | 22 | 2,916 | Houston, TX (28,477) |
| 203 | 2,000 × 104,369 | 2,025,126 | 429 | 408 | Cleveland, OH (4,836) |
| 204 | 2,000 × 101,419 | 4,116,266 | 0 | 5,534 | Seattle, WA (9,050) |
| 205 | 5,000 × 5,342,266 | 13,614,899 | 253 | 86,578 | Los Angeles, CA (38,519) |

**Exact duplicates in the register** (records identical on all five fields) are kept, because real registers contain them. The reference implementation keeps duplicate using records as separate rows.

## 6. Files written

`OUT/development/<id>_master.parquet`, `<id>_using.parquet` and `manifest_<id>.json`, and the same under `OUT/heldout/`. Each manifest records:
- the case and design;
- record counts, candidate pairs, exact matches and register duplicates;
- the largest register city;
- the share of non-empty values per field, for master and using;
- any fields emptied on purpose;
- the time estimates (section 7 of the README);
- the SHA-256 of both files and the build date.
