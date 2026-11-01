# Real-data workloads for reclink (task 003)

This folder is the complete package for the real-data workloads for task 003: linking company records on name, street address, city, ZIP and state, as Stata's `reclink2` does. It contains the workload design, the data, a loader, the task's reference implementation and the build scripts.

The task itself (what is computed, inputs, outputs, tolerances) was written by the task author. It's reproduced unchanged in **Appendix A** and in `code/task.md`. **Appendix B** is the full data lineage.

## 1. Goal

Speed up `reclink2`-style record linkage and show reviewers that a faster method:

1. gives the **same matches and scores** as the reference, and
2. is faster on **the linkage problems management researchers actually face, with real company records**, not only on the five synthetic workloads in `code/workloads.py`.

## 2. What the computation is (short version)

```stata
reclink2 name addr city zip state using using.dta, idmaster(idm) idusing(idu) gen(score) ///
    wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)
```

For each record in the **master** file, find the best-matching record in the **using** file (the register):

1. **Exact matches:** records equal on all five fields get score 1.
2. **Candidates:** using records in the **same state** that agree **exactly on at least one** of name, address, city or ZIP.
3. **Scoring:** each candidate gets a score from bigram similarities of the five fields, weighted 10 (name), 8 (address), 5, 5, 5.
4. **Best match:** keep the best candidate if its score is at least 0.6.

The company name carries the most weight, but the address and location must agree too. Name-only matching is task 007.

**Where the time goes:**
- **Stata `reclink2`** rescans the whole register for every master record, so its time grows with master size × register size.
- **Faster implementations** score all candidate pairs together, so their time grows with the **number of candidate pairs**.

In real data, the candidate pairs are dominated by **big cities**: every master record in Chicago is a candidate for every register record in Chicago.

## 3. The groups of workloads

Think of it like studying for an exam.

| Group | Like… | Data | When to use it |
|---|---|---|---|
| **Synthetic 1–5** (`code/workloads.py`) | worked examples | made-up companies in five sizes | any time |
| **100s: development** | practice problems | **PPP loan borrowers** (second-draw loans linked to first-draw loans, the same firms recorded twice by different lenders), plus the **IRS nonprofit register** for 103 | any time, while building and improving the method |
| **200s: held out** | the final exam, in a different subject | **OSHA-inspected establishments → EPA's Facility Registry Service**: two other agencies, different entities (workplaces and regulated facilities) | only on a frozen method |

**Rules:**
- **Develop and tune only** on the 100s and synthetic 1–5.
- **The 200s are for final evaluation only.** Don't use them for tuning, debugging or choosing settings, and don't look at their results while developing.
- **Freeze the method first:** make a dated git commit (or a dated copy of the code and settings) before running the 200s. Keep to two or three rounds, and report every round, including failures.
- **Disclosure:** the held-out data are in this folder, so the evidence rests on this protocol. The paper will state that the held-out workloads were available to the developers but, by protocol, used only to evaluate frozen versions.

## 4. Common linkage situations in management research, and where each is covered

| Case | Typical example | Development | Held out |
|---|---|---|---|
| A. Small sample → large register | survey respondents' employers, hand-collected firms | 101 | 201 |
| B. Panel linkage across rounds or years | the same firms recorded again later | 102 | – |
| C. Cross-source linkage | records of one agency or vendor linked to another's register | 103 | 201–205 (OSHA → EPA) |
| D. Dense big-city blocks | firms in Chicago or Houston | 104 | 202 |
| E. Chains, franchises, multi-site firms | restaurants, retail, manufacturers with many sites | 105 | 203 |
| F. Sparse fields | startup or VC data with name, city and state only | 106 | 204 |
| G. Person-named businesses | sole proprietors ("JOHN SMITH LLC") | 107 | – |
| H. Nonprofits | grantees or loan recipients → the IRS register | 103 | – |
| I. National scale | linking across the whole country | 108 | 205 |

## 5. Workloads

All 13 workloads are built and checked. **No linkage method has been run:** times are **estimates** (section 7), and candidate pairs were **counted with joins**, not scored.

### Development: PPP loans (and the IRS register for 103)

| # | Case | Master | Using (register) | Master × using | Candidate pairs | Exact matches | Stata `reclink2` | R port | Python reference |
|---|---|---|---|---|---|---|---|---|---|
| 101 | A | 2,000 second-draw borrowers, Oregon | Oregon first-draw borrowers | 2,000 × 84,657 | 16.5M | 555 | ~3–6 min | ~1–2 min | ~16–26 s |
| 102 | B | all second-draw borrowers, Vermont | Vermont first-draw borrowers | 5,680 × 16,016 | 1.9M | 1,024 | ~30–45 s | ~9–15 s | ~2–3 s |
| 103 | C + H | first-draw nonprofit borrowers, New England | IRS nonprofit register, New England | 12,033 × 101,179 | 5.9M | 903 | ~3–4 min | ~30–50 s | ~6–10 s |
| **104** | **D** | **500 second-draw borrowers, Chicago** | **Illinois first-draw borrowers** | 500 × 469,711 | **72.6M** | 237 | ~12–25 min | ~6–10 min | ~1–2 min |
| 105 | E | 2,000 franchise or food/lodging borrowers, Tennessee | Tennessee first-draw borrowers | 2,000 × 180,402 | 21.1M | 616 | ~4–8 min | ~2–3 min | ~21–34 s |
| 106 | F | as 102, street and ZIP emptied | as 102 | 5,680 × 16,016 | 1.7M | 0 | ~25–45 s | ~9–14 s | ~2–3 s |
| 107 | G | 2,000 sole proprietors and self-employed, Louisiana | Louisiana first-draw borrowers | 2,000 × 169,475 | 14.0M | 1,206 | ~3–5 min | ~1–2 min | ~14–22 s |
| **108** | **I** | **5,000 second-draw borrowers, nationwide** | **all first-draw borrowers** | 5,000 × **8,612,370** | **68.8M** | 1,744 | **~1.3–1.5 h** | ~6–9 min | ~1–2 min |

### Held out: OSHA establishments → EPA Facility Registry Service

| # | Case | Master | Using (register) | Master × using | Candidate pairs | Exact matches | Stata `reclink2` | R port | Python reference |
|---|---|---|---|---|---|---|---|---|---|
| 201 | A + C | 2,000 establishments, Minnesota | EPA facilities, Minnesota | 2,000 × 249,328 | 7.4M | 282 | ~2–3 min | ~40–60 s | ~7–12 s |
| 202 | D | 500 establishments, Houston | EPA facilities, Texas | 500 × 340,776 | 14.2M | 22 | ~3–5 min | ~1–2 min | ~14–23 s |
| 203 | E | 2,000 manufacturing establishments, Ohio | EPA facilities, Ohio | 2,000 × 104,369 | 2.0M | 429 | ~40–60 s | ~10–16 s | ~2–3 s |
| 204 | F | 2,000 establishments, Washington, street and ZIP emptied | EPA facilities, Washington | 2,000 × 101,419 | 4.1M | 0 | ~1–2 min | ~20–35 s | ~4–7 s |
| **205** | **I** | **5,000 establishments, nationwide** | **all EPA facilities** | 5,000 × **5,342,266** | 13.6M | 253 | **~45 min** | ~1–2 min | ~14–22 s |

**What the estimates suggest:**
- **Stata `reclink2` is slowest when the register is large** (108, 205: about 45 minutes to 1.5 hours), because it rescans the register for every master record.
- **Faster implementations are slowest when candidate sets are large,** in dense cities (104) and at national scale (108).
- **Real data are much denser than the synthetic workloads.** Just 500 Chicago records create 72.6M candidate pairs, more than synthetic workload 5's 42.8M from 100,000 records.

## 6. What real data adds

The synthetic workloads miss several features of real company records:
- **Huge city blocks:** Chicago has 145,108 first-draw borrowers in one city, versus about 300 firms per city in the synthetic data.
- **Real spelling variation between sources:** different lenders and agencies abbreviate, punctuate and format names and addresses differently ("INC" vs "INC.", "STREET" vs "ST", suite numbers).
- **Duplicates in registers:** up to 86,578 identical records in the national EPA register.
- **Few exact matches across agencies:** for example, 22 of 500 in Houston (202).
- **Person-named businesses and franchises,** with many near-identical candidates.

## 7. How the times were estimated

The times are scaled from the measured synthetic workloads 1–5 (Stata/MP 19.5 with 12 cores, R 4.6 and Python on an Apple M3 Pro; see the study report in the shared repository). **No method was run on the real workloads.**

| Tool | Rule fitted to synthetic workloads 1–5 |
|---|---|
| Stata `reclink2` | about 0.09 µs × master × register (the per-record rescan) **plus** 10–20 µs per candidate pair |
| R port (`data.table` + `stringi`) | about 5–8 µs per candidate pair |
| Python reference | about 1.0–1.6 µs per candidate pair |

Real strings are longer and more varied than synthetic ones, so expect the estimates to be off by up to 2–3×. Measured runs should replace them.

## 8. Data sources

| Source | What | Access |
|---|---|---|
| **SBA Paycheck Protection Program loans** (FOIA release 2024-09-30) | 11.5M loans with borrower name, address, city, state, ZIP, first or second draw, business type, industry, franchise name, nonprofit flag | free, data.sba.gov |
| **IRS Exempt Organizations Business Master File** (03/10/2026) | 1.98M tax-exempt organizations with name and address | free, irs.gov |
| **OSHA inspections** (Department of Labor, 2026-10-21) | 5.2M inspections with establishment name and site address | free, dataportal.dol.gov (bulk download, no key) |
| **EPA Facility Registry Service** (national single file, 2026-10-12) | 5.36M regulated facilities with name and location address | free, epa.gov/frs |

**Raw data aren't in this folder.** They're about 7.2 GB, archived by the package author in `Documents/Data/{PPP, IRS EO BMF, OSHA, EPA FRS}`, with SHA-256 checksums in the build's `staging/checksums.json`. These sources are updated over time, so a fresh download may differ slightly.

## 9. How the workloads were built (summary)

- **Fields are cleaned only to the task's input format:** uppercase ASCII, accents removed, whitespace collapsed, placeholders (`N/A`, PPP's `EXEMPTION 6` redaction, EPA's `(UNKNOWN ...)`) emptied, 5-digit ZIP, 2-letter state. **No name or address standardization**, which `task.md` puts out of scope, so real spelling differences remain.
- **Samples are deterministic** (ordered by a hash of the record ID and the workload number). Registers are complete for the stated states.
- **PPP:** first-draw and second-draw loans are separate records of the same businesses, often entered by different lenders, which makes a realistic linkage.
- **OSHA:** private-sector inspections opened 2015–2025, reduced to one row per establishment (694,568 establishments).
- **Emptied fields:** in 106 and 204, the master's street and ZIP are emptied on purpose, to represent sources with name, city and state only.

### Data lineage: summary

| # | From raw data to the workload |
|---|---|
| PPP base | 11,468,210 loans → 11,468,115 with name and state → **8,612,370 first-draw** (registers) and **2,855,745 second-draw** (master pools) |
| 101 | Oregon second-draw 31,213 → sample **2,000** · register: Oregon first-draw **84,657** |
| 102 | Vermont second-draw **5,680** (all) · register: Vermont first-draw **16,016** |
| 103 | New England first-draw nonprofits **12,033** (all) · register: IRS organizations in New England **101,179** (from 1,975,975) |
| 104 | Chicago second-draw 42,141 → sample **500** · register: Illinois first-draw **469,711** |
| 105 | Tennessee second-draw franchise or NAICS 72: 5,248 → sample **2,000** · register: Tennessee first-draw **180,402** |
| 106 | the 102 master with street and ZIP emptied · register as 102 |
| 107 | Louisiana second-draw sole proprietors, self-employed and independent contractors: 18,440 → sample **2,000** · register: Louisiana first-draw **169,475** |
| 108 | all second-draw 2,855,745 → sample **5,000** · register: all first-draw **8,612,370** |
| OSHA base | 5,202,096 inspections → 5,195,057 with name and state → 3,953,314 private → 749,414 opened 2015–2025 → **694,568 establishments** |
| EPA base | 5,359,938 facilities → **5,342,266** with name and state |
| 201 | Minnesota establishments 17,465 → **2,000** · register: Minnesota facilities **249,328** |
| 202 | Houston establishments 3,382 → **500** · register: Texas facilities **340,776** |
| 203 | Ohio manufacturing establishments 5,142 → **2,000** · register: Ohio facilities **104,369** |
| 204 | Washington establishments 52,603 → **2,000**, street and ZIP emptied · register: Washington facilities **101,419** |
| 205 | all establishments 694,568 → **5,000** · register: all facilities **5,342,266** |

**Known limitations:**
- **No true links are known.** Correctness is defined as matching `reference.py`, so ground truth isn't needed, but match quality can't be reported.
- **About 1,900 malformed lines in EPA's national file** (under 0.04%) were skipped.
- **The held-out set doesn't cover B (panels), G (person names) or H (nonprofits).** Those cases appear only in development.
- **OSHA–EPA overlap is limited to industries both agencies cover,** which is why the held-out exact-match shares are low. That's realistic for cross-agency linkage.

## 10. Common features this task doesn't cover yet

From `task.md` ("Not covered in this version"):
- the `manytoone` and `npairs()` options, `exactstr()`, `exclude()`, and hand-set `orblock()`;
- preprocessing (`stnd_compname`, `stnd_address`) and clerical review;
- other linkage tools with other scoring rules (`dtalink`, `reclin2`, `fastLink`, `recordlinkage`, Splink).

Name and address standardization is common in practice. It's a natural candidate for a task variant: the same workloads, with standardized fields.

## 11. The folder

```
reclink_real_data_package/
├── README.md                this guide (task.md is Appendix A; the data lineage is Appendix B)
├── requirements.txt
├── code/
│   ├── task.md              task definition (by the task author, unchanged)
│   ├── task.json            machine-readable contract (by the task author, unchanged)
│   ├── reference.py         reference implementation: run(master, using, vars, wmatch, wnomatch, required, ...)
│   ├── workloads.py         synthetic workloads 1-5 and the task's constants (by the task author, unchanged)
│   ├── realdata.py          loader: load(id, data_root) -> reference.run() inputs
│   └── example.py           load a workload; with --run, also run the reference
├── data/
│   ├── development/         101-108: <id>_master.parquet, <id>_using.parquet, manifest_<id>.json
│   └── heldout/             201-205: same layout. FINAL EVALUATION ONLY
└── build/                   scripts that built everything from the raw downloads, and DATA_LINEAGE.md
```

**Getting started:**

```bash
cd reclink_real_data_package
python -m pip install -r requirements.txt
python code/example.py 102            # load and describe a workload
python code/example.py 102 --run      # also run the reference (~1.9M candidate pairs)
```

```python
import sys; sys.path.insert(0, "code")
from realdata import load
import reference
d = load(104, data_root="data")      # dict(master, using, vars, wmatch, wnomatch, required, minscore, minbigram)
i, j, score, n_perfect = reference.run(**d)
```

A faster method must accept the same inputs as `reference.run` and return the same pairs and scores, within the tolerances in Appendix A.

## 12. Next steps

1. Measure Stata `reclink2`, the R port and the reference on the development workloads, replacing the estimates.
2. Develop the faster method on 101–108 and synthetic 1–5.
3. Freeze it, then run 201–205 and report all results.
4. Consider a task variant with name and address standardization.

---

## Appendix A: task definition (`task.md`, written by the task author, reproduced unchanged)

# Task 003: Record linkage using names and addresses

## Computation

Record linkage of firms on several fields at once (name, street address, city, ZIP code, state) when the two files share no identifier. For every record in the master file, find the record in the using file that agrees best across the fields. This is the computation behind

```stata
use master.dta, clear        // idm name addr city zip state
reclink2 name addr city zip state using using.dta, idmaster(idm) idusing(idu) gen(score) ///
    wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)
```

with the other options at their defaults (`minscore(.6)`, `minbigram(.6)`, `npairs(1)`, no `manytoone`). The weights are the ones in the example of `help reclink2`. `reclink2` (Wasi & Flaaen 2015) extends Blasnik's `reclink`; without `manytoone` and `npairs()` both compute the same thing (see Correctness for the one difference). Typical uses are linking survey respondents' employers, startups, loan recipients or patent assignees to a business register when names alone are ambiguous, so the address and location must also agree. Task 007 is the name-only version of this problem.

What `reclink2` computes, from its source code:

1. **Perfect matches.** Master records equal to a using record on all five fields get score 1. Both records are removed from the rest of the search.
2. **Candidates (blocking).** For each remaining master record, the candidates are the using records that have the same `state` (`required()`) **and** the same value of at least one of `name`, `addr`, `city`, `zip` (or-blocking, the default when four or more fields are given). Empty master values are skipped. In practice most candidates are the firms in the same city.
3. **Score of each candidate.** For each field, a bigram similarity bi between 0 and 1 (below), turned into a match weight M and a non-match weight NM:
   - M += wmatch × (½ bi² + [bi > .6] × 0.679 × (bi − .6)^(1/3)), where 0.679 = .5 / .4^(1/3)
   - NM += wnomatch × (1 if bi < .4; 1 − bi² if bi < .6; (1 − bi)² otherwise), only if the using value is not empty
   - NM += .3 × wnomatch if exactly one of the two values is empty
   - `required()` fields add wmatch to M.

   The score is M ÷ (M + NM).
4. **Best candidate.** Keep each master record's highest-scoring candidate if its score is at least `minscore` (.6). All tied candidates are kept.

The bigram similarity of master value m and using value s (`reclink`'s `bigram1`):

- (number of the len(m) − 1 two-character pieces of m that occur anywhere in s) × 2 ÷ (len(m) + len(s) − 2);
- strings of fewer than 3 characters score min(len) ÷ (len(m) + len(s) − 1) if one contains the other, else 0;
- then a bonus for a common start: score + k × (1 − score) ÷ 10, where k counts the i in 1–4 with the same first i characters (as in the Jaro–Winkler comparator);
- equal strings score 1, an empty using value 0.

Example: m = "ACME INC", s = "ACME CORP". Four of m's seven pieces (AC, CM, ME, "E ") occur in s, so 4 × 2 ÷ (8 + 9 − 2) = 0.533. The first four characters agree (k = 4), so the score is 0.533 + 4 × 0.467 ÷ 10 = 0.720.

`reclink2` stores bi, M, NM and the score as Stata `float`s (single precision), so the reference rounds every intermediate result to single precision too.

**Where the time goes.** `reclink2` is an ado-file loop over master records. For each record it (1) loads that record from a temporary file (`use ... in i`), (2) loads the **whole** using file with an `if` condition that selects the candidates (`use ... if (name == "..." | addr == "..." | ...) & state == "..."`), and (3) computes each bigram score with one `replace` command per two-character piece of the master value. Step 2 reads all n_using records for every master record, so the time grows with n_master × n_using even though only a few hundred candidates are scored. The R port and the reference instead find all candidates at once with joins (`data.table`, pandas) and score all candidate pairs together.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| master: name, addr, city, zip, state | [n_master] each | string (uppercase ASCII; "" = missing) |
| using: name, addr, city, zip, state | [n_using] each | string; standard generated workloads are unique, development fixtures also exercise duplicates |
| wmatch, wnomatch | [5] | float64 (10 8 5 5 5; 2 3 4 4 4) |
| required | list | `state` |
| minscore, minbigram | scalar | float64 (.6, .6) |

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| i | [n_pairs] | int64 | `idm` of master records with a match |
| j | [n_pairs] | int64 | `idu` |
| score | [n_pairs] | float32-rounded values | `score` (1 for perfect matches) |
| n_perfect | scalar | integer | number of perfect pairs, including duplicate-record combinations |

Pair indices are zero-based. Contract version 2 compares the complete pair set after sorting, every corresponding score and the perfect-pair count. Row order is irrelevant. Equal using records retain distinct row identities.

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6)
    -> (i, j, score, n_perfect)

# master, using: dicts of equal-length string arrays keyed by variable name
# vars = ["name", "addr", "city", "zip", "state"], wmatch = [10, 8, 5, 5, 5],
# wnomatch = [2, 3, 4, 4, 4], required = ["state"]  (constants in workloads.py)
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. The using file is a register of unique firms: a name of one to three made-up words plus a legal suffix, a street address, and a city, ZIP code and state. About 300 firms share a city on average, with lognormal city sizes; larger states have more cities. The master file holds records of some of these firms as another source would record them, plus firms that are not in the register:

- 25% are exact copies (perfect matches);
- the other 75% can differ in several fields: the name is changed with probability .7 (another suffix, a typo, a dropped word, punctuation, swapped letters); the address is missing (8%) or altered (52%: a spelled-out street type, a house-number typo, no street type, or an added suite); the city has a typo (5%); the ZIP code is missing (10%) or different (5%).

| # | design | n_master | n_using | cities | true links | candidate pairs scored |
|---|---|---|---|---|---|---|
| 1 | sanity check | 500 | 2,000 | 6 | 300 | 149,610 |
| 2 | survey employers to business register | 2,000 | 20,000 | 66 | 1,200 | 691,342 |
| 3 | startups to state business registrations | 10,000 | 100,000 | 333 | 5,000 | 4,592,859 |
| 4 | loan recipients to establishment database | 50,000 | 500,000 | 1,666 | 35,000 | 22,719,376 |
| 5 | establishments to establishment database (national) | 100,000 | 2,000,000 | 6,666 | 70,000 | 42,778,502 |

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| complete set of (i, j) pairs, with no duplicate pairs; n_perfect | exact |
| each corresponding score | relative error ≤ 1e-6 |

Notes for implementers:

- **Single precision.** `reclink2` stores bigram scores, M, NM and the final score as `float`. The reference and the R port round each intermediate result to single precision and reproduce `reclink2`'s scores exactly. This matters more than rounding usually does. A bigram score of exactly .6 (e.g. 3 × 2 ÷ 10) is stored as 0.60000002, which is above `minbigram`, so it earns the cube-root bonus 0.679 × (0.00000002)^(1/3) ≈ 0.002. In double precision it would earn nothing. On workload 3, a double-precision version keeps the same pairs, but individual scores differ by up to 3e-4 and the score sum by 2e-6, which fails the tolerance above. Ties could also break differently.
- **`reclink` vs `reclink2`.** `reclink` keeps a best candidate if its score is **>** `minscore`; `reclink2` if it is **≥** `minscore`. This never matters with the default .6, because a single-precision score cannot equal the double-precision .6 exactly (the nearest float is 0.60000002). So both commands return identical results here.
- **Or-blocking only with four or more variables.** With three or fewer variables and no `orblock()`, every using record that agrees on `required()` is a candidate, so the number of candidates (and the time) grows much faster.

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy/pandas implementation that finds candidates with joins and scores all candidate pairs together. It defines what is computed and is not a performance target.

## Not covered in this version

- The `manytoone` and `npairs()` options of `reclink2`, `exactstr()`, `exclude()` and `orblock()` set by hand.
- Preprocessing (`stnd_compname`, `stnd_address`) and clerical review (`clrevmatch`).
- Other linkage tools with other scoring rules (`dtalink`, R `reclin2` and `fastLink`, Python `recordlinkage`, Splink).

## Shared evaluator (contract version 2)

`task.json` identifies the prompt files, generator, validator and tolerances. Inputs are read-only. Outputs must have the documented shapes and numeric types and contain only finite values. Relative errors use `abs(actual - reference) / max(abs(reference), 1e-12)` element by element, including near zero.

Run `python -m harness evaluate --task 003 --solution solutions/003/candidate/solution.json --smoke` from the repository root for public development fixtures. Omit `--smoke` to validate those fixtures and time official workload 1; use `--ids` or `--all-workloads` for wider coverage. Every first-call, warmup and measured output must pass. The score uses the ratio of reference and candidate median call times on the same host. Generation, imports, first-call setup, validation and serialization are excluded; input-dependent preprocessing and conversion to host-ready outputs are included. See `harness/README.md` for the full protocol. Historical baseline results predate this evaluator and are not automatically certified by it.

---

## Appendix B: full data lineage (`build/DATA_LINEAGE.md`)

### Data lineage: from raw downloads to workloads (task 003, reclink)

This document traces every workload back to the raw files: which tables and columns are used, each transformation step, how many records remain after it, and how the five linkage fields are formed. The authoritative code is in this folder: `stage_sources.py`, `build_workloads.py` and `candidates.py`. Counts come from read-only queries on the staged tables. No linkage method was run.

Every workload is a **master** file (records to link) and a **using** file (the register). Both have the task's five fields, `name`, `addr`, `city`, `zip`, `state`, plus the source's record identifier in `id`.

#### 1. Raw files

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

#### 2. Turning raw fields into the task's five fields (`stage_sources.py`)

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

#### 3. Development workloads (101–108): PPP and the IRS register

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

#### 4. Held-out workloads (201–205): OSHA establishments → EPA FRS

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

#### 5. Candidate pairs (`candidates.py`)

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

#### 6. Files written

`OUT/development/<id>_master.parquet`, `<id>_using.parquet` and `manifest_<id>.json`, and the same under `OUT/heldout/`. Each manifest records:
- the case and design;
- record counts, candidate pairs, exact matches and register duplicates;
- the largest register city;
- the share of non-empty values per field, for master and using;
- any fields emptied on purpose;
- the time estimates (section 7 of the README);
- the SHA-256 of both files and the build date.
