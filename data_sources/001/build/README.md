# Real-data workloads for task 001: build scripts

**Step-by-step lineage:** [`DATA_LINEAGE.md`](DATA_LINEAGE.md) traces every workload from the raw files, with each transformation step, the rows or units remaining after it, and the formula for every variable.

These scripts build the workloads described in the main [`README.md`](../README.md) from the official PatentsView and SEC downloads. Nothing in the task definition, reference, generator or evaluator is changed.

## Raw data

| Source | Files | How to get them |
|---|---|---|
| PatentsView, release 2026-04-10 (USPTO Open Data Portal, product PVGPATDIS) | `g_patent`, `g_application`, `g_inventor_disambiguated`, `g_assignee_disambiguated`, `g_location_disambiguated`, `g_cpc_current`, `g_us_patent_citation` (`.tsv.zip`) | Sign in at data.uspto.gov (free account with multi-factor authentication) and download from the PVGPATDIS page, or run `download_patentsview.sh` with an ODP API key. |
| SEC Insider Transactions Data Sets | `2006q1_form345.zip` … `2026q3_form345.zip` (83 files) | sec.gov/dera/data/form-345. Automated downloads must send a `User-Agent` header with a contact email. |

The staging scripts record SHA-256 checksums of every raw file in `staging/*/checksums.json`.

## Build

```bash
python stage_patentsview.py RAW_PATENTSVIEW_DIR OUT/staging/patentsview
python stage_sec.py RAW_SEC_DIR OUT/staging/sec
python build_patentsview.py OUT/staging/patentsview OUT
python build_sec.py OUT/staging/sec OUT
```

The full build takes about 10 minutes on an Apple M3 Pro with 18 GB of RAM. It uses DuckDB with an 11 GB memory limit and writes:

- `OUT/development/101.parquet` … `108.parquet`: real ID structures (fixed-effect and cluster codes; real event-time dummies for 108).
- `OUT/heldout/201.parquet` … `206.parquet`, `301.parquet` … `304.parquet`: real outcomes and controls.
- `manifest_<id>.json` next to each file: unit of observation, sample, absorbed columns, cluster column, row count, SHA-256 and build date.

Keep `heldout/` out of any workspace used for candidate development (see the design, section 6).

## Load

```python
from realdata import load
inputs = load(201, data_root=OUT)          # dict(y, X, fe_ids, cluster_ids), the task-001 inputs
inputs = load(104, data_root=OUT, n_regressors=50)
```

- **Development workloads (101–108):** synthetic `y` and `X` are added with the same construction as `workloads.py`, seeded with `1000 + id`.
- **Workload 109:** fully synthetic (`synthetic_extra.py`). Workload 109's parameters were fixed before any SEC file was read.

## Checks

- **`validate_features.py`:** recomputes `y` and all 20 controls of the inventor–patent workloads with plain pandas for a random sample of rows and reports the largest difference per variable. On 500 rows, every variable matched exactly, and technology breadth to within 2e-16.
- **`validate_workloads.py`:** loads every workload, checks values, computes descriptors (singletons, levels, clusters, mobility, connected components) and solves it with `reference.run`. Results go to `OUT/validation.jsonl` and `OUT/reference_outputs/<id>.npz`.

## Notes from building

- **Controls fully determined by the fixed effects were replaced.** Examples are firm or career age (year minus a first year, absorbed by unit and year fixed effects) and single-firm-by-year variables in the dyad designs (absorbed by `firm#year`). Each workload's controls vary within its fixed effects, so the regressor matrix is full rank.
- **Workload 302** uses `state#year` instead of `industry#year`, because SIC codes are not in the insider data.
- **DuckDB 1.5.6 returned wrong results for a correlated `EXISTS` subquery with inequality conditions** (905 vs 7,283 firms in a test). All correlated subqueries were rewritten as joins.
