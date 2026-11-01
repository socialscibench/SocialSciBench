# Build scripts for the task-003 real-data workloads

**Step-by-step lineage:** [`DATA_LINEAGE.md`](DATA_LINEAGE.md) traces every workload from the raw files, with each step, the records remaining after it, and how the five linkage fields are formed.

These scripts rebuild every workload in `../data/` from the official raw downloads. No linkage method is run during the build.

## Raw data (download into one folder, `RAW_ROOT`)

| Folder | Files | Where |
|---|---|---|
| `PPP/` | `public_150k_plus_240930.csv`, `public_up_to_150k_1…12_240930.csv`, `ppp-data-dictionary.xlsx` | data.sba.gov/dataset/ppp-foia |
| `IRS EO BMF/` | `eo1.csv`, `eo2.csv`, `eo3.csv`, `eo4.csv`, `eo-info.pdf` | irs.gov, "Exempt Organizations Business Master File Extract (EO BMF)" |
| `OSHA/` | `OSHA_inspection.zip` | dataportal.dol.gov/datasets/10334, "Download Complete Dataset" (no API key needed) |
| `EPA FRS/` | `national_single.zip` | epa.gov/frs, "FRS Facilities State Single File CSV Download" |

About 7.2 GB in total. Scripted downloads should send a `User-Agent` header with a contact email.

**The sources are updated over time,** for example the IRS register monthly, OSHA daily and EPA monthly. A fresh download may give slightly different workloads. `staging/checksums.json` records the exact files used.

## Order

```bash
python stage_sources.py RAW_ROOT OUT/staging            # ppp, bmf, osha, frs -> Parquet with the five fields
python build_workloads.py OUT/staging OUT               # 101-108 (development), 201-205 (held out)
```

The full build takes a few minutes on an Apple M3 Pro with 18 GB of RAM, using DuckDB with a 10 GB memory limit.

## Files

| File | What it does |
|---|---|
| `stage_sources.py` | reads each raw source, forms `name`, `addr`, `city`, `zip`, `state` in the task's format (uppercase ASCII, placeholders emptied, 5-digit ZIP, 2-letter state), and keeps the attributes used for slicing |
| `build_workloads.py` | defines each workload (the master pool and sample, the register and any emptied fields) and writes the master and using Parquet files with a manifest |
| `candidates.py` | counts the candidate pairs `reclink2` would score, using joins, without scoring anything |
| `DATA_LINEAGE.md` | the step-by-step lineage with counts |
