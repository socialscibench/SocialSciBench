# Build scripts for the task-002 real-data workloads

**Step-by-step lineage:** [`DATA_LINEAGE.md`](DATA_LINEAGE.md) traces every workload from the raw files, with each transformation step, the rows or units remaining after it, and the formula for every variable.

These scripts rebuild every workload in `../data/` from the official raw downloads. See the main `README.md`, sections 7–8, for sources and definitions.

## Order

```bash
# 1. PatentsView and SEC base tables, built by the task-001 (reghdfe) package:
#    reghdfe_real_data_package/build/stage_patentsview.py, stage_sec.py, build_patentsview.py
#    -> 001 work database (_work_patentsview.duckdb) and staged SEC tables

# 2. QCEW: keep county-level, private-sector rows
python stage_qcew.py RAW_QCEW_DIR OUT/staging/qcew

# 3. Workloads
python build_patentsview.py PATH/TO/_work_patentsview.duckdb PATH/TO/g_cpc_current.parquet OUT   # 101-108, 201-203
python build_sec.py PATH/TO/001/staging/sec OUT                                                   # 301
python build_qcew.py OUT/staging/qcew RAW_MINWAGE_DIR OUT                                         # 401-403

# 4. Checks (data only; add no flag to also solve each workload with the reference)
python validate_workloads.py OUT ../code --no-reference
```

`RAW_MINWAGE_DIR` must contain the extracted Vaghul–Zipperer v1.4.0 state files (`mw_state_excel/`) and the saved U.S. Department of Labor history page (`dol_state_minimum_wage_history.html`).

## Outputs

- `OUT/development/<id>.parquet` and `OUT/heldout/<id>.parquet`: one row per unit, with columns `unit`, `gvar`, the covariates, and outcomes `y001` … `yT`.
- `manifest_<id>.json` next to each file: unit of observation, periods, treatment and outcome definitions, sample rule, covariates, units, cohorts, share treated, smallest cohort, cells, always-treated units dropped, source, SHA-256 and build date.

## Notes

- **Workload 106** has no file of its own: the loader reads 101 and drops the covariates.
- **The build uses DuckDB** with an 11 GB memory limit. On an Apple M3 Pro, the full build takes a few minutes once the task-001 base tables exist.
- **Correlated `EXISTS` subqueries are avoided.** DuckDB 1.5.6 returned wrong results for one such query during the task-001 build, so the CEO-turnover events use a join.
