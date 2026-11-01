# Task 005: One-to-one and many-to-one table joins

## Computation

Merge two tables on one or more key variables, keeping every row and recording where each row came from. This is the computation behind

```stata
use master.dta, clear
merge 1:1 firm year using using.dta     // or: merge m:1 cusip using using.dta
```

which produces one row per matched key combination plus the unmatched rows of both tables, and a variable `_merge`:

| `_merge` | Meaning |
|---|---|
| 1 | in master only |
| 2 | in using only |
| 3 | matched (in both) |

In R this is `data.table::merge(x, y, by = keys, all = TRUE)`, and in Python `pandas.merge(x, y, on = keys, how = "outer", indicator = True)`. Merging is one of the most frequent steps in empirical work: linking firm financials to executive data, stock returns to accounting data, job postings or résumés to employers.

The computation has three steps:

1. **Check the keys.** For `1:1`, each key combination must be unique in both tables; for `m:1`, in the using table. Stata stops with an error otherwise; pandas does the same with `validate`; `data.table::merge` does not check.
2. **Match.** Find, for every key combination, the rows of each table that have it. Stata sorts both tables by the keys and matches them in one pass; R and pandas use sorting or hashing.
3. **Combine.** Build the output: matched rows get the variables of both tables; unmatched rows get missing values for the other table's variables.

**Where the time goes.** Sorting or hashing the keys, which costs more for string keys than for numbers, and copying every variable into the output. Stata's `merge` also reads the using table from disk inside the command; R and pandas merge two tables already in memory. The table therefore reports Stata's reading time separately.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| master | [n_master, n_keys + master_vars] | key columns (int64 or string) + float64 variables |
| using | [n_using, n_keys + using_vars] | key columns + float64 variables |
| keys | n_keys | column names |
| kind | – | "1:1" or "m:1" |

Variables have about 3% missing values. In `m:1`, the key is unique in the using table and repeated in the master table.

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| merged | [rows, n_keys + master_vars + using_vars + 1] | as inputs | the data in memory after `merge` |
| _merge | [rows] | int8 | `_merge` |

The row order is not part of the output; correctness is checked on order-independent summaries (below).

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(master, using, keys, kind) -> DataFrame with a _merge column (1, 2, 3)
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Key coverage overlaps only partly, so every workload has master-only, using-only and matched rows.

| # | design | merge | keys | key type | n_master | n_using |
|---|---|---|---|---|---|---|
| 1 | sanity check | 1:1 | `id` | numeric | 1,000 | 1,000 |
| 2 | firm-year 1:1 (Compustat + ExecuComp) | 1:1 | `firm year` | numeric | 200,000 | 44,000 |
| 3 | firm-year m:1 firm data, string key (CUSIP) | m:1 | `cusip` | string | 200,000 | 10,000 |
| 4 | stock-month m:1 firm-year (CRSP + Compustat) | m:1 | `permno year` | numeric | 3,000,000 | 200,000 |
| 5 | job postings m:1 employers, string key | m:1 | `employer` | string | 10,000,000 | 475,000 |
| 6 | stock-day m:1 stock-month (CRSP daily + monthly) | m:1 | `permno month` | numeric | 20,160,000 | 1,088,296 |

## Correctness

Correctness is checked on summaries: `checksums` in `reference.py` reports the number of output rows, the number of rows with each `_merge` value, and, for every non-key variable, the number of non-missing values and their sum. These do not depend on row order. An implementation must match the reference on every workload:

| Output | Tolerance |
|---|---|
| rows, `_merge` counts, non-missing counts | exact |
| column sums | max relative error ≤ 1e-10 |

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) uses `pandas.merge(..., how="outer", indicator=True, validate=...)`. It defines what is computed and is not a performance target.

## Not covered in this version

- Many-to-many merges (`merge m:m`, which Stata's own manual advises against; `joinby` forms all pairwise combinations).
- `merge` options `keep()`, `assert()`, `update` and `replace`, and Stata frames (`frlink`/`frget`).
- Fuzzy matching on names (e.g. `reclink`, `matchit`) and cleaning of string keys before merging.
