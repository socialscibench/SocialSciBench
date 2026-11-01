# Task 010: Preparing data larger than memory under a memory budget

## Computation

Prepare analysis datasets from yearly Stata files whose total size exceeds the allowed memory. The data imitate commercial workforce position data (résumé-based job positions): seven yearly `.dta` files, 24 variables.

Each workload specifies a memory budget **M**. A solution must produce the required output and stay within the measured peak-memory budget.

Three recipes follow Stata's semantics:

1. **extract:** retain US positions with `startdate >= td(01jan2015)`, keeping missing start
   dates because Stata numeric missing values exceed every ordinary date. Missing start dates
   occur in about 16% of the input; a naive SQL/pandas comparison would exclude those qualifying
   US rows. Derive `year = year(startdate)`, missing when the date is missing. Encode distinct
   nonempty role/state strings as 1, 2, … in byte order, with value labels; empty strings become
   missing. A Stata formulation is:

   ```stata
   * Read each year, append the filtered years, then derive/categorize:
   use user_id rcid startdate enddate seniority salary role_k1500_v2 state country ///
       if country == "United States" & startdate >= td(01jan2015) using file, clear
   gen int year = year(startdate)
   encode role_k1500_v2, gen(role) label(role)
   encode state, gen(state_code) label(state)
   ```

2. **aggregate:** use the same filtered rows, grouped by `rcid` and `quarter = qofd(startdate)`.
   Return position count, nonmissing salary count, mean of nonmissing salaries, and count of
   **nonmissing** seniority ≥4 (`seniority >= 4 & seniority < .`). Mean salary is missing if
   its count is zero. Missing quarter is a separate group; sort by `rcid`, then quarter,
   with missing quarter last.
3. **append:** stack all files and all 24 variables, following Stata's type-promotion rules.
   Unlike aggregation, this recipe retains an output approximately as large as its input.

## Inputs and outputs

| Name | Shape | Dtype |
|---|---|---|
| yearly files | 7 `.dta` files (format 118), N_y rows each | 24 variables: 9 double, 2 long, 3 int, 1 byte, 9 str# (str10–str55); 457 bytes per row |
| memory budget M | scalar | GiB |

| Recipe | Output | Variables |
|---|---|---|
| extract | `.dta`, one row per kept position | `user_id rcid startdate enddate year seniority salary role state` (`role`, `state` with value labels) |
| aggregate | `.dta`, one row per firm-quarter | `rcid quarter n_positions n_salary mean_salary n_senior` |
| append | `.dta`, all rows of all files | the 24 input variables |

Row order is irrelevant for extract and append; aggregate must be sorted as specified.
The supplied reader (`dta.py`) supports little-endian `.dta` format 118 and fixed-width strings.
Padded strings account for 81% of the 457 bytes per row.

## Interface

Provide three functions with the same signatures as those in `reference.py`; each reads the yearly files in `paths` and writes its output `.dta` file to `out`, using `workdir` for any temporary files:

```python
extract(paths, out, workdir, chunk_rows=100_000)
aggregate(paths, out, workdir=None, chunk_rows=100_000)
append(paths, out, workdir=None, chunk_rows=100_000)
```

## Workloads

`generate.py` (in this folder) deterministically writes seven synthetic yearly files of a given total size (`python generate.py --gb 2 --out data/`), with the real schema, firm-grouped rows, and the patterns of missingness, cardinality and string length measured on a 300,000-row sample of the real data. Names are made up except the US country and state names. The development tier supports quick iteration; the standard tier is scored.

| # | tier | data | memory budget | data ÷ budget | rows | files |
|---|---|---|---|---|---|---|
| 1 | development | 0.5 GB | 1 GiB | 0.5 | 1,094,089 | 7 |
| 2 | development | 2 GB | 1 GiB | 2 | 4,376,365 | 7 |
| 3 | development | 8 GB | 1 GiB | 8 | 17,505,467 | 7 |
| 4 | standard | 2 GB | 4 GiB | 0.5 | 4,376,365 | 7 |
| 5 | standard | 8 GB | 4 GiB | 2 | 17,505,467 | 7 |
| 6 | standard | 32 GB | 4 GiB | 8 | 70,021,878 | 7 |

Input sizes are decimal GB (`1e9` bytes); budgets are binary GiB (`2**30` bytes).

## Correctness

A solution must match the reference's order-independent checksums on every workload, as computed by `checksums.py` (in this folder):

| Checksum | Tolerance |
|---|---|
| number of rows; per variable, number of non-missing values; string lengths; label text totals | exact |
| per variable, sum of values | relative error ≤ 1e-10 |

The checksums are per-variable nonmissing counts, numeric sums, string byte-length totals, and label-text byte-length totals across rows. Aggregate sorting is required but not checked by these summaries.

- Numeric missing values must be retained by the extract date filter, and empty strings must
  become missing codes; use byte order for `encode`, not a locale-dependent order.
- Read fixed-width strings only through the first NUL.

## Scoring

A run is valid only if it passes the correctness checks and the process's peak memory stays within the workload's budget. Peak memory is the OS-reported peak footprint of the process (not the system's file cache). Valid runs are scored by wall-clock time of each recipe, including writing the output and excluding process startup, relative to `reference.py` on the same machine.

## Reference implementation

`reference.py` (in this folder) is a readable streaming implementation using NumPy and `dta.py` (streaming format-118 reading and writing). It defines what is computed and is not a performance target. It retains one chunk of rows plus category and group tables, so its memory can grow with the number of distinct values or firm-quarter cells.

## Not covered in this version

strL; large merges, reshape, deduplication or rolling windows; regressions.
