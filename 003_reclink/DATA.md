# Task 003 real workloads

The record-linkage estimator, reference implementation and synthetic generators are those of the
supplied real-data package (`data_sources/003/`).

| Suite | IDs | Permitted use |
|---|---|---|
| `synthetic` | 1–5 | Development and measurement |
| `real-development` | 101–108 | PPP/IRS development inputs |
| `real-heldout` | 201–205 | Frozen final evaluation on OSHA/EPA inputs |

> **Not evaluated in this submission:** the `real-heldout` workloads are defined for a future frozen
> final evaluation and were not run. Every result reported with this repository uses the development
> and synthetic workloads.

## Development coverage

Start with 102, then cover cross-source, dense-block, sparse-field and national-scale inputs.
These are input descriptions; no expected pair counts, scores or timings are provided.
A staged workspace contains only the inputs selected in its `stage.json`.

| ID | Master rows | Using rows | Design and purpose |
|---|---:|---:|---|
| 101 | 2,000 | 84,657 | Oregon PPP second-draw → first-draw borrowers; small sample to register |
| 102 | 5,680 | 16,016 | Vermont PPP second-draw → first-draw borrowers; initial cross-round check |
| 103 | 12,033 | 101,179 | New England PPP nonprofits → IRS register; cross-source spelling variation |
| 104 | 500 | 469,711 | Chicago PPP sample → Illinois register; dense city blocks |
| 105 | 2,000 | 180,402 | Tennessee franchises/food/lodging → first-draw register; similar names across sites |
| 106 | 5,680 | 16,016 | Variant of 102 with master street and ZIP empty; sparse-field behavior |
| 107 | 2,000 | 169,475 | Louisiana sole proprietors/self-employed → first-draw register; person-named firms |
| 108 | 5,000 | 8,612,370 | Nationwide PPP sample → national register; indexing and memory scale |

Registers retain duplicate records as distinct rows. Large registers stress index size, while
dense location blocks stress candidate generation and scoring; master row count alone does
not describe workload difficulty.

Each workload has master and using Parquet files, pinned by `real_workloads.json`.
`realdata.py` preserves row order, duplicate register records and all five supplied fields.
Source IDs are metadata; candidate outputs use zero-based row positions. No extra string
standardization or deduplication is performed. In 106 and 204, the master street and ZIP
are already empty in the supplied files. Float32 intermediate rounding, global perfect-match
removal, duplicate-row identities and all best-score ties remain part of correctness.

## Running and reporting

Run from the repository or staged-workspace root after installing the root requirements.
Create the candidate once, then run the checks below. See the [shared protocol](../harness/README.md)
for baseline comparisons, CPU budgets and final evaluation.

```bash
python -m harness init-solution --task 003 --out solutions/003/first
python -m harness evaluate --task 003 --solution solutions/003/first/solution.json --smoke
python -m harness evaluate --task 003 --solution solutions/003/first/solution.json --suite real-development --ids 102
```

The default development data root is `datasets/003/data`; override it with `--data-root`.
Final evaluation requires an external root, normally `--data-root ~/SocialSciBench-private/003/data`.
Real measurements default to five repetitions and one thread. National registers and dense
city blocks can use substantial memory; select IDs and whole-worker timeout deliberately.
Loading is excluded from timing; blocking/index construction and all input-dependent work
remain inside `run`.

Never use held-out data/results for tuning, debugging or settings selection. Freeze code and
settings with `harness freeze`, then use `harness final` and report every round, including
failures. The package contains timing estimates, not measured real-data performance results.
There are no ground-truth links: matching the reference establishes computational agreement,
not match quality. See [the protocol](../harness/README.md) for staging and final evaluation.
