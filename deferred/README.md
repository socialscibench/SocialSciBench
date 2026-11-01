# Planned tasks

These computations are planned for the shared evaluator. Each directory holds a task description (`task.md`), a
reference implementation, deterministic synthetic workloads, baseline scripts and early results. None is in the
evaluator yet, and the results reported for tasks 001–003 do not depend on them.

| Task | Computation |
|---|---|
| [004](004_winsorize/) | Pooled and grouped winsorization |
| [005](005_merge/) | One-to-one and many-to-one table joins |
| [006](006_matching/) | Propensity-score and coarsened exact matching |
| [007](007_fuzzymatch/) | Fuzzy name matching with bigram cosine similarity |
| [009](009_melogit/) | Multilevel (random-intercept) logistic regression |
| [010](010_larger_than_memory/) | Preparing data larger than memory under a memory budget |
| [011](011_logithdfe/) | Fixed-effects logit and average marginal effects |

**The results here are historical.** The timings and results in these directories (each `results/` folder and the
tables in each README) were recorded early in the project with each task's own scripts, before the shared evaluator
existed, and have not been re-checked under it.

Run a task's scripts from its directory. Install the Python dependencies with
`python -m pip install -r deferred/requirements.txt` from the repository root; task 010 also needs `pyarrow` and
`duckdb` for its Python baselines. Its real-data workload reads licensed files that are not included (set
`SSB_TASK010_REAL_DIR` to their folder); its synthetic workloads need nothing else. The baseline scripts also need the
external software they compare (Stata, R).
