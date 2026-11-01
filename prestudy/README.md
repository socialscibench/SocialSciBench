# Pre-study records

Exploratory comparisons and candidates from before the development study, restored as they were recorded. They show where the development checks found the differences between existing tools that the proposal's introduction describes:

- **Convergence:** Julia's FixedEffectModels and R's fixest failed the accuracy checks on synthetic workloads 9 and 11 ([task 001 record](published/001/same-session-2026-10-18.md)).
- **Propensity scores:** R's did, ModernDiD, Python's csdid and Stata's csdid stopped with errors on the test case with a comparison unit above the 0.995 trimming threshold ([task 002 record](published/002/same-session-2026-10-18.md)).
- **Duplicate records:** Stata's reclink2 returned 7 pairs instead of 9 on the test case whose register holds identical records ([task 003 record](published/003/same-session-2026-10-19.md)).

The proposal's numbers come from the [development study](../solutions/baselines/development-study.md), not from these records.

Read them with these limits in mind:
- They keep their dates and task 003's earlier number, 008.
- The earliest studies used each task's own scripts, and the comparisons in `published/` used an earlier version of the evaluator. Their task 002 results describe that task before it was combined into the current one.
- The reports refer to raw data and reproduction code in a local companion repository that was not published, and the run bundles listed in `published/` were not uploaded.
- The candidates are records, not code to run. The 008-era candidates declare a task number that the evaluator no longer accepts, and some candidates depend on files in the companion repository.
- Six links to files outside this folder were changed to follow the move. For blind review, local paths that contained a user name now begin with `/path/to/`. Nothing else was changed.

| Folder | Contents |
|---|---|
| [reports/](reports/README.md) | Reports on the early comparisons for tasks 001, 002 and 003 |
| [published/](published/README.md) | Records of the comparisons of candidates, one per task |
| [baselines/](baselines/README.md) | Runner scripts and raw timings for Stata, fixest and an R implementation on the synthetic workloads |
| `solutions/` | Ten earlier candidates for tasks 001 and 002, and six record-linkage candidates from when the task was numbered 008 |
