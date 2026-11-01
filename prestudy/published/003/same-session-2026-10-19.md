# Task 003 same-session comparison, 19 October 2026

Interpreted in [`reports/003_reclink.md`](../../reports/003_reclink.md#same-session-comparison-19-october-2026).

- Task 003, then numbered 008; contract version 2, evaluator protocol version 2.
- Candidates: `solutions/003/legacy-pre-rename/bigram-table-scan`, `cpp-pipeline`, `numba-comparator`, `r-rcpp`,
  `r-port`, `stata-reclink2`. They were in `solutions/008/` when these runs were made; the commands below are as run.
- Source revision: `a5f7aa5` with the candidates uncommitted (`git_dirty: true` in each
  `run.json`). The committed candidate files are byte-identical to the ones snapshotted by the
  runs below, except `research.md`, which was updated with these results afterwards (and, for the
  first `bigram-table-scan` smoke run, `build.md`, written after it).
- Host: Intel Core i7-8700B (6 cores, 12 logical CPUs), 32 GiB, macOS 15.7.3, x86_64. Not a
  dedicated machine; CPU frequency not locked. A crash-looping desktop helper and its crash
  reporter used 15-30% of one core throughout.
- Environment: one shared Python 3.12.13 environment (`solutions/008/.venv`, recorded in each
  candidate's `build.md` and `requirements.lock`, identical to the task 002 environment) for the
  reference and all candidates. NumPy 2.3.5, SciPy 1.18.1, pandas 3.0.6, Numba 0.62.1; Apple clang
  17.0.0 for the C++ pipeline; R 4.1.2 with data.table 1.18.6.1, stringi 1.8.9 and Rcpp 1.1.2 via
  rpy2 3.5.17; Stata 18 MP (`set processors 1`) with reclink2 from the Stata Journal's dm0082
  package (`reclink2.ado` SHA-256 `5ed4aae093ffc2d74059a5fe47205441dbb76da641124d12b7855f1f3cc4ad0b`)
  via pystata.
- Settings: one thread, validation seed 1729 (smoke runs also 11, 12, 13), one warmup, three
  measured calls. Timing is per complete `run` call; generation, imports and the first call are
  excluded.

## Commands

Smoke checks, serially from the repository root, 2026-10-19 02:33-02:47 UTC:

```bash
PY=solutions/008/.venv/bin/python
$PY -m harness evaluate --task 008 --solution solutions/008/bigram-table-scan/solution.json --smoke
for c in cpp-pipeline numba-comparator r-port r-rcpp stata-reclink2; do
  $PY -m harness evaluate --task 008 --solution solutions/008/$c/solution.json --smoke
done
for s in 11 12 13; do
  $PY -m harness evaluate --task 008 --solution solutions/008/bigram-table-scan/solution.json \
    --smoke --validation-seed $s
done
```

Timed comparison (`runs/compare-008.sh`), 2026-10-19 02:47-06:43 UTC:

```bash
for c in bigram-table-scan cpp-pipeline numba-comparator r-rcpp r-port; do
  $PY -m harness evaluate --task 008 --solution solutions/008/$c/solution.json \
    --ids 1 2 3 4 5 --timeout 10800
done
```

## Runs

| Run | Candidate | Cases | Result | Bundle SHA-256 (`<run>.tar.gz`) |
|---|---|---|---|---|
| `20261019T024724.672285Z-63751f80` | bigram-table-scan | fixtures, W1-W5 | passed | `e3d567dc2b4d8405a37687adb068be4b4354cbade908e411d98c57806c2b6e15` |
| `20261019T031114.806653Z-b41b2449` | cpp-pipeline | fixtures, W1-W5 | passed | `cc7faefe1913466a6d467c4d923511294a9dbd3eb6e49a4e438ec283b0ba1ce3` |
| `20261019T033818.357802Z-ab03784d` | numba-comparator | fixtures, W1-W5 | passed | `95b45399e5341fbfcbe7ac10d7dd31642923d4b9c004a9e110a6cdbd4eff3120` |
| `20261019T040719.483227Z-8aecb21f` | r-rcpp | fixtures, W1-W5 | passed | `9bc10a73c0cc2d3aa3396b6333d0484494b817a43c26239df7f99b44b061a724` |
| `20261019T050131.508302Z-86e2d914` | r-port | fixtures, W1-W5 | passed | `b5308be9040f4de65063af878dc6aff5d5c4bb0b7c3f7064936c86096f9f5087` |
| `20261019T023320.928531Z-a715f349` | bigram-table-scan | smoke | passed | `a304160776e4026c2d1995911dfd34aebf4f29a2236443fa8e133f084aaa3bfe` |
| `20261019T024359.362148Z-1c6d5364` | bigram-table-scan | smoke, seed 11 | passed | `edda2244db65fa2378f790c1195f22fafffe82d7984a495b51f1301122f6891d` |
| `20261019T024503.386714Z-5362232c` | bigram-table-scan | smoke, seed 12 | passed | `f765a2d18ccfbd60147c20b8263fa7d54f5dffdce7a2275c4369959d6df6e91e` |
| `20261019T024608.131421Z-3752f7ab` | bigram-table-scan | smoke, seed 13 | passed | `c0278b9aa104acd90f69440390025bd49025530796cbb00d5ebf44de7ae2ce0f` |
| `20261019T024143.461496Z-c75c9573` | cpp-pipeline | smoke | passed | `9ca40dcd7e455992bcd2008f8270f758b77f69c3503db69ed10192191c2a44cd` |
| `20261019T024154.604339Z-872b4d40` | numba-comparator | smoke | passed | `a15397bdc6c65ea45df5191c1eb0870b5c17e82d2d7757314e7bc8f5d5ca95dc` |
| `20261019T024213.713476Z-8598cc24` | r-port | smoke | passed | `c91df77d42a9b40feb867158eec23d9dd7bebb7139f6c73cb1bee1386bceb658` |
| `20261019T024225.187327Z-f15d2dfc` | r-rcpp | smoke | passed | `a1f90bb412bda19b8202d0b2fb5b413f2c36bc0c9af0d4017e4382896f4492f8` |
| `20261019T024248.349045Z-a8f85193` | stata-reclink2 | smoke | failed: `dev-ties-missing-perfect` | `0f2b1a7580313121ffecd575bc681a05f414be51245bc5f94bf66164590cde3e` |

The `stata-reclink2` failure is an incorrect output on the one fixture whose using file contains
identical records that are the best candidate of more than one master record (7 pairs against the
reference's 9); the other three fixtures passed. The cause is described in its `research.md`.

Bundles were made with `python -m harness export` and archived with `tar -czf`. **They have
not been uploaded yet**; download links will be added when they are attached to a release.

Earlier smoke attempts of the competitors that stopped before running (missing `research.md`
artifacts) produced no run directories. The development measurements of `bigram-table-scan`
listed in its `research.md` were made outside the evaluator and are not part of this comparison.
