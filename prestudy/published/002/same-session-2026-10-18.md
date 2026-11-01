# Task 002 same-session comparison, 18 October 2026

Interpreted in [`reports/002_csdid.md`](../../reports/002_csdid.md#same-session-comparison-18-october-2026).

- Task 002, contract version 2, evaluator protocol version 2.
- Candidates: `solutions/002/shared-nuisance`, `r-drdid-cells`, `r-did`, `moderndid`,
  `csdid-py`, `stata-csdid`.
- Source revision: `450ebb1` with the candidates uncommitted (`git_dirty: true` in each
  `run.json`). For `shared-nuisance` and `r-drdid-cells`, the committed files are
  byte-identical to the ones snapshotted by the runs below, except `research.md`, which was
  updated with these results afterwards. For `r-did`, `moderndid`, `csdid-py` and
  `stata-csdid`, `solution.py` is byte-identical to the snapshot. Their `solution.json`
  (dependency list and description), `build.md` and `requirements.lock` were still the
  `init-solution` starter files when they were smoke-tested and were filled in afterwards;
  the runs used the shared environment described below either way.
- Host: Intel Core i7-8700B (6 cores, 12 logical CPUs), 32 GiB, macOS 15.7.3, x86_64. Not a
  dedicated machine; CPU frequency not locked.
- Environment: one shared Python 3.12.13 environment (`solutions/002/.venv`, recorded in each
  candidate's `build.md` and `requirements.lock`) for the reference and all candidates.
  NumPy 2.3.5, Numba 0.62.1, ModernDiD 0.2.0, csdid 0.4.2; R 4.1.2 with DRDID 1.3.0 and
  did 2.5.1 via rpy2 3.5.17; Stata 18 MP (`set processors 1`) with csdid 2.0.0 (commit
  `31a5773`) via pystata.
- Settings: one thread, validation seed 1729, one warmup, three measured calls. Timing is per
  complete `run` call; generation, imports and the first call are excluded.

## Commands

Smoke checks, serially from the repository root, 2026-10-18 19:47–19:51 UTC:

```bash
PY=solutions/002/.venv/bin/python
for c in moderndid r-drdid-cells r-did csdid-py stata-csdid; do
  $PY -m harness evaluate --task 002 --solution solutions/002/$c/solution.json --smoke
done
```

Timed comparison (`runs/compare-002.sh`), 2026-10-18 19:51–20:53 UTC:

```bash
for c in shared-nuisance r-drdid-cells; do
  $PY -m harness evaluate --task 002 --solution solutions/002/$c/solution.json \
    --ids 1 2 3 4 5 6 --timeout 3600
done
```

## Runs

| Run | Candidate | Cases | Result | Bundle SHA-256 (`<run>.tar.gz`) |
|---|---|---|---|---|
| `20261018T195143.154197Z-1fb46e17` | shared-nuisance | fixtures, W1–W6 | passed | `023fbe4b676d193c0632afe16b5238771b913f00ed521487aae8ff366e48adae` |
| `20261018T201053.813449Z-760179da` | r-drdid-cells | fixtures, W1–W6 | passed | `25d883516bd5c4c9078c7946d9ba5316326b66bbd65a42eac70c4b798bcc0210` |
| `20261018T194824.674460Z-163e29dd` | r-drdid-cells | smoke | passed | `ce3a4255f343da8fc5268d1747542dbbd35ffbf20a0ec38ceb57571c3d114688` |
| `20261018T194834.590901Z-a60386a3` | r-did | smoke | failed: `dev-propensity-trimming` | `94bd4ca9a83ef661dba481ececefa6829cdeed62a69240a36dd8f4d87a030242` |
| `20261018T194732.166248Z-17651f3e` | moderndid | smoke | failed: `dev-propensity-trimming` | `1de43ccb2a1e87ac9cbce488b5f462ad579700f304f106f1eaf40c28a6c6d2d4` |
| `20261018T194906.396206Z-fd98c158` | csdid-py | smoke | failed: `dev-propensity-trimming` | `8c758a4feee5e840b37f9e85dc481dbd15b8161ceb4512f89a9272ff0b360875` |
| `20261018T195027.807806Z-16881323` | stata-csdid | smoke | failed: `dev-propensity-trimming` | `690c14988a84f1cfddd9ae45dc11914dbfcb9b6e85687e735e9a133a6b27ef09` |

The four failures are runtime errors on the one fixture with a comparison unit above the 0.995
propensity trimming threshold; the other three fixtures passed. Error messages are in each
candidate's `research.md` and the run's `traces.jsonl`.

Bundles were made with `python -m harness export` and archived with `tar -czf`. **They have
not been uploaded yet**; download links will be added when they are attached to a release.

Earlier runs of `shared-nuisance` (smoke and validation seeds, listed in its `research.md`)
and the adapter-error smoke runs of `r-drdid-cells` and `r-did` are development measurements,
not part of this comparison.
