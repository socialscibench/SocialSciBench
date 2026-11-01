# Task 001 same-session comparison, 18 October 2026

Interpreted in [`reports/001_reghdfe.md`](../../reports/001_reghdfe.md#same-session-comparison-18-october-2026).

- Task 001, contract version 2, evaluator protocol version 2.
- Candidates: `solutions/001/eliminate-largest-fe`, `xhdfe`, `within`, `julia-fem`, `fixest`.
- Source revision: `1dfdd0e` with the candidates uncommitted (`git_dirty: true` in each
  `run.json`). The candidate files committed alongside this record are byte-identical to the
  ones snapshotted by the runs below, except `research.md`, which was updated with these
  results afterwards.
- Host: Intel Core i7-8700B (6 cores, 12 logical CPUs), 32 GiB, macOS 15.7.3, x86_64. Not a
  dedicated machine; CPU frequency not locked.
- Environment: one shared Python 3.12.13 environment (`solutions/001/.venv`, recorded in each
  candidate's `build.md` and `requirements.lock`) for the reference and all candidates.
  xhdfe 2.28.1.20260929 (built from the study's verified source archive), within-py 0.4.0,
  Julia 1.13.1 with FixedEffectModels 2.1.0 via juliacall 0.9.36, R 4.1.2 with fixest 0.14.2
  via rpy2 3.5.17.
- Settings: one thread, validation seed 1729, one warmup, three measured calls; W11 one
  measured call. Timing is per complete `run` call; generation, imports and the first call are
  excluded.

## Commands

Run serially from the repository root, 2026-10-18 02:34–16:38 UTC:

```bash
PY=solutions/001/.venv/bin/python
for c in eliminate-largest-fe xhdfe within julia-fem fixest; do
  $PY -m harness evaluate --task 001 --solution solutions/001/$c/solution.json \
    --ids 1 3 4 6 7 8 9 10 12 --timeout 3600
done
for c in eliminate-largest-fe xhdfe within julia-fem fixest; do
  $PY -m harness evaluate --task 001 --solution solutions/001/$c/solution.json \
    --ids 11 --repeats 1 --timeout 21600
done
```

## Runs

| Run | Candidate | Workloads | Result | Bundle SHA-256 (`<run>.tar.gz`) |
|---|---|---|---|---|
| `20261018T023431.390119Z-65be5196` | eliminate-largest-fe | 1 3 4 6 7 8 9 10 12 | passed | `df57a33fc007a00741a7abd528aeaacd7c55d23279b3379d0856673456a9d083` |
| `20261018T031236.191169Z-deeb54e3` | xhdfe | 1 3 4 6 7 8 9 10 12 | passed | `bb29e6fce5b4b302205a595f288a8ac767a0506c50ad22830c85dcdc6617f22a` |
| `20261018T034924.951843Z-48246f2c` | within | 1 3 4 6 7 8 9 10 12 | passed | `5c96a22ddade0d696159d9870113f6904503681e61651858f6a16c2ddbaca7a1` |
| `20261018T042523.272784Z-464d5031` | julia-fem | 1 3 4 6 7 8 9 10 12 | failed: W9 accuracy | `f266252271c22431514dc097edb593044e1e5b65c098e35dda11ee90a865fc33` |
| `20261018T050445.888229Z-ba8122f0` | fixest | 1 3 4 6 7 8 9 10 12 | failed: W9 accuracy | `fc47311ed5b0930f07af51a7728ac4088823ecbb3c2b5e29f3a019c9906ba15c` |
| `20261018T055637.090305Z-f59421e6` | eliminate-largest-fe | 11 | passed | `2a1bec43fcf456a1c97a4d69ea139bbd5d371c63b992aafe07b4d026e5514788` |
| `20261018T073339.204026Z-d42c9e78` | xhdfe | 11 | passed | `3d681a07100d389d3c1bf9a4383aee38c323ce96a0cf5ae6b2a252d8247d985e` |
| `20261018T103900.111502Z-1d86db38` | within | 11 | passed | `10ead5f2a8ef0f57ba4351ef7afca183fcb132246559295e919203ff86589db6` |
| `20261018T121935.984062Z-6d4c0e0a` | julia-fem | 11 | failed: accuracy | `6083c19365e50c62705bfb08c7f33a72af05817029b74a09c2b080079dfcff69` |
| `20261018T140933.944721Z-d6f2feb9` | fixest | 11 | failed: accuracy | `504fcc23078eeb494f74132395b8b42476582d558d9d971f7d04ff652b4a1451` |

Every run also passed the five development fixtures. Bundles were made with
`python -m harness export` and archived with `tar -czf`. **They have not been uploaded yet**;
download links will be added when they are attached to a release.

Earlier runs of `eliminate-largest-fe` (`20261017T…`, listed in its `research.md`) are
development measurements, not part of this comparison. One of them, W11
`20261017T224035.082271Z-b76c2624`, is incomplete: the session hosting it ended.
