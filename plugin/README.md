# turbo-socialsci: the SocialSciBench solutions as a plugin for agents

A Claude Code plugin with three skills. Each skill applies only when its computation fits; otherwise the
agent skips it and does the task its usual way.

| Skill | Computation | Engine function | Benchmark origin |
|---|---|---|---|
| `hdfe-regression` | OLS with absorbed fixed effects and one-way clustered SEs (reghdfe's conventions) | `feols` | task 001: schur-cg (speed), schur-lean (memory); one fixed effect: `hdfe_onefe` (0.2) |
| `staggered-did` | Callaway & Sant'Anna staggered DiD (csdid 1.81 / 2.0.1) | `att_gt` | task 002: cell-fused, cell-lean |
| `record-linkage` | name and address linkage (reclink2) | `reclink` | task 003: block-index, block-lean |

The skills install and call `engine/`, a Python package (`turbo_socialsci`, command `turbo-socialsci`).
Its computational cores are the frozen speed and lean candidates in `solutions/` (`schur-cg`,
`schur-lean`, `cell-fused`, `cell-lean`, `block-index`, `block-lean`), with the changes listed below. Version 0.2 adds what the pilot on the real-paper kits
(`eval/pilot-0.1.md`) showed was missing: a closed-form path for one fixed effect, a bootstrap
that never forms its multipliers as numbers, and text read as bytes instead of 4-byte characters.

For blind review, the author field in `turbo-socialsci/.claude-plugin/plugin.json` names no one. Every other file of the frozen 0.2.0 release is unchanged.

## Layout

```
plugin/
  .claude-plugin/marketplace.json     local marketplace: claude plugin marketplace add ./plugin
  turbo-socialsci/                    the plugin (claude --plugin-dir plugin/turbo-socialsci)
    .claude-plugin/plugin.json
    skills/<skill>/SKILL.md
    engine/                           pip-installable package
      src/turbo_socialsci/core/       the frozen cores, with the changes below; PROVENANCE.json;
                                      hdfe_onefe.py and _signs.py (new in 0.2)
  tools/export_cores.py               copies the frozen cores in; --check writes PROVENANCE.json and patches/
  tools/patches/<core>.diff           each core's diff against its frozen source
  tools/parity.py                     cores against the frozen candidates on the development inputs
  tests/                              engine tests (development fixtures only)
  eval/                               agent runs on the real-paper kits (see eval/README.md);
                                      bench_engine.py and compare_engine.py time and compare engine
                                      versions on a kit's data without an agent; grade_engine.py
                                      grades a frozen version's results with the kit's grader
```

This folder is not under `.claude/`, so sessions in this repository do not load the plugin.

## What changed from the frozen candidates

| Core | Change |
|---|---|
| all speed cores | `cache=True` on every Numba kernel (compiled once per installation); three task-001 kernels take the thread count as an argument, as schur-lean's do, so they can be cached |
| `hdfe_speed`, `hdfe_memory` | regressors collinear with the fixed effects or with regressors to their left are dropped, using reghdfe's order and a 1e-9 tolerance; R² of the full model and within R²; a regressor absorbed by the eliminated fixed effect gets a zero right-hand side, because its rounding residue made conjugate gradients run to the iteration limit |
| `did_speed`, `did_memory` | Rademacher multipliers drawn inside the call, from a counter-based generator keyed by (seed, unit, draw), so both variants draw the same ones; the event-study critical value over the event times with Pre_avg and Post_avg |
| `reclink_memory` | none (0.1) |

Version 0.2:

| Where | Change |
|---|---|
| `feols`, one fixed effect | `core/hdfe_onefe.py`, new: the partialled cross-products as sums over groups of within-group deviations, exact, with no iterative solver and no n × k floating-point matrix. Regressors are read in place in their own dtypes; columns holding only 0 and 1 are stored as lists of each row's ones. Codes of grouping variables come from a bitmap of the values present (`_codes.py`). Both variants compute the same numbers; "speed" copies the sample into group order first. Several fixed effects: unchanged. |
| `did_speed`, `did_memory` | the bootstrap's sums with the multipliers drawn inside the call come from `core/_signs.py`, new: tables of each 8 units' 256 signed sums, selected by the multipliers' bits (the same multipliers as 0.1), with explicit 4-double vector operations. `boot_tables=` limits the bootstrap to the tables reported; without the group-time table, the speed core draws the aggregates' own influence functions. The speed core no longer copies its inputs when every unit is used. |
| `reclink_speed`, `reclink_memory` | text as ASCII bytes, from Arrow's buffers (`_strings.py`): 2-D rows of bytes instead of 4-byte characters, with no Python-level loops; the memory core reads the using file's characters and offsets in place (a Numba structref), keeps record numbers in 32 bits and bigram codes in 16, and grows its block buffers 1.25× instead of 2× |
| wrappers | `att_gt` checks the panel with codes and counts instead of pandas' duplicated(); covariates are placed in one array without temporaries. `feols`' p-values come from scipy.special, so its first call no longer imports scipy.stats (1.6 s). `warmup()` goes through the public functions. |

With none of these conditions present, the cores run the frozen code's operations. `tools/parity.py` checks this on every development fixture, synthetic workload and real-development workload:
- integer outputs must match exactly;
- floating-point outputs must match within 1e-12 relative. That is the frozen code's own run-to-run variation: Apple Accelerate's BLAS is not bit-reproducible, and the frozen cell-lean differs from itself by up to 2e-15.

The tests compare the additions with independent computations:
- a dummy-variable OLS for collinearity and R²;
- task 002's reference given the same multipliers;
- (0.2) the one-fixed-effect path against the general solver, with float32 and float64 columns, dummies, missing values, singletons, collinear dummies and clusters that nest the groups or not;
- (0.2) the signed sums against the explicit product with the multipliers, and the bootstrap of some tables against the full bootstrap;
- (0.2) the byte and in-place text paths against the cores given 4-byte characters.

On the three kits' data, `eval/compare_engine.py` compares 0.2 with 0.1 call by call. Regression and linkage results, and the DiD estimates and analytic SEs, agree to rounding; the bootstrap results agree to rounding because the multipliers are the same.

## Use and test

```bash
claude --plugin-dir plugin/turbo-socialsci            # this session only
pip install plugin/turbo-socialsci/engine             # the engine alone
solutions/.venv/bin/python -m pytest -q -p no:cacheprovider plugin/tests
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  solutions/.venv/bin/python plugin/tools/parity.py 001 real-development
```

## Limits

- **Validation:** results are validated on the benchmark's development inputs only; held-out evaluation of the frozen candidates is pending.
- **Compilation:** the kernels compile on first use after installation, on the development host: about 45 s (regression), 45 s (DiD) and 80 s (linkage) for the speed variants, and 20 s and 70 s more for the regression and linkage memory variants. Later processes load them in 1–2.5 s.
- **Memory variants and Numba:** in 0.2 the DiD memory variant also loads Numba's runtime, about 65 MB, for its bootstrap when the multipliers are drawn inside the call.
- **Not covered:** each skill lists what it does not cover, and the engine raises `NotApplicable` for those requests.
