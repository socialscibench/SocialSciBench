# SocialSciBench

**A benchmark for reliable, scalable computation in social science.**

SocialSciBench provides computational contracts, real-data and deterministic synthetic workloads, readable reference implementations and a shared evaluator for agent-generated optimizations. The active scope is three task families:

| Task | Computation | Contract |
|---|---|---|
| 001 | Linear regression with high-dimensional fixed effects and clustered standard errors | [task.md](001_reghdfe/task.md) |
| 002 | Staggered-adoption difference-in-differences and event studies | [task.md](002_csdid/task.md) |
| 003 | Record linkage using names and addresses | [task.md](003_reclink/task.md) |

Planned tasks are in [deferred/](deferred/README.md): each has a task description, reference code, synthetic workloads and early baseline timings, but none is in the shared evaluator, and no result reported here depends on them.

## Where the proposal's results are

This repository accompanies the Stage 1 proposal to the *Organization Science* special issue. The proposal's results come from these files:

| In the proposal | In this repository |
|---|---|
| Where the development checks found differences between existing tools (introduction) | [The pre-study records](prestudy/README.md) |
| Task contracts and the evaluator (Figure 1) | The three task directories and [the evaluation protocol](harness/README.md) |
| Speed and peak memory against existing implementations (Table 2 and the per-workload results in the appendix) | [The development study](solutions/baselines/development-study.md) and [how the gains were made](solutions/baselines/optimizations.md). The candidates and the existing tools' adapters are in `solutions/001/`, `solutions/002/`, and `solutions/003/` |
| The skill and its engine, turbo-socialsci 0.2.0 | [plugin/](plugin/README.md), frozen at tag `turbo-socialsci-v0.2.0` |
| The engine on the three complete analyses (Figure 2) | [The engine report](plugin/eval/engine-0.2.md) and `plugin/eval/bench_engine.py`. The existing implementations' timings in Figure 2 were measured with scripts that are not in this repository |
| Agents with and without the skill (Figure 3) | [The agent study](plugin/eval/agent-study-0.2.md), [the pilot](plugin/eval/pilot-0.1.md), and [their protocol and tools](plugin/eval/README.md) |
| Workloads and data | [Download instructions](datasets/README.md), [provenance and lineage](data_sources/README.md), and each task's `DATA.md` |

The inputs of the three complete analyses are not in this repository. Two of them use public replication data. The record-linkage analysis uses licensed data that cannot be redistributed. The held-out workloads defined in each task's `DATA.md` were not evaluated for this submission. The run records that the study reports cite under `runs/` are local and are not included.

The two peak-memory candidates, `schur-lean` and `block-lean`, load compiled code from a Numba cache that is built once per host. Run their `build_cache.py` before evaluating them, as [the development study's reproduction steps](solutions/baselines/development-study.md#reproduction) show.

## Evaluate an implementation

Create a Python 3.12 or 3.13 environment on macOS or Linux and install the dependencies. On Windows, use WSL, because the evaluator measures memory with a Unix-only module. No Stata or R installation is needed for candidate evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m harness list
python -m harness init-solution --task 001 --out solutions/001/first
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --smoke
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --ids 1
```

The starter is an executable copy of the reference implementation. Edit its source to try a different implementation, then rerun the evaluator. Native implementations can use a Python adapter. See [the evaluation protocol](harness/README.md) for submission fields, accuracy rules, timing boundaries, thread settings, timeouts, accelerator hooks and complete-workload evaluation. `requirements.lock` records the tested dependency versions; `requirements.txt` allows resolution on other supported Python environments.

The evaluator runs the reference implementation and the candidate in separate processes, checks every output, and records raw timing samples, environment details, source hashes and snapshots in a unique `runs/<run-id>/` directory. Public development fixtures must pass before official workloads can be scored. Smoke checks do not establish full-task correctness. Task 003 validation compares complete pair identities and individual scores.

## Real-data study and agent research

The real-data study covers **001 (reghdfe), 002 (csdid), and 003 (reclink)**.

Development inputs live outside Git in `datasets/<task>/data/`. Held-out inputs, supplied
reference answers and validation logs live outside the repository in `~/SocialSciBench-private/`.
The held-out workloads are defined for a future frozen evaluation and were not evaluated for this
submission. Their inputs and the packages' answer files are in the same public Drive folders as the
development inputs (file IDs in `data_sources/packages.json`), so they are not hidden from developers.
See [download instructions](datasets/README.md),
[provenance and lineage](data_sources/README.md), and each task's `DATA.md`.

```bash
python scripts/download_real_data.py
python -m harness list --suite real-development
python -m harness init-solution --task 003 --out solutions/003/first
python -m harness evaluate --task 003 --solution solutions/003/first/solution.json --smoke
python -m harness evaluate --task 003 --solution solutions/003/first/solution.json --suite real-development --ids 102
```

Synthetic workloads remain available. `--all-workloads` applies only to the chosen suite;
held-out data are excluded from ordinary development evaluation. Final evaluation requires
a frozen snapshot and recorded rounds, described in [the protocol](harness/README.md).
Reference outputs are recomputed; supplied answers and historical performance results are
never candidate inputs. Package timing estimates are not new measured baselines.

[AGENTS.md](AGENTS.md) describes the research, implementation, validation and reporting workflow. Agents can investigate algorithms and libraries, implement candidates, and use the evaluator's structured feedback to iterate. The external agent supplies research tools and controls the experiment budget; this repository does not launch an LLM or perform web research itself.

For an independent-agent experiment, export an explicit task allowlist without historical results or prior solutions:

```bash
python -m harness stage --task 001 --out /tmp/socialscibench-001
```

Give the agent access to that exported workspace. Staging does not enforce operating-system or network isolation; use an external runner for those access policies. In an informed research session, the study reports listed above can be useful context.

## Repository layout

| Location | Purpose |
|---|---|
| `001_reghdfe/`, `002_csdid/`, `003_reclink/` | Task contracts, manifests, reference implementations, workload generators |
| [tasks.json](tasks.json) | Active task registry |
| [harness/](harness/README.md) | Candidate execution, public development fixtures, correctness validators, provenance |
| [solutions/](solutions/README.md) | Candidates, existing-tool adapters and research notes. [solutions/baselines/](solutions/baselines/README.md) holds the comparison scripts and the development study's reports |
| [plugin/](plugin/README.md) | The turbo-socialsci skill and engine for agents, and [the agent study's tools and reports](plugin/eval/README.md) |
| `runs/` | Development traces, snapshots and logs; ignored by Git. Final runs and freeze records use external directories |
| [datasets/](datasets/README.md) | Ignored development inputs and package code/docs; instructions for external private storage |
| [data_sources/](data_sources/README.md) | Pinned file index, original guides and build lineage |
| [deferred/](deferred/README.md) | Planned tasks, not yet in the evaluator: descriptions, references, workloads and early results |
| [prestudy/](prestudy/README.md) | Exploratory comparisons and candidates from before the development study, kept as recorded |

Protocol version 3 snapshots include the task registry, evaluator, dependency specifications and candidate research/build artifacts. Use `python -m harness export --run runs/RUN_ID --out NEW_DIRECTORY` to prepare a portable bundle, and `python -m harness replay --run BUNDLE_DIRECTORY --out runs/replays` to repeat its saved cases and settings in a new run. Install the recorded environment first. Exported bundles can be attached to versioned releases. Local scratch runs stay ignored. See the [evaluation protocol](harness/README.md) for replay and measurement limits.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

In a fresh environment the first full test run took about 12 minutes on our test machine (a six-core Intel i7-8700B), because Numba compiles the candidates' and the plugin's code first. Once that compiled code is cached, the suite takes about 3 minutes. The quickstart commands above take under a minute.
