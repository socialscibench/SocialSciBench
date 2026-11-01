# Candidate evaluation

## Real-data study

Tasks **001, 002 and 003** support real workloads as well as their unchanged synthetic generators.
Task 003 is record linkage. The planned fixed-effects logit task is in `deferred/011_logithdfe`.

```bash
python -m harness list --task 001 --suite real-development
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --suite real-development --ids 101
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --suite real-development --ids 104 --n-regressors 50 --timeout 1800
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --suite real-smoke
```

The default suite remains `synthetic`; explicit suites prevent ID ambiguity. `--ids` and
`--all-workloads` select only within that suite. Real development defaults to its first ID
(101); for a small linkage check choose 003/102. Task 001/109 is available only in
`real-smoke`, with a single compatibility call and no performance score. Do not tune speed
or accuracy on it. `--smoke` always means the small public synthetic correctness fixtures.

Install `pyarrow` through the root requirements. Default development input roots are
`datasets/001/data`, `datasets/002/data`, and `datasets/003/data`. Held-out inputs and saved
answers live outside the repository, by default in `~/SocialSciBench-private/<task>/data`.
`--data-root` takes the directory containing the chosen split (`development/` or `heldout/`);
both splits need not be present. See the local data download instructions in the full
repository. Real runs default to **five measured calls**, one additional warmup
and one thread. Loading, schema conversion and development-value generation occur before
timing. Candidate input-dependent indexing/preprocessing still belongs inside `run`.
The default 300-second timeout covers the whole worker, not each repetition; budget larger
workloads explicitly. Baseline comparisons must use the same data, accuracy and hardware.

Each input file is checked against the task catalog's size and SHA-256 before evaluation
and again afterward. A mismatch invalidates results. The run records external input paths,
hashes, suite and workload variants; both workers must also produce identical in-memory
input hashes. **Protocol 3 snapshots/export bundles do not embed large dataset files.**
Download or transfer the matching pinned files and use:

```bash
python -m harness replay --run /path/to/bundle --out runs/replays --data-root /path/to/package/data
```

No data-loading or checksum check executes a reference solve on held-out data.

## Frozen final evaluation

> **In this submission,** no frozen final evaluation was run and no held-out workload was evaluated.

Develop only on synthetic and real-development inputs. Do not inspect held-out results or
use held-out inputs for debugging, tuning or choosing settings. When the method is finished,
create a dated snapshot of code, task, evaluator, selected final cases and measurement settings:

```bash
python -m harness freeze --task 001 --solution solutions/001/first/solution.json --out ~/SocialSciBench-private/freezes/001-final-v1 --rounds 3 --threads 1 --repeats 5 --timeout 1800
python -m harness final --freeze ~/SocialSciBench-private/freezes/001-final-v1 --data-root ~/SocialSciBench-private/001/data --out ~/SocialSciBench-private/runs/001 --note 'First frozen evaluation'
```

Freezing does not load or evaluate held-out data. It selects every held-out workload by
default; `freeze --ids ...` can predeclare a narrower scope. `final` executes the frozen
evaluator and candidate, with no settings overrides. It reserves a numbered attempt before
execution; failures and interrupted attempts remain in `rounds/` and consume a round.
Source hashes are checked before execution. Another final invocation uses another round;
ordinary `replay` cannot rerun held-out cases. Held-out inputs, freeze records and final output
directories must be outside the repository/development workspace; paths inside are rejected.
If `final --out` is omitted, results go to a `runs/` sibling of `--data-root`, for example
`~/SocialSciBench-private/001/runs/`. Preserve the freeze and all run directories.
If a process is interrupted, inspect its attempt/logs before removing a stale
`active-round.lock`; never delete the attempt record to recover a round.

Keep the study to two or three final rounds **across versions**, report every round including
failures, and disclose that developers could access the held-out data. The software caps
rounds per freeze; it cannot enforce the study-wide policy against new freezes or direct
Python calls. These are research protocol checks, not a security boundary. Do not run
benchmarks concurrently on the host. Task 001/206 overlaps development-area patent records
and is a frozen full-sample evaluation, not wholly unseen data. Each task's `DATA.md` explains
its splits and limitations. Do not claim full-task correctness from selected development runs.

## Agent access to real inputs

```bash
python -m harness stage --task 002 --suite real-development --ids 101 106 --out /tmp/socialscibench-002
```

This exports only the selected development files (101 is copied once for 101 and 106), a
catalog limited to those cases, task files and the evaluator. Omitting `--ids` includes all
development workloads. `stage --suite real-smoke` exports 001/109 without any data files.
Default synthetic staging copies no dataset files and removes held-out catalog entries.
Staging never copies package guides, source manifests with performance estimates, reference
answers, validation logs, historical results or prior candidates. Give an independent agent
only that exported directory; use an external container/runner for filesystem and network
visibility. Keep the private data/results directory unmounted and prevent retrieval of held-out
packages or prior results through the network. An external path alone does not prevent access
by another process under the same user account. A staged workspace cannot create a final
freeze from its redacted catalog.

The evaluator supports tasks 001, 002, and 003. It runs Python adapters to arbitrary implementations without Stata or R. Run commands from the workspace root with Python 3.11 or later. Install `requirements.txt`, or `requirements.lock` for the tested package versions. The workspace's `tasks.json` lists the available tasks.

```bash
python -m harness list
python -m harness init-solution --task 001 --out solutions/001/first
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --smoke
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --ids 1
python -m harness evaluate --task 001 --solution solutions/001/first/solution.json --all-workloads --timeout 1800
```

The starter copies the reference implementation and should pass unchanged. It also creates `research.md`, `build.md` and a starting dependency lock, declared as artifacts in `solution.json`. Update these as the candidate evolves. Replace the computational implementation while retaining the task's interface. Task 003 returns a four-tuple `(i, j, score, n_perfect)`, as the executable reference does. Pair indices are zero-based. Generated `truth` labels are not passed to candidates. Task 003's weights, variable names and thresholds are supplied as keyword arguments.

## Task manifest hooks

Task manifests are benchmark definitions, not candidate settings. The evaluator resolves and validates every declared hook before listing, staging or evaluating a task. Task-local hook files must be in `prompt_files` so they survive staging and snapshotting.

| Field | Value and interface |
|---|---|
| `reference_entry_point` | Task-local `filename.py::function`; accepts the documented candidate keyword arguments |
| `workload_catalog` | Task-local `filename.py::symbol`; a sequence of workload dictionaries with IDs |
| `workload_entry_point` | Task-local `filename.py::function`; receives one workload dictionary |
| `realdata_entry_point` | Optional task-local `filename.py::load`; receives `(id, data_root=..., **variant_options)` |
| `real_workload_catalog` | Optional JSON filename in `prompt_files`; schema 1, task ID, split membership and pinned input files |
| `development_entry_point` | `module::function` or task-local `filename.py::function`; receives `(task_id, seed)` and returns cases |
| `adapter` | Same hook syntax; receives `(task_dir, task_id, case, generate)` and returns candidate keyword arguments, using the declared generator for generated cases |
| `validator` | Same hook syntax; receives `(task_id, actual, expected, tolerances, dimensions)` and returns a JSON-compatible report with `passed` and `checks` |

The built-in adapters, output normalization and dimension metadata support these four task contracts. The built-in 003 adapter reads linkage constants from its task's `workloads.py`. Adding a different contract can require changes to those implementations as well as its registry entry and manifest. Candidate code must not change task hooks.

## Submission format

```json
{
  "schema_version": 1,
  "name": "my-implementation",
  "task": "001",
  "entry_point": "solution.py::run",
  "sources": ["solution.py", "helpers.py"],
  "artifacts": ["research.md", "requirements.lock", "build.md"],
  "target_hardware": "cpu",
  "dependencies": ["numpy", "scipy"],
  "description": "Algorithm and implementation choices"
}
```

Sources and artifacts are files relative to `solution.json`; paths cannot escape that directory. Include helper modules, native sources and local build inputs in `sources`, and additional research/build records in `artifacts`. Conventional `research.md`, `requirements.lock` and `build.md` files are captured automatically when present, including for older solution manifests. All captured files are hashed. For compiled libraries, install/build the adapter's environment before evaluation and include reproducible build instructions. Dependencies are declarations, not an automatic installer; the worker records all installed distribution versions. Snapshots do not capture undeclared external files or native system libraries: record their versions/build hashes in research notes. Update the starter's dependency lock for the actual candidate environment.

For an accelerator, use a descriptive `target_hardware` such as `cuda` and add `"synchronize": "solution.py::synchronize"`. The no-argument hook must wait for all device work. The adapter must return host-ready arrays, so host/device transfers, preprocessing and output conversion are included in `run`. GPU timing/hooks have not been validated on hardware by the CPU test suite.

## Correctness and coverage

Each invocation first runs all public development fixtures. A failure prevents scoring of the selected official workloads. `--smoke` runs only those fixtures and produces no performance score. In the default synthetic suite, the default is official workload 1; use `--ids ...` or `--all-workloads` explicitly for wider coverage within the chosen suite. `--validation-seed N` changes development inputs without changing the official workloads or seeds.

Contract version 2 checks shapes, numeric types, finite values, exact counts and labels. Integer arrays may use any integer width, including the reference's sparse-index dtype. Float arrays may use any floating width if they pass accuracy checks. Numeric relative error is `abs(actual - reference) / max(abs(reference), 1e-12)` element by element; the floor defines behavior near zero. Tolerances are stored in each `task.json`.

- 001: coefficients, diagonal-covariance standard errors, counts; covariance must also be finite, square and symmetric with nonnegative diagonal. Off-diagonal differences are diagnostic because the original contract does not specify their tolerance.
- 002: every group-time and aggregate effect and standard error, the event-time subsets and the event-study covariance, and (with bootstrap multipliers) the bootstrap standard errors and simultaneous critical values, plus exact cohort/time labels and event times. Estimates are scaled by max(|reference|, its reference SE), the covariance by SE_i SE_j (`002_csdid/task.md`); the `event_sets` echo of the argument is not compared.
- 003: the complete pair set after sorting, no duplicate or out-of-range pairs, every pair's score, and the perfect-pair count. This strengthens the previous count-and-sum checks; historical baseline measurements are not retroactively certified under version 2.

The reference and candidate receive independently generated inputs. Input hashes must agree; arrays are read-only and input changes are checked after every call. Every first-call, warmup and measured output is compared. An incorrect, missing, non-finite, timed-out or failed output receives no performance score. Separate worker errors/logs distinguish candidate failures from invalid references. The development fixtures cover multiple seeds, fixed-effect graph structure, recursive singletons and label recoding, cohort patterns, similarity boundaries, empty/short strings, perfect matches, global exclusion and ties. They are not exhaustive or secret. Correctness is established only on the cases actually evaluated.

## Measurement

For a study of benefits to real research, prioritize real-development workloads while retaining
synthetic correctness and performance coverage. Use synthetic workloads to explain sensitivity
to size, connectivity, cohort structure and candidate density; use frozen real held-out cases
as the main generalization evidence. Report the suites separately and include every workload
in the claimed scope, including slowdowns and failures. Task 001/109 remains compatibility-only.

Research current papers, libraries and optimized implementations using primary sources, and
identify the strongest existing baselines compatible with the task's exact computation.
Record source URLs and versions in the candidate's research notes. Compare against those
tools using identical inputs, accuracy requirements, timing boundaries, hardware and CPU
budgets. Report absolute runtime and memory alongside speedups; distinguish these practical
comparisons from the harness score against the readable reference. Disclose unavailable or
incompatible baselines and label estimates separately from measurements. Independent research
must not retrieve prior SocialSciBench candidates or results.

A recommended CPU study has one-thread and explicitly budgeted multicore configurations of
the same method. First remove redundant computation and improve the serial implementation;
then parallelize measured bottlenecks. Multiple threads and multiple processes are alternative
ways to use multiple cores; implementing both is not a benchmark requirement. Freeze each
planned configuration before final evaluation and keep all configurations within the study's
declared evaluation rounds. Compare each tool at matching core budgets, numerical accuracy
and hardware; report absolute times, memory and scaling as well as speedups.

`--threads` controls supported runtime thread pools; it does not implement parallelism or cap
arbitrary subprocesses. Candidate-specific pools must respect the declared total CPU budget.
Avoid multiplying process workers by unrestricted BLAS/OpenMP threads. If a candidate uses
subprocesses, measure their aggregate memory separately: the worker RSS below excludes them.

The default is one requested compute thread, one first call, one additional warmup, and three measured calls per synthetic workload (five for real development and frozen final evaluation). Development fixtures use one correctness call. Reference and candidate run sequentially in fresh processes. Protocol versions 2 and 3 run the reference first to establish the comparison output, then validate and release each subsequent output before the worker continues. With default output retention, at most two full output files exist at a time: the first reference and the current repetition. `--keep-outputs` explicitly retains every output and can require substantial disk space. Standard BLAS/OpenMP/Numba/Rayon environment limits and `threadpoolctl` are applied; arbitrary native/accelerator libraries may need adapter-specific thread controls. Worker threadpool information is recorded.

Timing uses `time.perf_counter()` around `run` and completion synchronization. Generation, imports, garbage collection, input hashing, output serialization and validation are excluded. Input-dependent preprocessing and output conversion remain inside `run`. First-call latency is recorded separately; imports and first-call JIT setup are excluded from steady-state samples. Caches may contain compiled code but must not retain input-dependent answers between calls. All repetition samples are retained; speedup is reference median / candidate median on the same host. First calls, warmups and between-call validation can affect caches. Reference-first execution can introduce order effects; no CPU clock locking or cold-cache guarantee is provided. Protocol version 1 alternated worker order and retained all outputs until both workers finished, so record the protocol version when comparing measurements.

Peak RSS is the OS process high-water mark over the worker lifetime, including interpreter, imports, input generation, outputs and serialization. It is not incremental algorithm memory and excludes child-process/device memory and controller validation memory. Validation still loads complete outputs and sorts matching pairs; bounded temporary storage does not make validation memory constant in pair count. No memory budget is enforced for these four tasks. `--timeout` bounds worker time including generation, imports, warmups, all measured calls and serialization, excluding controller validation pauses. Controller `validation_s` and total worker `wall_s` are recorded separately. A timeout kills the worker process group on POSIX. Linux/macOS are supported; Windows process control is not implemented.

## Run records and staging

`--out` selects a parent directory; each run creates a unique child. `run.json` records task/contract versions, source SHA-256 hashes and snapshots, Git revision/dirty status, case parameters, command and measurement settings. `traces.jsonl` records every case's status, per-call checks, raw timing samples, worker environments and memory peaks. `summary.json` distinguishes requested workloads from successfully scored workloads. Worker configs and stdout/stderr logs remain under `workers/`; operational paths in configs are relative to their config file. Original command/environment paths remain as historical provenance. Arrays are removed incrementally unless `--keep-outputs` is requested; the first reference is removed at case completion. A passing selected subset is not a full-task certification. Exit codes: 0 for all selected cases passing, 1 for evaluation failure, 2 for invalid CLI/configuration.

Snapshots include the selected registry entry, task files, evaluator, root dependency specifications and declared candidate sources/artifacts. Export a completed protocol version 2 or 3 run to a **new directory outside that run**, or repeat its exact cases and settings:

```bash
python -m harness export --run runs/RUN_ID --out /tmp/socialscibench-bundle
python -m harness replay --run /tmp/socialscibench-bundle --out runs/replays
```

Export verifies source hashes and preserves traces, logs and metadata, including failures; transient output arrays are excluded. After moving a bundle to another machine, install the recorded candidate dependencies and native build requirements, then run the replay command from its `snapshot/` directory with `--run ..` and an output parent outside the bundle. Replay uses the saved evaluator code and settings and creates a new run with newly observed timings and environment metadata. It never installs dependencies automatically or overwrites the original run. Old protocol version 1 runs remain historical records; they lack the portable snapshot required by these commands.

```bash
python -m harness stage --task 001 --out /tmp/socialscibench-001
```

This exports the task allowlist plus evaluator, public fixtures, dependencies and agent instructions, excluding reports, historical baselines and prior solutions. Use the exported directory as an agent's workspace. Filesystem/network isolation and orchestration of agent research remain the responsibility of the external runner. This evaluator is intended for trusted research code; process separation is not a hostile-code sandbox.
