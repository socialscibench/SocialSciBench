# Candidate implementations

Create a candidate with `python -m harness init-solution --task 001 --out solutions/001/my-candidate`. This copies the reference into a standalone, editable submission and writes `solution.json`, `research.md`, `build.md` and a starting `requirements.lock`. Update the lock and build instructions for the actual candidate environment.

Keep all implementation sources, dependency/build instructions and research notes with the candidate. Use separate names for experiments you want to preserve. The evaluator snapshots and hashes declared sources and artifacts on each invocation, including conventional research/build files when present; traces and logs go to unique directories under `runs/`, not here. Additional artifacts must be declared in `solution.json`. See [the evaluator documentation](../harness/README.md) and [agent workflow](../AGENTS.md).
