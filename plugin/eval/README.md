# Measuring the plugin's gain on the real-paper kits

The three speed-comparison kits on Google Drive each hold the following. Their folder IDs are in `~/SocialSciBench-kits/drive-folders.json`, outside the repository, because the `reference/` folders hold the answers.
- `task/TASK.md`, the specification in words;
- `data/`;
- `reference/`, with the answers, a grader and an implementation.

The measurement compares one agent with and without the plugin:
- **Agent:** Claude Code, model `claude-opus-5-5`, in headless mode.
- **Inputs:** only copies of the kit's `task/` and `data/`, with the same prompt (`kits.json`) in both runs.
- **The prompt** doesn't mention the plugin, so the result includes whether the agent picks it up.

## Protocol

The scripts and `kits.json` hold the study host's locations: `/Users/Shared/` for the runs and the plugin copy, `~/SocialSciBench-kits/` for the kits, and Claude Code's temporary directory for that host's user (`/private/tmp/claude-502`, in `run_agent.py` and `audit.py`). Change them for another host.

1. **Freeze the plugin first:** commit, tag, and a SHA-256 manifest outside the repository. `export_plugin.py TAG MANIFEST` copies the frozen plugin to `/Users/Shared/turbo-socialsci-plugin` and checks it against the manifest.
2. **Fetch `task` and `data`** with `fetch_kit.py KIT task data`, into `~/SocialSciBench-kits/KIT/`. Check the data against the kit's manifest. Fetch `reference` only after step 1.
3. **Run each agent** with `run_agent.py KIT with|without RUN`, one at a time with nothing else running:
   - **Workspace:** `/Users/Shared/turbo-socialsci-runs/KIT-ARM-RUN`, holding APFS clones of `task/` and `data/`.
   - **002 only:** its TASK.md gets the maintainer's clarification that "doubly robust" means csdid's dripw.
   - **Time limit:** 2 hours, then SIGINT.
   - **Sleep:** run each run and rerun under `caffeinate -dims`, and check `sysctl kern.sleeptime` afterwards. During the study the host went into "Low Power Sleep" during a run despite an idle-sleep assertion, and the time limit never fired, because its timer does not count sleep.
   - **Isolation,** since there's no container on this host:
     - Claude Code's sandbox lets Bash write only to the workspace, the uv cache and Claude Code's temporary directory. That directory, `/tmp/claude-502` (the agents' `$TMPDIR`), is shared by all runs and stays readable, a gap. The transcripts show no run used another's files (`agent-study-0.2.md`), but each run should get its own;
     - it blocks reads of the home directory, of this session's scratch area and, for the run without the plugin, of the plugin copy;
     - network access goes only to package indexes;
     - permission rules block the same paths for the Read and Edit tools;
     - WebFetch, WebSearch and MCP servers are off, and the user's synced skills are turned off.
4. **Audit:** `audit.py META` lists the tool calls, any path outside the workspace, sandbox denials and plugin use.
5. **Rerun:** `rerun.py RUN --cmd "<the agent's one command>"` reruns the agent's code from its workspace:
   - after the thermal gate, at 4 threads, with a 2-hour limit per estimate;
   - it records the process tree's peak memory and a timeline of result-file changes;
   - the agent's own result files are kept separately.
6. **Grade** the agent's results and the rerun's results with the kit's `reference/grade.py`, outside the workspace.
7. **Move the run** into `~/SocialSciBench-kits/agent-runs/` with `finish_run.py RUN` before the next run.

## What is measured

- **Correctness:** estimates passing each kit's grader.
- **Agent effort:** wall time to the end of the run, turns, tokens and estimated cost, from the transcript's result message.
- **Compute:** for the code each agent produced, seconds per estimate under the kits' timing protocol and peak memory, from the rerun. `run_times.py` reads the seconds per estimate from each run's timing files.

## Engine benchmarks without an agent

`bench_engine.py` times the engine's own calls on a kit's data, the way the pilot's agents called it, so engine versions can be compared directly:
- **Calls:** 001: `feols` for columns 1, 7 and 8; 002: `att_gt` for calls 1–7, at 25,000 draws; 003: `reclink`, at the kit's sizes. The inputs are built before the clock, as the agents built them.
- **Isolation:** each call runs in a fresh process, after `warmup()`, with `--reps` repetitions and the given thread count. `--gate 30` waits for 30 s of unthrottled samples before each process.
- **Memory:** the process's resident memory, sampled every 5 ms; the increase over the call's start is reported for the first repetition.
- **Versions:** `--python` and `--pythonpath` choose the engine, for example an installed 0.1 wheel or the repository's sources.
- **002:** `--tables` passes `boot_tables=` the table each call reports (0.2).

`compare_engine.py DIR_A DIR_B` compares the saved results of two runs: N, dropped regressors, coefficients, SEs and R² (001); estimates, analytic and bootstrap SEs and critical values (002); pairs and scores (003). The saved results come from licensed data: keep the output directories outside the repository.

`grade_engine.py KIT DIR OUT --variant V` writes a run's saved results in the kit's result formats (`task/TASK.md`) and grades them with the kit's `reference/grade.py`, as `grade_run.py` grades an agent's files. Use it only on a frozen engine.
