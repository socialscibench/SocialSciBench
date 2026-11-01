# turbo-socialsci 0.2 on the three real-paper kits: grading and three agent runs per arm

**Where:** the development host: Mac mini, i7-8700B, 6 cores, 32 GB, macOS 15.7.3.

**Plugin:** frozen at tag `turbo-socialsci-v0.2.0`. The runs installed it from a local copy whose 28 files match the frozen release's manifest.

**Two parts:**
- the kits' own graders on the frozen engine's results, every cell and both variants;
- the pilot's agent comparison (`pilot-0.1.md`) repeated with 0.2, three runs with and three without the plugin per kit.

**In-sample:** 0.2's changes came from the pilot on these same three kits, and its speed was tuned and measured on their data (`engine-0.2.md`). The speed results below show what 0.2 does on the workloads it was built for. How it does on other data is untested.

## The frozen engine, graded

The frozen 0.2.0 wheel was installed in a fresh environment, as the skills install it (NumPy 2.3.5, Numba 0.62.1, pandas 3.1.0, pyarrow 26.0.0, SciPy 1.18.1). It ran every kit cell in both variants: `bench_engine.py`, one repetition, 4 threads. `grade_engine.py` wrote the results in the kits' formats and graded them with each kit's `reference/grade.py`.

| Kit | Cells | Speed variant | Memory variant |
|---|---|---|---|
| 001 | column × size | 15/15 | 15/15 |
| 002, every table bootstrapped | call × scale | 35/35 | 35/35 |
| 002, only the reported table (`boot_tables=`) | call × scale | 35/35 | 35/35 |
| 003 | sizes | 5/5 | 5/5 |

- **003:** at every size the pairs are the reference's, with none missing or extra; scores are within 1.1e-16.
- **Same results as the measured sources:** the frozen wheel reproduces the sources measured in `engine-0.2.md`. 001 and 003 are bit for bit the same; 002's bootstrap results are within 1e-14, from summation order across threads.

## Agent runs

**Design:** as in the pilot:
- Claude Code 2.1.304, model `claude-opus-5-5`, headless;
- the same prompt (`kits.json`), isolation (`run_agent.py`) and 2-hour limit;
- only copies of the kit's `task/` and `data/`.

**Runs:** three per arm per kit, 18 in all, one at a time, in three rounds. Each round ran the kits in order. The run with the plugin went first in rounds 1 and 3, the run without it in round 2.

**Two settings were added:**
- `caffeinate -dims` around each run and rerun from the replacement run `001-without-r1b` on; see the first deviation. The first run, `001-with-r1`, and its rerun ran without it and finished before the host slept;
- `DISABLE_AUTOUPDATER`, which kept Claude Code at the pilot's version.

**Reruns and grading:** as in the pilot. Each agent's one command was rerun alone after the thermal gate, at 4 threads, and both the agent's results and the rerun's were graded.

**Records:** transcripts, audits, reruns, grades and summaries (runs `*-r1`, `*-r1b`, `*-r2`, `*-r3`) and the coefficient check (finding 2) are kept outside the repository, because some of them hold results from the licensed linkage data or the kits' answers.

### Effort and correctness

The table gives the median over the three runs, with the range in parentheses. The "pilot" rows are the pilot's single runs.

**Correct** counts the cells passing the kit's grader, in the agent's own results files and in the rerun's: 001, column × size cells; 002, call × scale cells; 003, sizes.

| Kit | Plugin | Agent time | Turns | Est. cost | Correct, every run |
|---|---|---|---|---|---|
| 001 | 0.2 | 2.8 min (2.7–3.0) | 11 (10–11) | $0.37 (0.37–0.44) | 15/15, 15/15 |
| | none | 34.8 min (31.0–44.5) | 39 (37–46) | $1.84 (1.69–2.25) | 15/15, 15/15 |
| | 0.1, pilot | 27.7 min | 14 | $0.50 | 15/15, 15/15 |
| | none, pilot | 32.6 min | 47 | $2.46 | 15/15, 15/15 |
| 002 | 0.2 | 3.1 min (3.1–3.8) | 12 (12–15) | $0.39 (0.38–0.41) | 35/35, 35/35 |
| | none | 20.6 min (20.5–22.7) | 51 (47–53) | $3.15 (3.09–3.44) | 35/35, 35/35 |
| | 0.1, pilot | 7.6 min | 12 | $0.47 | 35/35, 35/35 |
| | none, pilot | 27.6 min | 67 | $3.95 | 35/35, 35/35 |
| 003 | 0.2 | 5.1 min (3.8–10.0) | 15 (9–20) | $0.44 (0.28–0.65) | 5/5, 5/5 |
| | none | 10.2 min (9.6–15.0) | 17 (16–33) | $1.07 (1.03–1.82) | 0/5, 0/5 |
| | 0.1, pilot | 8.5 min | 37 | $0.69 | 5/5, 5/5 |
| | none, pilot | 16.3 min | 28 | $1.28 | 0/5, 0/5 |

### Compute

**Seconds per estimate at full size:** each code's own timing file in the rerun gives the median over its repetitions; the table gives the median over the three runs, with the range.

| Kit, estimate | With 0.2 | Without | 0.1, pilot |
|---|---|---|---|
| 001 column 8 (348,338 assignee × year groups) | 0.81 s (0.76–1.51) | 0.26 s (0.11–0.28) | 16.3 s |
| 001 columns 1 and 7 | 0.54 s and 0.59 s | 0.21 s and 0.24 s | 19.4 s and 19.5 s |
| 001, all 15 cells (sum) | 3.2 s (3.2–4.2) | 1.4 s (0.66–1.8) | 77 s |
| 002 call 7 at 100 copies (25,000 draws) | 2.58 s (2.54–2.58) | 1.18 s (1.15–2.25) | 14.1 s |
| 002, all 35 cells (sum) | 17.8 s (17.5–17.9) | 8.4 s (7.6–12.3) | 107 s |
| 003 at 100% | 8.75 s (8.29–8.82) | 60 s (7.8–61), incorrect | 26.6 s |
| 003 at 1% | 0.67 s (0.64–0.78) | 0.66 s (0.24–0.96), incorrect | 16.6 s |

**The 0.1 column is inflated for 002 and 003:** the pilot's reruns were throttled on 25 of 64 samples (002, down to 54% speed) and 14 of 60 (003, down to 60%); 001's on 1 of 48. Gated, 0.1 takes 15 s for 001's column 8 and 18 s for columns 1 and 7, 9.97 s for 002's call 7, and 24 s and 16 s for 003 at 100% and 1% (`engine-0.2.md`). The findings compare with these.

**Peak memory of the rerun:** the summed resident memory of the command's process tree, end to end, over every estimate.

| Kit | With 0.2 | Without | 0.1, pilot |
|---|---|---|---|
| 001 | 3.4 GiB (3.3–4.8) | 2.7 GiB (2.1–2.9) | 16.4 GiB |
| 002 | 3.2 GiB (3.1–3.2) | 1.4 GiB (1.0–1.4) | 3.6 GiB |
| 003 | 2.1 GiB (2.0–2.1) | 0.8 GiB (0.4–1.4) | 7.2 GiB |

## Findings

1. **Agent effort: with the plugin, every agent invoked the matching skill (9 of 9) and finished in 2.7–10 minutes, for $0.28–0.65.**
   - **Without the plugin:** 9.6–44.5 minutes and $1.03–3.44.
   - **Ratio of medians, without to with:** time 12× (001), 6.6× (002) and 2.0× (003); cost 4.9×, 8.1× and 2.5×.
   - **Against 0.1:** agent time with the plugin fell from 27.7 to 2.8 minutes (001), 7.6 to 3.1 (002) and 8.5 to 5.1 (003). Part of this is the code the agents ran: one pass over every 001 estimate, with three repetitions, took 237 s with 0.1 and 14–18 s with 0.2 (the reruns).
   - **`boot_tables=`:** the three DiD agents with the plugin all passed it for the reported table, as the skill shows.
2. **Correctness: all runs with the plugin passed every cell, in their own results and in the reruns.**
   - **Without the plugin:** every 001 and 002 run passed; every 003 run failed all five sizes, as in the pilot.
   - **The cause, every time:** the agent took the definition's coefficient, `0.679 × (bi − .6)^(1/3), where 0.679 = .5 / .4^(1/3)`, as the literal 0.679. Two of the three agents said so in their notes; one named `.5/.4^(1/3)` as the alternative.
   - **Confirmed after the study:** in copies of the three programs, changing only that coefficient to `.5/.4^(1/3)` makes all three pass all five sizes, with scores within 5e-10.
   - **Across the pilot and this study,** 4 of 4 agents without the plugin fell into this. Writing the coefficient as `.5/.4^(1/3)` in the kit's task definition would remove the trap.
   - **Not an independent test of correctness:** the plugin's linkage core was validated against the benchmark's task 003 reference implementation. That is the computation the kit's grader uses: the kit was written for an earlier version of task 003. So the plugin passes partly by construction. It has the coefficient the grader expects, which the agents had to infer from the text.
3. **Compute: 0.2 closed most of 0.1's gap.**
   - **Speed per estimate, against 0.1's gated timings:** 19–33× faster on 001, 3.9× on 002's call 7, and 2.7× (100%) to 24× (1%) on 003. Against the pilot's throttled reruns these would read 20–36×, 5.5× and 3–25×.
   - **Peak memory, against 0.1:** 16.4 to 3.4 GiB on 001, and 7.2 to 2.1 GiB on 003.
   - **Against the agents' own code:**
     - 001 and 002: the plugin is still 2.5–3× slower per estimate on 001 and about 2× slower on 002;
     - 003: at 100% it was about 7× faster than two of the three agents' programs (C, about 60 s) and about 12% slower than the third (C++ with exact pruning, 7.8 s). The three were wrong only in the coefficient. With it corrected, the third took 9.0 s at 100% and 0.31 s at 1% in one ungated repetition: about as fast as the plugin at full size and about twice as fast at 1%. On 003, the plugin's advantage over the agents' code is correctness, not speed.
4. **Where the remaining gaps come from:**
   - **001:**
     - **Before the clock:** two of the three agents without the plugin sorted the rows by group before the clock, one also building a compressed layout of the dummies.
     - **Inside the clock:** the agents with the plugin passed the engine a DataFrame; it builds the column-8 groups inside the call. One agent also copied the rows with an assignee inside its timer, which took column 8 from about 0.8 s to 1.5 s.
   - **002:**
     - all three agents without the plugin compiled their bootstrap: one in C++ through Rcpp, two in C with 8-wide AVX vectors. All three used 8-unit sign tables, in single precision;
     - the engine works in double precision. On its own (frozen wheel, 4 threads, median of 3), call 7 at 100 copies takes 2.33 s, of which 1.30 s comes before the bootstrap: the cell estimates, influence functions and analytic SEs.

## What this suggests for 0.3

- **002:** single-precision sign tables, flushed to double precision periodically (the agents' choice); less work outside the bootstrap.
- **001:** the column-8 groups and the assignee filter in one pass over the data.
- **003:** exact pruning by an upper bound on the score. One agent's program scored only about 3% of candidates in full.
- **Memory:** the DiD speed variant's working memory, up to 1.2 GiB at call 7. The agents' scripts also hold every copy and subset at once.

## Deviations and caveats

- **Excluded run, `001-without-r1`:**
  - **What happened:** the host entered macOS's "Low Power Sleep" at 00:30 and slept until 10:04, apart from 45-second maintenance wakes. That happened although Owly held an idle-sleep assertion.
  - **Effect on the run:** its 8.4 hours include about 8.0 hours asleep; the agent was awake for about 28 minutes. Claude Code's 10-minute command timeouts count sleep, so two of the agent's commands timed out as the host woke. The 2-hour limit never fired, because `run_agent.py`'s timer (Python's monotonic clock) does not count sleep.
  - **Its code:** it passed 15/15; column 8 in the rerun took 0.25 s.
  - **Replacement:** `001-without-r1b`. All later runs and reruns ran under `caffeinate -dims`, and the kernel recorded no sleep after 10:04.
- **Repetitions:**
  - one repetition in four reruns with the plugin: 002 runs 1–3 and 003 run 1, whose scripts had no repetition option;
  - three in the other reruns: `3` was passed where a script's default was 5 (001 without the plugin, runs 2 and 3, as in the pilot) or 1 (002 without the plugin, run 3);
  - so on 002, every rerun with the plugin timed one repetition and every rerun without it three. A single repetition can include first-call costs, so the 002 times with the plugin may be slightly high. Gated, the engine alone takes 2.32 s for call 7 (`engine-0.2.md`), against 2.54–2.58 s in the reruns.
- **Timing boundaries:**
  - **001 without the plugin:** runs 1b and 3 sorted the rows by group before the clock, as the pilot's agent did; run 3 also built a compressed layout of the dummies. Run 2 sorted inside it. The kit allows building the groups before the clock; sorting is less clear.
  - **001 with the plugin:** run 1 copied the rows with an assignee inside its timer.
  - **003 without the plugin:** all three converted the files to flat binary before the clock, as reading is untimed.
- **Throttling:** the two longest reruns, 003 without the plugin runs 1 and 3 (about 4.4 minutes each), were throttled in 16 and 11 of 53 samples, down to 63–65% speed. No other rerun in this study was throttled. The pilot's 0.1 reruns were: see the note under the compute table.
- **Isolation gap, also in the pilot:**
  - **The gap:** all runs shared Claude Code's temporary directory, `/tmp/claude-502` (the agents' `$TMPDIR`), which the sandbox left readable and writable. Earlier runs' code and results were in it, including the pilot's incorrect 003 matches.
  - **Checked in all 26 transcripts** (19 here, 7 in the pilot): every temporary file a run used was first written by that run. No run listed the directory or opened another run's task outputs, so no run saw another's files.
  - **Shared package cache:** four runs without the plugin kept a uv package cache there, which can shorten later runs' installs.
  - **Fix:** future runs should each get their own `TMPDIR`.
  - **The audits' other flags** were the agents' own temporary files and path-like text in code, such as `//` comments.
- **Compilation:** agent time with the plugin includes compiling the engine's kernels on first use, about 45–80 s.
- **Sample size:** three runs per arm, one model and one host. The 0.1 rows are the pilot's single runs.
- **Cost of the study:** $26.68 estimated: $3.72 with the plugin, $19.38 without it, and $3.57 for the excluded run.
