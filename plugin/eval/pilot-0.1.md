# Pilot: turbo-socialsci 0.1.0 on the three real-paper kits

**Where:** the development host: Mac mini, i7-8700B, 6 cores, 32 GB, macOS 15.7.3.

**Design:**
- **Agent:** Claude Code 2.1.304, model `claude-opus-5-5`, headless mode.
- **Runs:** one run with and one without the plugin per kit, for 6 runs. The same prompt (`kits.json`), a 2-hour limit, and only copies of the kit's `task/` and `data/`.
- **Plugin:** version 0.1.0, frozen before the pilot. This repository carries its successor, 0.2.0.
- **Isolation (`run_agent.py`, `probe_isolation.py`):**
  - Claude Code's sandbox confines Bash, with no unsandboxed retries and a strict network allowlist of package indexes;
  - deny rules keep the file tools out of the home directory and this session's scratch area, and keep the run without the plugin out of the plugin copy;
  - bypassPermissions, the user's synced skills turned off, and no MCP servers or web tools.
- **Run records:** transcripts, audits, reruns and grades are in `~/SocialSciBench-kits/agent-runs/`, outside the repository.

## Results

**Agent time** runs to the end of the session; **cost** is Claude Code's estimate.

**Correct** counts the cells passing the kit's own `reference/grade.py`:
- 001: column × size cells;
- 002: call × scale cells;
- 003: sizes.

The first figure grades the agent's own results files, the second the rerun's.

**Rerun:** the agent's one command, rerun alone on the host after the thermal gate, with 4 threads.

| Kit | Plugin | Agent time | Turns | Est. cost | Correct | Rerun, end to end | Peak memory |
|---|---|---|---|---|---|---|---|
| 001 Kelly et al. (reghdfe) | with | 27.7 min | 14 | $0.50 | 15/15, 15/15 | 237 s (3 reps) | 16.4 GiB |
| | without | 32.6 min | 47 | $2.46 | 15/15, 15/15 | 17 s (3 reps) | 3.5 GiB |
| 002 JEL DiD guide (csdid) | with | 7.6 min | 12 | $0.47 | 35/35, 35/35 | 322 s (3 reps) | 3.6 GiB |
| | without | 27.6 min | 67 | $3.95 | 35/35, 35/35 | 17 s (1 rep) | 1.5 GiB |
| 003 DCA to Capital IQ (reclink2) | with | 8.5 min | 37 | $0.69 | 5/5, 5/5 | 297 s (3 reps) | 7.2 GiB |
| | without | 16.3 min | 28 | $1.28 | 0/5, 0/5 | 186 s (3 reps) | 1.0 GiB |

**Seconds per estimate at full size, from each code's own timing file** (median of the rerun's repetitions):

| Kit, estimate | With the plugin | Without |
|---|---|---|
| 001 column 8 (348,338 assignee × year groups) | 16.3 s | 0.20 s, with sorting and packing the data before the clock (11.0 s, once) |
| 001 columns 1 and 7 | 19.5 s | 0.20 s and 0.24 s, same boundary |
| 002 call 7 at 100 copies (25,000 draws) | 14.1 s | 1.39 s |
| 002, all 35 calls | 106.6 s | 10.2 s |
| 003 at 100% | 26.6 s | 41.9 s (incorrect) |
| 003 at 1% | 16.6 s | 1.13 s (incorrect) |

For context, Stata on the collaborator's 12-core Mac (from the kits' notes, not measured here): 001 column 8 at 100% took 752 s; 002 took 3–5 h for all calls; 003 took about 5 h at 100%.

## Findings

1. **Agent effort: lower with the plugin in every kit.**
   - **Cost:** without the plugin, cost was 4.9× (001), 8.4× (002) and 1.9× (003) higher.
   - **Time:** without the plugin, time was 1.2×, 3.6× and 1.9× longer.
   - **Behavior:** with the plugin, each agent found and invoked the matching skill unprompted. Without it, the agents wrote 400–600-line implementations, two of them with native code: C, and Numba.
2. **Correctness: the plugin prevented one failure.**
   - Without the plugin, the 003 agent misread the definition's coefficient `0.679 × (bi − .6)^(1/3), where 0.679 = .5 / .4^(1/3)` as the literal 0.679. Every score was off by up to 4.4e-5, against a tolerance of 1e-6, and one pair was lost at 100%: 0 of 5 sizes.
   - The agent saw the ambiguity and noted it, but could not check it.
   - Writing the coefficient as `.5/.4^(1/3)` in the kit's task definition would remove the trap.
3. **Compute: the plugin's code was slower than the agents' own on two kits.**
   - **001:** the data have one fixed effect and sparse 0/1 dummies. The agent wrote an exact within transformation, with cross-products of the dummies packed into bits. The plugin's general multi-fixed-effect solver builds a dense float64 copy of the 6M × 131 regressors in every call, and its speed core keeps a second one. Its "auto" variant estimated 13 GB and peaked at 16.4 GiB.
   - **002:** with ±1 multipliers, the agent tabulated each block of 8 units' sums over all 256 sign patterns and added them in float32 with AVX2. That is about 10× faster than the plugin's approach of drawing the multipliers, then multiplying matrices.
   - **003:** the plugin re-prepares the 2.78M-record register in every call: string conversion, an ASCII check and lookups. That is a fixed cost of about 16 s per call, so the plugin is slower at small sizes and faster at full size.

## What this suggests for version 0.2

These are measured bottlenecks. None needs a change to the computation:
- 001: detect a single fixed effect and binary regressors (exact demeaning, sparse cross-products), and avoid the per-call float64 copy.
- 002: a bit-packed Rademacher bootstrap, the tabulation above.
- 003: a prepared-register API, so the register is converted and indexed once.

## Deviations and caveats

- **Sample size:** one run per arm per kit, one model and one host. Agent runs vary, so these are case studies, not estimates of averages.
- **Shakedown run:** run 002-with-1 used the first settings, acceptEdits; it took 4.6 min, an estimated $0.51 and passed 35/35. Three Bash commands were refused there because, with Read() deny rules, acceptEdits needs approval for commands whose reads it cannot resolve, and nobody can approve. The six pilot runs then used bypassPermissions with the strict allowlist.
- **Timing boundaries:** these differ. The 001 agent without the plugin sorted and packed the data before its clock (11.0 s, once); the kit allows only building the groups untimed. The end-to-end rerun times have no such boundary.
- **Repetitions:**
  - 001 without the plugin: rerun with `./run.sh 3` (its default is 5), to match the with-plugin rerun;
  - 002 without the plugin: its command ran one repetition, so its end-to-end time covers one repetition and a self-check;
  - the with-plugin reruns ran three.
- **Throttling:** the host throttled during some reruns, for example 002 with the plugin: 25 of 64 samples, down to a 54% speed limit. Each rerun record lists its throttling.
- **Stata:** Stata was installed but unusable without packages, because SSC downloads were blocked. No run invoked it; two runs looked for it, and the 003 agent tried to download reclink2's source.
