# turbo-socialsci 0.2: the engine on the three real-paper kits

**Where:** the development host: Mac mini, i7-8700B, 6 cores, 32 GB, macOS 15.7.3. Python 3.12.13 with NumPy 2.3.5, Numba 0.62.1, pandas 3.1.0 and pyarrow 26.0.0, the versions the pilot's agents installed.

**What was measured:** the engine's own calls on each kit's data, made the way the pilot's agents made them (`bench_engine.py`), without an agent:
- **Versions:** 0.1, the frozen wheel (tag `turbo-socialsci-v0.1.0`), and 0.2, this branch's sources.
- **Calls:**
  - 001: `feols` for columns 1, 7 and 8;
  - 002: `att_gt` for the seven calls, 25,000 draws;
  - 003: `reclink`.
- **Protocol:**
  - 4 threads;
  - each call in a fresh process after `warmup()`, with 3 repetitions (v0.1's memory variant at 100 copies of kit 002: 1);
  - before each process, a thermal gate that waits for 30 s of unthrottled samples.
- **Seconds:** the median over the repetitions, from inputs in memory to the result.
- **Memory:** the process's resident memory sampled every 5 ms; the increase over the call's start, in the first repetition.

The kits' graders were not run. The kits' rule is to grade an implementation only after it is frozen; correctness here is agreement with 0.1, whose results passed every grader in the pilot.

## Results

At full size, with 4 threads, median of 3 repetitions; memory is the increase during the call:

- **001**, one regression on 6.0 million rows:
  - time: 15–23 s fall to 0.49–0.65 s, 23–47× faster;
  - memory: 7.6–12.3 GiB fall to 0.30–0.43 GiB.
- **002**, the seven calls at 100 copies:
  - speed variant: 55 s fall to 14.6 s in all, or 12 s when each call bootstraps only its reported table;
  - memory variant: 328 s fall to 26 s;
  - memory per call: the speed variant uses 0.27–1.22 GiB (0.1: 0.37–1.60 GiB) and the memory variant 0.21–0.42 GiB (0.1: 0.25–0.71 GiB).
- **003**, linking DCA records to Capital IQ:
  - time: 16 s fall to 1.2 s at 1%, and 24 s to 9.0 s at 100%;
  - memory: 3.7–3.8 GiB fall to 1.2–1.3 GiB in the speed variant and 0.42–0.57 GiB in the memory variant.

In the tables, bold marks 0.2's result; in parentheses, how many times faster and leaner it is than 0.1. No measurement was taken while the CPU was throttled.

### 001: Table A.2 columns at 100% (6.0M rows; column 8: 348,338 assignee × year groups) and 10%

| Size | Column | Speed, 0.1 | Speed, 0.2 | Memory, 0.1 | Memory, 0.2 |
|---|---|---|---|---|---|
| 100% | 1 | 18 s, 12.14 GiB | **0.527 s**, 0.35 GiB (34×, 34×) | 23 s, 9.50 GiB | **0.489 s**, 0.30 GiB (47×, 32×) |
| 100% | 7 | 18 s, 12.32 GiB | **0.588 s**, 0.43 GiB (30×, 29×) | 23 s, 9.57 GiB | **0.575 s**, 0.32 GiB (40×, 30×) |
| 100% | 8 | 15 s, 9.85 GiB | **0.645 s**, 0.33 GiB (23×, 29×) | 19 s, 7.57 GiB | **0.607 s**, 0.30 GiB (32×, 25×) |
| 10% | 1 | 1.93 s, 1.25 GiB | **0.0857 s**, 0.05 GiB (23×, 27×) | 2.19 s, 0.75 GiB | **0.137 s**, 0.03 GiB (16×, 28×) |
| 10% | 7 | 2 s, 1.24 GiB | **0.0841 s**, 0.05 GiB (24×, 26×) | 2.29 s, 0.77 GiB | **0.106 s**, 0.03 GiB (22×, 22×) |
| 10% | 8 | 1.36 s, 0.99 GiB | **0.0988 s**, 0.04 GiB (14×, 28×) | 1.79 s, 0.68 GiB | **0.121 s**, 0.04 GiB (15×, 19×) |

### 002: the seven csdid calls at 100 copies (2,604 × 100 counties; 25,000 draws)

| Call | Speed, 0.1 | Speed, 0.2 | Speed, 0.2, reported table only | Memory, 0.1 | Memory, 0.2 |
|---|---|---|---|---|---|
| 1 | 6.05 s, 0.37 GiB | **1.16 s**, 0.27 GiB (5.2×) | **1.1 s**, 0.25 GiB (5.5×) | 25 s, 0.25 GiB | **1.43 s**, 0.21 GiB (18×) |
| 2 | 7.29 s, 1.22 GiB | **1.59 s**, 0.92 GiB (4.6×) | **1.45 s**, 0.89 GiB (5.0×) | 36 s, 0.56 GiB | **2.06 s**, 0.34 GiB (18×) |
| 3 | 6.92 s, 1.16 GiB | **1.69 s**, 0.92 GiB (4.1×) | **1.47 s**, 0.90 GiB (4.7×) | 36 s, 0.58 GiB | **2.2 s**, 0.32 GiB (16×) |
| 4 | 6.93 s, 1.20 GiB | **1.8 s**, 0.94 GiB (3.9×) | **1.5 s**, 0.92 GiB (4.6×) | 40 s, 0.59 GiB | **2.52 s**, 0.35 GiB (16×) |
| 5 | 8.97 s, 0.54 GiB | **2.52 s**, 0.46 GiB (3.6×) | **2.54 s**, 0.44 GiB (3.5×) | 48 s, 0.34 GiB | **4.7 s**, 0.27 GiB (10×) |
| 6 | 8.58 s, 0.51 GiB | **2.37 s**, 0.49 GiB (3.6×) | **1.57 s**, 0.46 GiB (5.5×) | 55 s, 0.31 GiB | **4.34 s**, 0.30 GiB (13×) |
| 7 | 9.97 s, 1.60 GiB | **3.44 s**, 1.22 GiB (2.9×) | **2.32 s**, 1.25 GiB (4.3×) | 86 s, 0.71 GiB | **9.02 s**, 0.42 GiB (10×) |

All seven calls, total seconds:

| Copies | Speed, 0.1 | Speed, 0.2 | Speed, 0.2, reported table only | Memory, 0.1 | Memory, 0.2 |
|---|---|---|---|---|---|
| 1 | 0.825 s | 0.445 s | 0.371 s | 2.57 s | 0.72 s |
| 10 | 5.45 s | 1.64 s | 1.4 s | 26.4 s | 2.68 s |
| 100 | 54.7 s | 14.6 s | 12 s | 328 s | 26.3 s |

### 003: DCA → Capital IQ (2,775,671 register records)

| Master records | Speed, 0.1 | Speed, 0.2 | Memory, 0.1 | Memory, 0.2 |
|---|---|---|---|---|
| 419 (1%) | 16 s, 3.69 GiB | **1.19 s**, 1.18 GiB (13.1×, 3.1×) | 16 s, 3.71 GiB | **1.3 s**, 0.42 GiB (12.2×, 8.8×) |
| 4,183 (10%) | 17 s, 3.71 GiB | **2.15 s**, 1.26 GiB (7.8×, 2.9×) | 17 s, 3.71 GiB | **2.66 s**, 0.47 GiB (6.4×, 7.8×) |
| 41,824 (100%) | 24 s, 3.78 GiB | **8.99 s**, 1.32 GiB (2.7×, 2.9×) | 24 s, 3.78 GiB | **10 s**, 0.57 GiB (2.3×, 6.7×) |

## Correctness

- **Against 0.1 on the kits:** every size, call and variant (`compare_engine.py`):
  - **001**, 30 runs: N and the dropped regressors are the same. Coefficients differ by at most 1e-10 relative, standard errors by 1.3e-12, R² by 1.2e-11. The kit's tolerance is 1e-6.
  - **002**, 70 runs: estimates and analytic SEs are identical. Bootstrap SEs and critical values differ by at most 3.4e-15, since the multipliers are the same. With each call bootstrapping only its reported table, those tables differ by at most 1e-14.
  - **003**, 10 runs: the same pairs and the same scores, bit for bit.
- **Against the frozen candidates on the benchmark's development inputs** (`tools/parity.py`, tasks 002 and 003; task 001's cores are unchanged): 92 comparisons, no mismatch; 78 bit-identical, the rest within 6.9e-14, the frozen code's own BLAS variation.
- **Tests:** the plugin's 69 tests and the repository's 64 pass.

## What changed

| Kit | Bottleneck in 0.1 (pilot and profiles) | 0.2 |
|---|---|---|
| 001 | Each call converted the 131 regressors to a dense float64 matrix twice, then demeaned it with the general solver: 10–12 GiB and 15–18 s at full size. Its first call also imported scipy.stats (1.6 s). | One fixed effect is handled exactly by group sums, with no iterative solver (`core/hdfe_onefe.py`). Columns are read in place; 0/1 columns become lists of each row's ones, so 129 dummies cost about 1.6 entries per row. Group codes come from a bitmap of the values present. |
| 002 | The bootstrap formed 25,000 multipliers per unit as floating-point numbers and multiplied matrices: 7.6 of 9.7 s for call 7 at 100 copies. The memory variant drew them in NumPy: up to 86 s. | `core/_signs.py`: each 8 units' 256 signed sums are tabulated, and each draw adds the entry its 8 multipliers select (as bits), with explicit 4-double vector additions. Both variants use it, with the same multipliers as 0.1. `boot_tables=` bootstraps only the reported table; without the group-time table, the speed core draws the aggregates' own influence functions. |
| 003 | Text became NumPy arrays of 4-byte characters, and a Python loop checked each distinct value for ASCII: about 16 s and 3.7 GiB before any linking. | Text is read from Arrow's buffers as bytes (`_strings.py`): 2-D byte rows in the speed variant; in the memory variant, the using file's characters and offsets read in place, with smaller record numbers and block buffers. |

## Caveats

- **Scope:** these are engine timings. An agent's script adds its own reading, subsetting and output. The pilot's per-estimate times came from the agents' timing files.
- **001:** with one fixed effect both variants run the same computation. "speed" first copies the sample into the groups' order; that helped at 10%, but at full size it was 1–8% slower than "memory".
- **002:** the event-study calls are fastest with `boot_tables=["event"]`. The skill now shows it, but an agent may not pass it; the default bootstraps every table.
- **Memory:** the increase is measured while the call runs; the caller's tables are not counted. In the DiD memory variant, 0.2 loads Numba's runtime for the bootstrap, about 65 MB.
- **One host,** with 4 threads; the kits' Stata timings are from the collaborator's 12-core Mac.
