# 001: Fixed-effects regression — baselines and results

`CPU` · `estimator` · `L1` · `Stata reghdfe 6.12.3 · R fixest 0.14.2` · `workload set v2`

The task itself (computation, inputs, outputs, workloads, tolerances) is defined in [task.md](../../001_reghdfe/task.md), which is written to be given to a model as the task prompt. This file holds what is kept out of the prompt: baseline timings, agreement checks, notes on existing tools, and the October 2026 implementation study.

## Baseline timings (Apple M3 Pro)

Workload sizes are listed in [task.md](../../001_reghdfe/task.md#workloads).

| # | design | n_obs | n_regressors | mobility | Stata reghdfe (s) | R fixest (s) | Reference (s) |
|---|---|---|---|---|---|---|---|
| 1 | firm_year | 100,000 | 3 | – | 0.15200 | 0.06094 | 0.10604 |
| 3 | firm_year | 5,000,000 | 20 | – | 14.85600 | 1.85859 | 9.48110 |
| 4 | worker_firm | 1,000,000 | 20 | 10% | 17.55100 | 1.57228 | 6.56919 |
| 6 | gravity | 447,000 | 3 | – | 1.02500 | 0.09102 | 0.77691 |
| 7 | four_way | 2,000,000 | 20 | – | 13.10400 | 1.04316 | 10.14501 |
| 8 | worker_firm | 1,000,000 | 20 | 2% | 50.88400 | 8.26356 | 17.36380 |
| 9 | worker_firm | 1,000,000 | 20 | 0.5% | 127.23000 | 11.83408 | 40.15174 |
| 10 | firm_year | 1,000,000 | 50 | – | 9.74800 | 1.31059 | 2.70238 |
| 11 | worker_firm | 10,000,000 | 20 | 0.5% | 2828.57500 | 160.88785 | 711.78236 |
| 12 | firm_year | 10,000,000 | 40 | – | 342.58400 | 30.15275 | 97.25933 |

- **Stata reghdfe (s)** is the baseline: Stata/MP 19.5 (StataNow) with a 12-core license and reghdfe 6.12.3 / ftools 2.49.1, on an Apple M3 Pro with 18 GB of RAM. Stata's `timer` has 1 ms resolution, so the last two of the five decimals are always 0. The timer wraps only the `reghdfe` call; data loading is excluded, and a warm-up call loads the Mata library first. Each time is a single run.
- **R fixest (s)** is `fixest::feols(y ~ x1 + … | fe1 + fe2 + …, cluster = ~cl)` with fixest's default options, R 4.6.1 with fixest 0.14.2 on 12 threads, data loaded with `data.table::fread` before timing. It has the lowest runtime on every workload: 6–18× faster than reghdfe and 2–10× faster than the reference beyond the 100,000-row sanity check. See the notes below for where its results miss the tolerances. Results are in `../baselines/001_reghdfe/results/results_fixest.jsonl`.
- **Reference (s)** is `reference.py` on the same machine, timed with `time.perf_counter()`.

**Workload set v2** changes from v1: regressors raised to 20 in workloads 3, 4, 7, 8 and 9; workload 5 retired (it duplicated 8 except for size; id 2 was retired earlier for duplicating 1 and 3); long-running workloads 11 and 12 added; baselines re-timed on Stata 19.5.

## Agreement with Stata

The reference matches Stata reghdfe on every workload:

- `nobs` and `df_a` are identical.
- `beta` agrees to within 5e-7 relative error.
- Standard errors agree to within 2e-7 relative error.

The largest gaps are on workload 11, where convergence is hardest. With reghdfe's default tolerance (1e-8), its own coefficients are accurate only to about 1e-7 there, so the 1e-6 tolerance on `beta` should not be tightened.

Results are in `../baselines/001_reghdfe/results/results.jsonl`.

## Notes on existing tools

**fixest.** On the low-mobility workloads (9, 11), its coefficients miss the 1e-6 tolerance (up to 1e-05 relative error): its default tolerance for removing the fixed effects (`fixef.tol = 1e-6`) is too loose when the worker–firm network is poorly connected. Elsewhere its coefficients match the reference to 1e-10 or better. Its standard errors differ by up to 2e-05 on the gravity and four-way designs because it counts the degrees of freedom absorbed by nested fixed effects differently from reghdfe.

**Stata 19's built-in `areg, absorb()`.** Since Stata 19, `areg`, `xtreg, fe` and `ivregress 2sls` accept several variables in `absorb()`, using unaccelerated alternating projections. It is not a baseline here, for three reasons:

- It computes the same coefficients but counts absorbed degrees of freedom differently, so its standard errors do not match reghdfe's.
- **Its default limit is 50 iterations.** That is enough for balanced panels but not for poorly connected data.
- With the limit raised, it was slower than reghdfe in the one test below.

One test, on the v1 version of workload 9 (1M rows, 3 regressors, 0.5% mobility), Stata/MP 19.5 with 12 cores:

| Command | Time (s) | Converged | Result |
|---|---|---|---|
| `reghdfe ..., absorb(fe1 fe2 fe3) vce(cluster cl)` | 37.1 | yes (313 iterations) | baseline |
| `areg ..., absorb(fe1 fe2 fe3) vce(cluster cl)` | 4.9 | **no**: "Warning: Convergence not achieved" | coefficients wrong from about the 5th digit; SEs about 0.6% off |
| `areg ..., absorb(fe1 fe2 fe3) vce(cluster cl) iterate(100000) dfabsorb` | 49.5 | yes | coefficients match; SEs about 0.25% off (different df convention) |

## Implementation study (14 October 2026)

Measured against benchmark commit `25f279d` on an Intel Core i7-8700B Mac (6 physical cores, 12 logical CPUs, 32 GiB RAM; macOS 15.7.3).

The fastest qualifying implementation depends on the design. Exact absorbed-degree counts and clustered covariance matter alongside solver speed: matching coefficients alone does not satisfy this task.

Both rounds cover six of ten active workloads: W1, W4, W6, W7, W9 and W10. W3, W8, W11 and W12 remain unmeasured. Not every package was tested on every workload.

Main tables request **one compute thread** and report seconds, with measured repetitions in parentheses. Batches ran serially on an active desktop without locked CPU frequency or temperature. Generation, imports, startup, file transport and correctness comparison are excluded. **Each round has its own reference timings and its own warmup and adapter policy: do not compute speedups across rounds or against the M3 Pro table.**

### Round 1: R and Python

Python 3.12.13; R 4.1.2; fixest 0.14.2; PyFixest 0.60.0; lfe 3.1.1; PyHDFE 0.2.1. The R fixest build lacked OpenMP. Packages were warmed on small separate fixtures.

These are **strict adapted totals**: package fit medians plus separately measured input conversion, covariance extraction and optimized exact small-sample correction (SSC). They are composed measurements, not independently timed complete calls. References and PyHDFE's full custom estimator wrapper are integrated timings.

| Workload | Reference | R fixest default | PyFixest Rust MAP default | PyFixest within |
|---|---:|---:|---:|---:|
| W1 | 0.394 (3) | 0.066 (3) | 0.106 (3) | 0.169 (3) |
| W6 | 2.637 (3) | 0.498 (3) | 0.481 (3) | 1.642 (3) |
| W4 | 26.809 (3) | 12.875 (3) | 82.058 (1) | 22.236 (1) |
| W9 | 137.760 (1) | **Fails accuracy** | Not run | 13.762 (4) |
| W7 | 28.616 (1) | 9.449 (3) | 7.652 (3) | 41.938 (1) |
| W10 | 8.117 (1) | 8.764 (3) | 5.815 (3) | 8.651 (1) |

fixest and Rust MAP defaults use tolerance `1e-6`; the PyFixest within path uses `1e-10` and its bundled within 0.2.0 solver. Solver tolerances are not interchangeable accuracy guarantees. W9 within pools all four observations across two blocks; its native range was 12.692–16.316 s. The adapted 13.762 s result is approximately 10.0× faster than this round's single reference observation.

Additional challengers did not win the measured cases: the pruned lfe adapter took 0.222 s on W1 and 0.956 s on W6 (three fits each). PyHDFE symmetric-CG with integrated fast counts took 0.134 s on W1, 1.110 s on W6 (three each), and 196.718 s on W4 (one).

Native fixest/PyFixest counts fail W6 and W7: absorbed degrees are 5,998 versus the required 5,979, and 3,507 versus 3,489. The table charges adapters that restore the reference convention. Native lfe also needs SSC alignment and recursive singleton pruning. These corrections are calculated from input data, not copied from an oracle.

On W9, default fixest completed a 97.121 s native fit but failed coefficient accuracy (`1.14e-5` relative error versus `1e-6`). fixest at `1e-8` and `1e-10`, and pruned lfe at its default tolerance, reached a **300 s whole-process bound**, with no completed fit result. Tight Rust MAP on W4 was interrupted without persisted numerical output. These cases are not ranked. W9 default MAP and PyHDFE were not run.

### Round 2: C++/Rust and Julia

xhdfe **2.28.1.20260929**, standalone within-py **0.4.0**, and Julia **1.13.1 / FixedEffectModels 2.1.0 / FixedEffects 3.5.0**. Each isolated process had a full-workload warmup; a warmup exceeding 60 s reduced measured repetitions to one.

These are **integrated complete-estimator times**. xhdfe includes integer-label encoding and native estimation; its native outputs already satisfy the contract, so a redundant SSC audit is excluded from the ranked time. Julia includes table construction and required SSC correction. within includes encoding, singleton pruning, Rust residualization, OLS, clustered covariance and exact counts.

| Workload | Reference | xhdfe default | Julia, qualifying setting | within default |
|---|---:|---:|---:|---:|
| W1 | 0.214 (3) | 0.035 (3) | 0.053 (3) | 0.032 (3) |
| W6 | 1.725 (3) | 0.237 (3) | 0.492 (3) | 0.353 (3) |
| W4 | 18.272 (3) | 8.349 (3) | 6.991 (3) | 11.161 (3) |
| W9 | 137.426 (1) | 6.143 (3) | 117.220 (1), tight | 11.983 (3) |
| W7 | 27.035 (3) | 5.888 (3) | 6.848 (3) | 8.564 (3) |
| W10 | 7.295 (3) | 3.867 (3) | 2.008 (3) | 3.578 (3) |

xhdfe and standalone within use default tolerance `1e-8`; Julia defaults to `1e-6`. Default Julia W9 reported convergence but failed coefficient accuracy (`2.14e-5` relative error), taking 26.439 s. The separately tested `1e-10` setting passed at 117.220 s. Its correction still needs the reference's absorbed-degree convention. All 180 small adapted fixtures passed; xhdfe native outputs passed all 60 of its fixtures and all six full workloads after label encoding.

xhdfe W9 is approximately **22.4× faster** than this round's reference. A separate supplement verified **six active OpenMP workers** on every fit: W4 **5.018 s**, W9 **3.513 s**, W7 **3.779 s** (three repetitions each), respectively 1.66×, 1.75× and 1.56× faster than its one-thread results. All pass native and adapted correctness checks. Standalone within 0.4.0 and PyFixest's bundled 0.2.0 use different wrappers/settings, so the rounds do not isolate a version effect.

### Correctness and provenance

Qualifying results require coefficient relative error ≤ `1e-6`, SE error ≤ `1e-4`, exact retained observations/absorbed degrees/cluster counts, and convergence. Full covariance errors are retained as diagnostics; the specification provides no separate off-diagonal tolerance. The adapters preserve the executable reference's specialized rule beyond two nonnested effects, not a general proof of dummy-matrix rank.

Full reports, raw observations, ranges, memory measurements, source hashes, dependency locks and reproduction code are in the local companion repository `repro-socialscibench`, a sibling of this repository (`../../repro-socialscibench/research/2026-10-14/` from this folder): see `task001/RESULTS.md` and `native-extension/task001/RESULTS.md`. It retains archive commit **`af01b9c`** and has not been published.

Primary implementation documentation: [fixest covariance conventions](https://lrberge.github.io/fixest/articles/standard_errors.html), [PyFixest solver backends](https://pyfixest.org/how-to/demeaner-backends.html), [pinned xhdfe release](https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.28.1.20260929).

## Same-session comparison (18 October 2026)

A new candidate, `solutions/001/eliminate-largest-fe`, and four existing implementations wrapped as candidates (`xhdfe`, `within`, `julia-fem`, `fixest`, adapters following the 14 October study) were run through the shared evaluator (protocol v2) one after another, on the same Intel Core i7-8700B Mac (32 GiB, macOS 15.7.3) and in one Python environment. All ten active workloads are covered. Each evaluator run times its own reference.

`eliminate-largest-fe` eliminates the fixed effect with the most levels in closed form and solves the reduced normal equations on the remaining levels: dense, Jacobi-preconditioned CG, or an exact sparse factorization when the remaining graph is forest-like (low mobility). It makes two passes over the data with Numba kernels. Method, sources and failed attempts are in its `research.md`.

**Scope.** One thread; one warmup and three measured calls per workload (W11: one measured call); medians of complete `run` calls, including label encoding, singleton removal and, for R and Julia, copying the data into that language. Generation, imports and the first call are excluded. Tolerances are each package's defaults; fixest additionally drops singletons (`fixef.rm = "singleton"`). These timings are not comparable with the 14 October rounds or the M3 Pro table.

| Workload | Reference | eliminate-largest-fe | xhdfe | within | Julia FEM | R fixest | Best other / eliminate-largest-fe |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 0.329 | **0.0215** | 0.0342 | 0.0373 | 0.0719 | 0.133 | 1.59 (xhdfe) |
| W3 | 30.25 | **1.906** | 9.968 | 8.533 | 7.078 | 12.84 | 3.71 (Julia) |
| W4 | 27.19 | **1.013** | 8.435 | 10.14 | 7.207 | 11.11 | 7.11 (Julia) |
| W6 | 1.840 | 0.320 | **0.224** | 0.410 | 0.508 | 0.304 | 0.70 (xhdfe) |
| W7 | 24.71 | **1.558** | 5.644 | 8.262 | 7.253 | 8.178 | 3.62 (xhdfe) |
| W8 | 80.01 | **1.360** | 7.758 | 10.00 | 17.09 | 55.94 | 5.71 (xhdfe) |
| W9 | 119.8 | **0.477** | 6.130 | 9.780 | Fails accuracy | Fails accuracy | 12.9 (xhdfe) |
| W10 | 7.120 | **0.502** | 3.789 | 3.916 | 2.985 | 8.384 | 5.94 (Julia) |
| W11 | 1856 | **4.589** | 1819 | 141.0 | Fails accuracy | Fails accuracy | 30.7 (within) |
| W12 | 69.38 | **3.787** | 49.36 | 33.40 | 23.91 | 96.46 | 6.31 (Julia) |

Seconds. The reference column is from the `eliminate-largest-fe` runs; the other four runs measured the reference within 10% of it except on W1, W3, W4 and W8, where they measured 0.22–0.27, 22.1–22.4, 17.4–17.9 and 47.8–49.6 s. The last column is the fastest passing competitor's median divided by `eliminate-largest-fe`'s, so it does not depend on the reference. Per-candidate ranges, first calls, errors and memory are in each candidate's `research.md`.

- `eliminate-largest-fe` passes every workload and is fastest on nine of ten. On W6 (gravity) xhdfe is 1.43× faster.
- Julia FixedEffectModels (`1e-6`) and fixest defaults fail coefficient accuracy on W9 (2.1e-5 and 1.1e-5 relative error) and W11 (1.1e-6 and 3.4e-6), as in the 14 October study's W9. Tightened settings were not run in this session.
- xhdfe on W11 took 1,819 s per call (peak RSS 18.6 GB), about the reference's time; the cause was not investigated.
- On W9 and W11 all passing candidates differ from the reference by the same amount (5.5e-8 and 4.1e-7), consistent with the reference's own solver error; on W11 the recorded reference-versus-Stata gap is 4.13e-7.

**Interpretation.** The table measures this task's ten workloads at one thread; it does not rank general-purpose libraries.

- `eliminate-largest-fe` was designed in an informed session after inspecting these workloads, and its solver-selection thresholds were chosen on them. The competitors are general-purpose and untuned for them. Its speed on other structures is unmeasured. The approach is expected to gain little when several fixed effects are large and comparably sized, and an exact or dense reduced system can be slower than iterative demeaning or exhaust memory there. Iterative demeaning is what general-purpose packages default to, because its memory use and behavior are predictable on any structure.
- The candidate computes only the task's outputs. The library times include their own data handling, checks and extra results, and for fixest and Julia, copying the data into R or Julia.
- One thread hides the competitors' multithreading: xhdfe was 1.56–1.75× faster with six threads in the 14 October supplement. The candidate is not multithreaded either. xhdfe was built without CPU-specific optimization, and the R installation's fixest lacks OpenMP.
- Failing defaults are reported as failures; tighter Julia and fixest settings would pass W9 and W11 more slowly (Julia at `1e-10`: 117 s on W9 in the 14 October study).
- The same author wrote the candidate and the competitor adapters; the adapters follow the 14 October study's.

Fairer follow-ups: held-out workload designs evaluated blind (a benchmark change, so a separately scoped task); tightened Julia and fixest candidates on W9 and W11; a multithreaded round; and an AKM-specific competitor on the worker–firm workloads.

Run IDs, archive hashes and the batch commands are in `published/001/same-session-2026-10-18.md`.

## Reproduce

```bash
python baselines/001_reghdfe/bench.py --stata /Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp
python baselines/001_reghdfe/bench_fixest.py   # R fixest column
```

Run the reproduction commands from the repository root. New baseline runs default to unique directories under `runs/`. Historical raw measurements are preserved under `baselines/001_reghdfe/results/`.
