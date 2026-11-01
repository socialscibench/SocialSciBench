# Baselines for tasks 001, 002 and 003 (development study)

Existing implementations of each task's computation, wrapped as evaluator-compatible adapters
(`run` with the task's keyword arguments, outputs in the reference's form). They are
comparison points for the study's candidates, not optimized submissions. Task 002 is the
combined task (the former tasks 002, 002b and 002c merged by the package's author;
see 002_csdid/task.md); its adapters are new. The adapters of the former tasks 002 and 002b are
kept, unchanged, in `solutions/002/legacy-pre-merge/` with the runs that measured them. Adapters:

| Task | Directory | Implementation | Compatibility with the task contract |
|---|---|---|---|
| 001 | `solutions/001/pyfixest-within` | pyfixest 0.60.0 `feols`, LSMR demeaner on the `within` 0.4.0 backend (Rust) | exact computation; tolerance-dependent accuracy |
| 001 | `solutions/001/pyfixest-map` | pyfixest 0.60.0 `feols`, alternating projections (Rust) | exact computation; tolerance-dependent accuracy |
| 001 | `solutions/001/fixest-r` | R fixest 0.14.2 `feols` (rpy2) | exact computation; tolerance-dependent accuracy |
| 001 | `solutions/001/stata-reghdfe` | Stata 18 MP, reghdfe 6.14.1 (ftools 2.50.0) | the command the task specifies |
| 002 | `solutions/002/stata-csdid201` | Stata 18 MP, csdid 2.0.1 (github.com/pedrohcgs/csdid-stata, commit f98bcea), `bal(pair)` | the command the task's csdid201-rule workloads target; dripw/reg/stdipw only; bootstrap with Stata's own multipliers (not comparable) |
| 002 | `solutions/002/stata-csdid181` | Stata 18 MP, csdid 1.82 + drdid 1.91 (SSC), `long2` for the universal base (event window moved one period on the pre-treatment side, offsetting 1.82's long2 event pairing) | the command the task's csdid181-rule workloads target; bootstrap by `csdid_stats ..., wboot()` on the saved RIFs, with Stata's own multipliers (not comparable) |
| 002 | `solutions/002/did-r` | R did 2.5.1 + DRDID 1.3.0 (rpy2), the method's DRDID estimator with the task's trimming level, behind did's own overlap and conditioning checks | csdid201-rule (did's) rules; balanced panels and fixed covariates only; the universal base's reference cells dropped; bootstrap with did's own multipliers (not comparable) |
| 003 | `solutions/003/stata-reclink2` | Stata 18 MP, reclink2 (Stata Journal dm0082) | see its research.md |

Task 002's bootstrap standard errors and critical values depend on the multipliers. The task
passes them in (`boot_weights`); csdid and did draw their own. Their bootstrap outputs therefore
differ from the reference's by the bootstrap's sampling noise. compare.py validates everything
else and lists the bootstrap outputs as "not compared" (`--not-comparable se_boot_ crit_`); they
are still computed inside the timed call. csdid cannot report analytic and bootstrap standard
errors from one estimation, so the csdid adapters estimate twice on bootstrap workloads (an
analytic run and a run with `reps(B)`), both timed.

df_a and the reference's small-sample factor are not reported by fixest/pyfixest; those
adapters compute them (NumPy/SciPy/pandas, inside the timed call). The did/fastdid/moderndid
adapters form the event study's pre/post averages from the tools' influence functions.

## Environments (git-ignored, under `solutions/baselines/.venv`)

- **Python** `py/`: CPython 3.12.13 (Homebrew) with the candidates' pinned NumPy 2.3.5, SciPy
  1.18.1, pandas 3.0.6, pyarrow 26.0.0, threadpoolctl 3.7.0, Numba 0.62.1, plus rpy2 3.6.9 (ABI
  mode), pyfixest 0.60.0, within-py 0.4.0, moderndid 0.3.1, polars 2.0.0, statsmodels 0.15.0.
  Full list: `env/py.lock`.

  ```bash
  uv venv solutions/baselines/.venv/py --python /usr/local/bin/python3.12
  uv pip install --python solutions/baselines/.venv/py/bin/python -r solutions/baselines/env/py.lock
  ```

- **R** `r/`: conda-forge R 4.5.3 with fixest 0.14.2, did 2.5.1, DRDID 1.3.0, data.table
  1.18.6.1 (OpenMP), collapse 2.1.7, BMisc 1.4.10, dreamerr 1.5.0, OpenBLAS 0.3.34; fastdid 1.0.6
  from CRAN (source). Full list: `env/r-conda-list.txt`.

  ```bash
  mkdir -p solutions/baselines/.venv/bin
  curl -sL -o solutions/baselines/.venv/bin/micromamba \
    https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-osx-64
  chmod +x solutions/baselines/.venv/bin/micromamba   # sha256 1e71054b...
  export MAMBA_ROOT_PREFIX=$PWD/solutions/baselines/.venv/mamba
  solutions/baselines/.venv/bin/micromamba create -y -p solutions/baselines/.venv/r -c conda-forge \
    "r-base=4.5" r-fixest=0.14.2 r-did r-drdid r-data.table r-collapse r-bmisc r-dreamerr \
    r-ggplot2 r-stringr r-rcpp r-stringi
  solutions/baselines/.venv/r/bin/Rscript -e 'options(repos = c(CRAN = "https://cloud.r-project.org")); install.packages("fastdid")'
  ```

  The R adapters set `R_HOME` to this environment (override with `SOCIALSCIBENCH_R_HOME`).
  The R and Stata adapters find `solutions/baselines/.venv` from `SOCIALSCIBENCH_BASELINE_VENV`
  (set by `compare.py`) or by searching the directories above their own file, so they also run
  from the snapshot copy that `harness evaluate` executes.

- **Stata**: Stata 18 MP at `/Applications/Stata 18` (pystata; override with
  `SOCIALSCIBENCH_STATA`). Packages (SHA-256):
  - reclink2 (Wasi & Flaaen, Stata Journal 15(3), dm0082) from
    `https://www.stata-journal.com/software/sj15-3/dm0082/`: `reclink2.ado` 5ed4aae0...4ad0b,
    `reclink2.sthlp` 6fba67c6...9be7e, in `.venv/stata-ado/`.
  - reghdfe 6.14.1 (08Jul2026), ftools 2.50.0 (09jan2026), require 1.4.0 (10apr2024): GitHub
    archives of `sergiocorreia/reghdfe` (master; zip ab3dc6a9...6d69), `sergiocorreia/ftools`
    (master; zip 026957a6...97e2) and `sergiocorreia/stata-require` (zip 6d59430e...c246); their
    `src/` files are copied into `.venv/stata-ado/hdfe/` (`reghdfe.ado` e9b96033...6abc).
  - csdid 1.82 / drdid 1.91 / csdid2 1.3: GitHub archive of `friosavila/stpackages` (main; zip
    124edc34...6a82), in `.venv/stata-src/csdid-dl/`.
    csdid 1.82 passes its own `pscoretrim` default (1: no trimming) to drdid, as the corrected task
    does (before the correction the adapter requested .995); csdid2 1.3 has no trimming option. csdid2 uses its shipped compiled
    Mata library `lcsdid.mlib` (SHA-256 1fb2f323...3187).

  The adapters add only these directories to the ado path and point PLUS/PERSONAL at a
  private temporary directory, so no user-installed packages are used. Inputs are written to a
  temporary .dta file inside the timed call (pystata's `nparray_to_data` was 20 times slower).

## Measurements

`compare.py` runs an adapter through the evaluator's own worker (`harness.worker`: fresh
process per case, the same loaders, read-only inputs, first call, warmups and measured calls,
`perf_counter` around `run`, thread environment variables and threadpool limits as the
evaluator sets them) and validates every output with the task's validator against a saved
reference output, so the reference is not recomputed for each baseline. `make_refs.py`
creates those reference outputs (one call of `reference.run` per case, one BLAS thread, in
`solutions/.venv`, the candidates' environment):

```bash
solutions/.venv/bin/python solutions/baselines/make_refs.py 001 104k50 REFDIR/001
solutions/baselines/.venv/py/bin/python solutions/baselines/compare.py --task 001 \
  --solution solutions/001/pyfixest-within/solution.json --cases 101 104 s9 --threads 1 \
  --warmups 0 --repeats 1 --timeout 3600 --refdir REFDIR/001 --out results-001.jsonl \
  --python solutions/baselines/.venv/py/bin/python
```

Adapters can also be run with `python -m harness evaluate` (which recomputes the reference;
use the baseline interpreter), e.g. `--smoke` for the development fixtures.

Host behavior: the development host (Mac mini, i7-8700B) throttles under sustained load (CPU
speed limit 52-60% within about 4 minutes of six-core load, 78-80% intermittently at one core) and
was forced into sleep three times in one day under six-core loads. From then on all
measurements ran through `thermal_queue.py` (a gate before each case: speed limit 100 and thermal
level at most 90 for 60 s; a monitor of the speed limit, thermal level and sleep times; attempts
that overlap a sleep or severe throttling are excluded and repeated; per-case statistics in
`thermal-cases.jsonl`), with the plans `plan-resume.json`, `plan-resume-2.json` and
`plan-overnight.json` in `runs/baselines-20261023/`.

The study's exact sequences: `queue-evaluator.sh` (the candidates' evaluator runs),
`queue-baselines.sh` (every baseline measurement, after the evaluator queue; set `REFS` to the
reference outputs from `make_refs.py`, one directory per task, and `OUT` to a results directory)
and `followup.sh` (checks that explain baseline failures, and reclink2 on 003/104;
`followup_reclink2.py` and `followup_csdid.py`). `summarize.py` tabulates run directories and
result files; `tables.py` prints the report's tables (`--per-baseline`: one table per adapter):

```bash
solutions/baselines/.venv/py/bin/python solutions/baselines/tables.py RESULTS_DIR RUN_DIR...
```

**The final study round** (the combined task 002, the minimal-memory candidates and the four-thread
runs) used:
- **Queue plans:** `runs/study-20261025/plan-1.json` and `plan-2.json`, run with
  `thermal_queue.py` (results, logs and per-case thermal statistics in the same directory).
- **`check_adapter.py`:** runs a task 002 adapter on the evaluator's development fixtures against
  the reference, with the evaluator's validator.
- **`report_tables.py`:** prints the report's tables from `runs/study-20261025/report-config.json`.
- **`ratios.py`:** per-case peak and time ratios between each task's memory and speed candidates,
  from the same config.

Its report is `development-study.md`. `optimizations.md` summarizes the speed and
memory optimizations of the six candidates, with their measured effects.

**Results:**
- earlier results: `runs/baselines-20261023/` (tasks 001 and 003, and task 002 before its
  correction), `runs/baselines-002v2/` (the former task 002 after the correction) and
  `runs/baselines-002b/`;
- the combined task 002's: `runs/study-20261025/`.

All are git-ignored.

Thread budgets: the worker sets OMP, OpenBLAS, vecLib, Rayon and Numba thread variables to
`--threads`; R packages read OMP_NUM_THREADS (fixest `nthreads`, data.table `setDTthreads`);
Stata MP uses `set processors N`.
