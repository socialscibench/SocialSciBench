# Environment and build

Runs in the shared task 002 environment `solutions/002/.venv` (see
`../shared-nuisance/build.md`; `requirements.lock` is its freeze).

R 4.1.2 (`R_HOME`, default `/Library/Frameworks/R.framework/Resources`) with did 2.5.1 and DRDID 1.3.0
from the 14 October 2026 study's R library (`SOCIALSCIBENCH_R_LIBS`, default
`repro-socialscibench/research/2026-10-14/R-library`, used read only). rpy2 3.6 needs R 4.5+,
hence 3.5.17. R's bundled reference BLAS is single-threaded.
