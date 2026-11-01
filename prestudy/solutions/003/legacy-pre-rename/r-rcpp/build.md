# Environment and build

Runs in the shared task 008 environment `solutions/008/.venv` (see
`../bigram-table-scan/build.md`; `requirements.lock` is its freeze).

R 4.1.2 (`R_HOME`, default `/Library/Frameworks/R.framework/Resources`) with data.table 1.18.6.1,
stringi 1.8.9 and Rcpp 1.1.2 from the 14 October 2026 study's R library (`SOCIALSCIBENCH_R_LIBS`,
default `repro-socialscibench/research/2026-10-14/R-library`, used read only). rpy2 3.6 needs R
4.5+, hence 3.5.17. data.table is set to one thread (`setDTthreads(1)`); stringi and R's reference
BLAS are single-threaded. The Rcpp variant compiles its comparator with `Rcpp::cppFunction` at
import (Apple clang 17.0.0), excluded from timing.
