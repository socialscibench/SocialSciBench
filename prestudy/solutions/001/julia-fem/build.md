# Environment and build

Runs in the shared comparison environment `solutions/001/.venv` (git-ignored), used for every
task 001 candidate in the same-session comparison, so the reference and all candidates share
one interpreter: CPython 3.12.13, NumPy 2.3.5, SciPy 1.18.1 (`requirements.lock` is that
environment's freeze). Steps, from `solutions/001`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --no-build -r eliminate-largest-fe/requirements.lock \
    within-py==0.4.0 cmake==4.4.4 setuptools==84.0.0 wheel==0.48.0 packaging==26.3
uv pip install --python .venv/bin/python juliacall rpy2==3.5.17
```

xhdfe 2.28.1.20260929 is built from the release tarball kept by the study
(`repro-socialscibench/research/2026-10-14/native-extension/task001/downloads/`, SHA-256
4ee258a0aa85bccc67f672fdce2efa8cef62bc1e75966bdaa8f894244c848cef), extracted to a fresh
directory and installed with `uv pip install --no-build-isolation` using the study's CMake
arguments from `build_xhdfe.py` (CUDA, Metal and -march=native off; OpenMP required, linked to
that study's `downloads/openmp/lib/libomp.dylib`), Apple clang 17.0.0.

R 4.1.2 with fixest 0.14.2 comes from the study's R library
(`SOCIALSCIBENCH_R_LIBS`, default `repro-socialscibench/research/2026-10-14/R-library`); rpy2 3.6
needs R 4.5+, hence 3.5.17. Julia 1.13.1 and FixedEffectModels 2.1.0 come from the study's
Julia download and project; juliacall installs PythonCall into `.venv/julia-depot`, stacked in
front of the study's depot, which is read only.

Run the evaluator with `solutions/001/.venv/bin/python -m harness evaluate ...` from the
repository root. These are competitors for a same-session comparison, not optimized candidates.
