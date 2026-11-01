# Environment and build

Runs in the shared task 008 environment `solutions/008/.venv` (see
`../bigram-table-scan/build.md`; `requirements.lock` is its freeze). No native build; the
comparator is a Numba function compiled on the first call.
