# Environment and build

Runs in the shared task 008 environment `solutions/008/.venv` (see
`../bigram-table-scan/build.md`; `requirements.lock` is its freeze).

`linkage.cpp` is compiled when `solution.py` is imported (excluded from timing), with the study's
command and flags,

```bash
clang++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -pthread -dynamiclib linkage.cpp -o liblinkage-<hash>.dylib
```

into `$SOCIALSCIBENCH_BUILD_CACHE` (default `<environment>/build-cache`), keyed by the SHA-256 of
the source and flags; an existing library with the same key is reused. Compiler: Apple clang
17.0.0 (clang-1700.6.4.2), target x86_64-apple-darwin24.6.0, the study's compiler. Source
SHA-256 `537a929443bb14db1c43798f46d9ed0226c43cc83d4acaf18cf02983d1e46dc4`, identical to the
study's `research/2026-10-14/native-extension/task008/linkage.cpp`.
