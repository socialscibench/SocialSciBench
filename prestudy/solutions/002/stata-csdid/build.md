# Environment and build

Runs in the shared task 002 environment `solutions/002/.venv` (see
`../shared-nuisance/build.md`; `requirements.lock` is its freeze) with pystata from the
installed Stata 18 MP (`SOCIALSCIBENCH_STATA`, default `/Applications/Stata 18`; licensed for
2 cores, set to 1 processor). csdid 2.0.0 is the study's vendored copy of the official release,
pedrohcgs/csdid-stata commit 31a577382d7c09757ed3baa095659f45cc63c107
(`SOCIALSCIBENCH_CSDID_PKG`, default `repro-socialscibench/research/2026-10-14/native-extension/
task002/vendor/pedrohcgs-csdid-stata-31a5773/pkg`), added to the ado path; PERSONAL, PLUS and
OLDPLACE point at an empty directory, so no installed ado files are used.
