# Environment and build

Runs in the shared task 008 environment `solutions/008/.venv` (see
`../bigram-table-scan/build.md`; `requirements.lock` is its freeze) with pystata from the installed
Stata 18 MP (`SOCIALSCIBENCH_STATA`, default `/Applications/Stata 18`; licensed for 2 cores, set to
1 processor).

reclink2 is not installed in this Stata. Its ado file was downloaded from the Stata Journal's
dm0082 package (SJ 15-3) into `solutions/008/.venv/stata-ado` (git-ignored;
`SOCIALSCIBENCH_RECLINK2` overrides the directory), which the adapter puts first on the ado path;
PERSONAL, PLUS and OLDPLACE point at an empty directory, so no other installed ado files are used:

```bash
mkdir -p solutions/008/.venv/stata-ado && cd solutions/008/.venv/stata-ado
curl -L -O https://www.stata-journal.com/software/sj15-3/dm0082/reclink2.ado \
        -O https://www.stata-journal.com/software/sj15-3/dm0082/reclink2.sthlp
shasum -a 256 reclink2.ado   # 5ed4aae093ffc2d74059a5fe47205441dbb76da641124d12b7855f1f3cc4ad0b
```

Downloaded 2026-10-19 (`*! last modified August 12,2014` in the file).
