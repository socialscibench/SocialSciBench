# Baseline: stata-csdid2-fra (task 002b)

Baseline: Stata csdid2 1.3 with the task's options and short (csdid's varying base period), estat attgt, simple, group, calendar and event within the window, via pystata; long panel with one row per observed unit-period passed as a temporary .dta file; no ipw. Threads: Stata MP set processors.

csdid2 1.3 with `short`. The task lists two known differences from csdid: the cohort-average (GAverage) standard error and the standard errors of weighted dripw; csdid2 has no `ipw`.

The adapter's docstring (`solution.py`) gives the exact commands and the conversions it performs.

## Sources (accessed 2026-10-23)

- csdid 1.82, drdid 1.91 and csdid2 1.3 (F. Rios-Avila, P. H. C. Sant'Anna, B. Callaway), GitHub
  archive of https://github.com/friosavila/stpackages (main branch, zip SHA-256 124edc34...6a82),
  the same download as the task 002 adapters.
- The task package's `stata/run_002b.do` (the commands and options used for the package's Stata
  comparison).

## Compatibility and measurements

(added after the comparison runs)
