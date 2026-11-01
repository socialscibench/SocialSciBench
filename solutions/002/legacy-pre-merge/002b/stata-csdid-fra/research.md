# Baseline: stata-csdid-fra (task 002b)

Baseline: Stata csdid 1.82 with drdid 1.91 and the task's options (iweights, notyet, method), estat simple, group, calendar and event within the window, via pystata; long panel with one row per observed unit-period passed as a temporary .dta file. Threads: Stata MP set processors.

csdid 1.82 is the command the task targets (the reference matches csdid 1.81 on seven development and held-out workloads, per the package's author). No propensity-score trimming: csdid passes its default `pscoretrim(1)` to drdid.

The adapter's docstring (`solution.py`) gives the exact commands and the conversions it performs.

## Sources (accessed 2026-10-23)

- csdid 1.82, drdid 1.91 and csdid2 1.3 (F. Rios-Avila, P. H. C. Sant'Anna, B. Callaway), GitHub
  archive of https://github.com/friosavila/stpackages (main branch, zip SHA-256 124edc34...6a82),
  the same download as the task 002 adapters.
- The task package's `stata/run_002b.do` (the commands and options used for the package's Stata
  comparison).

## Compatibility and measurements

(added after the comparison runs)
