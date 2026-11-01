# Baseline: stata-csdid-fra

Baseline: Stata csdid (method dripw, never-treated controls, varying base period, pscoretrim(.995), analytic SEs) with estat simple and estat event, via pystata; panel passed as a temporary .dta file. Threads: Stata MP set processors.

csdid 1.82 declares `pscoretrim(real 1.0)` and passes it to drdid, so by default no comparison unit
is trimmed; drdid's own default (.995) is the task's rule, so the adapter requests it explicitly.

The adapter's docstring (`solution.py`) gives the exact commands and the conversions it performs.

## Sources (accessed 2026-10-22)

- csdid 1.82 and drdid 1.91 (F. Rios-Avila, P. H. C. Sant'Anna, B. Callaway), GitHub archive of
  https://github.com/friosavila/stpackages (main branch, zip SHA-256 124edc34...6a82); the SSC
  copies (fmwww.bc.edu/repec/bocode/c) could not be downloaded from this host.
- Callaway & Sant'Anna (2021); Sant'Anna & Zhao (2020).

## Compatibility and measurements

(added after the comparison runs)
