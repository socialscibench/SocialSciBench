# Baseline: stata-csdid2-fra

Baseline: Stata csdid2 (the Mata rewrite of csdid; method dripw, never-treated controls, `short`
= csdid's varying base period, analytic SEs) with `estat attgt`, `estat simple` and `estat event`,
via pystata; panel passed as a temporary .dta file. Threads: Stata MP `set processors`.

The task names this command as csdid's equivalent. csdid2 1.3's syntax has no propensity-score
trimming option and its Mata code (`csdid.mata`, `drdid.mata`) does not trim, so comparison units
with a propensity score above .995 keep their weight; where that rule binds (the development
trimming fixture), results differ from the task's.

The adapter's docstring (`solution.py`) gives the exact commands and the conversions it performs.

## Sources (accessed 2026-10-22)

- csdid2 1.3 (F. Rios-Avila; `csdid2.pkg` distribution date 20230414), GitHub archive of
  https://github.com/friosavila/stpackages (main branch, zip SHA-256 124edc34...6a82):
  `csdid2.ado` 916a2fb5...270b, `csdid2_estat.ado` f7c04157...9b16, `lcsdid.mlib`
  1fb2f323...3187 (the compiled Mata library shipped with the package).
- Callaway & Sant'Anna (2021); Sant'Anna & Zhao (2020).

## Compatibility and measurements

(added after the comparison runs)
