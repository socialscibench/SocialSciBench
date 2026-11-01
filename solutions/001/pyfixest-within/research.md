# Baseline: pyfixest-within

Baseline: pyfixest.feols with the LSMR demeaner on the `within` Rust backend (additive Schwarz preconditioning, approximate-Cholesky local solvers); reghdfe df_a and small-sample factor computed in the adapter. Threads: RAYON_NUM_THREADS.

The adapter's docstring (`solution.py`) gives the exact call and the conversions it performs.

## Sources (accessed 2026-10-22)

- pyfixest 0.60.0, https://github.com/py-econometrics/pyfixest (feols, demeaner options)
- within 0.4.0, https://github.com/py-econometrics/within

## Compatibility and measurements

(added after the comparison runs)
