# Baseline: did-r (task 002b)

Baseline: R did att_gt and aggte (simple, group, calendar, dynamic within the window) via rpy2 with the task's comparison group and weights; est_method calls the DRDID panel estimator of the method (drdid_panel, reg_did_panel, ipw_did_panel, std_ipw_did_panel, drdid_imp_panel) with trim.level = 1; balanced panels only. Threads: data.table only.

The adapter's docstring (`solution.py`) gives the exact call and the conversions it performs.

## Sources (accessed 2026-10-23)

- did 2.5.1, https://bcallaway11.github.io/did/ (att_gt accepts a function as est_method:
  https://github.com/cran/did/blob/2.5.1/R/compute.att_gt.R); DRDID 1.3.0, https://psantanna.com/DRDID/
  (https://github.com/cran/DRDID/tree/1.3.0/R: drdid_panel, reg_did_panel, ipw_did_panel,
  std_ipw_did_panel, drdid_imp_panel and their trim.level arguments).

## Compatibility and measurements

(added after the comparison runs)
