"""Task 002b baseline: R did 2.5.1 with DRDID 1.3.0, in-process via rpy2, on the balanced workloads.

    att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = ~ x1 + ... + xk,
           data = <long panel>, control_group = "nevertreated" / "notyettreated",
           weightsname = "w" (weighted workloads), base_period = "varying",
           est_method = <DRDID estimator below>, bstrap = FALSE, cband = FALSE)
    aggte(<att_gt>, type = "simple" / "group" / "calendar" / "dynamic" (min_e, max_e = window))

est_method is a function calling the DRDID panel estimator of the task's method with
trim.level = 1 (no comparison unit is trimmed, as csdid and the task; DRDID's default trims at
0.995): dripw drdid_panel, reg reg_did_panel, ipw ipw_did_panel (Abadie's unnormalized weights),
stdipw std_ipw_did_panel (normalized; did's own "ipw"), drimp drdid_imp_panel. att_gt passes y1,
y0, D, covariates, i.weights and inffunc to it. Unbalanced panels are not supported: did either
drops units with missing periods or, with allow_unbalanced_panel, uses repeated cross-section
estimators, while csdid uses every unit observed in both periods of a cell (pair balance); the
adapter raises for them. The pre- and post-period averages of the event study are formed from the
dynamic aggregation's influence functions (as in the task 002 adapter). Building the long panel
and the transfers to R are inside the timed call.

Threads: data.table uses the evaluator's thread limit; did itself is single-threaded here.
"""
import os

import numpy as np

def _venv():
    """The baseline environments: SOCIALSCIBENCH_BASELINE_VENV (set by compare.py), else
    solutions/baselines/.venv of the repository above this file (also from an evaluator run's snapshot)."""
    if os.environ.get("SOCIALSCIBENCH_BASELINE_VENV"):
        return os.environ["SOCIALSCIBENCH_BASELINE_VENV"]
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, "solutions", "baselines", ".venv")):
        if os.path.dirname(d) == d:
            raise RuntimeError("solutions/baselines/.venv not found above " + os.path.abspath(__file__))
        d = os.path.dirname(d)
    return os.path.join(d, "solutions", "baselines", ".venv")


R_HOME = os.path.join(_venv(), "r", "lib", "R")
os.environ["R_HOME"] = os.environ.get("SOCIALSCIBENCH_R_HOME", os.path.abspath(R_HOME))
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import numpy2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402

ro.r('suppressMessages({library(did); library(data.table)})')
ro.r(r'''
est002b <- list(
  dripw = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...)
    DRDID::drdid_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc, trim.level = 1),
  reg = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...)
    DRDID::reg_did_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc),
  ipw = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...)
    DRDID::ipw_did_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc, trim.level = 1),
  stdipw = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...)
    DRDID::std_ipw_did_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc, trim.level = 1),
  drimp = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...)
    DRDID::drdid_imp_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc, trim.level = 1))
fit002b <- function(y, X, g, w, control, method, lo, hi, nthreads) {
  setDTthreads(nthreads)
  N <- nrow(y); Tn <- ncol(y); k <- ncol(X)
  g <- as.integer(g)
  long <- data.table(id = rep.int(seq_len(N), Tn), t = rep(seq_len(Tn), each = N),
                     y = as.vector(y), g = rep.int(g, Tn))
  if (!is.null(w)) set(long, j = "w", value = rep.int(w, Tn))
  xn <- character(0)
  if (k > 0) {
    xn <- paste0("x", seq_len(k))
    for (j in seq_len(k)) set(long, j = xn[j], value = rep.int(X[, j], Tn))
  }
  xf <- if (k > 0) as.formula(paste("~", paste(xn, collapse = " + "))) else ~1
  a <- att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = xf, data = long,
              control_group = if (control == "notyet") "notyettreated" else "nevertreated",
              weightsname = if (is.null(w)) NULL else "w", base_period = "varying",
              est_method = est002b[[method]], bstrap = FALSE, cband = FALSE, panel = TRUE,
              print_details = FALSE)
  s <- aggte(a, type = "simple", bstrap = FALSE, cband = FALSE)
  gr <- aggte(a, type = "group", bstrap = FALSE, cband = FALSE)
  ca <- aggte(a, type = "calendar", bstrap = FALSE, cband = FALSE)
  e <- aggte(a, type = "dynamic", min_e = lo, max_e = hi, bstrap = FALSE, cband = FALSE)
  IFe <- e$inf.function$dynamic.inf.func.e
  pre <- e$egt < 0
  avg <- function(sel) {
    if (!any(sel)) return(c(NA_real_, NA_real_))
    f <- rowMeans(IFe[, sel, drop = FALSE])
    c(mean(e$att.egt[sel]), sqrt(mean(f^2) / N))
  }
  list(group = a$group, t = a$t, att = a$att, se = a$se, simple = c(s$overall.att, s$overall.se),
       gid = gr$egt, att_g = gr$att.egt, se_g = gr$se.egt, gavg = c(gr$overall.att, gr$overall.se),
       cid = ca$egt, att_c = ca$att.egt, se_c = ca$se.egt, cavg = c(ca$overall.att, ca$overall.se),
       egt = e$egt, att_e = e$att.egt, se_e = e$se.egt, pre = avg(pre), post = avg(!pre))
}
''')
_fit = ro.r["fit002b"]


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5)):
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("unbalanced panels: did does not estimate csdid's pair-balanced cells")
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    N = y.shape[0]
    X = np.asarray(X, dtype=np.float64).reshape(N, -1)
    with localconverter(ro.default_converter + numpy2ri.converter):
        conv = ro.conversion.get_conversion().py2rpy
        args = [conv(np.asfortranarray(y)), conv(np.asfortranarray(X)), conv(np.asarray(gvar).astype(np.int32)),
                ro.NULL if w is None else conv(np.asarray(w, dtype=np.float64))]
    r = _fit(*args, control, method, int(event_window[0]), int(event_window[1]), threads)
    f = lambda name: np.asarray(r.rx2(name), dtype=np.float64)
    gt = np.column_stack([f("group"), f("t")]).astype(np.int64)
    pre, post, simple, gavg, cavg = f("pre"), f("post"), f("simple"), f("gavg"), f("cavg")
    return dict(gt=gt, att_gt=f("att"), se_gt=f("se"), att_simple=float(simple[0]), se_simple=float(simple[1]),
                group_ids=f("gid").astype(np.int64), att_group=f("att_g"), se_group=f("se_g"),
                att_group_avg=float(gavg[0]), se_group_avg=float(gavg[1]),
                calendar_periods=f("cid").astype(np.int64), att_calendar=f("att_c"), se_calendar=f("se_c"),
                att_calendar_avg=float(cavg[0]), se_calendar_avg=float(cavg[1]),
                event_times=f("egt").astype(np.int64), att_event=f("att_e"), se_event=f("se_e"),
                att_pre_avg=float(pre[0]), se_pre_avg=float(pre[1]),
                att_post_avg=float(post[0]), se_post_avg=float(post[1]))
