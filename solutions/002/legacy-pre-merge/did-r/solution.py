"""Task 002 baseline: R did (Callaway & Sant'Anna's package; DRDID's drdid_panel), in-process via rpy2.

    att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = ~ x1 + ... + xk,
           data = <long panel>, control_group = "nevertreated", base_period = "varying",
           est_method = dr_notrim, bstrap = FALSE, cband = FALSE)
    aggte(<att_gt>, type = "simple" / "dynamic", bstrap = FALSE, cband = FALSE)

`dr_notrim` is DRDID::drdid_panel (what est_method = "dr" calls) with trim.level = 1: no comparison
unit is trimmed, as in Stata's csdid and the task (DRDID's own default trims scores at or above
0.995). att_gt passes y1, y0, D, covariates, i.weights and inffunc to a user-supplied function.

`did` needs a long panel (unit x period rows); building it from the wide inputs is part of the
timed call, as are the transfers. The pre- and post-period averages of the event study (Stata's
`estat event` Pre_avg/Post_avg, not reported by `did`) are formed from the dynamic aggregation's
influence functions: the mean of the event-time effects and of their influence functions, with
standard error sqrt(mean(IF^2) / n). Standard errors are the analytic influence-function ones.

Threads: data.table uses the evaluator's thread limit; `did` itself is single-threaded here
(its `pl`/`cores` option parallelizes only the bootstrap).
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
dr_notrim <- function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...) {
  DRDID::drdid_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE,
                     inffunc = inffunc, trim.level = 1)
}
fit002 <- function(y, X, g, nthreads) {
  setDTthreads(nthreads)
  N <- nrow(y); Tn <- ncol(y); k <- ncol(X)
  g <- as.integer(g)
  long <- data.table(id = rep.int(seq_len(N), Tn), t = rep(seq_len(Tn), each = N),
                     y = as.vector(y), g = rep.int(g, Tn))
  xn <- character(0)
  if (k > 0) {
    xn <- paste0("x", seq_len(k))
    for (j in seq_len(k)) set(long, j = xn[j], value = rep.int(X[, j], Tn))
  }
  xf <- if (k > 0) as.formula(paste("~", paste(xn, collapse = " + "))) else ~1
  a <- att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = xf, data = long,
              control_group = "nevertreated", base_period = "varying", est_method = dr_notrim,
              bstrap = FALSE, cband = FALSE, panel = TRUE, print_details = FALSE)
  s <- aggte(a, type = "simple", bstrap = FALSE, cband = FALSE)
  e <- aggte(a, type = "dynamic", bstrap = FALSE, cband = FALSE)
  IFe <- e$inf.function$dynamic.inf.func.e
  pre <- e$egt < 0
  avg <- function(sel) {
    if (!any(sel)) return(c(NA_real_, NA_real_))
    f <- rowMeans(IFe[, sel, drop = FALSE])
    c(mean(e$att.egt[sel]), sqrt(mean(f^2) / N))
  }
  list(group = a$group, t = a$t, att = a$att, se = a$se,
       simple = c(s$overall.att, s$overall.se), egt = e$egt, att_e = e$att.egt, se_e = e$se.egt,
       pre = avg(pre), post = avg(!pre))
}
''')
_fit = ro.r["fit002"]


def run(y, X, gvar):
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    y = np.asarray(y, dtype=np.float64)
    N = y.shape[0]
    X = np.asarray(X, dtype=np.float64).reshape(N, -1)
    with localconverter(ro.default_converter + numpy2ri.converter):
        args = [ro.conversion.get_conversion().py2rpy(a) for a in
                (np.asfortranarray(y), np.asfortranarray(X), np.asarray(gvar).astype(np.int32))]
    r = _fit(*args, threads)
    f = lambda name: np.asarray(r.rx2(name), dtype=np.float64)
    gt = np.column_stack([f("group"), f("t")]).astype(np.int64)
    pre, post, simple = f("pre"), f("post"), f("simple")
    return dict(gt=gt, att_gt=f("att"), se_gt=f("se"), att_simple=float(simple[0]), se_simple=float(simple[1]),
                event_times=f("egt").astype(np.int64), att_event=f("att_e"), se_event=f("se_e"),
                att_pre_avg=float(pre[0]), se_pre_avg=float(pre[1]),
                att_post_avg=float(post[0]), se_post_avg=float(post[1]))
