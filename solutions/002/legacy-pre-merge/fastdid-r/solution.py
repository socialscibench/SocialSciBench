"""Task 002 baseline: R fastdid (Tsai; fast implementation of Callaway & Sant'Anna), in-process via rpy2.

    fastdid(<long panel>, timevar = "t", cohortvar = "g", unitvar = "id", outcomevar = "y",
            control_option = "never", control_type = "dr", base_period = "varying",
            covariatesvar = c("x1", ...), result_type = "dynamic", full = TRUE)   # group-time + event study
    fastdid(..., result_type = "simple")                                          # simple ATT

Never-treated units are coded Inf (fastdid's convention). fastdid's doubly robust estimator follows
DRDID's formulas but does not trim comparison units with propensity scores above 0.995 and clips
scores at 1 - 1e-10 (the reference: 1 - 1e-6). The long panel is built inside the timed call. The
pre/post averages of the event study are the mean event-time effects and the influence functions'
mean, with standard errors computed like fastdid's own (checked against its event-time SEs).

Threads: data.table uses the evaluator's thread limit; fastdid's `parallel` option is off.
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

ro.r('suppressMessages({library(fastdid); library(data.table)})')
ro.r(r'''
fit_fastdid <- function(y, X, g, nthreads) {
  setDTthreads(nthreads)
  N <- nrow(y); Tn <- ncol(y); k <- ncol(X)
  gg <- as.numeric(g); gg[gg == 0] <- Inf
  dt <- data.table(id = rep.int(seq_len(N), Tn), t = rep(seq_len(Tn), each = N),
                   g = rep.int(gg, Tn), y = as.vector(y))
  xn <- NA
  if (k > 0) {
    xn <- paste0("x", seq_len(k))
    for (j in seq_len(k)) set(dt, j = xn[j], value = rep.int(X[, j], Tn))
  }
  d <- fastdid(dt, timevar = "t", cohortvar = "g", unitvar = "id", outcomevar = "y", control_option = "never",
               control_type = "dr", base_period = "varying", covariatesvar = xn, result_type = "dynamic",
               full = TRUE)
  s <- fastdid(dt, timevar = "t", cohortvar = "g", unitvar = "id", outcomevar = "y", control_option = "never",
               control_type = "dr", base_period = "varying", covariatesvar = xn, result_type = "simple")
  gte <- d$gt_estimate$y
  og <- order(gte$gt$G, gte$gt$time)
  ev <- d$estimate
  IF <- d$agg_inf_func$y
  if (nrow(IF) != N) IF <- t(IF)
  o <- order(ev$event_time); ev <- ev[o]; IF <- IF[, o, drop = FALSE]
  sef <- function(f) sqrt(sum(f^2)) / N
  chk <- max(abs(apply(IF, 2, sef) / ev$se - 1))
  pre <- ev$event_time < 0
  avg <- function(sel) if (!any(sel)) c(NA_real_, NA_real_) else c(mean(ev$att[sel]), sef(rowMeans(IF[, sel, drop = FALSE])))
  se_gt <- apply(gte$inf_func, 2, sef)
  list(cohort = gte$gt$G[og], time = gte$gt$time[og], att = as.vector(gte$att)[og], se = se_gt[og],
       simple = c(s[type == "post"]$att, s[type == "post"]$se),
       egt = ev$event_time, att_e = ev$att, se_e = ev$se, pre = avg(pre), post = avg(!pre), se_check = chk)
}
''')
_fit = ro.r["fit_fastdid"]


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
    if f("se_check")[0] > 1e-6:
        raise RuntimeError("fastdid influence functions do not reproduce its standard errors")
    gt = np.column_stack([f("cohort"), f("time")]).astype(np.int64)
    pre, post, simple = f("pre"), f("post"), f("simple")
    return dict(gt=gt, att_gt=f("att"), se_gt=f("se"), att_simple=float(simple[0]), se_simple=float(simple[1]),
                event_times=f("egt").astype(np.int64), att_event=f("att_e"), se_event=f("se_e"),
                att_pre_avg=float(pre[0]), se_pre_avg=float(pre[1]),
                att_post_avg=float(post[0]), se_post_avg=float(post[1]))
