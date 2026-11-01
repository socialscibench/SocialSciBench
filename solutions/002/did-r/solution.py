"""Task 002 baseline: R did 2.5.1 with DRDID 1.3.0 (Callaway & Sant'Anna), in-process via rpy2, for
the balanced workloads that follow csdid 2.0.1's rules (rules="csdid201"), which are R did's: the
not-yet-treated comparison units are untreated in both periods of a cell, the cohort average's
standard error includes the estimated cohort shares, and a cell whose unweighted propensity score
reaches 0.999 for some unit is not estimated.

    att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = ~ x1 + ... + xk,
           data = <long panel>, control_group = "nevertreated" / "notyettreated",
           weightsname = "w" (weighted workloads), base_period = "varying" / "universal",
           est_method = <DRDID estimator below>, bstrap = FALSE, cband = FALSE)
    aggte(<att_gt>, type = "simple" / "group" / "calendar" / "dynamic", na.rm = TRUE)

and, with bootstrap multipliers, the one estimation with bstrap = TRUE, biters = B, cband = TRUE,
the same aggregations again with bstrap = TRUE, and did's mboot() for the pre- and post-period
averages.

- **Estimators:** est_method calls the DRDID panel estimator of the task's method with the
  task's trimming level (`trim.level`; DRDID's default is 0.995): dripw is drdid_panel, reg is
  reg_did_panel, stdipw is std_ipw_did_panel (did's own "ipw"). 1.81's unnormalized ipw and
  drimp are not 2.0.1-rule estimators.
- **Cell checks:** did runs its overlap check (an unweighted score of 0.999) and its covariate
  conditioning check only before its built-in estimators, not before a function. The functions
  call the same checks (did:::overlap_check_fail, did:::rcond_check_fail) and stop with did's
  conditions, so did leaves those cells unestimated as it does for "dr", "ipw" and "reg".
- **Universal base:** did also returns the reference cells (g, g - 1) and event time -1, at 0;
  the task does not estimate them, so they are dropped (and left out of the averages).
- **Analytic standard errors with the bootstrap:** att_gt's bootstrap replaces the cells'
  standard errors, so the analytic ones are its formula on the stored influence functions,
  sqrt(diag(IF'IF / n) / n).
- **Unbalanced panels:** not supported. did either drops units with missing periods or, with
  allow_unbalanced_panel, uses repeated cross-section estimators, while csdid uses every unit
  observed in both periods of a cell.
- **Outputs from the influence functions:** the pre- and post-period averages, the event-time
  subsets and the event-study covariance are formed from the dynamic aggregation's influence
  functions.
- **Bootstrap:** did draws its own multipliers, so its bootstrap standard errors and critical
  values differ from the reference's (computed from the given multipliers) by the bootstrap's
  sampling noise. They are timed but not comparable (compare.py --not-comparable se_boot_ crit_).

Building the long panel and the transfers to R are inside the timed call. Threads: data.table
uses the evaluator's thread limit; did itself runs in one thread here.
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
## did checks a cell before its built-in "dr", "ipw" and "reg" estimators but not before an
## estimator given as a function, so these run did's own checks as the built-ins do: the 0.999
## overlap check (dr, ipw) and the control units' covariate conditioning (dr, reg). stop("overlap")
## and stop("singular") are the conditions did catches to leave the cell unestimated.
guard002 <- function(covariates, D, overlap, conditioning) {
  io <- ncol(covariates) == 1                       # did's intercept_only (xformla = ~1)
  if (overlap && did:::overlap_check_fail(covariates, D, io)) stop("overlap")
  if (conditioning && did:::rcond_check_fail(covariates, D, io)) stop("singular")
}
est002 <- function(method, trim) {
  f <- switch(method,
    dripw = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...) {
      guard002(covariates, D, TRUE, TRUE)
      DRDID::drdid_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc, trim.level = trim)
    },
    reg = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...) {
      guard002(covariates, D, FALSE, TRUE)
      DRDID::reg_did_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc)
    },
    stdipw = function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...) {
      guard002(covariates, D, TRUE, FALSE)
      DRDID::std_ipw_did_panel(y1, y0, D, covariates, i.weights = i.weights, boot = FALSE, inffunc = inffunc, trim.level = trim)
    })
  f
}
fit002 <- function(y, X, g, w, control, method, base, trim, lo, hi, sets, B, alp, nthreads) {
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
  est <- est002(method, trim)
  boot <- B > 0
  universal <- base == "universal"
  if (boot) set.seed(20261025)
  ## one estimation; with the bootstrap, att_gt's multiplier bootstrap and simultaneous band
  a <- att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = xf, data = long,
              control_group = if (control == "notyet") "notyettreated" else "nevertreated",
              weightsname = if (is.null(w)) NULL else "w", base_period = base, est_method = est,
              bstrap = boot, cband = boot, biters = if (boot) B else 1000, alp = alp, panel = TRUE,
              print_details = FALSE)
  agg <- function(type, bs, ...) aggte(a, type = type, na.rm = TRUE, bstrap = bs, cband = bs,
                                       biters = if (bs) B else 1000, alp = alp, ...)
  s <- agg("simple", FALSE); gr <- agg("group", FALSE); ca <- agg("calendar", FALSE)
  e <- agg("dynamic", FALSE, min_e = lo, max_e = hi)
  ## the cells' analytic standard errors: att_gt's own, or (after its bootstrap, which replaces
  ## them) its formula on the stored influence functions, sqrt(diag(IF'IF / n) / n)
  se_a <- a$se
  if (boot) {
    se_a <- sqrt(Matrix::colSums(a$inffunc^2)) / nrow(a$inffunc)
    se_a[se_a <= sqrt(.Machine$double.eps) * 10] <- NA
  }
  ## under a universal base did also returns the reference cells (g, g - 1) and e = -1, at 0; the
  ## task does not estimate them
  keep <- !is.na(a$att) & !(universal & a$t == a$group - 1)
  ke <- function(egt) !(universal & egt == -1)
  k1 <- ke(e$egt)
  IFe <- e$inf.function$dynamic.inf.func.e[, k1, drop = FALSE]
  egt <- e$egt[k1]; att_e <- e$att.egt[k1]
  avg <- function(cols) {
    if (length(cols) == 0) return(c(NA_real_, NA_real_))
    f <- rowMeans(IFe[, cols, drop = FALSE])
    c(mean(att_e[cols]), sqrt(mean(f^2) / N))
  }
  cov_e <- crossprod(IFe) / N^2
  pre <- which(egt < 0); post <- which(egt >= 0)
  set_out <- matrix(NA_real_, nrow = 0, ncol = 2)
  if (length(sets) > 0) {
    lo2 <- min(c(lo, unlist(sets))); hi2 <- max(c(hi, unlist(sets)))
    e2 <- agg("dynamic", FALSE, min_e = lo2, max_e = hi2)
    k2 <- ke(e2$egt)
    IF2 <- e2$inf.function$dynamic.inf.func.e[, k2, drop = FALSE]
    egt2 <- e2$egt[k2]; att2 <- e2$att.egt[k2]
    for (s_ in sets) {
      cols <- which(egt2 %in% s_)
      f <- rowMeans(IF2[, cols, drop = FALSE])
      set_out <- rbind(set_out, c(mean(att2[cols]), sqrt(mean(f^2) / N)))
    }
  }
  out <- list(group = a$group[keep], t = a$t[keep], att = a$att[keep], se = se_a[keep],
       simple = c(s$overall.att, s$overall.se),
       gid = gr$egt, att_g = gr$att.egt, se_g = gr$se.egt, gavg = c(gr$overall.att, gr$overall.se),
       cid = ca$egt, att_c = ca$att.egt, se_c = ca$se.egt, cavg = c(ca$overall.att, ca$overall.se),
       egt = egt, att_e = att_e, se_e = e$se.egt[k1], cov_e = cov_e, pre = avg(pre), post = avg(post),
       sets = set_out)
  if (boot) {
    out$se_boot_gt <- a$se[keep]; out$crit_gt <- a$c
    bs <- agg("simple", TRUE); bg <- agg("group", TRUE); bc <- agg("calendar", TRUE)
    be <- agg("dynamic", TRUE, min_e = lo, max_e = hi)
    out$se_boot_simple <- bs$overall.se
    out$se_boot_group <- bg$se.egt; out$crit_group <- bg$crit.val.egt; out$se_boot_group_avg <- bg$overall.se
    out$se_boot_calendar <- bc$se.egt; out$crit_calendar <- bc$crit.val.egt
    out$se_boot_calendar_avg <- bc$overall.se
    kb <- ke(be$egt)
    out$se_boot_event <- be$se.egt[kb]; out$crit_event <- be$crit.val.egt
    ## the averages' bootstrap standard errors: did's multiplier bootstrap of their influence
    ## functions, as aggte bootstraps its overall averages
    IFb <- be$inf.function$dynamic.inf.func.e[, kb, drop = FALSE]; egb <- be$egt[kb]
    bavg <- function(cols) if (length(cols) == 0) NA_real_ else
      mboot(as.matrix(rowMeans(IFb[, cols, drop = FALSE])), DIDparams = be$DIDparams)$se
    out$se_boot_pre_avg <- bavg(which(egb < 0)); out$se_boot_post_avg <- bavg(which(egb >= 0))
  }
  out
}
''')
_fit = ro.r["fit002"]
METHODS = ("dripw", "reg", "stdipw")


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5), base_period="varying",
        trim=1.0, boot_weights=None, alpha=0.05, event_sets=None, rules="csdid181"):
    if rules != "csdid201":
        raise ValueError("this baseline follows csdid 2.0.1's (R did's) rules (rules='csdid201')")
    if method not in METHODS:
        raise ValueError(f"{method} is not a csdid 2.0.1 estimator")
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("unbalanced panels: did does not estimate csdid's pair-balanced cells")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        raise ValueError("covariates that vary over time are not supported by this adapter")
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    N = y.shape[0]
    X = X.reshape(N, -1)
    B = 0 if boot_weights is None else int(np.asarray(boot_weights).shape[1])
    sets = [sorted(set(int(v) for v in s)) for s in (event_sets or [])]
    with localconverter(ro.default_converter + numpy2ri.converter):
        conv = ro.conversion.get_conversion().py2rpy
        args = [conv(np.asfortranarray(y)), conv(np.asfortranarray(X)), conv(np.asarray(gvar).astype(np.int32)),
                ro.NULL if w is None else conv(np.asarray(w, dtype=np.float64))]
    rsets = ro.r["list"](*[ro.IntVector(s) for s in sets])
    r = _fit(*args, control, method, base_period, float(trim), int(event_window[0]), int(event_window[1]),
             rsets, B, float(alpha), threads)
    f = lambda name: np.asarray(r.rx2(name), dtype=np.float64)
    gt = np.column_stack([f("group"), f("t")]).astype(np.int64)
    pre, post, simple, gavg, cavg = f("pre"), f("post"), f("simple"), f("gavg"), f("cavg")
    so = f("sets").reshape(-1, 2) if sets else np.zeros((0, 2))
    out = dict(gt=gt, att_gt=f("att"), se_gt=f("se"), att_simple=float(simple[0]), se_simple=float(simple[1]),
               group_ids=f("gid").astype(np.int64), att_group=f("att_g"), se_group=f("se_g"),
               att_group_avg=float(gavg[0]), se_group_avg=float(gavg[1]),
               calendar_periods=f("cid").astype(np.int64), att_calendar=f("att_c"), se_calendar=f("se_c"),
               att_calendar_avg=float(cavg[0]), se_calendar_avg=float(cavg[1]),
               event_times=f("egt").astype(np.int64), att_event=f("att_e"), se_event=f("se_e"),
               cov_event=f("cov_e"), att_pre_avg=float(pre[0]), se_pre_avg=float(pre[1]),
               att_post_avg=float(post[0]), se_post_avg=float(post[1]), event_sets=sets,
               att_event_sets=so[:, 0].copy(), se_event_sets=so[:, 1].copy())
    if B:
        for name in ("se_boot_gt", "se_boot_group", "se_boot_calendar", "se_boot_event"):
            out[name] = f(name)
        for name in ("crit_gt", "crit_group", "crit_calendar", "crit_event", "se_boot_simple", "se_boot_group_avg",
                     "se_boot_calendar_avg", "se_boot_pre_avg", "se_boot_post_avg"):
            out[name] = float(f(name)[0]) if len(f(name)) else float("nan")
    return out
