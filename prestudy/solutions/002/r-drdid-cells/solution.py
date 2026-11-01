"""Task 002 competitor: R DRDID 1.3.0 panel cells with task aggregation, in-process via rpy2.

Follows the 14 October 2026 study's `R_DRDID_adapter` (research/2026-10-14/task002/bench_did.R
in the companion repository), its fastest engine on W3 and W4: DRDID::drdid_panel for every
(cohort, period) cell with 0.995 trimming, the cell influence functions rescaled to the full
sample, and the task's aggregation. The study built the wide outcome matrix from long data
outside its timer; here the task's wide inputs are passed directly, and the copies into R are
inside the timed call.
"""
import os

import numpy as np

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import numpy2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402

R_LIBS = os.environ.get("SOCIALSCIBENCH_R_LIBS",
                        "/path/to/repro-socialscibench/research/2026-10-14/R-library")
ro.r(f'.libPaths(c("{R_LIBS}", .libPaths())); suppressPackageStartupMessages(library(DRDID))')
_fit = ro.r("""
function(Y, X, G) {
  aggregate_if <- function(rif, wgt) {
    ma <- colMeans(rif); mw <- colMeans(wgt); sw <- sum(mw); att <- sum(ma * mw) / sw
    as.numeric(sweep(rif, 2, ma) %*% (mw / sw) + sweep(wgt, 2, mw) %*% ((ma - att) / sw)) + att
  }
  se <- function(rif) sqrt(colSums(sweep(rif, 2, colMeans(rif))^2)) / nrow(rif)
  N <- nrow(Y); TT <- ncol(Y)
  X <- cbind(X, 1)
  cohorts <- sort(unique(G[G > 0]))
  ngt <- length(cohorts) * (TT - 1L)
  gt <- matrix(0L, nrow = ngt, ncol = 2)
  rif <- matrix(0, N, ngt)
  wgt <- matrix(0, N, ngt)
  j <- 0L
  for (g in cohorts) {
    base <- 1L
    for (tt in 2:TT) {
      j <- j + 1L
      s <- G == 0 | G == g
      fit <- DRDID::drdid_panel(y1 = Y[s, tt], y0 = Y[s, base], D = as.numeric(G[s] == g),
        covariates = X[s, , drop = FALSE], boot = FALSE, inffunc = TRUE, trim.level = .995)
      rif[, j] <- fit$ATT
      cif <- fit$att.inf.func
      rif[s, j] <- fit$ATT + (cif - mean(cif)) * N / sum(s)
      wgt[, j] <- as.numeric(G == g)
      gt[j, ] <- c(g, tt)
      if (tt < g) base <- tt
    }
  }
  e <- gt[, 2] - gt[, 1]
  ev <- sort(unique(e))
  post <- e >= 0
  rs <- aggregate_if(rif[, post, drop = FALSE], wgt[, post, drop = FALSE])
  re <- sapply(ev, function(k) aggregate_if(rif[, e == k, drop = FALSE], wgt[, e == k, drop = FALSE]))
  rpre <- rowMeans(re[, ev < 0, drop = FALSE])
  rpost <- rowMeans(re[, ev >= 0, drop = FALSE])
  list(gt = gt, att_gt = colMeans(rif), se_gt = se(rif), att_simple = mean(rs),
    se_simple = se(matrix(rs, ncol = 1)), event_times = ev, att_event = colMeans(re), se_event = se(re),
    att_pre_avg = mean(rpre), se_pre_avg = se(matrix(rpre, ncol = 1)),
    att_post_avg = mean(rpost), se_post_avg = se(matrix(rpost, ncol = 1)))
}
""")

SCALARS = ("att_simple", "se_simple", "att_pre_avg", "se_pre_avg", "att_post_avg", "se_post_avg")


def run(y, X, gvar):
    with localconverter(ro.default_converter + numpy2ri.converter):
        out = _fit(np.asfortranarray(y, dtype=float), np.asfortranarray(X, dtype=float),
                   np.asarray(gvar, dtype=np.int32))
        res = {k: np.asarray(v) for k, v in out.items()}
    res = {k: (float(v.reshape(-1)[0]) if k in SCALARS else v) for k, v in res.items()}
    res["gt"] = res["gt"].astype(np.int64).reshape(-1, 2)
    res["event_times"] = res["event_times"].astype(np.int64).reshape(-1)
    for k in ("att_gt", "se_gt", "att_event", "se_event"):
        res[k] = res[k].astype(float).reshape(-1)
    return res
