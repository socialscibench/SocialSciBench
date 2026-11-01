"""Task 002 competitor: R did 2.5.1 att_gt (faster_mode) and aggte, in-process via rpy2.

Follows the 14 October 2026 study's `R_did_fast` engine (research/2026-10-14/task002/
bench_did.R in the companion repository): never-treated controls, varying base period,
traditional doubly robust estimator, analytic standard errors, no bootstrap, one core.
`aggte(type = "simple")` and `aggte(type = "dynamic")` give the simple and event-time results;
the equal-weight pre-period average and its standard error come from the exported event-time
influence functions, as in the study. The long data frame did requires is built in R from the
task's wide inputs inside the timed call (the study loaded it from a file outside its timer).
"""
import os

import numpy as np

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import numpy2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402

R_LIBS = os.environ.get("SOCIALSCIBENCH_R_LIBS",
                        "/path/to/repro-socialscibench/research/2026-10-14/R-library")
ro.r(f'.libPaths(c("{R_LIBS}", .libPaths())); suppressPackageStartupMessages(library(did))')
ro.r("data.table::setDTthreads(1)")
_fit = ro.r("""
function(Y, X, G) {
  N <- nrow(Y); TT <- ncol(Y); k <- ncol(X)
  d <- data.frame(id = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N),
                  y = as.vector(t(Y)), g = rep(G, each = TT))
  for (j in seq_len(k)) d[[paste0("x", j)]] <- rep(X[, j], each = TT)
  formula <- reformulate(paste0("x", seq_len(k)))
  fit <- did::att_gt(yname = "y", tname = "t", idname = "id", gname = "g", xformla = formula,
    data = d, panel = TRUE, control_group = "nevertreated", base_period = "varying",
    est_method = "dr", bstrap = FALSE, cband = FALSE, faster_mode = TRUE, cores = 1, pl = FALSE)
  simple <- did::aggte(fit, type = "simple", bstrap = FALSE, cband = FALSE)
  event <- did::aggte(fit, type = "dynamic", bstrap = FALSE, cband = FALSE)
  pre <- event$egt < 0
  ipre <- rowMeans(event$inf.function$dynamic.inf.func.e[, pre, drop = FALSE])
  list(gt = cbind(fit$group, fit$t), att_gt = fit$att, se_gt = fit$se,
    att_simple = simple$overall.att, se_simple = simple$overall.se,
    event_times = event$egt, att_event = event$att.egt, se_event = event$se.egt,
    att_pre_avg = mean(event$att.egt[pre]),
    se_pre_avg = sqrt(sum((ipre - mean(ipre))^2)) / length(ipre),
    att_post_avg = event$overall.att, se_post_avg = event$overall.se)
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
