"""Task 001 baseline: R fixest::feols (Berge, Butts & McDermott), called in-process through rpy2.

    feols(y ~ x1 + ... + xk | fe1 + ... + feG, cluster = ~cl, fixef.rm = "perfect_fit",
          fixef.tol = TOL, ssc = ssc(K.adj = FALSE, G.adj = FALSE), nthreads = N, lean = TRUE)

fixest drops singleton fixed-effect levels recursively (its default), partials out the fixed
effects (its accelerated fixed-point algorithm) and returns the coefficients and the unadjusted
clustered sandwich. The task's outputs follow reghdfe's conventions, which fixest does not
report: df_a (absorbed degrees of freedom: nesting in the cluster variable, connected components
of the first two non-nested fixed effects, nesting among the others) is computed here with
NumPy/SciPy/pandas, and the sandwich is multiplied by the reference's small-sample factor
(n - 1) / (n - K) * G / (G - 1), K = k + 1 + df_a. The array transfers to R, this computation
and the conversion of the results are inside the timed call.

Threads: fixest and data.table use N = the evaluator's thread limit (OMP_NUM_THREADS).
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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

TOL = 1e-8

ro.r('suppressMessages({library(fixest); library(data.table)})')
ro.r(r'''
fit001 <- function(y, X, fe, cl, nthreads, tol) {
  setFixest_nthreads(nthreads); setDTthreads(nthreads)
  y <- as.numeric(y); cl <- as.integer(cl)
  k <- ncol(X); G <- ncol(fe)
  colnames(X) <- paste0("x", seq_len(k)); colnames(fe) <- paste0("fe", seq_len(G))
  df <- data.table(y = y, X, fe, cl = cl)
  fml <- as.formula(paste("y ~", paste(colnames(X), collapse = " + "), "|", paste(colnames(fe), collapse = " + ")))
  m <- feols(fml, df, cluster = ~cl, ssc = ssc(K.adj = FALSE, G.adj = FALSE), fixef.tol = tol,
             fixef.iter = 100000L, nthreads = nthreads, lean = TRUE, notes = FALSE, warn = FALSE)
  removed <- m$obs_selection$obsRemoved
  if (is.null(removed)) removed <- integer(0)
  list(coef = unname(coef(m)), vcov = unname(as.matrix(vcov(m))), nobs = nobs(m), removed = -as.integer(removed))
}
''')
_fit = ro.r["fit001"]


def _codes(a):
    return pd.factorize(a, sort=False)[0]


def _nested(a, b):
    """Every level of a lies within one level of b."""
    return bool((pd.Series(b).groupby(a).nunique() == 1).all())


def _absorbed_df(fe, cl):
    """reghdfe's df_a, as in reference.absorbed_df (codes of the kept rows)."""
    df_a = 0
    seen = []
    for g in range(fe.shape[1]):
        a = fe[:, g]
        La = int(a.max()) + 1
        if _nested(a, cl):
            continue
        if not seen:
            df_a += La
        elif len(seen) == 1:
            h = seen[0]
            Lh = int(h.max()) + 1
            graph = coo_matrix((np.ones(a.size), (h, a + Lh)), shape=(Lh + La, Lh + La))
            df_a += La - connected_components(graph, directed=False)[0]
        else:
            nested_levels = [int(h.max()) + 1 for h in seen if _nested(a, h)]
            df_a += La - max([1] + nested_levels)
        seen.append(a)
    return df_a - 1 if seen else 0


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    n0 = len(y)
    X = np.asarray(X, dtype=np.float64).reshape(n0, -1)
    fe = np.asarray(fe_ids).reshape(n0, -1)
    if np.abs(fe).max(initial=0) >= 2**31 or np.abs(cluster_ids).max(initial=0) >= 2**31:
        raise ValueError("fixed-effect codes must fit in R integers")
    with localconverter(ro.default_converter + numpy2ri.converter):
        args = [ro.conversion.get_conversion().py2rpy(a) for a in (
            np.ascontiguousarray(y, dtype=np.float64), np.asfortranarray(X),
            np.asfortranarray(fe.astype(np.int32)), np.asarray(cluster_ids).astype(np.int32))]
    res = _fit(*args, threads, TOL)
    beta = np.asarray(res.rx2("coef"), dtype=np.float64)
    vcov = np.asarray(res.rx2("vcov"), dtype=np.float64).reshape(beta.size, beta.size, order="F")
    keep = np.ones(n0, bool)
    removed = np.asarray(res.rx2("removed"), dtype=np.int64)
    if removed.size:
        keep[removed - 1] = False
    n = int(keep.sum())
    k = X.shape[1]
    codes = np.column_stack([_codes(fe[keep, g]) for g in range(fe.shape[1])])
    cl = _codes(np.asarray(cluster_ids)[keep])
    Gc = int(cl.max()) + 1
    df_a = _absorbed_df(codes, cl)
    q = (n - 1) / (n - (k + 1 + df_a)) * Gc / (Gc - 1)
    return dict(beta=beta, vcov=q * vcov, nobs=n, df_a=int(df_a), n_clusters=Gc)
