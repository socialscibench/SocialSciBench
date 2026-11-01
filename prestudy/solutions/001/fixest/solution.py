"""Task 001 competitor: R fixest 0.14.2 feols, called in-process through rpy2 3.5.17.

Follows the 14 October 2026 study's Round 1 adapter (research/2026-10-14/task001/benchmark.R
and fast_ssc.py in the companion repository): a default `feols(..., cluster = ~cl)` fit, the
unadjusted sandwich recovered from fixest's own small-sample factor, and the task's exact
correction q = (N-1)/(N-K) * G/(G-1) with K = k + 1 + df_a under the reference's df rule.
`fixef.rm = "singleton"` is set because fixest keeps singletons by default and the task drops
them. Everything, including the Python-to-R copies, is inside the timed call.
"""
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

os.environ.setdefault("R_HOME", "/Library/Frameworks/R.framework/Resources")
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import numpy2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402

R_LIBS = os.environ.get("SOCIALSCIBENCH_R_LIBS",
                        "/path/to/repro-socialscibench/research/2026-10-14/R-library")
ro.r(f'.libPaths(c("{R_LIBS}", .libPaths())); suppressPackageStartupMessages(library(fixest))')
ro.r(f'setFixest_nthreads({int(os.environ.get("OMP_NUM_THREADS", "1"))})')
_fit = ro.r("""
function(y, X, ids, cl) {
  k <- ncol(X); nf <- ncol(ids)
  df <- data.frame(y = y, X, cl = cl, ids)
  names(df) <- c("y", paste0("x", seq_len(k)), "cl", paste0("fe", seq_len(nf)))
  fml <- as.formula(paste("y ~", paste0("x", seq_len(k), collapse = "+"), "|",
                          paste0("fe", seq_len(nf), collapse = "+")))
  m <- feols(fml, df, cluster = ~cl, fixef.rm = "singleton", notes = FALSE)
  V <- vcov(m, attr = TRUE)
  n <- nobs(m)
  G <- attr(V, "G")
  K <- attr(V, "df.K")
  if (is.null(K)) {
    raw <- vcov(m, ssc = ssc(K.adj = FALSE, G.adj = FALSE))
  } else {
    raw <- V / ((n - 1) / (n - K) * G / (G - 1))
  }
  list(beta = unname(coef(m)), raw = unname(as.matrix(raw)), nobs = n, G = G)
}
""")


def _recode(ids):
    return np.unique(ids, return_inverse=True)[1].astype(np.int64)


def _drop_singletons(fe):
    keep = np.ones(fe.shape[0], dtype=bool)
    while True:
        changed = False
        for g in range(fe.shape[1]):
            counts = np.bincount(fe[keep, g], minlength=fe[:, g].max() + 1)
            single = keep & (counts[fe[:, g]] == 1)
            if single.any():
                keep &= ~single
                changed = True
        if not changed:
            return keep


def _nested_in(fe, cluster):
    representative = np.empty(int(fe.max()) + 1, dtype=cluster.dtype)
    representative[fe] = cluster
    return bool(np.all(representative[fe] == cluster))


def _absorbed_df(fe_ids, cluster_ids):
    df_a, seen = 0, []
    for g in range(fe_ids.shape[1]):
        fe = fe_ids[:, g]
        L = int(fe.max()) + 1
        if _nested_in(fe, cluster_ids):
            continue
        if not seen:
            df_a += L
        elif len(seen) == 1:
            a = seen[0]
            La = int(a.max()) + 1
            graph = coo_matrix((np.ones(a.size), (a, fe + La)), shape=(La + L, La + L))
            n_comp, _ = connected_components(graph, directed=False)
            df_a += L - n_comp
        else:
            levels = [int(h.max()) + 1 for h in seen if _nested_in(fe, h)]
            df_a += L - max([1] + levels)
        seen.append(fe)
    return df_a - 1 if seen else 0


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    fe0 = np.column_stack([_recode(fe_ids[:, j]) for j in range(fe_ids.shape[1])])
    cl0 = _recode(cluster_ids)
    with localconverter(ro.default_converter + numpy2ri.converter):
        out = _fit(np.ascontiguousarray(y, dtype=float), np.asfortranarray(X, dtype=float),
                   np.asfortranarray(fe0, dtype=np.int32), cl0.astype(np.int32))
        beta = np.asarray(out["beta"], dtype=float)
        raw = np.asarray(out["raw"], dtype=float)
        n_fixest, G_fixest = int(np.asarray(out["nobs"])[0]), int(np.asarray(out["G"])[0])
    keep = _drop_singletons(fe0)
    fe = fe0 if keep.all() else np.column_stack([_recode(fe0[keep, j]) for j in range(fe0.shape[1])])
    cl = _recode(cl0[keep])
    n, k, G = int(keep.sum()), X.shape[1], int(cl.max()) + 1
    if (n, G) != (n_fixest, G_fixest):
        raise RuntimeError(f"fixest sample differs from the task's: nobs {n_fixest} vs {n}, clusters {G_fixest} vs {G}")
    df_a = int(_absorbed_df(fe, cl))
    q = (n - 1) / (n - k - 1 - df_a) * G / (G - 1)
    return dict(beta=beta, vcov=raw * q, nobs=n, df_a=df_a, n_clusters=G)
